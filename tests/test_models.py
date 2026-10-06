"""Shape / correctness tests for the neural components (Eqs. 1-10)."""

import math

import numpy as np
import pytest
import torch

from k12agent.content.llama_layers import (GroupedQueryAttention, HierarchicalRotaryEmbedding, RotaryEmbedding,
                                           SwiGLU, TinyLlama, apply_rope)
from k12agent.speech.ge2e import GE2ELoss, SpeakerEncoder
from k12agent.speech.prosody_decoder import ConcatMLPAttention, ProsodyAwareDecoder
from k12agent.video.alignment import (ModalityDiscriminator, discriminator_loss, dtw, dtw_retime, shift_sequence,
                                      windowed_sync_loss)
from k12agent.video.models import EnhancedLipGenerator
from k12agent.video.syncnet import EnhancedSyncNet, av_offset


def test_rope_is_rotation_and_relative():
    d = 8
    rope = RotaryEmbedding(d)
    q = torch.randn(1, 1, 1, d)
    k = torch.randn(1, 1, 1, d)
    # norm preserved (R is orthogonal)
    qr = apply_rope(q, rope.angles(torch.tensor([5])))
    assert torch.allclose(qr.norm(), q.norm(), atol=1e-5)
    # q_m . k_n depends only on m - n
    s1 = (apply_rope(q, rope.angles(torch.tensor([7]))) * apply_rope(k, rope.angles(torch.tensor([3])))).sum()
    s2 = (apply_rope(q, rope.angles(torch.tensor([14]))) * apply_rope(k, rope.angles(torch.tensor([10])))).sum()
    assert torch.allclose(s1, s2, atol=1e-4)


def test_swiglu_matches_equation():
    m = SwiGLU(16, hidden_dim=32)
    x = torch.randn(4, 16)
    ref = m.w2(m.w(x) * torch.sigmoid(m.w(x)) * m.v(x))
    assert torch.allclose(m(x), ref, atol=1e-6)


def test_gqa_and_tiny_llama():
    attn = GroupedQueryAttention(64, n_heads=8, n_kv_heads=2)
    assert attn(torch.randn(2, 10, 64)).shape == (2, 10, 64)
    lm = TinyLlama(100, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, hierarchical_rope=True)
    tokens = torch.randint(0, 100, (2, 12))
    steps = torch.arange(12).div(4, rounding_mode="floor").expand(2, -1)
    assert lm(tokens, steps).shape == (2, 12, 100)
    assert isinstance(lm.layers[0].attn.rope, HierarchicalRotaryEmbedding)


def test_ge2e_loss_prefers_clustered_embeddings():
    loss = GE2ELoss()
    n, m, d = 4, 5, 32
    centres = torch.nn.functional.normalize(torch.randn(n, 1, d), dim=-1)
    clustered = torch.nn.functional.normalize(centres + 0.05 * torch.randn(n, m, d), dim=-1)
    random = torch.nn.functional.normalize(torch.randn(n, m, d), dim=-1)
    assert loss(clustered) < loss(random)
    enc = SpeakerEncoder(n_mels=40)
    e = enc(torch.randn(3, 50, 40))
    assert e.shape == (3, 256) and torch.allclose(e.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_prosody_attention_eq5():
    att = ConcatMLPAttention(16, 8)
    c, a = att(torch.randn(2, 16), torch.randn(2, 7, 8))
    assert c.shape == (2, 8) and torch.allclose(a.sum(-1), torch.ones(2), atol=1e-5)
    dec = ProsodyAwareDecoder(memory_dim=24, n_mels=10, prenet_dim=16, rnn_dim=32, prosody_dim=8)
    mels, stops, al = dec(torch.randn(2, 6, 24), torch.randn(2, 5, 10), torch.randn(2, 20), torch.randn(2, 20))
    assert mels.shape == (2, 5, 10) and stops.shape == (2, 5) and al.shape == (2, 5, 6)


def test_enhanced_generator_shapes():
    gen = EnhancedLipGenerator()
    out, f_a, f_v, alpha = gen(torch.randn(2, 80, 16), torch.rand(2, 5, 6, 96, 96))
    assert out.shape == (2, 5, 3, 96, 96)
    assert f_a.shape == f_v.shape == (2, 5, 512)        # Eqs. (6)-(7): R^{T x 512}
    assert torch.allclose(alpha.sum(-1), torch.ones(2, 5), atol=1e-5)  # Eq. (8)


def test_syncnet_and_offset():
    net = EnhancedSyncNet().eval()
    a, v = net(torch.randn(3, 1, 80, 16), torch.rand(3, 15, 48, 96))
    assert a.shape == v.shape == (3, 512)
    off, conf = av_offset(net, torch.randn(12, 1, 80, 16), torch.rand(12, 15, 48, 96), max_shift=3)
    assert -120 <= off <= 120 and math.isfinite(conf)


def test_sync_losses():
    a = torch.randn(2, 8, 16)
    assert windowed_sync_loss(a, a, 2).item() == pytest.approx(0.0, abs=1e-7)
    assert windowed_sync_loss(a, a + 1.0, 2).item() == pytest.approx(16.0, rel=1e-5)  # ||1||^2 over 16 dims
    D = ModalityDiscriminator(16)
    v_async = shift_sequence(a, min_shift=2)
    assert discriminator_loss(D, a.flatten(0, 1), a.flatten(0, 1), v_async.flatten(0, 1)).item() > 0


def test_dtw_recovers_shift():
    t = np.linspace(0, 6 * np.pi, 80)
    audio = np.sin(t) + 0.3 * np.sin(3 * t)
    video = np.roll(audio, 2)                  # video lags by 2 frames
    cost, path = dtw(np.abs(audio[:, None] - video[None, :]))
    assert path[0].tolist() == [0, 0] and path[-1].tolist() == [79, 79]
    mapping = dtw_retime(audio, video, max_offset_frames=3)
    assert np.all(np.diff(mapping) >= 0)
    assert np.median(mapping[10:70] - np.arange(10, 70)) == 2
