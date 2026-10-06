"""Prosody-aware Tacotron-2 style decoder components (Section 3.2, Eq. 5, Figure 4).

    c_t = sum_i a_{t,i} e_i ,     a_{t,i} = softmax_i( MLP(h_t || e_i) )

h_t is the decoder hidden state, e_i the i-th encoder output (text encoding
concatenated with the speaker embedding).  A dynamic prosody-control branch
encodes the pitch contour and energy of the reference audio and conditions the
decoder (Figure 4: "Dynamic prosody control", "Pitch Contour").
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatMLPAttention(nn.Module):
    """Eq. (5): additive attention computed by an MLP over [h_t ; e_i]."""

    def __init__(self, query_dim: int, memory_dim: int, attn_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(query_dim + memory_dim, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1, bias=False))

    def forward(self, h_t: torch.Tensor, memory: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """h_t: (B, Dq); memory e: (B, L, Dm) -> context c_t (B, Dm), weights a_t (B, L)."""
        L = memory.shape[1]
        energy = self.mlp(torch.cat([h_t[:, None].expand(-1, L, -1), memory], dim=-1)).squeeze(-1)
        if mask is not None:
            energy = energy.masked_fill(~mask, float("-inf"))
        a_t = F.softmax(energy, dim=-1)
        c_t = torch.bmm(a_t[:, None], memory).squeeze(1)
        return c_t, a_t


class ProsodyEncoder(nn.Module):
    """Encodes frame-level (log-F0, energy) of the reference into a prosody vector."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(2, 64, 3, padding=1), nn.ReLU(), nn.Conv1d(64, 64, 3, padding=1), nn.ReLU())
        self.gru = nn.GRU(64, out_dim, batch_first=True)

    def forward(self, pitch: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        x = self.conv(torch.stack([pitch, energy], dim=1)).transpose(1, 2)
        _, h = self.gru(x)
        return h[-1]


class ProsodyAwareDecoder(nn.Module):
    """Autoregressive mel decoder with Eq. (5) attention and prosody conditioning."""

    def __init__(self, memory_dim: int = 512 + 256, n_mels: int = 80, prenet_dim: int = 256,
                 rnn_dim: int = 1024, prosody_dim: int = 64):
        super().__init__()
        self.n_mels = n_mels
        self.prenet = nn.Sequential(nn.Linear(n_mels, prenet_dim), nn.ReLU(), nn.Dropout(0.5),
                                    nn.Linear(prenet_dim, prenet_dim), nn.ReLU(), nn.Dropout(0.5))
        self.rnn = nn.LSTMCell(prenet_dim + memory_dim + prosody_dim, rnn_dim)
        self.attention = ConcatMLPAttention(rnn_dim, memory_dim)
        self.mel_proj = nn.Linear(rnn_dim + memory_dim, n_mels)
        self.stop_proj = nn.Linear(rnn_dim + memory_dim, 1)
        self.prosody = ProsodyEncoder(prosody_dim)
        self.prosody_dim = prosody_dim

    def forward(self, memory: torch.Tensor, mel_targets: torch.Tensor,
                pitch: Optional[torch.Tensor] = None, energy: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None):
        """Teacher-forced decoding. memory: (B, L, Dm); mel_targets: (B, T, n_mels)."""
        b, t, _ = mel_targets.shape
        p = self.prosody(pitch, energy) if pitch is not None else memory.new_zeros(b, self.prosody_dim)
        h = memory.new_zeros(b, self.rnn.hidden_size)
        c = memory.new_zeros(b, self.rnn.hidden_size)
        ctx = memory.new_zeros(b, memory.shape[-1])
        prev = torch.cat([memory.new_zeros(b, 1, self.n_mels), mel_targets[:, :-1]], dim=1)
        mels, stops, aligns = [], [], []
        for step in range(t):
            x = torch.cat([self.prenet(prev[:, step]), ctx, p], dim=-1)
            h, c = self.rnn(x, (h, c))
            ctx, a = self.attention(h, memory, memory_mask)
            out = torch.cat([h, ctx], dim=-1)
            mels.append(self.mel_proj(out))
            stops.append(self.stop_proj(out).squeeze(-1))
            aligns.append(a)
        return torch.stack(mels, 1), torch.stack(stops, 1), torch.stack(aligns, 1)
