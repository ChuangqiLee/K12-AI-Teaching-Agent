"""Reference implementation of the LLaMA building blocks used in Section 3.1.

* Rotary position encoding, Eqs. (1)-(2):  q_m = (W_q x_m) R(m theta),  k_n = (W_k x_n) R(n theta)
* SwiGLU gated activation, Eq. (3):         SwiGLU(x) = Swish_beta(W x) * (V x),  beta = 1
* Grouped multi-query attention (Figure 2) and RMSNorm.

The production model is LLaMA-2-13B loaded through ``transformers`` (which
already contains these blocks); this module makes the equations in the paper
explicit, is used by the unit tests, and can be used to train small models.

``HierarchicalRotaryEmbedding`` is our reading of "hierarchical RoPE": the
paper only gives Eqs. (1)-(2), so the two-level variant below (half of the
frequency pairs rotate with the global token index, the other half with the
index of the reasoning step the token belongs to) is a documented
reconstruction choice and is disabled by default.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_frequencies(head_dim: int, base: float = 10000.0) -> torch.Tensor:
    """theta_i = base^(-2(i-1)/d), i = 1..d/2."""
    return base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Apply the 2x2 rotation R(m theta) = [[cos, -sin], [sin, cos]] to each feature pair.

    x: (B, H, T, D); angles: (T, D/2) or (B, 1, T, D/2).
    """
    cos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1)
    sin = torch.repeat_interleave(torch.sin(angles), 2, dim=-1)
    return x * cos + rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        self.register_buffer("inv_freq", rope_frequencies(head_dim, base), persistent=False)

    def angles(self, positions: torch.Tensor) -> torch.Tensor:
        return positions.float()[..., None] * self.inv_freq  # (..., T, D/2)

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor,
                step_ids: Optional[torch.Tensor] = None):
        a = self.angles(positions)
        if a.dim() == 3:  # batched positions -> broadcast over heads
            a = a[:, None]
        return apply_rope(q, a), apply_rope(k, a)


class HierarchicalRotaryEmbedding(RotaryEmbedding):
    """Two-level RoPE: global token position + reasoning-step position (reconstruction)."""

    def __init__(self, head_dim: int, base: float = 10000.0, step_base: float = 100.0):
        super().__init__(head_dim, base)
        n = head_dim // 2
        self.n_global = n - n // 2
        self.register_buffer("step_freq", step_base ** (-torch.arange(0, n // 2, dtype=torch.float32) / max(1, n // 2)),
                             persistent=False)

    def forward(self, q, k, positions, step_ids=None):
        if step_ids is None:
            return super().forward(q, k, positions)
        positions = positions.expand_as(step_ids)
        g = positions.float()[..., None] * self.inv_freq[: self.n_global]
        s = step_ids.float()[..., None] * self.step_freq
        a = torch.cat([g, s], dim=-1)
        if a.dim() == 3:
            a = a[:, None]
        return apply_rope(q, a), apply_rope(k, a)


class SwiGLU(nn.Module):
    """FFN(x) = W_2 (Swish_beta(W x) * V x), Eq. (3)."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, beta: float = 1.0, multiple_of: int = 256):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * dim) / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w = nn.Linear(dim, hidden_dim, bias=False)
        self.v = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.beta = beta

    @staticmethod
    def swish(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        return x * torch.sigmoid(beta * x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.swish(self.w(x), self.beta) * self.v(x))


class GroupedQueryAttention(nn.Module):
    """Multi-head attention with n_kv_heads <= n_heads shared key/value heads."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: Optional[int] = None,
                 rope: Optional[RotaryEmbedding] = None):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        assert n_heads % self.n_kv_heads == 0
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.rope = rope or RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None,
                step_ids: Optional[torch.Tensor] = None, causal: bool = True) -> torch.Tensor:
        b, t, _ = x.shape
        if positions is None:
            positions = torch.arange(t, device=x.device)
        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, positions, step_ids)
        rep = self.n_heads // self.n_kv_heads
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if causal:
            mask = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
        out = F.softmax(scores.float(), dim=-1).type_as(q) @ v
        return self.wo(out.transpose(1, 2).reshape(b, t, -1))


class LlamaBlock(nn.Module):
    """Pre-norm decoder block: x + Attn(RMSNorm(x)); x + SwiGLU(RMSNorm(x))."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: Optional[int] = None,
                 hierarchical_rope: bool = False):
        super().__init__()
        head_dim = dim // n_heads
        rope = HierarchicalRotaryEmbedding(head_dim) if hierarchical_rope else RotaryEmbedding(head_dim)
        self.attn_norm = RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, n_heads, n_kv_heads, rope)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim)

    def forward(self, x, positions=None, step_ids=None):
        x = x + self.attn(self.attn_norm(x), positions, step_ids)
        return x + self.ffn(self.ffn_norm(x))


class TinyLlama(nn.Module):
    """Small LLaMA-style LM for experiments / tests (same blocks as LLaMA-13B)."""

    def __init__(self, vocab_size: int, dim: int = 256, n_layers: int = 4, n_heads: int = 8,
                 n_kv_heads: Optional[int] = 2, hierarchical_rope: bool = False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(LlamaBlock(dim, n_heads, n_kv_heads, hierarchical_rope) for _ in range(n_layers))
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, step_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed(tokens)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        for layer in self.layers:
            x = layer(x, positions, step_ids)
        return self.lm_head(self.norm(x))
