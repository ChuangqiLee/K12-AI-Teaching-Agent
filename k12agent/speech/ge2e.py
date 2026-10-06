"""Speaker encoder trained with the Generalised End-to-End loss (Section 3.2, Eq. 4).

Wan et al. (2018): for utterance embedding e_ji of speaker j and centroid c_k,

    S_ji,k = w * cos(e_ji, c_k) + b             (w > 0, b learnable)
    L(e_ji) = -S_ji,j + log sum_k exp(S_ji,k)   (softmax variant)

where c_j is computed *excluding* e_ji for numerical stability.  Eq. (4) in the
paper is the compact form  L = -1/N sum_i log( exp(s cos h_ii) / sum_j exp(s cos h_ij) ).

The encoder follows the paper's description: a convolutional front-end that
analyses the spectrogram, followed by a recurrent layer and a projection to a
256-d L2-normalised "d-vector" (same output size as Real-Time-Voice-Cloning,
so trained weights can be compared with its pretrained encoder).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerEncoder(nn.Module):
    def __init__(self, n_mels: int = 40, conv_channels: int = 128, hidden: int = 256,
                 embed_dim: int = 256, n_layers: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, conv_channels, 5, padding=2), nn.BatchNorm1d(conv_channels), nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, 5, padding=2), nn.BatchNorm1d(conv_channels), nn.ReLU(),
        )
        self.lstm = nn.LSTM(conv_channels, hidden, num_layers=n_layers, batch_first=True)
        self.proj = nn.Linear(hidden, embed_dim)

    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        """mels: (B, T, n_mels) -> embeddings (B, embed_dim), L2-normalised."""
        x = self.conv(mels.transpose(1, 2)).transpose(1, 2)
        _, (h, _) = self.lstm(x)
        e = F.relu(self.proj(h[-1]))
        return F.normalize(e, p=2, dim=-1)

    @torch.no_grad()
    def embed_utterance(self, mels: torch.Tensor, window: int = 160, hop: int = 80) -> torch.Tensor:
        """Average of sliding-window embeddings (inference-time d-vector). mels: (T, n_mels)."""
        if mels.shape[0] <= window:
            return self(mels[None])[0]
        chunks = [mels[s:s + window] for s in range(0, mels.shape[0] - window + 1, hop)]
        e = self(torch.stack(chunks)).mean(0)
        return F.normalize(e, dim=-1)


class GE2ELoss(nn.Module):
    def __init__(self, init_w: float = 10.0, init_b: float = -5.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def similarity_matrix(self, embeds: torch.Tensor) -> torch.Tensor:
        """embeds: (N speakers, M utterances, D) -> S: (N, M, N)."""
        n, m, _ = embeds.shape
        centroids_incl = F.normalize(embeds.mean(1), dim=-1)                       # (N, D)
        centroids_excl = F.normalize((embeds.sum(1, keepdim=True) - embeds) / (m - 1), dim=-1)  # (N, M, D)
        e = F.normalize(embeds, dim=-1)
        sim = torch.einsum("nmd,kd->nmk", e, centroids_incl)                         # (N, M, N)
        own = (e * centroids_excl).sum(-1)                                           # (N, M)
        idx = torch.arange(n, device=embeds.device)
        sim[idx, :, idx] = own
        return self.w.clamp(min=1e-6) * sim + self.b

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        n, m, _ = embeds.shape
        s = self.similarity_matrix(embeds).reshape(n * m, n)
        target = torch.arange(n, device=embeds.device).repeat_interleave(m)
        return F.cross_entropy(s, target)

    @torch.no_grad()
    def eer_proxy(self, embeds: torch.Tensor) -> float:
        """Fraction of utterances whose own centroid is not the most similar (quick sanity metric)."""
        n, m, _ = embeds.shape
        pred = self.similarity_matrix(embeds).argmax(-1)
        return float((pred != torch.arange(n, device=embeds.device)[:, None]).float().mean())


def cosine_similarity(a, b) -> float:
    """Speaker-embedding cosine similarity used as the voice-match metric (target >= 0.85)."""
    a = torch.as_tensor(a, dtype=torch.float32).flatten()
    b = torch.as_tensor(b, dtype=torch.float32).flatten()
    return float(F.cosine_similarity(a, b, dim=0))
