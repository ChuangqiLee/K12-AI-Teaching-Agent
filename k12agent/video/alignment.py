"""Dual-alignment mechanism for multimodal fusion (Section 3.4, Figure 6).

1. Windowed synchronisation loss, Eq. (9):
       L_sync = 1/(2 Dt + 1) * sum_{tau=-Dt}^{Dt} || phi(A_{t+tau}) - psi(V_{t+tau}) ||_2^2
2. Adversarial modality discriminator, Eq. (10):
       min_{phi,psi} max_D  E[log D(phi(A), psi(V))] + E[log(1 - D(phi(A), psi(V~)))]
   where V~ is a temporally shifted (asynchronous) visual sequence.
3. Dynamic Time Warping between audio and visual feature sequences, used at
   inference time to re-time generated frames so that residual A/V offsets
   stay below the 100 ms criterion.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- Eq. (9)
def windowed_sync_loss(phi_a: torch.Tensor, psi_v: torch.Tensor, delta_t: int = 2) -> torch.Tensor:
    """phi_a, psi_v: (B, T, D) projected features. Averages Eq. (9) over all centre steps t."""
    b, t, _ = phi_a.shape
    sq = (phi_a - psi_v).pow(2).sum(-1)                         # (B, T): ||.||_2^2 per step
    kernel = torch.ones(1, 1, 2 * delta_t + 1, device=sq.device) / (2 * delta_t + 1)
    windowed = F.conv1d(F.pad(sq[:, None], (delta_t, delta_t), mode="replicate"), kernel)  # (B, 1, T)
    return windowed.mean()


# --------------------------------------------------------------------------- Eq. (10)
class ModalityDiscriminator(nn.Module):
    """D(phi(A), psi(V)) -> probability that both come from the same time step."""

    def __init__(self, d: int = 512, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, hidden), nn.LeakyReLU(0.2, True),
                                 nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, True), nn.Linear(hidden, 1))

    def forward(self, a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([a, v], dim=-1)).squeeze(-1)   # logits


def shift_sequence(v: torch.Tensor, min_shift: int = 3, max_shift: Optional[int] = None) -> torch.Tensor:
    """Build the asynchronous sequence V~ by rolling each item by >= min_shift steps."""
    b, t = v.shape[:2]
    max_shift = max_shift or max(min_shift + 1, t - min_shift)
    shifts = torch.randint(min_shift, max(min_shift + 1, max_shift), (b,))
    return torch.stack([torch.roll(v[i], int(shifts[i]), dims=0) for i in range(b)])


def discriminator_loss(D: ModalityDiscriminator, a: torch.Tensor, v: torch.Tensor, v_async: torch.Tensor) -> torch.Tensor:
    """max_D part of Eq. (10) written as a BCE minimisation (features detached)."""
    real = D(a.detach(), v.detach())
    fake = D(a.detach(), v_async.detach())
    return F.binary_cross_entropy_with_logits(real, torch.ones_like(real)) + \
        F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))


def projection_adversarial_loss(D: ModalityDiscriminator, a: torch.Tensor, v: torch.Tensor,
                                v_async: torch.Tensor) -> torch.Tensor:
    """min_{phi,psi} part of Eq. (10): make matched/unmatched pairs indistinguishable."""
    real = D(a, v)
    fake = D(a, v_async)
    return -(F.binary_cross_entropy_with_logits(real, torch.ones_like(real)) +
             F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake)))


# --------------------------------------------------------------------------- DTW
def dtw(cost: np.ndarray, band: Optional[int] = None) -> Tuple[float, np.ndarray]:
    """Classic DTW with an optional Sakoe-Chiba band. Returns (total cost, path (K, 2))."""
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        j_lo, j_hi = 1, m
        if band is not None:
            centre = int(round(i * m / n))
            j_lo, j_hi = max(1, centre - band), min(m, centre + band)
        for j in range(j_lo, j_hi + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    i, j, path = n, m, [(n - 1, m - 1)]
    while i > 1 or j > 1:
        steps = [(acc[i - 1, j - 1], i - 1, j - 1), (acc[i - 1, j], i - 1, j), (acc[i, j - 1], i, j - 1)]
        _, i, j = min((s for s in steps if s[1] >= 1 and s[2] >= 1), key=lambda s: s[0])
        path.append((i - 1, j - 1))
    return float(acc[n, m]), np.asarray(path[::-1])


def cosine_cost(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return 1.0 - a @ b.T


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean(0)) / (x.std(0) + 1e-8)


def dtw_retime(audio_feats: np.ndarray, video_feats: np.ndarray, max_offset_frames: int = 3) -> np.ndarray:
    """Return, for every audio step, the index of the visual frame to display.

    Features are (T, D) arrays (SyncNet embeddings, or 1-D envelopes reshaped to
    (T, 1)).  The band restricts warping to +-max_offset_frames (100 ms @ 25 fps
    ~= 2.5 frames), so DTW only corrects small residual drifts and never
    reorders speech.
    """
    a = np.asarray(audio_feats, dtype=np.float64).reshape(len(audio_feats), -1)
    v = np.asarray(video_feats, dtype=np.float64).reshape(len(video_feats), -1)
    cost = cosine_cost(a, v) if a.shape[1] > 1 else np.abs(zscore(a) - zscore(v).T)
    band = max(max_offset_frames, abs(a.shape[0] - v.shape[0]) + 1)
    _, path = dtw(cost, band=band)
    mapping = np.zeros(a.shape[0], dtype=int)
    for ai, vi in path:
        mapping[ai] = vi          # last match wins -> monotone non-decreasing
    return mapping


def audio_envelope(wav: np.ndarray, sr: int, fps: float, n_frames: int) -> np.ndarray:
    """RMS energy per video frame (fallback audio feature for DTW)."""
    hop = sr / fps
    env = np.zeros(n_frames)
    for i in range(n_frames):
        seg = wav[int(i * hop):int((i + 1) * hop)]
        env[i] = np.sqrt(np.mean(seg ** 2)) if seg.size else 0.0
    return env


def mouth_opening(frames_bgr: np.ndarray, boxes) -> np.ndarray:
    """Cheap visual feature: darkness of the inner-mouth region per frame (higher = more open)."""
    vals = []
    for f, (y1, y2, x1, x2) in zip(frames_bgr, boxes):
        h = y2 - y1
        roi = f[y1 + int(0.65 * h):y2 - int(0.1 * h), x1 + int(0.3 * (x2 - x1)):x2 - int(0.3 * (x2 - x1))]
        vals.append(255.0 - float(roi.mean()) if roi.size else 0.0)
    return np.asarray(vals)
