"""Enhanced SyncNet lip-sync expert (Section 3.3 / 3.4, Figure 6).

Like the Wav2Lip expert, it embeds a 0.2 s window (5 video frames, lower
half of the face; 16 mel frames) into a shared space and scores sync by cosine
similarity.  "Enhanced" = (i) explicit projection networks phi (speech) and psi
(vision) into the shared representation used by Eq. (9)/(10), and (ii) helper
functions to measure the A/V offset in milliseconds (module-level benchmark,
Section 4.3: target < 100 ms).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(cin, cout, k, s, p, residual=False):
    return _ConvBlock(cin, cout, k, s, p, residual)


class _ConvBlock(nn.Module):
    def __init__(self, cin, cout, k, s, p, residual):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(cin, cout, k, s, p), nn.BatchNorm2d(cout))
        self.residual = residual

    def forward(self, x):
        out = self.block(x)
        if self.residual:
            out = out + x
        return F.relu(out)


class EnhancedSyncNet(nn.Module):
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.face_encoder = nn.Sequential(          # input (B, 15, 48, 96): 5 RGB lower-half frames
            _conv(15, 32, (7, 7), 1, 3),
            _conv(32, 64, 5, (1, 2), 1), _conv(64, 64, 3, 1, 1, True), _conv(64, 64, 3, 1, 1, True),
            _conv(64, 128, 3, 2, 1), _conv(128, 128, 3, 1, 1, True), _conv(128, 128, 3, 1, 1, True),
            _conv(128, 256, 3, 2, 1), _conv(256, 256, 3, 1, 1, True), _conv(256, 256, 3, 1, 1, True),
            _conv(256, 512, 3, 2, 1), _conv(512, 512, 3, 1, 1, True),
            nn.AdaptiveAvgPool2d(1))
        self.audio_encoder = nn.Sequential(         # input (B, 1, 80, 16)
            _conv(1, 32, 3, 1, 1), _conv(32, 32, 3, 1, 1, True),
            _conv(32, 64, 3, (3, 1), 1), _conv(64, 64, 3, 1, 1, True),
            _conv(64, 128, 3, 3, 1), _conv(128, 128, 3, 1, 1, True),
            _conv(128, 256, 3, (3, 2), 1), _conv(256, 256, 3, 1, 1, True),
            _conv(256, 512, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1))
        # projection networks into the shared representation (phi: speech, psi: vision)
        self.phi = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, embed_dim))
        self.psi = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, embed_dim))

    def embed_audio(self, mel: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.phi(self.audio_encoder(mel).flatten(1)), dim=-1)

    def embed_video(self, faces: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.psi(self.face_encoder(faces).flatten(1)), dim=-1)

    def forward(self, mel: torch.Tensor, faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embed_audio(mel), self.embed_video(faces)


def sync_bce_loss(a: torch.Tensor, v: torch.Tensor, is_synced: torch.Tensor) -> torch.Tensor:
    """Wav2Lip expert loss: BCE on cosine similarity (clamped to (0, 1))."""
    d = F.cosine_similarity(a, v).clamp(1e-7, 1 - 1e-7)
    return F.binary_cross_entropy(d, is_synced.float())


@torch.no_grad()
def av_offset(syncnet: EnhancedSyncNet, mels: torch.Tensor, faces: torch.Tensor, fps: float = 25.0,
              max_shift: int = 15) -> Tuple[float, float]:
    """SyncNet-style offset estimation.

    mels:  (T, 1, 80, 16) mel window per video frame;  faces: (T, 15, 48, 96).
    Returns (offset_ms, confidence) where offset_ms > 0 means video lags audio.
    """
    a = syncnet.embed_audio(mels)
    v = syncnet.embed_video(faces)
    t = a.shape[0]
    dists = []
    for shift in range(-max_shift, max_shift + 1):
        lo, hi = max(0, shift), min(t, t + shift)
        if hi - lo < 1:
            dists.append(np.inf)
            continue
        dists.append(float((a[lo - shift:hi - shift] - v[lo:hi]).norm(dim=-1).mean()))
    dists = np.asarray(dists)
    best = int(np.argmin(dists))
    conf = float(np.median(dists[np.isfinite(dists)]) - dists[best])
    return (best - max_shift) * 1000.0 / fps, conf
