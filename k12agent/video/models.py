"""Enhanced Wav2Lip generator with dual-stream features and cross-modal attention.

Section 3.3:
    F_A = Conv1D(A_mel)   in R^{T x 512},   A_mel in R^{80 x T'}            (Eq. 6)
    F_V = ResNet34(V_roi) in R^{T x 512},   V_roi in R^{96 x 96 x T}        (Eq. 7)
    alpha_t = softmax( Q(t_F^A) K(t_F^V)^T / sqrt(d) ),  d = 512            (Eq. 8)

Figure 5: face encoder + audio encoder -> fused features -> face decoder
(with skip connections, as in Wav2Lip) -> lip-synced mouth-region frames.

The visual input follows Wav2Lip: the target frame with its lower half masked,
concatenated channel-wise with a random reference frame of the same identity
(6 channels).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioConv1DEncoder(nn.Module):
    """Eq. (6): temporal Conv1D stack over the mel spectrogram, pooled to T video frames."""

    def __init__(self, n_mels: int = 80, d: int = 512):
        super().__init__()
        ch = [n_mels, 256, 384, d, d]
        layers = []
        for i in range(len(ch) - 1):
            layers += [nn.Conv1d(ch[i], ch[i + 1], 5, padding=2), nn.BatchNorm1d(ch[i + 1]), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, mel: torch.Tensor, n_frames: int) -> torch.Tensor:
        """mel: (B, 80, T_mel) -> (B, T, 512)."""
        x = self.net(mel)
        return F.adaptive_avg_pool1d(x, n_frames).transpose(1, 2)


class VisualResNet34Encoder(nn.Module):
    """Eq. (7): ResNet-34 over each 96x96 frame; also returns skip features for the decoder."""

    def __init__(self, in_ch: int = 6, d: int = 512):
        super().__init__()
        from torchvision.models import resnet34
        net = resnet34(weights=None)
        net.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)          # 48x48
        self.pool = net.maxpool                                          # 24x24
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.out = nn.Linear(512, d)

    def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """frames: (B*T, 6, 96, 96) -> vec (B*T, 512), skips [48, 24, 12, 6, 3]."""
        s0 = self.stem(frames)
        s1 = self.layer1(self.pool(s0))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        vec = self.out(F.adaptive_avg_pool2d(s4, 1).flatten(1))
        return vec, [s0, s1, s2, s3, s4]


class CrossModalAttention(nn.Module):
    """Eq. (8): audio queries attend over visual keys/values along time."""

    def __init__(self, d: int = 512):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(inplace=True))
        self.d = d

    def forward(self, f_a: torch.Tensor, f_v: torch.Tensor):
        """f_a, f_v: (B, T, d) -> fused (B, T, d), alpha (B, T, T)."""
        alpha = torch.softmax(self.q(f_a) @ self.k(f_v).transpose(1, 2) / math.sqrt(self.d), dim=-1)
        attended = alpha @ self.v(f_v)
        return self.fuse(torch.cat([f_a, attended], dim=-1)), alpha


def _up(in_ch, out_ch):
    return nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                         nn.Conv2d(out_ch, out_ch, 3, 1, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class FaceDecoder(nn.Module):
    def __init__(self, d: int = 512):
        super().__init__()
        self.seed = nn.Linear(d, 512 * 3 * 3)
        self.up4 = _up(512 + 512, 256)   # 3 -> 6   (+ layer4 skip)
        self.up3 = _up(256 + 256, 128)   # 6 -> 12  (+ layer3 skip)
        self.up2 = _up(128 + 128, 64)    # 12 -> 24 (+ layer2 skip)
        self.up1 = _up(64 + 64, 64)      # 24 -> 48 (+ layer1 skip)
        self.up0 = _up(64 + 64, 32)      # 48 -> 96 (+ stem skip)
        self.head = nn.Sequential(nn.Conv2d(32, 3, 1), nn.Sigmoid())

    def forward(self, z: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        s0, s1, s2, s3, s4 = skips
        x = self.seed(z).view(-1, 512, 3, 3)
        x = self.up4(torch.cat([x, s4], 1))
        x = self.up3(torch.cat([x, s3], 1))
        x = self.up2(torch.cat([x, s2], 1))
        x = self.up1(torch.cat([x, s1], 1))
        x = self.up0(torch.cat([x, s0], 1))
        return self.head(x)


class EnhancedLipGenerator(nn.Module):
    def __init__(self, d: int = 512, n_mels: int = 80):
        super().__init__()
        self.audio_enc = AudioConv1DEncoder(n_mels, d)
        self.face_enc = VisualResNet34Encoder(6, d)
        self.xattn = CrossModalAttention(d)
        self.decoder = FaceDecoder(d)

    def forward(self, mel: torch.Tensor, faces: torch.Tensor):
        """mel: (B, 80, T_mel); faces: (B, T, 6, 96, 96) in [0, 1] -> frames (B, T, 3, 96, 96)."""
        b, t = faces.shape[:2]
        f_a = self.audio_enc(mel, t)                                      # (B, T, 512)
        vec, skips = self.face_enc(faces.flatten(0, 1))                   # (B*T, 512)
        f_v = vec.view(b, t, -1)
        fused, alpha = self.xattn(f_a, f_v)
        out = self.decoder(fused.reshape(b * t, -1), skips)
        return out.view(b, t, 3, out.shape[-2], out.shape[-1]), f_a, f_v, alpha


def mask_lower_half(frames: torch.Tensor) -> torch.Tensor:
    """Wav2Lip input masking: zero the lower half (mouth region) of the target frame."""
    masked = frames.clone()
    masked[..., frames.shape[-2] // 2:, :] = 0
    return masked
