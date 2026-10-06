"""Training of the enhanced lip-sync models (Sections 3.3-3.4).

Stage 1 - ``--stage syncnet``: Enhanced SyncNet expert
    L = BCE(sync / off-sync pairs) + lambda_win * L_sync (Eq. 9) + lambda_adv * Eq. (10)
    over sequences of K consecutive 5-frame windows; phi / psi are the
    projection heads of :class:`EnhancedSyncNet` and D is a
    :class:`ModalityDiscriminator` trained on matched vs. shifted pairs.

Stage 2 - ``--stage generator``: EnhancedLipGenerator (Eqs. 6-8)
    L = (1 - lambda_sync) * L1 + lambda_sync * expert sync loss (frozen stage-1 expert)

Data layout = Wav2Lip's preprocessed LRS2 format (``preprocess.py`` of Wav2Lip):
    <root>/<video_id>/{0.jpg, 1.jpg, ..., audio.wav}
with face crops resized to 96x96 on the fly.  ``--filelist`` is a text file of
``<video_id>`` lines (relative to root).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .alignment import (ModalityDiscriminator, discriminator_loss, projection_adversarial_loss, shift_sequence,
                        windowed_sync_loss)
from .audio_features import load_wav, mel_window, melspectrogram
from .models import EnhancedLipGenerator, mask_lower_half
from .syncnet import EnhancedSyncNet, sync_bce_loss

FPS, WIN, SIZE = 25.0, 5, 96


class LipDataset(Dataset):
    def __init__(self, root: str, filelist: str, k_windows: int = 1, samples_per_epoch: int = 20000):
        import cv2
        self.cv2 = cv2
        self.root = Path(root)
        self.videos = [l.strip() for l in open(filelist, encoding="utf-8") if l.strip()]
        self.k = k_windows
        self.n = samples_per_epoch
        self._mel_cache: dict = {}

    def __len__(self):
        return self.n

    def _frames(self, vdir: Path):
        return sorted(vdir.glob("*.jpg"), key=lambda p: int(p.stem))

    def _load(self, paths):
        imgs = [self.cv2.resize(self.cv2.imread(str(p)), (SIZE, SIZE)) for p in paths]
        return torch.from_numpy(np.stack(imgs)[..., ::-1].copy()).permute(0, 3, 1, 2).float() / 255.0

    def _mel(self, vdir: Path):
        key = str(vdir)
        if key not in self._mel_cache:
            self._mel_cache[key] = melspectrogram(load_wav(str(vdir / "audio.wav")))
        return self._mel_cache[key]

    def __getitem__(self, _):
        while True:
            vdir = self.root / random.choice(self.videos)
            frames = self._frames(vdir)
            need = self.k + WIN - 1
            if len(frames) < need + 3 * WIN:
                continue
            s = random.randint(0, len(frames) - need - 1)
            far = [i for i in range(0, len(frames) - need) if abs(i - s) >= WIN]
            if not far:
                continue
            wrong = random.choice(far)
            r = random.randint(0, len(frames) - WIN)
            try:
                mel = self._mel(vdir)
            except Exception:  # noqa: BLE001 - corrupt audio, resample
                continue
            return {
                "frames": self._load(frames[s:s + need]),                               # (K+4, 3, 96, 96)
                "ref": self._load(frames[r:r + WIN]),                                   # (5, 3, 96, 96)
                "mels": torch.from_numpy(np.stack([mel_window(mel, s + k) for k in range(self.k)]))[:, None],
                "wrong_mels": torch.from_numpy(np.stack([mel_window(mel, wrong + k) for k in range(self.k)]))[:, None],
            }


def lower_half_windows(frames: torch.Tensor, k: int) -> torch.Tensor:
    """frames (B, K+4, 3, 96, 96) -> SyncNet inputs (B, K, 15, 48, 96)."""
    wins = [frames[:, i:i + WIN, :, SIZE // 2:, :].flatten(1, 2) for i in range(k)]
    return torch.stack(wins, 1)


def train_syncnet(args, device):
    ds = LipDataset(args.data_root, args.filelist, k_windows=args.k_windows, samples_per_epoch=args.samples)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, drop_last=True)
    net = EnhancedSyncNet().to(device)
    D = ModalityDiscriminator(512).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr)
    step = 0
    for epoch in range(args.epochs):
        for batch in dl:
            frames = batch["frames"].to(device)
            mels, wrong = batch["mels"].to(device), batch["wrong_mels"].to(device)
            b, k = mels.shape[:2]
            faces = lower_half_windows(frames, k)                                   # (B, K, 15, 48, 96)
            a = net.embed_audio(mels.flatten(0, 1)).view(b, k, -1)
            a_wrong = net.embed_audio(wrong.flatten(0, 1)).view(b, k, -1)
            v = net.embed_video(faces.flatten(0, 1)).view(b, k, -1)
            # BCE on matched vs. mismatched pairs (Wav2Lip expert objective)
            l_bce = sync_bce_loss(a.flatten(0, 1), v.flatten(0, 1), torch.ones(b * k, device=device)) + \
                sync_bce_loss(a_wrong.flatten(0, 1), v.flatten(0, 1), torch.zeros(b * k, device=device))
            # Eq. (9) windowed alignment in the shared space
            l_win = windowed_sync_loss(a, v, delta_t=min(args.delta_t, max(0, (k - 1) // 2))) if k > 1 else a.new_zeros(())
            # Eq. (10): update D, then phi/psi
            v_async = shift_sequence(v, min_shift=1) if k > 1 else v[torch.randperm(b)]
            opt_d.zero_grad()
            discriminator_loss(D, a.flatten(0, 1), v.flatten(0, 1), v_async.flatten(0, 1)).backward()
            opt_d.step()
            l_adv = projection_adversarial_loss(D, a.flatten(0, 1), v.flatten(0, 1), v_async.flatten(0, 1))
            loss = l_bce + args.lambda_win * l_win + args.lambda_adv * l_adv
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % args.log_every == 0:
                print(f"[syncnet] ep {epoch} step {step} bce {l_bce.item():.4f} win {float(l_win):.4f} adv {l_adv.item():.4f}")
        torch.save({"syncnet": net.state_dict(), "discriminator": D.state_dict()}, Path(args.out) / f"syncnet_ep{epoch}.pth")


def train_generator(args, device):
    ds = LipDataset(args.data_root, args.filelist, k_windows=1, samples_per_epoch=args.samples)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, drop_last=True)
    gen = EnhancedLipGenerator().to(device)
    expert = EnhancedSyncNet().to(device).eval()
    expert.load_state_dict(torch.load(args.syncnet_ckpt, map_location=device)["syncnet"])
    for p in expert.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    step = 0
    for epoch in range(args.epochs):
        for batch in dl:
            gt = batch["frames"].to(device)                                         # (B, 5, 3, 96, 96)
            ref = batch["ref"].to(device)
            mel = batch["mels"].to(device)[:, 0]                                    # (B, 1, 80, 16)
            inp = torch.cat([mask_lower_half(gt), ref], dim=2)                     # (B, 5, 6, 96, 96)
            pred, *_ = gen(mel[:, 0], inp)
            l1 = F.l1_loss(pred, gt)
            a = expert.embed_audio(mel)
            v = expert.embed_video(pred[:, :, :, SIZE // 2:, :].flatten(1, 2))
            l_sync = sync_bce_loss(a, v, torch.ones(a.shape[0], device=device))
            loss = (1 - args.lambda_sync) * l1 + args.lambda_sync * l_sync
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % args.log_every == 0:
                print(f"[generator] ep {epoch} step {step} l1 {l1.item():.4f} sync {l_sync.item():.4f}")
        torch.save({"generator": gen.state_dict()}, Path(args.out) / f"enhanced_generator_ep{epoch}.pth")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["syncnet", "generator"], required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--filelist", required=True)
    p.add_argument("--out", default="checkpoints/lipsync")
    p.add_argument("--syncnet-ckpt", help="stage-1 checkpoint (required for --stage generator)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--samples", type=int, default=20000, help="samples per epoch")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--k-windows", type=int, default=5, help="consecutive windows per sequence (Eq. 9 / 10)")
    p.add_argument("--delta-t", type=int, default=2, help="half window Dt of Eq. (9)")
    p.add_argument("--lambda-win", type=float, default=0.5)
    p.add_argument("--lambda-adv", type=float, default=0.05)
    p.add_argument("--lambda-sync", type=float, default=0.03)
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.stage == "syncnet":
        train_syncnet(args, device)
    else:
        if not args.syncnet_ckpt:
            p.error("--syncnet-ckpt is required for the generator stage")
        train_generator(args, device)


if __name__ == "__main__":
    main()
