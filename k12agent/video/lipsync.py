"""Audio-driven talking-face generation (Section 2.3 / 3.3, Figure 5).

Wraps the original Wav2Lip implementation (Prajwal et al., 2020) - cloned into
``third_party/Wav2Lip`` - and adds:

* optional use of :class:`~k12agent.video.models.EnhancedLipGenerator`
  (dual-stream + cross-modal attention) when ``video.enhanced_checkpoint`` is set,
* DTW-based re-timing of the generated frames (Section 3.4) constrained to
  +-``max_offset_ms``,
* an A/V offset estimate for the module-level benchmark (Section 4.3).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..config import resolve_path
from ..schemas import PipelineError
from ..utils import IsolatedImporter, run_ffmpeg
from .alignment import audio_envelope, dtw_retime, mouth_opening

log = logging.getLogger(__name__)

MEL_STEP = 16


def xcorr_offset_ms(env: np.ndarray, mouth: np.ndarray, fps: float, max_shift: int = 10) -> float:
    """Lag (ms) maximising the correlation between audio energy and mouth opening (>0: video lags)."""
    a = (env - env.mean()) / (env.std() + 1e-8)
    v = (mouth - mouth.mean()) / (mouth.std() + 1e-8)
    best, best_s = -np.inf, 0
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            c = np.mean(a[: len(a) - s] * v[s:]) if len(a) > s else -np.inf
        else:
            c = np.mean(a[-s:] * v[: len(v) + s]) if len(a) > -s else -np.inf
        if c > best:
            best, best_s = c, s
    return best_s * 1000.0 / fps


def _smooth_boxes(boxes: np.ndarray, T: int = 5) -> np.ndarray:
    out = boxes.astype(np.float64).copy()
    for i in range(len(boxes)):
        window = boxes[i:i + T] if i + T <= len(boxes) else boxes[len(boxes) - T:]
        out[i] = np.mean(window, axis=0)
    return out.astype(int)


class Wav2LipRunner:
    def __init__(self, repo_dir: str, checkpoint: str, fps: float = 25.0, img_size: int = 96,
                 face_det_batch_size: int = 16, batch_size: int = 128, pads=(0, 10, 0, 0),
                 enhanced_checkpoint: Optional[str] = None, dtw_refine: bool = True, max_offset_ms: float = 100.0,
                 device: Optional[str] = None):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        repo = resolve_path(repo_dir)
        if repo is None or not repo.exists():
            raise PipelineError("video", f"Wav2Lip repo not found at {repo}; run scripts/setup_third_party.sh")
        self.ctx = IsolatedImporter(repo, ["models", "audio", "hparams", "face_detection"])
        with self.ctx:
            import audio as w2l_audio
            import face_detection
            from models import Wav2Lip
            self.audio = w2l_audio
            self.detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                                         flip_input=False, device=self.device)
            self.model = None
            if not enhanced_checkpoint:
                ckpt = torch.load(str(resolve_path(checkpoint)), map_location=self.device)
                state = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
                self.model = Wav2Lip()
                self.model.load_state_dict(state)
                self.model = self.model.to(self.device).eval()
        self.enhanced = None
        if enhanced_checkpoint:
            from .models import EnhancedLipGenerator
            self.enhanced = EnhancedLipGenerator().to(self.device).eval()
            state = torch.load(str(resolve_path(enhanced_checkpoint)), map_location=self.device)
            self.enhanced.load_state_dict(state.get("generator", state))
        self.fps, self.img_size = fps, img_size
        self.face_det_batch_size, self.batch_size, self.pads = face_det_batch_size, batch_size, pads
        self.dtw_refine, self.max_offset_ms = dtw_refine, max_offset_ms

    @classmethod
    def from_config(cls, cfg) -> "Wav2LipRunner":
        v = cfg.video
        return cls(v.repo_dir, v.checkpoint, v.fps, v.img_size, v.face_det_batch_size, v.wav2lip_batch_size,
                   tuple(v.pads), v.enhanced_checkpoint, v.dtw_refine, v.max_offset_ms)

    # ----------------------------------------------------------------- inputs
    def _read_face(self, face_path: str) -> Tuple[List[np.ndarray], float]:
        import cv2
        p = str(face_path)
        if Path(p).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            img = cv2.imread(p)
            if img is None:
                raise PipelineError("video", f"cannot read face image {p}")
            return [img], self.fps
        cap = cv2.VideoCapture(p)
        fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if not frames:
            raise PipelineError("video", f"cannot read face video {p}")
        return frames, fps

    def _detect(self, frames: List[np.ndarray]) -> List[Tuple[int, int, int, int]]:
        preds = []
        with self.ctx:
            for i in range(0, len(frames), self.face_det_batch_size):
                preds.extend(self.detector.get_detections_for_batch(np.array(frames[i:i + self.face_det_batch_size])))
        py1, py2, px1, px2 = self.pads
        boxes = []
        for rect, img in zip(preds, frames):
            if rect is None:
                raise PipelineError("video", "face not detected in the avatar image/video")
            boxes.append([max(0, rect[0] - px1), max(0, rect[1] - py1),
                          min(img.shape[1], rect[2] + px2), min(img.shape[0], rect[3] + py2)])
        boxes = np.array(boxes)
        if len(boxes) >= 5:
            boxes = _smooth_boxes(boxes)
        return [(int(y1), int(y2), int(x1), int(x2)) for x1, y1, x2, y2 in boxes]

    def _mel_chunks(self, wav_path: str, fps: float):
        with self.ctx:
            wav = self.audio.load_wav(wav_path, 16000)
            mel = self.audio.melspectrogram(wav)
        if np.isnan(mel.reshape(-1)).any():
            raise PipelineError("video", "mel contains NaN (try adding a small amount of noise to the wav)")
        chunks, mult, i = [], 80.0 / fps, 0
        while True:
            start = int(i * mult)
            if start + MEL_STEP > mel.shape[1]:
                chunks.append(mel[:, -MEL_STEP:])
                break
            chunks.append(mel[:, start:start + MEL_STEP])
            i += 1
        return wav, mel, chunks

    # ----------------------------------------------------------------- generation
    def _predict_wav2lip(self, faces: np.ndarray, mels: np.ndarray) -> np.ndarray:
        torch = self.torch
        s = self.img_size
        masked = faces.copy()
        masked[:, s // 2:] = 0
        img = np.concatenate((masked, faces), axis=3) / 255.0
        img_t = torch.FloatTensor(np.transpose(img, (0, 3, 1, 2))).to(self.device)
        mel_t = torch.FloatTensor(mels[:, None]).to(self.device)   # (B, 1, 80, 16)
        with torch.no_grad():
            pred = self.model(mel_t, img_t)
        return pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

    def _predict_enhanced(self, faces: np.ndarray, mel_full: np.ndarray, start_frame: int, fps: float,
                          window: int = 5) -> np.ndarray:
        torch = self.torch
        from .models import mask_lower_half
        out = []
        mult = 80.0 / fps
        for s in range(0, len(faces), window):
            f = torch.FloatTensor(faces[s:s + window] / 255.0).permute(0, 3, 1, 2)       # (T, 3, H, W)
            ref = f[torch.randperm(f.shape[0])]
            inp = torch.cat([mask_lower_half(f), ref], dim=1)[None].to(self.device)       # (1, T, 6, H, W)
            m0 = int((start_frame + s) * mult)
            m1 = max(m0 + MEL_STEP, int((start_frame + s + f.shape[0]) * mult))
            mel = mel_full[:, m0:m1]
            if mel.shape[1] < MEL_STEP:
                mel = mel_full[:, -MEL_STEP:]
            with torch.no_grad():
                pred = self.enhanced(torch.FloatTensor(mel)[None].to(self.device), inp)[0][0]
            out.append(pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0)
        return np.concatenate(out)

    def generate(self, face_path: str, wav_path: str, out_path: str) -> dict:
        import cv2
        frames, fps = self._read_face(face_path)
        boxes = self._detect(frames)
        wav, mel, mel_chunks = self._mel_chunks(wav_path, fps)
        n = len(mel_chunks)
        out_frames: List[np.ndarray] = []
        out_boxes = []
        for b0 in range(0, n, self.batch_size):
            idx = [i % len(frames) for i in range(b0, min(n, b0 + self.batch_size))]
            faces = np.stack([cv2.resize(frames[i][boxes[i][0]:boxes[i][1], boxes[i][2]:boxes[i][3]],
                                         (self.img_size, self.img_size)) for i in idx])
            if self.enhanced is not None:
                preds = self._predict_enhanced(faces, mel, b0, fps)
            else:
                preds = self._predict_wav2lip(faces, np.stack(mel_chunks[b0:b0 + len(idx)]))
            for p, i in zip(preds, idx):
                f = frames[i].copy()
                y1, y2, x1, x2 = boxes[i]
                f[y1:y2, x1:x2] = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                out_frames.append(f)
                out_boxes.append(boxes[i])

        env = audio_envelope(wav, 16000, fps, len(out_frames))
        mouth = mouth_opening(np.stack(out_frames), out_boxes)
        metrics = {"av_offset_ms_before": xcorr_offset_ms(env, mouth, fps)}
        if self.dtw_refine and len(out_frames) > 3:
            band = max(1, int(round(self.max_offset_ms / 1000.0 * fps)))
            mapping = dtw_retime(env, mouth, max_offset_frames=band)
            out_frames = [out_frames[j] for j in mapping]
            out_boxes = [out_boxes[j] for j in mapping]
            mouth = mouth_opening(np.stack(out_frames), out_boxes)
        metrics["av_offset_ms"] = xcorr_offset_ms(env, mouth, fps)
        metrics["frames"] = len(out_frames)
        metrics["fps"] = fps

        h, w = out_frames[0].shape[:2]
        with tempfile.TemporaryDirectory() as tmp:
            silent = str(Path(tmp) / "silent.avi")
            vw = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"DIVX"), fps, (w, h))
            for f in out_frames:
                vw.write(f)
            vw.release()
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            run_ffmpeg(["-i", wav_path, "-i", silent, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", out_path])
        return metrics
