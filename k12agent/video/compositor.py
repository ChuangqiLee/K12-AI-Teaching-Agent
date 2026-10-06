"""Dynamic presentation: overlay the virtual teacher onto the slides (Section 2.3).

For every slide i the compositor shows the slide frame for ``timings[i].duration_s``
seconds, overlays the lip-synced avatar clip (or the static face image) in
the avatar box (bottom-right, Figure 8), optionally burns in subtitles, and
concatenates the narration WAVs padded with silence to the same timeline, so
slides never advance before the explanation is complete (Section 3.4).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ..schemas import AudioSegment, SlideTiming
from ..utils import ffmpeg_exe, run_ffmpeg

log = logging.getLogger(__name__)


def _read_clip(path: Optional[str]):
    import cv2
    if not path or not Path(path).exists():
        return []
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize-and-crop to exactly (w, h) keeping the aspect ratio."""
    import cv2
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    r = cv2.resize(img, (max(w, int(iw * s)), max(h, int(ih * s))))
    y0, x0 = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return r[y0:y0 + h, x0:x0 + w]


def compose_lecture(slide_images: Sequence[str], audio: Sequence[AudioSegment], timings: Sequence[SlideTiming],
                    out_path: str, avatar_clips: Optional[Sequence[Optional[str]]] = None,
                    face_image: Optional[str] = None, avatar_box=(0.66, 0.50, 0.30, 0.46), fps: float = 25.0,
                    sample_rate: int = 16000, size=(1280, 720)) -> str:
    import cv2
    import soundfile as sf

    W, H = size
    ax, ay, aw, ah = int(W * avatar_box[0]), int(H * avatar_box[1]), int(W * avatar_box[2]), int(H * avatar_box[3])
    face = cv2.imread(face_image) if face_image and Path(face_image).exists() else None
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        silent = str(Path(tmp) / "slides.avi")
        vw = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"DIVX"), fps, (W, H))
        track: List[np.ndarray] = []
        for i, timing in enumerate(timings):
            slide = cv2.resize(cv2.imread(slide_images[i]), (W, H))
            clip = _read_clip(avatar_clips[i]) if avatar_clips and i < len(avatar_clips) else []
            n_frames = int(round(timing.duration_s * fps))
            for k in range(n_frames):
                frame = slide.copy()
                src = clip[k] if k < len(clip) else (clip[-1] if clip else face)
                if src is not None:
                    frame[ay:ay + ah, ax:ax + aw] = _fit(src, aw, ah)
                vw.write(frame)
            seg = audio[i] if i < len(audio) else None
            wav, sr = sf.read(seg.wav_path, dtype="float32") if seg else (np.zeros(0, np.float32), sample_rate)
            if wav.ndim > 1:
                wav = wav.mean(1)
            if sr != sample_rate and wav.size:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
            total = int(round(n_frames / fps * sample_rate))
            track.append(np.pad(wav[:total], (0, max(0, total - len(wav)))))
        vw.release()
        wav_path = str(Path(out_path).with_suffix(".wav"))
        sf.write(wav_path, np.concatenate(track) if track else np.zeros(1, np.float32), sample_rate)
        if ffmpeg_exe() is None:
            log.warning("ffmpeg not available: writing silent video + separate WAV")
            Path(silent).replace(Path(out_path).with_suffix(".avi"))
            return str(Path(out_path).with_suffix(".avi"))
        run_ffmpeg(["-i", silent, "-i", wav_path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", out_path])
    return out_path
