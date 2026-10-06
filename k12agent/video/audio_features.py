"""Mel-spectrogram front-end identical to Wav2Lip's ``audio.py`` / ``hparams.py``.

16 kHz, n_fft = win = 800, hop = 200 (80 mel frames per second), 80 mel bands,
fmin = 55 Hz, fmax = 7600 Hz, pre-emphasis 0.97, symmetric normalisation to
[-4, 4].  Keeping the exact settings lets the enhanced models consume the same
inputs as the original Wav2Lip checkpoints.
"""

from __future__ import annotations

import numpy as np

SR, N_FFT, HOP, WIN, N_MELS, FMIN, FMAX = 16000, 800, 200, 800, 80, 55, 7600
MIN_LEVEL_DB, REF_LEVEL_DB, MAX_ABS = -100.0, 20.0, 4.0
_mel_basis = None


def load_wav(path: str, sr: int = SR) -> np.ndarray:
    import librosa
    return librosa.load(path, sr=sr)[0]


def melspectrogram(wav: np.ndarray) -> np.ndarray:
    import librosa
    from scipy import signal
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    emphasized = signal.lfilter([1, -0.97], [1], wav)
    D = librosa.stft(y=emphasized, n_fft=N_FFT, hop_length=HOP, win_length=WIN)
    S = 20 * np.log10(np.maximum(1e-5, _mel_basis @ np.abs(D))) - REF_LEVEL_DB
    return np.clip(2 * MAX_ABS * ((S - MIN_LEVEL_DB) / -MIN_LEVEL_DB) - MAX_ABS, -MAX_ABS, MAX_ABS).astype(np.float32)


def mel_window(mel: np.ndarray, frame_idx: int, fps: float = 25.0, step: int = 16) -> np.ndarray:
    start = int(80.0 * frame_idx / fps)
    if start + step > mel.shape[1]:
        return mel[:, -step:]
    return mel[:, start:start + step]
