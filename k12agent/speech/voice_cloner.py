"""Voice cloning & speech synthesis (Section 2.2 / 3.2, Figure 4).

Pipeline: 5-second reference clip -> speaker encoder (GE2E d-vector) ->
Tacotron-2 synthesizer (text + embedding -> mel) -> vocoder (mel -> waveform).

Backends
--------
``rtvc``   Real-Time-Voice-Cloning (CorentinJ, English) or its Mandarin fork
           MockingBird (babysor) - same encoder/synthesizer/vocoder layout.
           Clone either repo into ``third_party/`` (see scripts/setup_third_party.sh)
           and point ``speech.repo_dir`` and the checkpoint paths at it.
``dummy``  Produces correctly-timed silence; lets the whole pipeline run
           without any model (tests, CI, UI development).
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ..config import resolve_path
from ..schemas import AudioSegment, ContentSection, PipelineError
from ..slides.pacing import count_words, narration_seconds
from ..utils import IsolatedImporter
from .ge2e import cosine_similarity
from .prosody import apply_emphasis, concat_with_pauses, plan_prosody

log = logging.getLogger(__name__)


def write_wav(path: str | Path, wav: np.ndarray, sr: int) -> None:
    import soundfile as sf
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(wav, dtype=np.float32), sr)


def time_stretch(wav: np.ndarray, rate: float) -> np.ndarray:
    """rate > 1 speeds up. Clamped to [0.6, 1.6] to avoid artefacts."""
    rate = float(np.clip(rate, 0.6, 1.6))
    if abs(rate - 1.0) < 0.03 or wav.size == 0:
        return wav
    import librosa
    return librosa.effects.time_stretch(wav.astype(np.float32), rate=rate)


class TTSBackend(ABC):
    sample_rate: int = 16000

    @abstractmethod
    def speaker_embedding(self, wav_or_path) -> np.ndarray: ...

    @abstractmethod
    def synthesize(self, texts: Sequence[str], embedding: np.ndarray) -> List[np.ndarray]: ...


class RTVCBackend(TTSBackend):
    def __init__(self, repo_dir: str, encoder_ckpt: str, synthesizer_ckpt: str, vocoder_ckpt: str):
        repo = resolve_path(repo_dir)
        if repo is None or not repo.exists():
            raise PipelineError("speech", f"voice-cloning repo not found at {repo}; run scripts/setup_third_party.sh")
        self.ctx = IsolatedImporter(repo, ["encoder", "synthesizer", "vocoder", "models", "utils", "control"])
        with self.ctx:
            if (repo / "models" / "encoder").exists():  # MockingBird >= 2023 layout
                from models.encoder import inference as encoder
                from models.synthesizer.inference import Synthesizer
                try:
                    from models.vocoder.hifigan import inference as vocoder
                except ImportError:
                    from models.vocoder.wavernn import inference as vocoder
            else:  # Real-Time-Voice-Cloning / old MockingBird layout
                from encoder import inference as encoder
                from synthesizer.inference import Synthesizer
                try:
                    from vocoder import inference as vocoder
                except ImportError:
                    from vocoder.hifigan import inference as vocoder
            self.encoder = encoder
            self.encoder.load_model(Path(resolve_path(encoder_ckpt)))
            self.synthesizer = Synthesizer(Path(resolve_path(synthesizer_ckpt)))
            self.vocoder = vocoder
            self.vocoder.load_model(Path(resolve_path(vocoder_ckpt)))
        self.sample_rate = int(self.synthesizer.sample_rate)

    def speaker_embedding(self, wav_or_path) -> np.ndarray:
        with self.ctx:
            src = Path(wav_or_path) if isinstance(wav_or_path, (str, Path)) else wav_or_path
            wav = self.encoder.preprocess_wav(src)
            return np.asarray(self.encoder.embed_utterance(wav))

    def synthesize(self, texts: Sequence[str], embedding: np.ndarray) -> List[np.ndarray]:
        with self.ctx:
            specs = self.synthesizer.synthesize_spectrograms(list(texts), [embedding] * len(texts))
            wavs = []
            for spec in specs:
                out = self.vocoder.infer_waveform(spec)
                wav = out[0] if isinstance(out, tuple) else out   # hifigan returns (wav, sr)
                wavs.append(np.asarray(wav, dtype=np.float32))
        return wavs


class DummyBackend(TTSBackend):
    """Silent audio whose length follows the narration rate (no model needed)."""

    def __init__(self, sample_rate: int = 16000, wpm: float = 110.0):
        self.sample_rate = sample_rate
        self.wpm = wpm

    def speaker_embedding(self, wav_or_path) -> np.ndarray:
        seed = int(hashlib.md5(str(wav_or_path).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        e = rng.normal(size=256).astype(np.float32)
        return e / np.linalg.norm(e)

    def synthesize(self, texts, embedding):
        return [np.zeros(int(self.sample_rate * max(0.3, narration_seconds(t, self.wpm))), dtype=np.float32)
                for t in texts]


class VoiceCloner:
    """High-level API used by the pipeline: one WAV per slide section."""

    def __init__(self, backend: TTSBackend, default_voice: Optional[str] = None, language: str = "zh",
                 emotional_modulation: bool = True, similarity_threshold: float = 0.85):
        self.backend = backend
        self.default_voice = default_voice
        self.language = language
        self.emotional_modulation = emotional_modulation
        self.similarity_threshold = similarity_threshold
        self._cache: dict = {}

    @classmethod
    def from_config(cls, cfg, wpm: float = 110.0) -> "VoiceCloner":
        s = cfg.speech
        if s.backend == "rtvc":
            backend: TTSBackend = RTVCBackend(s.repo_dir, s.encoder_ckpt, s.synthesizer_ckpt, s.vocoder_ckpt)
        elif s.backend == "dummy":
            backend = DummyBackend(s.sample_rate, wpm)
        else:
            raise ValueError(f"unknown speech backend {s.backend!r}")
        default_voice = resolve_path(s.default_voice)
        return cls(backend, str(default_voice) if default_voice and default_voice.exists() else None,
                   cfg.language, s.emotional_modulation, s.similarity_threshold)

    def embedding_for(self, voice_ref: Optional[str]) -> np.ndarray:
        ref = voice_ref or self.default_voice
        if ref is None:
            if isinstance(self.backend, DummyBackend):
                ref = "default"
            else:
                raise PipelineError("speech", "no reference voice: upload a 5-second clip or set speech.default_voice")
        if ref not in self._cache:
            self._cache[ref] = self.backend.speaker_embedding(ref)
        return self._cache[ref]

    def speak_section(self, section: ContentSection, embedding: np.ndarray, target_wpm: Optional[float] = None) -> np.ndarray:
        chunks = plan_prosody(section.narration, section.emphasis, self.language, self.emotional_modulation)
        if not chunks:
            return np.zeros(int(0.5 * self.backend.sample_rate), dtype=np.float32)
        wavs = self.backend.synthesize([c.text for c in chunks], embedding)
        sr = self.backend.sample_rate
        if target_wpm and not isinstance(self.backend, DummyBackend):
            speech_s = sum(len(w) for w in wavs) / sr
            natural_wpm = 60.0 * sum(count_words(c.text) for c in chunks) / max(speech_s, 1e-3)
            wavs = [time_stretch(w, target_wpm / natural_wpm) for w in wavs]
        wavs = [apply_emphasis(w, sr, c) if c.emphasis else w for w, c in zip(wavs, chunks)]
        return concat_with_pauses(wavs, chunks, sr)

    def similarity(self, ref_embedding: np.ndarray, wav: np.ndarray) -> Optional[float]:
        if isinstance(self.backend, DummyBackend) or wav.size < self.backend.sample_rate:
            return None
        return cosine_similarity(ref_embedding, self.backend.speaker_embedding(wav))

    def synthesize_lesson(self, sections: Sequence[ContentSection], out_dir: str | Path,
                          voice_ref: Optional[str] = None, target_wpm: Optional[float] = None) -> List[AudioSegment]:
        out_dir = Path(out_dir)
        emb = self.embedding_for(voice_ref)
        segments = []
        for i, sec in enumerate(sections):
            wav = self.speak_section(sec, emb, target_wpm)
            path = out_dir / f"narration_{i:02d}.wav"
            write_wav(path, wav, self.backend.sample_rate)
            sim = self.similarity(emb, wav)
            if sim is not None and sim < self.similarity_threshold:
                log.warning("section %d: voice-match similarity %.3f below threshold %.2f", i, sim, self.similarity_threshold)
            segments.append(AudioSegment(i, str(path), len(wav) / self.backend.sample_rate, sim))
        return segments
