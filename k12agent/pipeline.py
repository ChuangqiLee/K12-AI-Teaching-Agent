"""End-to-end pedagogical intelligent agent (Figure 1, Figure 2).

question -> [adaptive settings (+ teacher override)] -> LLaMA structured content
-> slides (PPTX + frames) -> cloned-voice narration -> adaptive pacing
-> Wav2Lip avatar clips (+ DTW refinement) -> composited lecture video.

Every stage communicates through the dataclasses in :mod:`k12agent.schemas`;
failures in optional stages (lip-sync) degrade gracefully and are recorded in
``LessonPackage.errors`` instead of aborting the answer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .adaptive.diagnostic import AdaptationPolicy
from .adaptive.override import Override, OverridePanel
from .config import Config, load_config, resolve_path
from .content.generator import ContentGenerator, build_generator
from .schemas import GenerationSettings, LearnerProfile, LessonPackage, PipelineError, Question
from .slides.builder import build_pptx, render_slide_images
from .slides.pacing import PacingController
from .speech.voice_cloner import VoiceCloner
from .subjects.hooks import get_subject

log = logging.getLogger(__name__)


class TeachingAgent:
    def __init__(self, cfg: Optional[Config] = None, generator: Optional[ContentGenerator] = None,
                 voice: Optional[VoiceCloner] = None, lipsync=None):
        self.cfg = cfg or load_config()
        self._generator = generator
        self._voice = voice
        self._lipsync = lipsync
        self.policy = AdaptationPolicy()
        self.overrides = OverridePanel(Path(self.cfg.output_dir) / "logs" / "overrides.jsonl")

    # lazily constructed heavy components ------------------------------------------------
    @property
    def generator(self) -> ContentGenerator:
        if self._generator is None:
            self._generator = build_generator(self.cfg)
        return self._generator

    @property
    def voice(self) -> VoiceCloner:
        if self._voice is None:
            self._voice = VoiceCloner.from_config(self.cfg)
        return self._voice

    @property
    def lipsync(self):
        if self._lipsync is None and self.cfg.video.backend == "wav2lip":
            from .video.lipsync import Wav2LipRunner
            self._lipsync = Wav2LipRunner.from_config(self.cfg)
        return self._lipsync

    # ------------------------------------------------------------------------------------
    def settings_for(self, profile: LearnerProfile, override: Optional[Override] = None) -> GenerationSettings:
        band = override.band if override and override.band else profile.band
        s = self.policy.settings_for(profile, band, self.cfg.language)
        if override:
            s = self.overrides.apply_to_settings(s, override)
        return s

    def answer(self, question: Union[str, Question], profile: Optional[LearnerProfile] = None,
               settings: Optional[GenerationSettings] = None, override: Optional[Override] = None,
               render_video: bool = True, out_dir: Optional[str] = None) -> LessonPackage:
        q = Question(question) if isinstance(question, str) else question
        profile = profile or LearnerProfile(student_id=q.student_id)
        settings = settings or self.settings_for(profile, override)
        out = Path(out_dir or Path(self.cfg.output_dir) / q.qid)
        out.mkdir(parents=True, exist_ok=True)
        voice_ref = (override.voice_ref if override and override.voice_ref else None) or profile.voice_ref
        face = profile.face_ref or str(resolve_path(self.cfg.video.default_face))
        face = face if face and Path(face).exists() else None

        # 1. content -------------------------------------------------------------------
        lesson = self.generator.generate(q.text, settings)
        lesson = get_subject(settings.subject).apply(lesson, settings.language)
        if override:
            lesson = self.overrides.apply_to_content(lesson, override)
        pkg = LessonPackage(question=q, content=lesson)

        # 2. slides ----------------------------------------------------------------------
        box = tuple(self.cfg.slides.avatar_box)
        size = (int(self.cfg.slides.width_px), int(self.cfg.slides.height_px))
        pkg.slide_images = render_slide_images(lesson, out / "slides", size, settings.language,
                                               self.cfg.slides.font_cjk, box)
        pacing = PacingController.from_config(self.cfg, settings.pace)

        # 3. speech ----------------------------------------------------------------------
        try:
            pkg.audio = self.voice.synthesize_lesson(lesson.sections, out / "audio", voice_ref, pacing.wpm)
        except PipelineError as err:
            pkg.errors.append(str(err))
            log.error("%s", err)
        sims = [a.speaker_similarity for a in pkg.audio if a.speaker_similarity is not None]
        if sims:
            pkg.metrics["voice_similarity_mean"] = float(np.mean(sims))
            pkg.metrics["voice_similarity_sd"] = float(np.std(sims))

        # 4. pacing ----------------------------------------------------------------------
        durations = [a.duration_s for a in pkg.audio] if len(pkg.audio) == len(lesson.sections) else None
        pkg.timings = pacing.schedule(lesson.sections, durations)

        # 5. avatar ----------------------------------------------------------------------
        clips = [None] * len(lesson.sections)
        if render_video and face and pkg.audio and self.cfg.video.backend != "none":
            try:
                runner = self.lipsync
                offsets = []
                for a in pkg.audio:
                    path = str(out / "avatar" / f"avatar_{a.section_index:02d}.mp4")
                    m = runner.generate(face, a.wav_path, path)
                    clips[a.section_index] = path
                    offsets.append(abs(m["av_offset_ms"]))
                pkg.metrics["av_offset_ms_mean"] = float(np.mean(offsets))
                pkg.metrics["av_offset_ms_max"] = float(np.max(offsets))
            except (PipelineError, RuntimeError, ImportError, FileNotFoundError) as err:
                pkg.errors.append(f"[video] lip-sync disabled: {err}")
                log.warning("lip-sync failed, using static avatar: %s", err)
                clips = [None] * len(lesson.sections)
        pkg.avatar_video_path = next((c for c in clips if c), None)

        # 6. PPTX + lecture video ----------------------------------------------------------------
        pkg.pptx_path = str(build_pptx(lesson, out / "lesson.pptx", settings.language, box, face, clips,
                                       out / "slides" / "formulas", self.cfg.slides.formula_dpi))
        if render_video and pkg.audio:
            from .video.compositor import compose_lecture
            try:
                pkg.video_path = compose_lecture(pkg.slide_images, pkg.audio, pkg.timings, str(out / "lecture.mp4"),
                                                 clips, face, box, self.cfg.video.fps, self.voice.backend.sample_rate)
            except (RuntimeError, OSError) as err:
                pkg.errors.append(f"[compose] {err}")
        pkg.metrics["total_duration_s"] = round(sum(t.duration_s for t in pkg.timings), 2)
        pkg.save(out / "lesson.json")
        return pkg
