"""Unified I/O contracts shared by all modules (Section 3, Figure 2).

Each stage of the pipeline consumes and produces one of these dataclasses so the
data flow "semantic parsing -> slides -> speech -> video" is deterministic and
serialisable (every :class:`LessonPackage` can be dumped to JSON for logging).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional


class Stage(str, Enum):
    """School stage (factor B of the 2 x 3 x 2 design)."""

    PRIMARY = "primary"   # Grade 5
    JUNIOR = "junior"     # Grade 8
    SENIOR = "senior"     # Grade 11

    @classmethod
    def from_grade(cls, grade: int) -> "Stage":
        if grade <= 6:
            return cls.PRIMARY
        if grade <= 9:
            return cls.JUNIOR
        return cls.SENIOR


class Band(str, Enum):
    """Proficiency band from the baseline diagnostic (Section 4.1.1)."""

    LOW = "low"                # < 50 %
    EMERGING = "emerging"      # 50-79 %
    PROFICIENT = "proficient"  # >= 80 %


class Difficulty(str, Enum):
    REMEDIAL = "remedial"
    ON_LEVEL = "on_level"
    ENRICHMENT = "enrichment"


class SectionKind(str, Enum):
    """Instructional sequence: definition -> derivation -> application."""

    DEFINITION = "definition"
    DERIVATION = "derivation"
    APPLICATION = "application"
    EXERCISE = "exercise"
    SUMMARY = "summary"


@dataclass
class Scaffolds:
    worked_examples: bool = True
    stepwise_derivation: bool = True
    visual_signaling: bool = True
    targeted_hints: bool = False
    fading: bool = False


@dataclass
class GenerationSettings:
    """Everything the content generator needs besides the question itself.

    Produced by :class:`k12agent.adaptive.diagnostic.AdaptationPolicy` and
    optionally modified by the instructor override panel.
    """

    stage: Stage = Stage.JUNIOR
    difficulty: Difficulty = Difficulty.ON_LEVEL
    scaffolds: Scaffolds = field(default_factory=Scaffolds)
    pace: str = "medium"                 # slow | medium | fast
    micro_chunk_sentences: int = 3       # max sentences per slide bullet group
    pause_prompts: bool = True
    reading_level: Optional[str] = None  # free text, e.g. "Grade 5"
    terminology: Optional[str] = None
    localization: Optional[str] = None   # e.g. "use RMB and Henan school context"
    language: str = "zh"
    subject: str = "math"


@dataclass
class LearnerProfile:
    student_id: str = "anonymous"
    grade: int = 8
    region: str = "urban"                # urban | rural (factor C)
    band: Band = Band.EMERGING
    voice_ref: Optional[str] = None      # 5-second reference clip for cloning
    face_ref: Optional[str] = None       # image / short clip of the preferred "teacher"
    pace: str = "medium"
    subtitle_size: int = 32
    theme: str = "light"

    @property
    def stage(self) -> Stage:
        return Stage.from_grade(self.grade)


@dataclass
class Question:
    text: str
    student_id: str = "anonymous"
    modality: str = "text"               # text | speech
    timestamp: float = field(default_factory=time.time)
    qid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class Formula:
    latex: str
    spoken: Optional[str] = None         # verbalised form used by TTS


@dataclass
class ContentSection:
    kind: SectionKind
    title: str
    bullets: List[str]
    narration: str
    formulas: List[Formula] = field(default_factory=list)
    example: Optional[str] = None
    emphasis: List[str] = field(default_factory=list)   # words to stress prosodically


@dataclass
class LessonContent:
    question: str
    topic: str
    stage: Stage
    concepts: List[str]
    sections: List[ContentSection]
    extension: Optional[str] = None      # enrichment for advanced learners (Section 2.1)
    raw_model_output: Optional[str] = None


@dataclass
class SlideTiming:
    index: int
    start_s: float
    duration_s: float


@dataclass
class AudioSegment:
    section_index: int
    wav_path: str
    duration_s: float
    speaker_similarity: Optional[float] = None


@dataclass
class LessonPackage:
    """Final artefact returned to the student UI (Figure 1, "lecture video")."""

    question: Question
    content: LessonContent
    pptx_path: Optional[str] = None
    slide_images: List[str] = field(default_factory=list)
    audio: List[AudioSegment] = field(default_factory=list)
    timings: List[SlideTiming] = field(default_factory=list)
    video_path: Optional[str] = None
    avatar_video_path: Optional[str] = None
    metrics: dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, default=_json_default)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")


def _json_default(o):
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


class PipelineError(RuntimeError):
    """Raised by a module when it cannot fulfil its I/O contract."""

    def __init__(self, module: str, message: str):
        super().__init__(f"[{module}] {message}")
        self.module = module
