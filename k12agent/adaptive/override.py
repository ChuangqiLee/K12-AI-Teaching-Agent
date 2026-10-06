"""Instructor-in-the-loop override panel (Section 2, Section 4.1.2).

Teachers may (i) adjust difficulty, (ii) toggle scaffolds, (iii) change pacing,
(iv) edit / replace problem stems and localise examples, and (v) set
personalisation preferences (voice profile, terminology, reading level).
Edits are previewed before a one-click publish, and every override is logged
(JSONL) for transparency and reproducibility.  Intervention points: after the
baseline diagnostic, mid-session (off-task / confusion signals), post-session.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from ..schemas import Band, ContentSection, Difficulty, GenerationSettings, LessonContent


class InterventionPoint(str, Enum):
    AFTER_DIAGNOSTIC = "after_diagnostic"
    MID_SESSION = "mid_session"
    POST_SESSION = "post_session"


@dataclass
class Override:
    teacher_id: str
    student_id: str
    point: InterventionPoint = InterventionPoint.AFTER_DIAGNOSTIC
    band: Optional[Band] = None                       # re-classify a suspected misclassification
    difficulty: Optional[Difficulty] = None
    scaffolds: Dict[str, bool] = field(default_factory=dict)
    pace: Optional[str] = None
    micro_chunk_sentences: Optional[int] = None
    pause_prompts: Optional[bool] = None
    localization: Optional[str] = None
    reading_level: Optional[str] = None
    terminology: Optional[str] = None
    voice_ref: Optional[str] = None
    section_edits: Dict[int, Dict[str, object]] = field(default_factory=dict)   # {slide idx: {"bullets": [...], ...}}
    replace_sections: Optional[List[Dict[str, object]]] = None
    note: str = ""


class OverridePanel:
    def __init__(self, log_path: str | Path = "outputs/logs/overrides.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._pending: Dict[str, Dict] = {}

    # --- settings ----------------------------------------------------------------
    @staticmethod
    def apply_to_settings(settings: GenerationSettings, ov: Override) -> GenerationSettings:
        s = copy.deepcopy(settings)
        if ov.difficulty:
            s.difficulty = ov.difficulty
        for k, v in ov.scaffolds.items():
            if hasattr(s.scaffolds, k):
                setattr(s.scaffolds, k, bool(v))
        for name in ("pace", "micro_chunk_sentences", "pause_prompts", "localization", "reading_level", "terminology"):
            v = getattr(ov, name)
            if v is not None:
                setattr(s, name, v)
        return s

    # --- content -----------------------------------------------------------------
    @staticmethod
    def apply_to_content(lesson: LessonContent, ov: Override) -> LessonContent:
        from ..schemas import Formula, SectionKind
        out = copy.deepcopy(lesson)
        if ov.replace_sections is not None:
            out.sections = [ContentSection(kind=SectionKind(s.get("kind", "definition")), title=str(s["title"]),
                                           bullets=list(s.get("bullets", [])), narration=str(s.get("narration", "")),
                                           formulas=[Formula(f) for f in s.get("formulas", [])])
                            for s in ov.replace_sections]
        for idx, edit in ov.section_edits.items():
            idx = int(idx)
            if not 0 <= idx < len(out.sections):
                continue
            sec = out.sections[idx]
            for k, v in edit.items():
                if k == "formulas":
                    sec.formulas = [Formula(f) for f in v]
                elif hasattr(sec, k):
                    setattr(sec, k, v)
        return out

    # --- preview / publish -----------------------------------------------------------
    def preview(self, lesson: LessonContent, settings: GenerationSettings, ov: Override) -> str:
        token = f"{ov.student_id}-{int(time.time() * 1000)}"
        self._pending[token] = {"lesson": self.apply_to_content(lesson, ov),
                                "settings": self.apply_to_settings(settings, ov), "override": ov}
        return token

    def get_preview(self, token: str):
        p = self._pending[token]
        return p["lesson"], p["settings"]

    def publish(self, token: str):
        p = self._pending.pop(token)
        self._log(p["override"], published=True)
        return p["lesson"], p["settings"]

    def discard(self, token: str) -> None:
        p = self._pending.pop(token, None)
        if p:
            self._log(p["override"], published=False)

    def _log(self, ov: Override, published: bool) -> None:
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "published": published, **asdict(ov)}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=lambda o: o.value if isinstance(o, Enum) else str(o)) + "\n")
