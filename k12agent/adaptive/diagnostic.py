"""Initial assessment and adaptive content delivery (Section 4.1.1).

* Baseline diagnostic quiz of 5-7 grade-aligned items, auto-scored.
* Proficiency bands: Low (< 50 %), Emerging (50-79 %), Proficient (>= 80 %).
* Adaptation policy: difficulty (prerequisite review -> on-level -> enrichment),
  scaffolds (worked examples with fading, stepwise derivation, hints, signalling),
  pacing (micro-chunk length, narration speed, pause prompts).
* Mastery gating: promote when the learner answers two consecutive items of the
  current subskill correctly, or reaches >= 80 % within a block.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..schemas import Band, Difficulty, GenerationSettings, LearnerProfile, Scaffolds, Stage

QUIZ_BANK = Path(__file__).resolve().parent / "data" / "quiz_bank.json"


@dataclass
class QuizItem:
    id: str
    stage: Stage
    subskill: str
    prompt: Dict[str, str]
    choices: List[str]
    answer: int


def load_item_bank(path: str | Path = QUIZ_BANK) -> List[QuizItem]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)["items"]
    return [QuizItem(i["id"], Stage(i["stage"]), i["subskill"], i["prompt"], i["choices"], int(i["answer"])) for i in raw]


@dataclass
class DiagnosticResult:
    score: float                               # proportion correct in [0, 1]
    band: Band
    per_subskill: Dict[str, float]
    weak_subskills: List[str]
    overridden_by: Optional[str] = None        # teacher id when the band was overridden


def band_from_score(score: float, low: float = 0.50, proficient: float = 0.80) -> Band:
    if score < low:
        return Band.LOW
    if score < proficient:
        return Band.EMERGING
    return Band.PROFICIENT


class BaselineDiagnostic:
    def __init__(self, items: Optional[Sequence[QuizItem]] = None, n_items: int = 6, low: float = 0.50,
                 proficient: float = 0.80, seed: Optional[int] = None):
        if not 5 <= n_items <= 7:
            raise ValueError("the baseline diagnostic uses 5-7 items")
        self.items = list(items) if items is not None else load_item_bank()
        self.n_items, self.low, self.proficient = n_items, low, proficient
        self.rng = random.Random(seed)

    def draw(self, stage: Stage) -> List[QuizItem]:
        pool = [i for i in self.items if i.stage == stage]
        by_skill: Dict[str, List[QuizItem]] = defaultdict(list)
        for it in pool:
            by_skill[it.subskill].append(it)
        chosen: List[QuizItem] = []
        skills = list(by_skill)
        self.rng.shuffle(skills)
        while len(chosen) < min(self.n_items, len(pool)):   # round-robin over subskills for coverage
            for s in skills:
                rest = [i for i in by_skill[s] if i not in chosen]
                if rest and len(chosen) < self.n_items:
                    chosen.append(self.rng.choice(rest))
        return chosen

    def score(self, items: Sequence[QuizItem], responses: Sequence[int]) -> DiagnosticResult:
        if len(items) != len(responses):
            raise ValueError("one response per item expected")
        correct = [int(r == it.answer) for it, r in zip(items, responses)]
        per: Dict[str, List[int]] = defaultdict(list)
        for it, c in zip(items, correct):
            per[it.subskill].append(c)
        per_skill = {k: sum(v) / len(v) for k, v in per.items()}
        score = sum(correct) / max(1, len(correct))
        return DiagnosticResult(score, band_from_score(score, self.low, self.proficient), per_skill,
                                sorted(k for k, v in per_skill.items() if v < 0.5))


class MasteryGate:
    """Promotion rule = two consecutive correct items on the current subskill OR >= 80 % within a block."""

    def __init__(self, consecutive: int = 2, block_ratio: float = 0.80, block_size: int = 5):
        self.consecutive, self.block_ratio, self.block_size = consecutive, block_ratio, block_size
        self.history: Dict[str, List[int]] = defaultdict(list)

    def record(self, subskill: str, correct: bool) -> bool:
        """Record one answer; return True when the learner is promoted on this subskill."""
        h = self.history[subskill]
        h.append(int(correct))
        if len(h) >= self.consecutive and all(h[-self.consecutive:]):
            return True
        block = h[-self.block_size:]
        return len(block) >= self.block_size and sum(block) / len(block) >= self.block_ratio

    def reset(self, subskill: str) -> None:
        self.history.pop(subskill, None)


class AdaptationPolicy:
    """Maps (stage, band) to GenerationSettings."""

    def settings_for(self, profile: LearnerProfile, band: Optional[Band] = None, language: str = "zh") -> GenerationSettings:
        band = band or profile.band
        stage = profile.stage
        if band == Band.LOW:
            s = GenerationSettings(stage, Difficulty.REMEDIAL,
                                   Scaffolds(worked_examples=True, stepwise_derivation=True, visual_signaling=True,
                                             targeted_hints=True, fading=True),
                                   pace="slow", micro_chunk_sentences=1, pause_prompts=True)
        elif band == Band.EMERGING:
            s = GenerationSettings(stage, Difficulty.ON_LEVEL,
                                   Scaffolds(worked_examples=True, stepwise_derivation=True, visual_signaling=True,
                                             targeted_hints=False, fading=True),
                                   pace=profile.pace or "medium", micro_chunk_sentences=2, pause_prompts=True)
        else:
            s = GenerationSettings(stage, Difficulty.ENRICHMENT,
                                   Scaffolds(worked_examples=False, stepwise_derivation=True, visual_signaling=True,
                                             targeted_hints=False, fading=False),
                                   pace=profile.pace or "medium", micro_chunk_sentences=3, pause_prompts=False)
        if stage == Stage.SENIOR:
            # Section 4.7 remediation: senior decks must always keep full step-by-step derivations
            s.scaffolds.stepwise_derivation = True
            s.micro_chunk_sentences = min(s.micro_chunk_sentences, 2)
        s.language = language
        return s
