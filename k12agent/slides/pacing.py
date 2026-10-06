"""Slide pacing: narration-rate presets and the adaptive nonlinear dwell function.

Section 4.5.1: three PPT auto-advance paces - slow (70 wpm), medium (110 wpm),
fast (150 wpm).

Section 3.4: "an adaptive nonlinear function to dynamically regulate the pace
of content display" so that slides never change before (or long after) the
corresponding explanation.  The paper does not print the function; we use

    dwell_i = max(min_dwell, audio_i) + max_extra * sigmoid(k * (rho_i - rho_0))

where audio_i is the narration length of slide i and rho_i in [0, 1] is its
visual density (formulas + amount of on-slide text).  Dense slides therefore
get up to ``max_extra`` seconds of extra reading time, sparse slides advance as
soon as the narration ends.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..schemas import ContentSection, SlideTiming

PACE_WPM = {"slow": 70, "medium": 110, "fast": 150}
_CJK = re.compile(r"[㐀-鿿]")
_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
CJK_CHARS_PER_WORD = 1.5   # average Mandarin word length in characters


def count_words(text: str) -> float:
    """Language-agnostic word count (CJK characters are converted to words)."""
    return len(_WORD.findall(text)) + len(_CJK.findall(text)) / CJK_CHARS_PER_WORD


def narration_seconds(text: str, wpm: float) -> float:
    return 60.0 * count_words(text) / max(wpm, 1e-6)


def visual_density(section: ContentSection, max_chars: int = 220, max_formulas: int = 3) -> float:
    chars = sum(len(b) for b in section.bullets)
    rho = 0.6 * min(1.0, chars / max_chars) + 0.4 * min(1.0, len(section.formulas) / max_formulas)
    return float(min(1.0, rho))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class PacingController:
    wpm: float = 110.0
    min_dwell_s: float = 3.0
    max_extra_s: float = 6.0
    density_midpoint: float = 0.5
    density_slope: float = 8.0

    @classmethod
    def from_config(cls, cfg, preset: Optional[str] = None) -> "PacingController":
        p = cfg.pacing
        preset = preset or p.preset
        return cls(wpm=float(p.wpm[preset]), min_dwell_s=p.min_dwell_s, max_extra_s=p.max_extra_s,
                   density_midpoint=p.density_midpoint, density_slope=p.density_slope)

    def extra_dwell(self, density: float) -> float:
        return self.max_extra_s * _sigmoid(self.density_slope * (density - self.density_midpoint))

    def slide_duration(self, section: ContentSection, audio_seconds: Optional[float] = None) -> float:
        speech = audio_seconds if audio_seconds is not None else narration_seconds(section.narration, self.wpm)
        return max(self.min_dwell_s, speech) + self.extra_dwell(visual_density(section))

    def schedule(self, sections: Sequence[ContentSection],
                 audio_seconds: Optional[Sequence[Optional[float]]] = None) -> List[SlideTiming]:
        timings, t = [], 0.0
        for i, s in enumerate(sections):
            a = audio_seconds[i] if audio_seconds is not None else None
            d = self.slide_duration(s, a)
            timings.append(SlideTiming(index=i, start_s=round(t, 3), duration_s=round(d, 3)))
            t += d
        return timings

    def tts_rate(self, natural_wpm: float) -> float:
        """Time-stretch factor that maps the TTS' natural rate to the chosen preset."""
        return self.wpm / max(natural_wpm, 1e-6)
