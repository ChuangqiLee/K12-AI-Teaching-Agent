"""Stage-specific slide themes (Figure 8) and CJK font discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from ..config import resolve_path
from ..schemas import Stage

RGB = Tuple[int, int, int]


@dataclass(frozen=True)
class Theme:
    name: str
    background: RGB
    title_bar: RGB
    title_color: RGB
    text_color: RGB
    accent: RGB          # emphasis / visual signalling colour
    label: dict          # stage label shown in the corner (zh / en)


THEMES = {
    # light green "primary maths" deck (Figure 8, top)
    Stage.PRIMARY: Theme("primary", (220, 245, 200), (46, 160, 67), (255, 255, 255), (30, 30, 30),
                         (214, 69, 65), {"zh": "小学数学", "en": "Primary Maths"}),
    # dark chalkboard "junior maths" deck (Figure 8, bottom)
    Stage.JUNIOR: Theme("junior", (38, 34, 32), (70, 62, 58), (255, 255, 255), (245, 245, 240),
                        (255, 214, 102), {"zh": "初中数学", "en": "Junior Maths"}),
    Stage.SENIOR: Theme("senior", (245, 247, 252), (28, 52, 110), (255, 255, 255), (20, 24, 40),
                        (200, 60, 40), {"zh": "高中数学", "en": "Senior Maths"}),
}

_FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def find_font(preferred: Optional[str] = None) -> Optional[str]:
    if preferred:
        p = resolve_path(preferred)
        if p is not None and p.exists():
            return str(p)
    for c in _FONT_CANDIDATES:
        if Path(c).exists():
            return c
    return None


def pptx_font_name(language: str) -> str:
    return "Microsoft YaHei" if language == "zh" else "Calibri"
