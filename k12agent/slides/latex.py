"""LaTeX formula rendering for slides (Section 2.1, "LaTeX-rendered formulas").

Uses matplotlib's mathtext so that no TeX installation is required.  If
``usetex=True`` and a TeX distribution is available, full LaTeX is used.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_MATHTEXT_FIXES = {
    r"\iff": r"\Leftrightarrow",
    r"\implies": r"\Rightarrow",
    r"\dfrac": r"\frac",
    r"\tfrac": r"\frac",
    r"\text": r"\mathrm",
    r"\left": "",
    r"\right": "",
}


def sanitize_for_mathtext(latex: str) -> str:
    s = latex.strip().strip("$")
    for a, b in _MATHTEXT_FIXES.items():
        s = s.replace(a, b)
    return s


def render_formula(latex: str, out_dir: str | Path, fontsize: int = 40, dpi: int = 200,
                   color: str = "black", usetex: bool = False) -> Path:
    """Render ``latex`` to a transparent PNG and return its path (cached by hash)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"{latex}|{fontsize}|{dpi}|{color}|{usetex}".encode()).hexdigest()[:16]
    path = out_dir / f"formula_{key}.png"
    if path.exists():
        return path
    expr = sanitize_for_mathtext(latex)
    for attempt in (f"${expr}$", expr):  # fall back to plain text if mathtext fails
        fig = plt.figure(figsize=(0.01, 0.01))
        try:
            with plt.rc_context({"text.usetex": usetex, "mathtext.fontset": "cm"}):
                fig.text(0, 0, attempt, fontsize=fontsize, color=color)
                fig.savefig(path, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.05)
            return path
        except Exception:  # noqa: BLE001 - mathtext raises many error types
            continue
        finally:
            plt.close(fig)
    raise ValueError(f"cannot render formula: {latex}")
