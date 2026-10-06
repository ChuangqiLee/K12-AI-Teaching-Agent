"""Text-side prosody planning and emotional modulation (Section 2.2, Section 3.2).

* verbalises formulas so the TTS can read them (``\\lim_{x\\to 0}\\frac{\\sin x}{x}=1``
  -> "当 x 趋近于 0 时，sin x 除以 x 的极限等于 1"),
* splits narration into prosodic chunks with strategic pauses,
* marks chunks that contain key operators ("lim", "积分", key concepts) for
  emphasis (gain + slight pitch raise, applied after synthesis).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

# ---------------------------------------------------------------- verbaliser
_ZH = {
    "frac": "{1} 分之 {0}", "frac_simple": "{0} 除以 {1}", "sqrt": "根号 {0}", "pow2": "{0} 的平方",
    "pow3": "{0} 的立方", "pow": "{0} 的 {1} 次方", "lim": "当 {0} 趋近于 {1} 时，{2} 的极限",
    "=": " 等于 ", "+": " 加 ", "-": " 减 ", r"\cdot": " 乘 ", r"\times": " 乘 ", r"\div": " 除以 ",
    ">": " 大于 ", "<": " 小于 ", r"\geq": " 大于等于 ", r"\leq": " 小于等于 ", r"\neq": " 不等于 ",
    r"\Delta": "德尔塔 ", r"\pi": "派", r"\infty": "无穷大", r"\iff": " 当且仅当 ", r"\Leftrightarrow": " 当且仅当 ",
    r"\angle": "角 ", r"\triangle": "三角形 ", "'": " 的导数", r"\sin": "sin ", r"\cos": "cos ", r"\tan": "tan ",
}
_EN = {
    "frac": "{0} over {1}", "frac_simple": "{0} divided by {1}", "sqrt": "the square root of {0}",
    "pow2": "{0} squared", "pow3": "{0} cubed", "pow": "{0} to the power {1}",
    "lim": "the limit as {0} approaches {1} of {2}", "=": " equals ", "+": " plus ", "-": " minus ",
    r"\cdot": " times ", r"\times": " times ", r"\div": " divided by ", ">": " is greater than ",
    "<": " is less than ", r"\geq": " is at least ", r"\leq": " is at most ", r"\neq": " is not equal to ",
    r"\Delta": "delta ", r"\pi": "pi", r"\infty": "infinity", r"\iff": " if and only if ",
    r"\Leftrightarrow": " if and only if ", r"\angle": "angle ", r"\triangle": "triangle ", "'": " prime",
    r"\sin": "sine ", r"\cos": "cosine ", r"\tan": "tangent ",
}

_BRACE = r"\{((?:[^{}]|\{[^{}]*\})*)\}"


def verbalize_latex(latex: str, language: str = "zh") -> str:
    v = _ZH if language == "zh" else _EN
    s = latex.strip().strip("$")
    s = s.replace(r"\left", "").replace(r"\right", "").replace(r"\,", " ").replace(r"\dfrac", r"\frac")

    def lim(m):
        var, to, rest = m.group(1).split(r"\to")[0].strip(), m.group(1).split(r"\to")[-1].strip(), m.group(2)
        return v["lim"].format(var, to, verbalize_latex(rest, language))

    s = re.sub(r"\\lim_" + _BRACE + r"\s*(.*?)(?==|$)", lim, s)
    for _ in range(3):  # nested fractions
        s = re.sub(r"\\frac" + _BRACE + _BRACE,
                   lambda m: v["frac" if len(m.group(1)) <= 2 and len(m.group(2)) <= 2 and language == "zh"
                               else "frac_simple"].format(m.group(1), m.group(2)), s)
    s = re.sub(r"\\sqrt" + _BRACE, lambda m: v["sqrt"].format(m.group(1)), s)
    s = re.sub(r"(\([^()]*\)|[A-Za-z0-9]+)\^\{?2\}?", lambda m: v["pow2"].format(m.group(1)), s)
    s = re.sub(r"(\([^()]*\)|[A-Za-z0-9]+)\^\{?3\}?", lambda m: v["pow3"].format(m.group(1)), s)
    s = re.sub(r"(\([^()]*\)|[A-Za-z0-9]+)\^" + _BRACE, lambda m: v["pow"].format(m.group(1), m.group(2)), s)
    s = re.sub(r"_\{?([A-Za-z0-9]+)\}?", r" \1", s)  # x_0 -> "x 0"
    for k in sorted((k for k in v if not k.isalpha() and k not in ("frac", "frac_simple")), key=len, reverse=True):
        s = s.replace(k, v[k])
    s = re.sub(r"\\[A-Za-z]+", "", s).replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------- chunks
@dataclass
class ProsodyChunk:
    text: str
    pause_after_s: float = 0.15
    emphasis: bool = False
    gain_db: float = 0.0
    pitch_semitones: float = 0.0


_KEY_OPERATORS_ZH = ("极限", "导数", "积分", "定理", "所以", "因此", "注意", "关键")
_KEY_OPERATORS_EN = ("limit", "derivative", "integral", "theorem", "therefore", "so ", "note", "key")
_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])|(?<=[.!?])\s+")


def plan_prosody(narration: str, emphasis_words: Sequence[str] = (), language: str = "zh",
                 modulate: bool = True) -> List[ProsodyChunk]:
    """Split narration into sentence chunks and assign pauses / emphasis."""
    text = re.sub(r"\$([^$]+)\$", lambda m: verbalize_latex(m.group(1), language), narration)
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    keys = tuple(emphasis_words) + (_KEY_OPERATORS_ZH if language == "zh" else _KEY_OPERATORS_EN)
    chunks = []
    for s in sentences:
        is_question = s.endswith(("？", "?"))
        is_key = modulate and any(k.lower() in s.lower() for k in keys)
        pause = 0.35
        if is_question:
            pause = 1.2      # "think about it" pause prompt
        elif is_key:
            pause = 0.6      # strategic pause after a key statement
        chunks.append(ProsodyChunk(
            text=s, pause_after_s=pause if modulate else 0.2, emphasis=is_key,
            gain_db=2.0 if is_key else 0.0, pitch_semitones=0.7 if is_key else 0.0))
    return chunks


def apply_emphasis(wav: np.ndarray, sr: int, chunk: ProsodyChunk) -> np.ndarray:
    if wav.size == 0 or float(np.abs(wav).max()) < 1e-6:   # silent (e.g. dummy backend)
        return wav
    if chunk.pitch_semitones:
        try:
            import librosa
            wav = librosa.effects.pitch_shift(wav.astype(np.float32), sr=sr, n_steps=chunk.pitch_semitones)
        except Exception:  # noqa: BLE001 - emphasis is best effort
            pass
    if chunk.gain_db:
        wav = wav * (10 ** (chunk.gain_db / 20))
    return np.clip(wav, -1.0, 1.0)


def concat_with_pauses(wavs: Iterable[np.ndarray], chunks: Sequence[ProsodyChunk], sr: int) -> np.ndarray:
    out = []
    for w, c in zip(wavs, chunks):
        out.append(w.astype(np.float32))
        out.append(np.zeros(int(sr * c.pause_after_s), dtype=np.float32))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
