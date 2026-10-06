"""Subject adaptation hooks (Section 3.5, Section 4.1.3).

The HCI levers (segmenting, signalling, social-agency cues) stay fixed; only
domain-facing components are swapped:  prompt-template additions, plug-ins
(unit-aware solver, reaction balancer, ...), representations/scaffolds,
assessment templates and safety reminders.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from math import lcm
from typing import Callable, Dict, List, Optional

from ..schemas import ContentSection, LessonContent, SectionKind


@dataclass
class SubjectPack:
    name: str
    prompt_addendum: Dict[str, str]
    representations: List[str]
    assessment: List[str]
    safety: Dict[str, str] = field(default_factory=dict)
    post_processors: List[Callable[[LessonContent], LessonContent]] = field(default_factory=list)

    def apply(self, lesson: LessonContent, language: str = "zh", lab_context: bool = False) -> LessonContent:
        for fn in self.post_processors:
            lesson = fn(lesson)
        if lab_context and self.safety:
            lesson.sections.insert(0, ContentSection(
                kind=SectionKind.SUMMARY, title="实验安全" if language == "zh" else "Lab safety",
                bullets=[self.safety.get(language, self.safety["en"])],
                narration=self.safety.get(language, self.safety["en"])))
        return lesson


# ------------------------------------------------------------------ physics plug-in
def check_units(expression: str, expected: str) -> bool:
    """Dimensional analysis with sympy.physics.units, e.g. check_units('kg*m/s**2', 'newton')."""
    from sympy.physics import units as u
    from sympy.physics.units.systems.si import SI, dimsys_SI

    ns = {n: getattr(u, n) for n in dir(u) if not n.startswith("_")}
    lhs, rhs = eval(expression, {"__builtins__": {}}, ns), eval(expected, {"__builtins__": {}}, ns)  # noqa: S307
    dim = lambda q: dimsys_SI.get_dimensional_dependencies(SI.get_dimensional_expr(q))  # noqa: E731
    return dim(lhs) == dim(rhs)


# ------------------------------------------------------------------ chemistry plug-in
_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)|(\()|(\))(\d*)")


def parse_formula(formula: str) -> Dict[str, int]:
    stack: List[Dict[str, int]] = [{}]
    for el, n, lpar, rpar, rn in _TOKEN.findall(formula):
        if lpar:
            stack.append({})
        elif rpar:
            group = stack.pop()
            mult = int(rn or 1)
            for k, v in group.items():
                stack[-1][k] = stack[-1].get(k, 0) + v * mult
        elif el:
            stack[-1][el] = stack[-1].get(el, 0) + int(n or 1)
    return stack[0]


def balance_reaction(equation: str) -> str:
    """'H2 + O2 -> H2O' -> '2H2 + O2 -> 2H2O' (null-space of the element matrix)."""
    from sympy import Matrix

    left, right = [side.split("+") for side in re.split(r"->|=|→", equation)]
    species = [s.strip() for s in left + right]
    comps = [parse_formula(s) for s in species]
    elements = sorted({e for c in comps for e in c})
    rows = [[c.get(e, 0) * (1 if i < len(left) else -1) for i, c in enumerate(comps)] for e in elements]
    ns = Matrix(rows).nullspace()
    if not ns:
        raise ValueError(f"cannot balance {equation}")
    vec = [Fraction(str(x)) for x in ns[0]]
    m = lcm(*[f.denominator for f in vec])
    coeffs = [int(abs(f * m)) for f in vec]
    fmt = lambda c, s: (str(c) if c != 1 else "") + s  # noqa: E731
    return " + ".join(fmt(c, s) for c, s in zip(coeffs[:len(left)], species[:len(left)])) + " -> " + \
        " + ".join(fmt(c, s) for c, s in zip(coeffs[len(left):], species[len(left):]))


def _balance_in_lesson(lesson: LessonContent) -> LessonContent:
    pattern = re.compile(r"([A-Z][A-Za-z0-9()]*(?:\s*\+\s*[A-Z][A-Za-z0-9()]*)*\s*(?:->|→)\s*[A-Z][A-Za-z0-9()]*(?:\s*\+\s*[A-Z][A-Za-z0-9()]*)*)")
    for sec in lesson.sections:
        new = []
        for b in sec.bullets:
            def repl(m):
                try:
                    return balance_reaction(m.group(1))
                except Exception:  # noqa: BLE001
                    return m.group(1)
            new.append(pattern.sub(repl, b))
        sec.bullets = new
    return lesson


# ------------------------------------------------------------------ registry
SUBJECTS: Dict[str, SubjectPack] = {
    "math": SubjectPack(
        "math",
        {"zh": "", "en": ""},
        ["LaTeX formulas", "step-by-step derivations", "worked examples"],
        ["pre/post standardised test", "mastery gating per subskill"]),
    "physics": SubjectPack(
        "physics",
        {"zh": "每个数值结果必须带单位并做量纲检查；同时给出示意图描述、公式和文字三种表征；矢量用箭头表示，受力分析给出受力图。",
         "en": "Every numeric result must carry units and pass a dimensional check; give diagram + equation + verbal representations; draw vectors as arrows and include free-body diagrams."},
        ["vector arrows", "free-body diagrams", "unit highlighting"],
        ["numeric problems with unit checks", "multi-step worked examples", "partial-credit feedback"],
        {"zh": "实验前确认电源已断开，佩戴护目镜，禁止用手直接触碰高温器材。",
         "en": "Before the experiment, disconnect power, wear goggles and never touch hot apparatus directly."}),
    "chemistry": SubjectPack(
        "chemistry",
        {"zh": "化学方程式必须配平；给出结构式与反应路径；涉及危险试剂时加安全标识。",
         "en": "All chemical equations must be balanced; show structural formulas and reaction pathways; add safety badges for hazardous reagents."},
        ["structural formulas", "reaction pathways", "safety badges"],
        ["balanced-reaction accuracy", "structural identification / nomenclature", "lab-safety compliance"],
        {"zh": "在通风橱中操作挥发性试剂，佩戴手套和护目镜，严禁品尝或直接闻试剂。",
         "en": "Handle volatile reagents in a fume hood, wear gloves and goggles, never taste or directly smell chemicals."},
        [_balance_in_lesson]),
    "sociology": SubjectPack(
        "sociology",
        {"zh": "按照“观点—证据—推理”（CER）结构组织讲解；引用材料需注明来源并提示检查偏见。",
         "en": "Organise explanations as claim-evidence-reasoning (CER); cite sources and prompt a bias/source check."},
        ["excerpts", "tables / figures", "concept maps", "bias / source-check prompts"],
        ["rubric-based short answers (argument structure, use of evidence)", "formative hints"]),
}


def get_subject(name: str) -> SubjectPack:
    try:
        return SUBJECTS[name]
    except KeyError as exc:
        raise ValueError(f"unknown subject {name!r}; available: {sorted(SUBJECTS)}") from exc


def subject_prompt_addendum(name: str, language: str = "zh") -> Optional[str]:
    text = get_subject(name).prompt_addendum.get(language, "")
    return text or None
