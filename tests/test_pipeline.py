"""Offline end-to-end tests (template content, dummy TTS, no lip-sync)."""

import json

import pytest

from k12agent.adaptive import AdaptationPolicy, BaselineDiagnostic, MasteryGate, Override, OverridePanel, band_from_score
from k12agent.config import load_config
from k12agent.content.generator import TemplateContentGenerator, _extract_json, lesson_from_json
from k12agent.content.knowledge_base import KnowledgeBase
from k12agent.content.prompts import build_prompt
from k12agent.config import REPO_ROOT
from k12agent.schemas import Band, Difficulty, GenerationSettings, LearnerProfile, SectionKind, Stage
from k12agent.slides.pacing import PacingController, count_words, narration_seconds
from k12agent.speech.prosody import plan_prosody, verbalize_latex
from k12agent.subjects import balance_reaction, check_units
from k12agent.tracking import OffTaskRule

KB = KnowledgeBase.load(REPO_ROOT / "k12agent/content/data/math_kg.json")


def test_retrieval_and_concepts():
    assert KB.retrieve("勾股定理是怎么推导出来的？")[0].id == "pythagorean"
    assert KB.retrieve("How is the Pythagorean theorem derived?")[0].id == "pythagorean"
    assert KB.retrieve("小明吃了蛋糕的1/4，小红吃了2/4，谁吃得多？", Stage.PRIMARY)[0].id == "fraction_compare"
    assert KB.retrieve("什么是导数？", Stage.SENIOR)[0].id == "derivative_concept"
    assert "斜边" in KB.extract_concepts("直角三角形的斜边怎么求")


def test_template_generator_sequence():
    gen = TemplateContentGenerator(KB)
    lesson = gen.generate("勾股定理", GenerationSettings(stage=Stage.JUNIOR))
    kinds = [s.kind for s in lesson.sections]
    assert kinds[0] == SectionKind.DEFINITION and SectionKind.DERIVATION in kinds and kinds[-1] == SectionKind.APPLICATION
    remedial = gen.generate("导数", GenerationSettings(stage=Stage.SENIOR, difficulty=Difficulty.REMEDIAL))
    assert remedial.sections[0].title.startswith("先复习")          # prerequisite review
    enrich = gen.generate("Pythagorean theorem", GenerationSettings(stage=Stage.JUNIOR, difficulty=Difficulty.ENRICHMENT,
                                                                    language="en"))
    assert enrich.sections[-1].title == "Extension"


def test_prompt_and_json_parsing():
    s = GenerationSettings(stage=Stage.SENIOR, subject="physics")
    p = build_prompt("什么是导数", KB.context_block(KB.retrieve("导数")), s)
    assert "不得跳步" in p and "量纲" in p
    raw = 'Sure! ```json\n{"topic": "t", "sections": [{"kind": "proof", "title": "x", "bullets": ["a {b}"], ' \
          '"formulas": ["$a^2$"], "narration": "n"}]}\n``` done'
    lesson = lesson_from_json(_extract_json(raw), "q", s)
    assert lesson.sections[0].kind == SectionKind.DERIVATION and lesson.sections[0].formulas[0].latex == "a^2"


def test_diagnostic_bands_and_policy():
    assert band_from_score(0.3) == Band.LOW and band_from_score(0.5) == Band.EMERGING
    assert band_from_score(0.79) == Band.EMERGING and band_from_score(0.8) == Band.PROFICIENT
    diag = BaselineDiagnostic(n_items=6, seed=1)
    items = diag.draw(Stage.JUNIOR)
    assert len(items) == 6 and len({i.subskill for i in items}) >= 4
    res = diag.score(items, [i.answer for i in items])
    assert res.score == 1.0 and res.band == Band.PROFICIENT
    s = AdaptationPolicy().settings_for(LearnerProfile(grade=5), Band.LOW)
    assert s.difficulty == Difficulty.REMEDIAL and s.pace == "slow" and s.scaffolds.targeted_hints


def test_mastery_gate():
    g = MasteryGate()
    assert not g.record("x", True)
    assert g.record("x", True)                         # two consecutive correct
    g2 = MasteryGate(block_size=5)
    for c in [True, False, True, True, False]:
        promoted = g2.record("y", c)
    assert not promoted
    g3 = MasteryGate(consecutive=10, block_size=5)
    results = [g3.record("z", c) for c in [True, True, False, True, True]]
    assert results[-1]                                 # 80 % within the block


def test_override_preview_publish(tmp_path):
    gen = TemplateContentGenerator(KB)
    s = GenerationSettings(stage=Stage.JUNIOR)
    lesson = gen.generate("勾股定理", s)
    panel = OverridePanel(tmp_path / "ov.jsonl")
    ov = Override("t1", "s1", difficulty=Difficulty.ENRICHMENT, pace="slow",
                  section_edits={0: {"bullets": ["新的题干：用人民币举例"]}})
    token = panel.preview(lesson, s, ov)
    new_lesson, new_s = panel.publish(token)
    assert new_s.pace == "slow" and new_s.difficulty == Difficulty.ENRICHMENT
    assert new_lesson.sections[0].bullets == ["新的题干：用人民币举例"]
    assert lesson.sections[0].bullets != new_lesson.sections[0].bullets      # original untouched
    rec = json.loads((tmp_path / "ov.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rec["published"] and rec["teacher_id"] == "t1"


def test_pacing():
    assert count_words("one two three") == 3
    assert narration_seconds("a " * 110, 110) == pytest.approx(60)
    gen = TemplateContentGenerator(KB)
    secs = gen.generate("勾股定理", GenerationSettings(stage=Stage.JUNIOR)).sections
    slow, fast = PacingController(wpm=70), PacingController(wpm=150)
    assert sum(t.duration_s for t in slow.schedule(secs)) > sum(t.duration_s for t in fast.schedule(secs))
    t = PacingController().schedule(secs, [10.0] * len(secs))
    assert all(x.duration_s >= 10.0 for x in t)                              # never before narration ends
    assert t[1].start_s == pytest.approx(t[0].duration_s)


def test_verbalizer_and_prosody():
    zh = verbalize_latex(r"\lim_{x \to 0} \frac{\sin x}{x} = 1", "zh")
    assert "趋近于 0" in zh and "极限" in zh and "等于 1" in zh
    assert verbalize_latex("a^2 + b^2 = c^2", "en") == "a squared plus b squared equals c squared"
    chunks = plan_prosody("因此 a² + b² = c²。想一想，下一步该怎么做？", language="zh")
    assert chunks[0].emphasis and chunks[-1].pause_after_s > chunks[0].pause_after_s


def test_subject_plugins():
    assert balance_reaction("H2 + O2 -> H2O") == "2H2 + O2 -> 2H2O"
    assert balance_reaction("CH4 + O2 -> CO2 + H2O") == "CH4 + 2O2 -> CO2 + 2H2O"
    assert check_units("kilogram*meter/second**2", "newton")
    assert not check_units("kilogram*meter/second", "newton")


def test_off_task_rule():
    r = OffTaskRule(5.0)
    assert r.update(False, 0.0) is None
    assert r.update(False, 4.9) is None
    assert r.update(False, 5.1) == "off_task"
    assert r.update(True, 6.0) == "on_task"


def test_end_to_end_offline(tmp_path):
    from k12agent.pipeline import TeachingAgent
    cfg = load_config(overrides={"content": {"backend": "template"}, "speech": {"backend": "dummy"},
                                 "video": {"backend": "none"}, "output_dir": str(tmp_path)})
    agent = TeachingAgent(cfg)
    pkg = agent.answer("勾股定理是怎么推导出来的？", LearnerProfile(grade=8, pace="fast"), render_video=False)
    assert pkg.pptx_path and (tmp_path / pkg.question.qid / "lesson.pptx").exists()
    assert len(pkg.slide_images) == len(pkg.content.sections) == len(pkg.audio) == len(pkg.timings)
    assert json.loads((tmp_path / pkg.question.qid / "lesson.json").read_text(encoding="utf-8"))["content"]["topic"] == "勾股定理"
