"""Gradio front-end (Section 4.5.1 student workflow + Section 2 override panel).

Student tab   : personal setting panel (default / 5-s cloned voice, teacher image,
                PPT pace slow 70 / medium 110 / fast 150 wpm, subtitle size, theme)
                and the interactive Q&A workspace (typed or spoken question).
Diagnostic tab: 5-7 item baseline quiz -> proficiency band.
Teacher tab   : override panel - difficulty, scaffolds, pacing, localisation,
                slide edits; preview before one-click publish; all edits logged.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Optional

from ..adaptive.diagnostic import BaselineDiagnostic
from ..adaptive.override import InterventionPoint, Override
from ..config import Config
from ..pipeline import TeachingAgent
from ..schemas import Band, Difficulty, LearnerProfile, Question
from ..tracking.logger import BehaviorLogger


def transcribe(audio_path: Optional[str], language: str = "zh") -> str:
    """Spoken questions: uses faster-whisper or openai-whisper if installed."""
    if not audio_path:
        return ""
    try:
        from faster_whisper import WhisperModel
        segs, _ = WhisperModel("small").transcribe(audio_path, language=language)
        return "".join(s.text for s in segs).strip()
    except ImportError:
        pass
    try:
        import whisper
        return whisper.load_model("small").transcribe(audio_path, language=language)["text"].strip()
    except ImportError:
        return ""


def launch(cfg: Config, port: int = 7860, share: bool = False):
    import gradio as gr

    agent = TeachingAgent(cfg)
    diag = BaselineDiagnostic(n_items=cfg.adaptive.quiz_items, low=cfg.adaptive.bands.low,
                              proficient=cfg.adaptive.bands.proficient)
    lang = cfg.language
    state_defaults = {"band": Band.EMERGING.value, "items": [], "last_lesson": None, "preview": None}

    def ask(question, spoken, student_id, grade, region, use_clone, voice_file, face_file, pace, subtitle, theme, state):
        text = (question or "").strip() or transcribe(spoken, lang)
        if not text:
            return None, None, [], "请输入或说出你的问题 / please type or speak a question", state
        logger = BehaviorLogger(cfg.tracking.log_dir, student_id or "anonymous")
        logger.log("settings", pace=pace, cloned_voice=bool(use_clone and voice_file), subtitle=subtitle, theme=theme)
        logger.log("question", text=text, modality="speech" if spoken and not question else "text")
        profile = LearnerProfile(student_id=student_id or "anonymous", grade=int(grade), region=region,
                                 band=Band(state["band"]), voice_ref=voice_file if use_clone else None,
                                 face_ref=face_file, pace=pace, subtitle_size=int(subtitle), theme=theme)
        pkg = agent.answer(Question(text, student_id=profile.student_id), profile)
        state["last_lesson"] = pkg.content
        state["profile"] = asdict(profile)
        logger.close()
        info = json.dumps({"topic": pkg.content.topic, "metrics": pkg.metrics, "errors": pkg.errors},
                          ensure_ascii=False, indent=2)
        return pkg.video_path, pkg.pptx_path, pkg.slide_images, info, state

    def start_quiz(grade, state):
        items = diag.draw(LearnerProfile(grade=int(grade)).stage)
        state["items"] = [it.id for it in items]
        text = "\n\n".join(f"**{k + 1}.** {it.prompt.get(lang, it.prompt['en'])}  \n" +
                           "  ".join(f"{chr(65 + j)}. {c}" for j, c in enumerate(it.choices))
                           for k, it in enumerate(items))
        return text, state

    def submit_quiz(answers, state):
        bank = {it.id: it for it in diag.items}
        items = [bank[i] for i in state["items"]]
        letters = [a for a in (answers or "").upper().replace(",", " ").split() if a]
        responses = [ord(a[0]) - 65 for a in letters] + [-1] * (len(items) - len(letters))
        res = diag.score(items, responses[:len(items)])
        state["band"] = res.band.value
        return f"score = {res.score:.0%} → band = **{res.band.value}**; weak subskills: {res.weak_subskills}", state

    def teacher_preview(teacher_id, point, band, difficulty, worked, stepwise, hints, pace, chunk, local, edits, state):
        if state.get("last_lesson") is None:
            return "No lesson generated yet.", state
        ov = Override(teacher_id=teacher_id or "teacher", student_id=state.get("profile", {}).get("student_id", "?"),
                      point=InterventionPoint(point), band=Band(band) if band else None,
                      difficulty=Difficulty(difficulty) if difficulty else None,
                      scaffolds={"worked_examples": worked, "stepwise_derivation": stepwise, "targeted_hints": hints},
                      pace=pace or None, micro_chunk_sentences=int(chunk) if chunk else None,
                      localization=local or None, section_edits=json.loads(edits) if edits and edits.strip() else {})
        profile = LearnerProfile(**{k: v for k, v in state.get("profile", {}).items() if k in LearnerProfile.__dataclass_fields__})
        settings = agent.settings_for(profile, ov)
        token = agent.overrides.preview(state["last_lesson"], settings, ov)
        state["preview"] = token
        lesson, s = agent.overrides.get_preview(token)
        preview = "\n".join(f"[{i}] {sec.title}: " + " / ".join(sec.bullets) for i, sec in enumerate(lesson.sections))
        return f"difficulty={s.difficulty.value}, pace={s.pace}\n\n{preview}", state

    def teacher_publish(state):
        if not state.get("preview"):
            return None, None, "Nothing to publish.", state
        lesson, settings = agent.overrides.publish(state.pop("preview"))
        profile = LearnerProfile(**{k: v for k, v in state.get("profile", {}).items() if k in LearnerProfile.__dataclass_fields__})

        class _Fixed:  # serve the edited lesson without regenerating it
            def generate(self, q, s):
                return lesson
        published = TeachingAgent(cfg, generator=_Fixed(), voice=agent._voice, lipsync=agent._lipsync)
        pkg = published.answer(Question(lesson.question, student_id=profile.student_id), profile, settings=settings)
        return pkg.video_path, pkg.pptx_path, "Published (override logged).", state

    with gr.Blocks(title="K-12 AI Teaching Agent") as demo:
        state = gr.State(dict(state_defaults))
        gr.Markdown("## K-12 AI Teaching Agent / 智能教学体")
        with gr.Tab("Student / 学生"):
            with gr.Row():
                with gr.Column(scale=1):
                    student_id = gr.Textbox(label="Student ID", value="s001")
                    grade = gr.Slider(1, 12, value=8, step=1, label="Grade")
                    region = gr.Radio(["urban", "rural"], value="urban", label="Region")
                    use_clone = gr.Checkbox(label="Use 5-second cloned voice", value=False)
                    voice_file = gr.Audio(type="filepath", label="Reference voice (5 s)")
                    face_file = gr.Image(type="filepath", label="Teacher appearance")
                    pace = gr.Radio(["slow", "medium", "fast"], value="medium", label="PPT pace (70/110/150 wpm)")
                    subtitle = gr.Slider(20, 56, value=32, step=2, label="Subtitle size")
                    theme = gr.Radio(["light", "dark"], value="light", label="Theme")
                with gr.Column(scale=2):
                    question = gr.Textbox(label="Question / 问题", placeholder="勾股定理是怎么推导出来的？")
                    spoken = gr.Audio(sources=["microphone"], type="filepath", label="…or ask by voice")
                    go = gr.Button("Ask / 提问", variant="primary")
                    video = gr.Video(label="Lecture video")
                    pptx = gr.File(label="Slides (.pptx)")
                    gallery = gr.Gallery(label="Slides", columns=3)
                    info = gr.Code(label="Info", language="json")
            go.click(ask, [question, spoken, student_id, grade, region, use_clone, voice_file, face_file, pace,
                           subtitle, theme, state], [video, pptx, gallery, info, state])
        with gr.Tab("Diagnostic / 诊断"):
            qgrade = gr.Slider(1, 12, value=8, step=1, label="Grade")
            qstart = gr.Button("Start quiz")
            qtext = gr.Markdown()
            qans = gr.Textbox(label="Answers (e.g. A C B A D B)")
            qsubmit = gr.Button("Submit")
            qres = gr.Markdown()
            qstart.click(start_quiz, [qgrade, state], [qtext, state])
            qsubmit.click(submit_quiz, [qans, state], [qres, state])
        with gr.Tab("Teacher override / 教师干预"):
            teacher_id = gr.Textbox(label="Teacher ID", value="t001")
            point = gr.Radio([p.value for p in InterventionPoint], value="after_diagnostic", label="Intervention point")
            band = gr.Dropdown([""] + [b.value for b in Band], value="", label="Re-classify band")
            difficulty = gr.Dropdown([""] + [d.value for d in Difficulty], value="", label="Difficulty")
            worked = gr.Checkbox(True, label="Worked examples")
            stepwise = gr.Checkbox(True, label="Stepwise derivation")
            hints = gr.Checkbox(False, label="Targeted hints")
            tpace = gr.Dropdown(["", "slow", "medium", "fast"], value="", label="Pace")
            chunk = gr.Number(value=None, label="Micro-chunk sentences")
            local = gr.Textbox(label="Localise examples (currency, units, school context)")
            edits = gr.Code(label='Slide edits JSON, e.g. {"1": {"bullets": ["..."]}}', language="json")
            prev = gr.Button("Preview")
            preview_out = gr.Textbox(label="Preview", lines=10)
            pub = gr.Button("Publish to student", variant="primary")
            tvideo, tpptx, tmsg = gr.Video(), gr.File(), gr.Markdown()
            prev.click(teacher_preview, [teacher_id, point, band, difficulty, worked, stepwise, hints, tpace, chunk,
                                         local, edits, state], [preview_out, state])
            pub.click(teacher_publish, [state], [tvideo, tpptx, tmsg, state])
    demo.launch(server_port=port, share=share)
