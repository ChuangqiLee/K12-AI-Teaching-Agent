"""Command-line entry point.

Examples::

    # offline smoke test - no GPU / models needed
    python -m k12agent ask "勾股定理是怎么推导出来的？" --grade 8 \
        --content template --speech dummy --video none

    # full system (LLaMA + voice cloning + Wav2Lip)
    python -m k12agent ask "How is the Pythagorean theorem derived?" --grade 8 --lang en \
        --voice my_teacher_5s.wav --face my_teacher.jpg --pace medium

    # baseline diagnostic in the terminal
    python -m k12agent quiz --grade 5

    # web UI (student panel + teacher override panel)
    python -m k12agent ui --port 7860
"""

from __future__ import annotations

import argparse
import json
import logging
import time

from .config import load_config
from .schemas import Band, LearnerProfile, Question


def _overrides(args) -> dict:
    o: dict = {}
    if getattr(args, "content", None):
        o.setdefault("content", {})["backend"] = args.content
    if getattr(args, "speech", None):
        o.setdefault("speech", {})["backend"] = args.speech
    if getattr(args, "video", None):
        o.setdefault("video", {})["backend"] = args.video
    if getattr(args, "lang", None):
        o["language"] = args.lang
    if getattr(args, "out", None):
        o["output_dir"] = args.out
    return o


def cmd_ask(args) -> None:
    from .pipeline import TeachingAgent
    cfg = load_config(args.config, _overrides(args))
    agent = TeachingAgent(cfg)
    profile = LearnerProfile(student_id=args.student, grade=args.grade, region=args.region,
                             band=Band(args.band), voice_ref=args.voice, face_ref=args.face, pace=args.pace)
    t0 = time.time()
    pkg = agent.answer(Question(args.question, student_id=args.student), profile, render_video=not args.no_video)
    print(json.dumps({
        "topic": pkg.content.topic, "slides": len(pkg.content.sections), "pptx": pkg.pptx_path,
        "video": pkg.video_path, "metrics": pkg.metrics, "errors": pkg.errors,
        "elapsed_s": round(time.time() - t0, 1)}, ensure_ascii=False, indent=2))


def cmd_quiz(args) -> None:
    from .adaptive.diagnostic import AdaptationPolicy, BaselineDiagnostic
    cfg = load_config(args.config, _overrides(args))
    lang = cfg.language
    profile = LearnerProfile(grade=args.grade)
    diag = BaselineDiagnostic(n_items=cfg.adaptive.quiz_items, low=cfg.adaptive.bands.low,
                              proficient=cfg.adaptive.bands.proficient)
    items = diag.draw(profile.stage)
    responses = []
    for k, it in enumerate(items, 1):
        print(f"\n{k}. {it.prompt.get(lang, it.prompt['en'])}")
        for j, c in enumerate(it.choices):
            print(f"   {chr(65 + j)}. {c}")
        ans = input("> ").strip().upper()[:1]
        responses.append(ord(ans) - 65 if ans else -1)
    res = diag.score(items, responses)
    settings = AdaptationPolicy().settings_for(profile, res.band, lang)
    print(json.dumps({"score": res.score, "band": res.band.value, "weak_subskills": res.weak_subskills,
                      "difficulty": settings.difficulty.value, "pace": settings.pace}, ensure_ascii=False, indent=2))


def cmd_ui(args) -> None:
    from .app.webui import launch
    launch(load_config(args.config, _overrides(args)), port=args.port, share=args.share)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="k12agent", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--content", choices=["hf", "template"])
        sp.add_argument("--speech", choices=["rtvc", "dummy"])
        sp.add_argument("--video", choices=["wav2lip", "none"])
        sp.add_argument("--lang", choices=["zh", "en"])
        sp.add_argument("--out")

    a = sub.add_parser("ask", help="answer one question end-to-end")
    a.add_argument("question")
    a.add_argument("--grade", type=int, default=8)
    a.add_argument("--region", choices=["urban", "rural"], default="urban")
    a.add_argument("--band", choices=[b.value for b in Band], default="emerging")
    a.add_argument("--voice", help="5-second reference clip for cloning")
    a.add_argument("--face", help="teacher image or short video")
    a.add_argument("--pace", choices=["slow", "medium", "fast"], default="medium")
    a.add_argument("--student", default="demo")
    a.add_argument("--no-video", action="store_true")
    common(a)
    a.set_defaults(fn=cmd_ask)

    q = sub.add_parser("quiz", help="run the baseline diagnostic")
    q.add_argument("--grade", type=int, default=8)
    common(q)
    q.set_defaults(fn=cmd_quiz)

    u = sub.add_parser("ui", help="launch the Gradio web UI")
    u.add_argument("--port", type=int, default=7860)
    u.add_argument("--share", action="store_true")
    common(u)
    u.set_defaults(fn=cmd_ui)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args.fn(args)


if __name__ == "__main__":
    main()
