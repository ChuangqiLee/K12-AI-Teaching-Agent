"""Minimal Python API example: diagnostic -> adaptive settings -> teacher override -> lesson.

    python examples/quickstart.py            # offline (no models)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from k12agent.adaptive import BaselineDiagnostic, Override  # noqa: E402
from k12agent.config import REPO_ROOT, load_config  # noqa: E402
from k12agent.pipeline import TeachingAgent  # noqa: E402
from k12agent.schemas import Difficulty, LearnerProfile  # noqa: E402

cfg = load_config(REPO_ROOT / "configs" / "offline_demo.yaml")
agent = TeachingAgent(cfg)

# 1. baseline diagnostic (here: simulated answers, 4 of 6 correct -> "emerging")
profile = LearnerProfile(student_id="s001", grade=8, region="rural", pace="medium")
diag = BaselineDiagnostic(n_items=6, seed=0)
items = diag.draw(profile.stage)
answers = [it.answer if k < 4 else -1 for k, it in enumerate(items)]
result = diag.score(items, answers)
profile.band = result.band
print(f"diagnostic: {result.score:.0%} -> {result.band.value}")

# 2. optional teacher override (logged to outputs/logs/overrides.jsonl when published via the panel)
override = Override(teacher_id="t001", student_id="s001", difficulty=Difficulty.ON_LEVEL,
                    localization="用河南本地学校操场举例 / use the local school playground as the example")

# 3. answer a question end-to-end
pkg = agent.answer("勾股定理是怎么推导出来的？", profile, override=override)
print("slides :", len(pkg.content.sections))
print("pptx   :", pkg.pptx_path)
print("video  :", pkg.video_path)
for t, s in zip(pkg.timings, pkg.content.sections):
    print(f"  [{t.start_s:6.1f}s +{t.duration_s:5.1f}s] {s.kind.value:<12} {s.title}")
