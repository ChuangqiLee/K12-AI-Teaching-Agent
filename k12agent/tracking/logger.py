"""Behavioural log capture (Table 2 "Behavioral Logs", Section 4.5).

Events: question, slide_view, pause, replay, scroll, quiz_response (with
response time), off_task, window_blur / window_focus, search / link_click /
notebook_edit (A2 condition), override.  One JSONL file per session.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

EVENT_TYPES = {
    "session_start", "session_end", "question", "slide_view", "pause", "resume", "replay", "scroll",
    "quiz_response", "off_task", "on_task", "window_blur", "window_focus", "search", "link_click",
    "notebook_edit", "idle", "override", "settings",
}


class BehaviorLogger:
    def __init__(self, log_dir: str | Path, student_id: str, condition: str = "A1", session_id: Optional[str] = None):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.student_id, self.condition = student_id, condition
        self.session_id = session_id or uuid.uuid4().hex[:10]
        self.path = self.dir / f"{student_id}_{self.session_id}.jsonl"
        self._lock = threading.Lock()
        self.log("session_start")

    def log(self, event: str, **payload: Any) -> Dict[str, Any]:
        if event not in EVENT_TYPES:
            raise ValueError(f"unknown event type {event!r}")
        rec = {"t": time.time(), "session": self.session_id, "student": self.student_id,
               "condition": self.condition, "event": event, **payload}
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def close(self) -> None:
        self.log("session_end")

    @staticmethod
    def read(path: str | Path) -> List[Dict[str, Any]]:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


def summarize(events: List[Dict[str, Any]]) -> Dict[str, float]:
    """Session-level indicators used in Section 4.7 (off-task events, dwell per slide, why/how queries)."""
    out: Dict[str, float] = {"off_task_events": 0, "pauses": 0, "replays": 0, "questions": 0, "why_how_questions": 0}
    dwell: List[float] = []
    last_slide_t: Optional[float] = None
    for e in events:
        ev = e["event"]
        if ev == "off_task":
            out["off_task_events"] += 1
        elif ev == "pause":
            out["pauses"] += 1
        elif ev == "replay":
            out["replays"] += 1
        elif ev == "question":
            out["questions"] += 1
            q = str(e.get("text", "")).lower()
            if any(w in q for w in ("why", "how", "为什么", "怎么", "如何")):
                out["why_how_questions"] += 1
        elif ev == "slide_view":
            if last_slide_t is not None:
                dwell.append(e["t"] - last_slide_t)
            last_slide_t = e["t"]
    if dwell:
        dwell.sort()
        out["median_dwell_s"] = dwell[len(dwell) // 2]
    rts = [e["response_time_s"] for e in events if e["event"] == "quiz_response" and "response_time_s" in e]
    if rts:
        out["mean_quiz_rt_s"] = sum(rts) / len(rts)
    return out
