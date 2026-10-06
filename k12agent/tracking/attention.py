"""Real-time face / gaze off-task monitoring (Section 4.5.1).

"Looking away from the screen for more than 5 s paused playback, displayed a
red warning bar, and logged an off_task event."  The same rule is applied in
both conditions (A1 and A2) to keep attentional tracking at parity.

The monitor uses MediaPipe FaceMesh (if installed) to estimate head yaw/pitch
and iris position; without MediaPipe it falls back to OpenCV's Haar frontal
face detector (no frontal face == looking away).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class GazeState:
    on_screen: bool
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    face_found: bool = False


class OffTaskRule:
    """Pure state machine (testable without a camera)."""

    def __init__(self, threshold_s: float = 5.0):
        self.threshold_s = threshold_s
        self.away_since: Optional[float] = None
        self.off_task = False

    def update(self, on_screen: bool, now: float) -> Optional[str]:
        """Returns "off_task" / "on_task" on state transitions, else None."""
        if on_screen:
            self.away_since = None
            if self.off_task:
                self.off_task = False
                return "on_task"
            return None
        if self.away_since is None:
            self.away_since = now
        if not self.off_task and now - self.away_since > self.threshold_s:
            self.off_task = True
            return "off_task"
        return None


class GazeEstimator:
    YAW_LIMIT, PITCH_LIMIT = 30.0, 25.0

    def __init__(self):
        self.mesh = None
        try:
            import mediapipe as mp
            self.mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        except Exception:  # noqa: BLE001
            import cv2
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def __call__(self, frame_bgr: np.ndarray) -> GazeState:
        import cv2
        if self.mesh is None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.haar.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
            return GazeState(on_screen=len(faces) > 0, face_found=len(faces) > 0)
        res = self.mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return GazeState(on_screen=False)
        lm = res.multi_face_landmarks[0].landmark
        nose, left, right, top, chin = lm[1], lm[234], lm[454], lm[10], lm[152]
        # crude head pose from landmark geometry (normalised image coordinates)
        mid_x = (left.x + right.x) / 2
        yaw = np.degrees(np.arcsin(np.clip((nose.x - mid_x) / max(1e-6, (right.x - left.x) / 2), -1, 1)))
        mid_y = (top.y + chin.y) / 2
        pitch = np.degrees(np.arcsin(np.clip((nose.y - mid_y) / max(1e-6, (chin.y - top.y) / 2), -1, 1)))
        on = abs(yaw) < self.YAW_LIMIT and abs(pitch) < self.PITCH_LIMIT
        return GazeState(on_screen=on, yaw_deg=float(yaw), pitch_deg=float(pitch), face_found=True)


class AttentionMonitor:
    """Background thread: webcam -> gaze -> OffTaskRule -> callbacks + log."""

    def __init__(self, on_off_task: Callable[[], None], on_back: Optional[Callable[[], None]] = None,
                 threshold_s: float = 5.0, camera_index: int = 0, logger=None, fps: float = 10.0):
        self.rule = OffTaskRule(threshold_s)
        self.on_off_task, self.on_back = on_off_task, on_back
        self.camera_index, self.logger, self.period = camera_index, logger, 1.0 / fps
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.window_focused = True   # set from the UI (window-focus monitoring)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        import cv2
        cap = cv2.VideoCapture(self.camera_index)
        est = GazeEstimator()
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                state = est(frame) if ok else GazeState(on_screen=False)
                transition = self.rule.update(state.on_screen and self.window_focused, time.time())
                if transition == "off_task":
                    if self.logger:
                        self.logger.log("off_task", yaw=state.yaw_deg, pitch=state.pitch_deg, face=state.face_found)
                    self.on_off_task()
                elif transition == "on_task":
                    if self.logger:
                        self.logger.log("on_task")
                    if self.on_back:
                        self.on_back()
                time.sleep(self.period)
        finally:
            cap.release()
