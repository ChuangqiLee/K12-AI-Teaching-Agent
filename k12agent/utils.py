"""Small shared helpers."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional


class IsolatedImporter:
    """Import a third-party repo whose top-level module names may clash with others.

    Wav2Lip and MockingBird both ship a top-level ``models`` package, Wav2Lip
    also has ``audio``/``hparams``.  Use the importer as a context manager around
    every import *and* call into the repo; it swaps the repo's modules in and out
    of ``sys.modules``.
    """

    def __init__(self, repo_dir: str | Path, top_level: Iterable[str]):
        self.repo_dir = str(repo_dir)
        self.top_level = tuple(top_level)
        self._own: Dict[str, object] = {}
        self._saved: Dict[str, object] = {}

    def _matches(self, name: str) -> bool:
        return any(name == t or name.startswith(t + ".") for t in self.top_level)

    def __enter__(self):
        self._saved = {k: v for k, v in sys.modules.items() if self._matches(k)}
        for k in self._saved:
            del sys.modules[k]
        sys.modules.update(self._own)
        sys.path.insert(0, self.repo_dir)
        return self

    def __exit__(self, *exc):
        if self.repo_dir in sys.path:
            sys.path.remove(self.repo_dir)
        self._own = {k: v for k, v in sys.modules.items() if self._matches(k)}
        for k in self._own:
            del sys.modules[k]
        sys.modules.update(self._saved)
        return False


def ffmpeg_exe() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def run_ffmpeg(args, quiet: bool = True) -> None:
    exe = ffmpeg_exe()
    if exe is None:
        raise RuntimeError("ffmpeg not found: install it or `pip install imageio-ffmpeg`")
    cmd = [exe, "-y"] + (["-loglevel", "error"] if quiet else []) + [str(a) for a in args]
    subprocess.run(cmd, check=True)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
