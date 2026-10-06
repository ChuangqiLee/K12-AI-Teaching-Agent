"""Configuration loading (YAML -> attribute-accessible dict)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


class Config(dict):
    """A dict whose keys are also attributes (``cfg.speech.backend``)."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _deep_update(base: dict, update: Mapping) -> dict:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _wrap(d: Any) -> Any:
    if isinstance(d, dict):
        return Config({k: _wrap(v) for k, v in d.items()})
    return d


def load_config(path: str | Path | None = None, overrides: Mapping | None = None) -> Config:
    """Load ``configs/default.yaml`` and optionally merge a user file and overrides."""
    with open(DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG:
        with open(path, "r", encoding="utf-8") as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        _deep_update(cfg, overrides)
    return _wrap(cfg)


def resolve_path(p: str | Path | None) -> Path | None:
    """Resolve a config path relative to the repository root."""
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p
