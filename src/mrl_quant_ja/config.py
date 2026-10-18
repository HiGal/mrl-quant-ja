"""Tiny YAML config loader with path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "grid.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    for key in ("cache_dir", "results_dir", "figures_dir"):
        p = Path(cfg[key])
        cfg[key] = p if p.is_absolute() else ROOT / p
        cfg[key].mkdir(parents=True, exist_ok=True)
    return cfg


def emb_dir(cfg: dict[str, Any], model_key: str) -> Path:
    d = cfg["cache_dir"] / "embeddings" / model_key
    d.mkdir(parents=True, exist_ok=True)
    return d
