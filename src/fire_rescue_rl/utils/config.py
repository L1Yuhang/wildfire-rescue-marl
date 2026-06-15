"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = ROOT / file_path
    with file_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_dirs(*paths: str | Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

