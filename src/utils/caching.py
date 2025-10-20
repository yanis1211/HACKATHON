"""Simple on-disk caching utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def file_is_fresh(path: Path, max_age_hours: float) -> bool:
    if not path.exists():
        return False
    mtime = path.stat().st_mtime
    from time import time

    age_hours = (time() - mtime) / 3600
    return age_hours <= max_age_hours
