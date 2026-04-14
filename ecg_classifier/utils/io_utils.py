from __future__ import annotations

import json
from pathlib import Path
from typing import Any

def ensure_dir(directory_path: Path) -> None:
    directory_path.mkdir(parents=True, exist_ok=True)

def save_json(json_path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(json_path.parent)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")