from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_test_metrics(metrics_path: str | Path) -> dict[str, Any]:
    path = Path(metrics_path)
    if not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", [])

    if not results:
        return {}

    if isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
        return dict(results[0])

    return {}