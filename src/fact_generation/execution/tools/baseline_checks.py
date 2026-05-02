from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Baseline:
    raw: dict[str, Any]

    @property
    def checks(self) -> list[dict[str, Any]]:
        checks = self.raw.get("checks")
        return checks if isinstance(checks, list) else []


def load_baseline(path: str) -> Baseline:
    if not path:
        return Baseline(raw={})
    p = Path(path)
    if not p.exists():
        return Baseline(raw={})
    try:
        raw = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        if not isinstance(raw, dict):
            return Baseline(raw={})
        return Baseline(raw=raw)
    except Exception:
        return Baseline(raw={})
