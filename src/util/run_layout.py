from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

RUN_ID_FORMAT = "%Y-%m-%d_%H%M%S"


def make_run_id() -> str:
    return datetime.now().strftime(RUN_ID_FORMAT)


def slugify_run_key(value: str | None, *, fallback: str = "paper") -> str:
    token = str(value or "").strip().lower()
    token = re.sub(r"[^a-z0-9._-]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("._-")
    return token or fallback


def build_run_dir(run_root: str | Path, paper_key: str, run_id: str | None = None) -> Path:
    rid = str(run_id or make_run_id()).strip()
    return Path(run_root).resolve() / f"{slugify_run_key(paper_key)}_{rid}"


def ensure_run_subdirs(run_dir: str | Path) -> dict[str, Path]:
    root = Path(run_dir).resolve()
    layout = {
        "root": root,
        "inputs": root / "inputs",
        "runtime": root / "runtime",
        "workspace": root / "workspace",
        "stages": root / "stages",
        "logs": root / "logs",
        "artifacts": root / "artifacts",
        "reports": root / "reports",
        "debug": root / "debug",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout
