from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .fs import ensure_dir, sha256_file, write_text


@dataclass(frozen=True)
class Meta:
    run_id: str
    platform: str
    python: str
    cwd: str
    paper_root: str
    tasks_path: str
    baseline_path: str
    llm: dict[str, Any]


def collect_meta(
    run_id: str,
    paper_root: str,
    tasks_path: str,
    baseline_path: str,
    llm_cfg: dict[str, Any],
) -> Meta:
    return Meta(
        run_id=run_id,
        platform=f"{platform.system()} {platform.release()} ({platform.version()})",
        python=sys.version.replace("\n", " "),
        cwd=os.getcwd(),
        paper_root=paper_root,
        tasks_path=tasks_path,
        baseline_path=baseline_path,
        llm=llm_cfg,
    )


def write_meta(meta: Meta, run_dir: str | Path) -> Path:
    p = ensure_dir(run_dir) / "meta.json"
    text = json.dumps(asdict(meta), ensure_ascii=False, indent=2)
    write_text(p, text + "\n")
    return p


def index_artifacts(run_artifacts_dir: str | Path) -> dict[str, Any]:
    root = Path(run_artifacts_dir)
    out: dict[str, Any] = {"files": []}
    if not root.exists():
        return out
    for f in sorted(root.rglob("*")):
        if f.is_dir():
            continue
        try:
            out["files"].append(
                {
                    "path": str(f.relative_to(root)).replace("\\", "/"),
                    "size": f.stat().st_size,
                    "sha256": sha256_file(f),
                }
            )
        except Exception:
            continue
    return out
