from __future__ import annotations

from pathlib import Path

from common.pipeline_context import (
    bootstrap_bridge_state,
    ensure_full_pipeline_context,
    load_bridge_state,
    load_job_state_snapshot,
    load_stage_assets_snapshot,
    positioning_stage_dir,
    read_json_file,
    write_json_file,
)
from schemas.stage import StageResult, StageStatus


def run_positioning_stage(
    *,
    repo_root: Path,
    run_dir: Path,
    paper_pdf: Path | None = None,
    paper_key: str = "",
    reuse_job_id: str = "",
) -> StageResult:
    ensure_full_pipeline_context(run_dir=run_dir, allow_standalone=True, stage="positioning")
    bridge = load_bridge_state(run_dir)
    if bridge is None:
        bridge = bootstrap_bridge_state(
            repo_root=repo_root,
            run_dir=run_dir,
            paper_pdf=paper_pdf,
            paper_key=paper_key,
            reuse_job_id=reuse_job_id,
        )

    job_state = load_job_state_snapshot(run_dir) or read_json_file(bridge.job_json_path)
    stage_assets = load_stage_assets_snapshot(run_dir)
    metadata = job_state.get("metadata") if isinstance(job_state.get("metadata"), dict) else {}

    semantic_snapshot_raw = str(stage_assets.get("semantic_scholar_candidates_snapshot_path") or "").strip()
    semantic_snapshot = Path(semantic_snapshot_raw).resolve() if semantic_snapshot_raw else None
    semantic_path = (
        semantic_snapshot
        if (semantic_snapshot is not None and semantic_snapshot.exists())
        else (bridge.job_dir / "semantic_scholar_candidates.json")
    )
    semantic_payload = read_json_file(semantic_path)
    if not semantic_payload:
        semantic_payload = {"success": False, "papers": []}

    paper_search_state = (
        metadata.get("paper_search_runtime_state")
        if isinstance(metadata.get("paper_search_runtime_state"), dict)
        else {}
    )
    search_started = bool(paper_search_state.get("started"))
    semantic_file_exists = semantic_path.exists()
    status: StageStatus = "ok" if (semantic_file_exists or (not search_started)) else "failed"
    error = (
        ""
        if status == "ok"
        else f"paper search reported started but semantic candidates file is missing ({semantic_path})"
    )

    positioning_out = positioning_stage_dir(run_dir) / "positioning.json"
    write_json_file(
        positioning_out,
        {
            "semantic_scholar": semantic_payload,
            "paper_search_runtime_state": paper_search_state,
            "job_id": bridge.job_id,
            "job_json_path": str(bridge.job_json_path),
        },
    )

    return StageResult(
        status=status,
        outputs={"main": str(positioning_out)},
        extra={"job_id": bridge.job_id},
        error=error,
    )


if __name__ == "__main__":
    raise SystemExit("Internal stage module. Use scripts/execute_review_pipeline.py.")
