from __future__ import annotations

from pathlib import Path
from typing import Any

from common.pipeline_context import (
    bootstrap_bridge_state,
    claim_extract_stage_dir,
    ensure_full_pipeline_context,
    load_bridge_state,
    load_job_state_snapshot,
    load_stage_assets_snapshot,
    read_json_file,
    resolve_artifact_path,
    write_json_file,
)
from schemas.stage import StageResult


def run_claim_extract_stage(
    *,
    repo_root: Path,
    run_dir: Path,
    paper_pdf: Path | None = None,
    paper_key: str = "",
    reuse_job_id: str = "",
) -> StageResult:
    ensure_full_pipeline_context(run_dir=run_dir, allow_standalone=True, stage="claim_extract")
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
    artifacts = job_state.get("artifacts") if isinstance(job_state.get("artifacts"), dict) else {}
    metadata = job_state.get("metadata") if isinstance(job_state.get("metadata"), dict) else {}
    annotation_count = int(job_state.get("annotation_count") or 0)
    annotations_raw = str(artifacts.get("annotations_path") or "").strip()

    annotations_snapshot_raw = str(stage_assets.get("annotations_snapshot_path") or "").strip()
    annotations_snapshot = Path(annotations_snapshot_raw).resolve() if annotations_snapshot_raw else None
    if annotations_snapshot is not None and annotations_snapshot.exists():
        annotations = annotations_snapshot
    else:
        annotations = resolve_artifact_path(repo_root, annotations_raw)
    has_annotations_file = annotations is not None and annotations.exists()
    annotations_payload: dict[str, Any] | list[Any]
    if has_annotations_file:
        annotations_payload = read_json_file(annotations)
    else:
        # Agent runtime can legitimately complete a job with zero annotations and no annotations.json.
        annotations_payload = []

    facts_out = claim_extract_stage_dir(run_dir) / "facts.json"
    write_json_file(
        facts_out,
        {
            "annotation_count": annotation_count,
            "annotations_path": annotations_raw if has_annotations_file else "",
            "annotations": annotations_payload,
            "usage": job_state.get("usage") or {},
            "metadata": {
                "final_report_sections_completed": metadata.get("final_report_sections_completed") or [],
                "final_report_source": metadata.get("final_report_source"),
            },
            "job_id": bridge.job_id,
            "job_json_path": str(bridge.job_json_path),
        },
    )

    ok = has_annotations_file or annotation_count == 0
    error = ""
    if not ok:
        looked_at = annotations_raw or (str(annotations) if annotations else "no path resolved")
        error = (
            f"job reported {annotation_count} annotations but annotations.json is missing "
            f"(looked at {looked_at})"
        )
    return StageResult(
        status="ok" if ok else "failed",
        outputs={"main": str(facts_out)},
        extra={"job_id": bridge.job_id},
        error=error,
    )


if __name__ == "__main__":
    raise SystemExit("Internal stage module. Use scripts/execute_review_pipeline.py.")
