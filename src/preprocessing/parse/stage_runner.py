"""Parse stage: PDF → structured Paper artifacts.

Wraps ``execute_review_runtime_job.py`` (via ``bootstrap_bridge_state``) to run
the agent-based ingestion runtime, then materialises run-local snapshots and
writes ``stages/preprocessing/parse/paper.json`` for downstream stages.
"""

from __future__ import annotations

from pathlib import Path

from common.pipeline_context import (
    bootstrap_bridge_state,
    ensure_full_pipeline_context,
    materialize_stage_inputs_snapshot,
    parse_stage_dir,
    resolve_artifact_path,
    write_json_file,
)
from schemas.stage import StageResult
from util.fs import copy_dir_if_exists, copy_file_if_exists


def _materialize_execution_paper_extract(
    *,
    run_dir: Path,
    job_dir: Path,
    mineru_md: Path | None,
    mineru_content: Path | None,
) -> dict[str, str]:
    """Bridge the runtime MinerU output into this run's execution input folder
    so the execution stage can reuse the first PDF parse instead of running
    MinerU again locally."""
    extracted_dir = run_dir / "inputs" / "paper_extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    md_dst = extracted_dir / "paper.mineru.md"
    content_dst = extracted_dir / "paper.mineru.content_list.json"
    assets_src = job_dir / "mineru_assets"
    assets_dst = extracted_dir / "mineru_assets"

    copied_md = copy_file_if_exists(mineru_md, md_dst)
    copied_content = copy_file_if_exists(mineru_content, content_dst)
    copied_assets = copy_dir_if_exists(assets_src, assets_dst)

    tables_dir = extracted_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    return {
        "paper_extracted_dir": str(extracted_dir.resolve()),
        "paper_extracted_md_path": str(md_dst.resolve()) if copied_md else "",
        "paper_extracted_content_list_path": str(content_dst.resolve()) if copied_content else "",
        "paper_extracted_assets_dir": str(assets_dst.resolve()) if copied_assets else "",
        "paper_extracted_tables_dir": str(tables_dir.resolve()),
    }


def run_parse_stage(
    *,
    repo_root: Path,
    run_dir: Path,
    paper_pdf: Path,
    paper_key: str,
    reuse_job_id: str = "",
    materialize_execution_extract: bool = True,
) -> StageResult:
    ensure_full_pipeline_context(run_dir=run_dir, allow_standalone=True, stage="parse")
    state = bootstrap_bridge_state(
        repo_root=repo_root,
        run_dir=run_dir,
        paper_pdf=paper_pdf,
        paper_key=paper_key,
        reuse_job_id=reuse_job_id,
    )
    own_payload = state.own_payload if isinstance(state.own_payload, dict) else {}
    materialize_stage_inputs_snapshot(
        repo_root=repo_root,
        run_dir=run_dir,
        state=state,
        own_payload=own_payload,
    )
    artifacts = own_payload.get("artifacts") if isinstance(own_payload.get("artifacts"), dict) else {}
    metadata = own_payload.get("metadata") if isinstance(own_payload.get("metadata"), dict) else {}
    usage = own_payload.get("usage") if isinstance(own_payload.get("usage"), dict) else {}
    annotation_count = int(own_payload.get("annotation_count") or 0)
    mineru_md_raw = str(artifacts.get("mineru_markdown_path") or "").strip()
    mineru_content_raw = str(artifacts.get("mineru_content_list_path") or "").strip()

    mineru_md = resolve_artifact_path(repo_root, mineru_md_raw)
    mineru_content = resolve_artifact_path(repo_root, mineru_content_raw)
    shared_extract = (
        _materialize_execution_paper_extract(
            run_dir=run_dir,
            job_dir=state.job_dir,
            mineru_md=mineru_md,
            mineru_content=mineru_content,
        )
        if materialize_execution_extract
        else {}
    )

    parse_out = parse_stage_dir(run_dir) / "paper.json"
    write_json_file(
        parse_out,
        {
            "source_pdf": str(state.paper_pdf),
            "mineru_markdown_path": mineru_md_raw if (mineru_md is not None and mineru_md.exists()) else "",
            "mineru_content_list_path": mineru_content_raw
            if (mineru_content is not None and mineru_content.exists())
            else "",
            "markdown_provider": metadata.get("markdown_provider"),
            "mineru_batch_id": metadata.get("mineru_batch_id"),
            "parse_warning": metadata.get("parse_warning"),
            "job_id": state.job_id,
            "job_json_path": str(state.job_json_path),
            "shared_execution_extract": shared_extract,
            "annotation_count": annotation_count,
            "annotations_path": str(artifacts.get("annotations_path") or ""),
            "semantic_scholar_candidates_path": str(
                (state.job_dir / "semantic_scholar_candidates.json").resolve()
            ),
            "final_markdown_path": str(artifacts.get("final_markdown_path") or ""),
            "report_pdf_path": str(artifacts.get("report_pdf_path") or ""),
            "latest_output_md": str(own_payload.get("latest_output_md") or ""),
            "latest_output_pdf": str(own_payload.get("latest_output_pdf") or ""),
            "runtime_status": own_payload.get("status"),
            "runtime_message": own_payload.get("message"),
            "runtime_error": own_payload.get("error"),
            "usage": usage,
            "paper_search_runtime_state": metadata.get("paper_search_runtime_state")
            if isinstance(metadata.get("paper_search_runtime_state"), dict)
            else {},
        },
    )

    md_ok = mineru_md is not None and mineru_md.exists()
    error = ""
    if not md_ok:
        runtime_error = str(own_payload.get("error") or "").strip()
        runtime_message = str(own_payload.get("message") or "").strip()
        runtime_status = str(own_payload.get("status") or "").strip()
        detail = runtime_error or runtime_message or f"runtime status={runtime_status!r}" or "no detail"
        error = f"MinerU markdown not produced ({detail})"
    return StageResult(
        status="ok" if md_ok else "failed",
        outputs={"main": str(parse_out)},
        extra={
            "job_id": state.job_id,
            "job_dir": str(state.job_dir),
            "shared_execution_extract": shared_extract,
        },
        error=error,
    )


if __name__ == "__main__":
    raise SystemExit("Internal stage module. Use scripts/execute_review_pipeline.py.")
