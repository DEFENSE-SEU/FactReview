from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from common.config import get_settings
from common.pipeline_context import (
    ensure_full_pipeline_context,
    execution_stage_dir,
    load_bridge_state,
    read_json_file,
    write_json_file,
)
from schemas.execution import ExecutionExitStatus, ExecutionPayload, ExecutionStageStatus
from schemas.stage import StageResult


def _load_execution_artifacts(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    run_info = state.get("run") if isinstance(state.get("run"), dict) else {}
    run_dir = Path(str(run_info.get("dir") or "")).resolve() if run_info.get("dir") else Path()
    summary = read_json_file(run_dir / "summary.json") if run_dir else {}
    alignment = read_json_file(run_dir / "artifacts" / "alignment" / "alignment.json") if run_dir else {}
    return summary, alignment, str(run_dir) if run_dir else ""


def _archive_prior_current_dir(*, stage_root: Path, current_dir: Path) -> None:
    """Set up an empty ``current/`` workspace, preserving the prior attempt
    by renaming it to ``current.<timestamp>`` instead of deleting outright.
    A separate ``history/`` tree (passed as ``run_root`` to the orchestrator)
    holds full per-attempt outputs; this archive only protects the most
    recent in-place workspace from being silently wiped."""
    resolved_stage = stage_root.resolve()
    resolved_current = current_dir.resolve()
    if resolved_current.parent != resolved_stage or resolved_current.name != "current":
        raise RuntimeError(f"refusing to reset unexpected execution current dir: {resolved_current}")
    if resolved_current.exists():
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        archived = resolved_current.with_name(f"current.{timestamp}")
        resolved_current.rename(archived)
    resolved_current.mkdir(parents=True, exist_ok=True)


async def _run_orchestrator_async(
    *,
    run_root: Path,
    paper_pdf: Path,
    paper_key: str,
    max_attempts: int,
    no_pdf_extract: bool,
    paper_extracted_dir: str = "",
    execution_run_dir: Path | None = None,
    enable_refcheck: bool = False,
) -> dict[str, Any]:
    from fact_generation.execution.graph import ExecutionOrchestrator

    orchestrator = ExecutionOrchestrator(
        run_root=str(run_root),
        max_attempts=max_attempts,
        enable_refcheck=enable_refcheck,
        paper_extracted_dir=str(paper_extracted_dir or ""),
        run_dir=str(execution_run_dir or ""),
    )
    return await orchestrator.run(
        paper_root="",
        paper_pdf=str(paper_pdf),
        paper_key=paper_key,
        tasks_path="",
        baseline_path="",
        local_source_path="",
        no_pdf_extract=no_pdf_extract,
    )


def run_execution_stage(
    *,
    run_dir: Path,
    paper_pdf: Path | None = None,
    paper_key: str | None = None,
    paper_extracted_dir: str = "",
    max_attempts: int = 5,
    no_pdf_extract: bool = False,
    enable_refcheck: bool | None = None,
) -> StageResult:
    ensure_full_pipeline_context(run_dir=run_dir, allow_standalone=True, stage="execution")
    bridge = load_bridge_state(run_dir)
    resolved_pdf = paper_pdf.resolve() if paper_pdf else (bridge.paper_pdf if bridge else None)
    resolved_key = (paper_key or "").strip() or (bridge.paper_key if bridge else "")

    if resolved_pdf is None or not resolved_pdf.exists():
        raise FileNotFoundError(
            "paper_pdf is required for execution stage when bridge state is missing or invalid."
        )
    if not resolved_key:
        resolved_key = resolved_pdf.stem.strip() or "paper"

    stage_root = execution_stage_dir(run_dir)
    stage_root.mkdir(parents=True, exist_ok=True)
    # ``current/`` = the in-place workspace the orchestrator scribbles into.
    # ``history/`` = the orchestrator's per-attempt outputs root (one
    # timestamped subdir per attempt). The names are intentionally distinct.
    execution_run_dir = stage_root / "current"
    stage_run_root = stage_root / "history"
    _archive_prior_current_dir(stage_root=stage_root, current_dir=execution_run_dir)
    settings = get_settings()
    resolved_refcheck = bool(
        settings.execution_enable_refcheck if enable_refcheck is None else enable_refcheck
    )

    run_result = asyncio.run(
        _run_orchestrator_async(
            run_root=stage_run_root,
            paper_pdf=resolved_pdf,
            paper_key=resolved_key,
            paper_extracted_dir=str(paper_extracted_dir or ""),
            execution_run_dir=execution_run_dir,
            max_attempts=max_attempts,
            no_pdf_extract=no_pdf_extract,
            enable_refcheck=resolved_refcheck,
        )
    )

    state = run_result.get("state") if isinstance(run_result.get("state"), dict) else {}
    summary, alignment, actual_run_dir = _load_execution_artifacts(state)
    raw_exit = str(run_result.get("exit_status") or "failed")
    exit_status: ExecutionExitStatus = (
        raw_exit if raw_exit in ("success", "inconclusive", "failed", "skipped") else "failed"  # type: ignore[assignment]
    )

    stage_status: ExecutionStageStatus = "failed"
    if exit_status == "success":
        stage_status = "ok"
    elif exit_status == "inconclusive":
        stage_status = "inconclusive"

    error = ""
    if stage_status == "failed":
        detail = str(run_result.get("error") or run_result.get("message") or "").strip()
        error = (
            f"execution orchestrator exit_status={exit_status!r}: {detail}"
            if detail
            else f"execution orchestrator exit_status={exit_status!r}"
        )

    payload = ExecutionPayload(
        paper_key=resolved_key,
        paper_pdf=str(resolved_pdf),
        status=stage_status,
        success=bool(run_result.get("success")),
        exit_status=exit_status,
        run_dir=actual_run_dir,
        summary=summary,
        alignment=alignment,
    )

    output_path = stage_root / "execution.json"
    write_json_file(output_path, payload.model_dump())

    return StageResult(
        status=stage_status,
        outputs={"main": str(output_path)},
        extra={"run_dir": actual_run_dir},
        error=error,
    )


if __name__ == "__main__":
    raise SystemExit("Internal stage module. Use scripts/execute_review_pipeline.py.")
