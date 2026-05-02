from __future__ import annotations

from pathlib import Path
from typing import Any

from common.config import get_settings
from common.pipeline_context import (
    bootstrap_bridge_state,
    ensure_full_pipeline_context,
    load_bridge_state,
    refcheck_stage_dir,
    write_json_file,
)
from fact_generation.refcheck.refcheck import check_references, format_reference_check_markdown
from schemas.stage import StageResult, StageStatus


def run_refcheck_stage(
    *,
    repo_root: Path,
    run_dir: Path,
    paper_pdf: Path | None = None,
    paper_key: str = "",
    reuse_job_id: str = "",
    enable_refcheck: bool | None = None,
) -> StageResult:
    ensure_full_pipeline_context(run_dir=run_dir, allow_standalone=True, stage="refcheck")
    bridge = load_bridge_state(run_dir)
    if bridge is None:
        bridge = bootstrap_bridge_state(
            repo_root=repo_root,
            run_dir=run_dir,
            paper_pdf=paper_pdf,
            paper_key=paper_key,
            reuse_job_id=reuse_job_id,
        )

    settings = get_settings()
    refcheck_enabled = bool(settings.reference_check_enabled if enable_refcheck is None else enable_refcheck)
    stage_dir = refcheck_stage_dir(run_dir)
    output_json = stage_dir / "reference_check.json"
    output_md = stage_dir / "reference_check.md"
    detail_txt = stage_dir / "reference_check_details.txt"
    payload: dict[str, Any] = {"enabled": refcheck_enabled, "attempted": False}

    if refcheck_enabled:
        resolved_pdf = paper_pdf.resolve() if paper_pdf else bridge.paper_pdf
        if resolved_pdf is None or not resolved_pdf.exists():
            payload.update(
                {
                    "attempted": True,
                    "ok": False,
                    "error_message": "paper PDF not found; reference check was not run",
                }
            )
        else:
            rc = check_references(
                paper=str(resolved_pdf),
                api_key=settings.semantic_scholar_api_key,
                output_file=str(detail_txt),
                debug=False,
            )
            payload.update({"attempted": True, **rc})

        markdown = format_reference_check_markdown(
            payload,
            max_issues=max(1, int(settings.reference_check_report_max_issues)),
        )
        if markdown.strip():
            output_md.parent.mkdir(parents=True, exist_ok=True)
            output_md.write_text(markdown, encoding="utf-8")

    write_json_file(output_json, payload)
    status: StageStatus = "skipped" if not refcheck_enabled else "ok"
    error = ""
    if refcheck_enabled and payload.get("ok") is False:
        status = "failed"
        detail = str(payload.get("error_message") or payload.get("message") or "").strip()
        error = f"reference check failed: {detail}" if detail else "reference check reported failure"

    outputs = {"main": str(output_json)}
    if output_md.exists():
        outputs["markdown"] = str(output_md)
    return StageResult(
        status=status,
        outputs=outputs,
        extra={"reference_check": payload},
        error=error,
    )


if __name__ == "__main__":
    raise SystemExit("Internal stage module. Use scripts/execute_review_pipeline.py.")
