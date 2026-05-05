"""Review report sub-stage.

Reads the agent-runner's final report artifacts (markdown + audit) from the
parse-stage snapshot, normalises image paths, optionally appends a
``RefCopilot`` summary, re-renders the PDF, and writes the canonical review
output to ``stages/review/report/``.

A clean copy of the markdown (without the refcheck section) is also written to
``final_review_clean.md`` so the teaser sub-stage can build its prompt from a
report that has not been polluted by reference-check findings.
"""

from __future__ import annotations

import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from common.config import get_settings
from common.pipeline_context import (
    ensure_full_pipeline_context,
    execution_stage_dir,
    load_job_state_snapshot,
    load_stage_assets_snapshot,
    read_json_file,
    refcheck_stage_dir,
    report_stage_dir,
    require_bridge_state,
    resolve_artifact_path,
    write_json_file,
)
from fact_generation.refcheck.refcheck import format_reference_check_markdown
from review.report.pdf_renderer import build_review_report_pdf
from schemas.stage import StageResult
from util.fs import copy_file_if_exists


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _strip_experiment_eval_status(text: str) -> str:
    """Remove Evaluation Status column and its legend from experiment tables."""
    sec = re.search(
        r"(?ims)(^##\s+(?:\*\*)?5\.\s+Experiment(?:\*\*)?\s*$\n)(.*?)(?=^##\s+|\Z)",
        text,
    )
    if not sec:
        return text

    body = sec.group(2)

    # Remove colored status-legend lines added by _augment_experiment_with_eval_status.
    body = re.sub(r"(?m)^\(Status legend:.*\)\s*$\n?", "", body)

    def _strip_last_col(line: str) -> str:
        stripped = line.rstrip("\n")
        if not (stripped.startswith("|") and stripped.endswith("|")):
            return line
        parts = stripped.split("|")
        # parts: ['', col1, col2, ..., colN, '']
        # Remove the second-to-last non-empty slot (the last column cell).
        if len(parts) >= 4:
            parts = parts[:-2] + parts[-1:]
        return "|".join(parts) + "\n"

    lines = body.splitlines(keepends=True)
    result: list[str] = []
    i = 0
    while i < len(lines):
        header = lines[i].rstrip()
        if i + 1 < len(lines) and header.startswith("|") and header.endswith("|"):
            sep = lines[i + 1].rstrip()
            if re.fullmatch(r"\|[ :\-|]+\|", sep):
                # Check if last header column is "Evaluation Status".
                cols = [c.strip().strip("*") for c in header.strip("|").split("|")]
                if cols and cols[-1].lower() == "evaluation status":
                    j = i + 2
                    while j < len(lines) and lines[j].rstrip().startswith("|"):
                        j += 1
                    result.append(_strip_last_col(lines[i]))
                    result.append(_strip_last_col(lines[i + 1]))
                    for k in range(i + 2, j):
                        result.append(_strip_last_col(lines[k]))
                    i = j
                    continue
        result.append(lines[i])
        i += 1

    new_body = "".join(result)
    return text[: sec.start(2)] + new_body + text[sec.end(2) :]


def _render_review_pdf(
    *, markdown_path: Path, pdf_path: Path, workspace_title: str, source_pdf_name: str
) -> tuple[bool, str]:
    """Render the review PDF. Returns ``(ok, error)``; ``error`` is empty on success."""
    if not markdown_path.exists() or not markdown_path.is_file():
        return False, f"markdown source not found at {markdown_path}"
    md_text = markdown_path.read_text(encoding="utf-8", errors="ignore")
    overview_path = markdown_path.parent / "overview_figure.jpg"
    if overview_path.exists() and overview_path.is_file():
        md_text = md_text.replace("./overview_figure.jpg", str(overview_path.resolve()))
    try:
        settings = get_settings()
        pdf_bytes = build_review_report_pdf(
            workspace_title=workspace_title,
            source_pdf_name=source_pdf_name,
            run_id=markdown_path.parents[3].name,
            status="completed",
            decision=None,
            estimated_cost=0,
            actual_cost=None,
            exported_at=datetime.now(UTC),
            meta_review={},
            reviewers=[],
            raw_output=None,
            final_report_markdown=md_text,
            source_pdf_bytes=None,
            source_annotations=[],
            review_display_id=None,
            owner_email=None,
            token_usage={},
            agent_model=str(settings.agent_model or "").strip() or "factreview-review",
        )
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(pdf_bytes)
        return True, ""
    except Exception as exc:
        # PDF rendering is best-effort; the markdown is the canonical artifact.
        # Route the message to stderr so single-stage CLI stdout stays clean,
        # and bubble it up to the caller so it can be surfaced in StageResult.
        message = f"{type(exc).__name__}: {exc}"
        print(f"[report] PDF render failed: {message}", file=sys.stderr)
        return False, message


def _absolutize_markdown_image_refs(*, markdown_path: Path, source_base_dirs: list[Path]) -> None:
    if not markdown_path.exists() or not markdown_path.is_file():
        return
    text = markdown_path.read_text(encoding="utf-8", errors="ignore")

    def _replace(match: re.Match[str]) -> str:
        whole = match.group(0) or ""
        src = (match.group(1) or "").strip()
        if not src:
            return whole
        src_path = Path(src).expanduser()
        if src_path.is_absolute():
            return whole
        for base_dir in source_base_dirs:
            resolved = (base_dir / src).resolve()
            if resolved.exists() and resolved.is_file():
                target = resolved
                if resolved.name.lower() == "overview_figure.jpg":
                    alias = resolved.with_name("technical_positioning_image.jpg")
                    try:
                        if not alias.exists() or not alias.is_file():
                            shutil.copy2(resolved, alias)
                        target = alias
                    except OSError:
                        target = resolved
                return whole.replace(src, str(target))
        return whole

    updated = re.sub(r"!\[[^\]]*\]\(([^)]+)\)", _replace, text)
    if updated != text:
        markdown_path.write_text(updated, encoding="utf-8")


def _load_reference_check_payload(run_dir: Path) -> dict[str, Any]:
    return read_json_file(refcheck_stage_dir(run_dir) / "reference_check.json")


def _append_reference_check_section(
    *,
    markdown_path: Path,
    reference_check: dict[str, Any],
    max_issues: int,
) -> str:
    if not reference_check.get("enabled"):
        return ""
    section = format_reference_check_markdown(reference_check, max_issues=max_issues).strip()
    if not section:
        return ""
    current = _read_text(markdown_path).rstrip()
    markdown_path.write_text(current + "\n\n" + section + "\n", encoding="utf-8")
    return section + "\n"


def run_report_stage(
    *,
    repo_root: Path,
    run_dir: Path,
) -> StageResult:
    ensure_full_pipeline_context(run_dir=run_dir, allow_standalone=True, stage="report")
    # Report is a pure consumer of the parse-stage snapshot. If the bridge
    # state is missing we fail clean rather than silently re-running the agent
    # runtime — keeping the documented invariant that every stage except
    # ``parse`` works from the run dir alone.
    bridge = require_bridge_state(run_dir=run_dir)

    job_state = load_job_state_snapshot(run_dir) or read_json_file(bridge.job_json_path)
    stage_assets = load_stage_assets_snapshot(run_dir)
    artifacts = job_state.get("artifacts") if isinstance(job_state.get("artifacts"), dict) else {}
    metadata = job_state.get("metadata") if isinstance(job_state.get("metadata"), dict) else {}
    final_md_raw = str(artifacts.get("final_markdown_path") or "").strip()
    final_audit_raw = str(artifacts.get("final_report_audit_path") or "").strip()
    final_pdf_raw = str(artifacts.get("report_pdf_path") or "").strip()

    final_md_snapshot_raw = str(stage_assets.get("final_markdown_snapshot_path") or "").strip()
    final_pdf_snapshot_raw = str(stage_assets.get("report_pdf_snapshot_path") or "").strip()
    final_md_snapshot = Path(final_md_snapshot_raw).resolve() if final_md_snapshot_raw else None
    final_pdf_snapshot = Path(final_pdf_snapshot_raw).resolve() if final_pdf_snapshot_raw else None
    final_md = (
        final_md_snapshot
        if (final_md_snapshot is not None and final_md_snapshot.exists())
        else resolve_artifact_path(repo_root, final_md_raw)
    )
    final_audit = resolve_artifact_path(repo_root, final_audit_raw) if final_audit_raw else None
    final_pdf = (
        final_pdf_snapshot
        if (final_pdf_snapshot is not None and final_pdf_snapshot.exists())
        else resolve_artifact_path(repo_root, final_pdf_raw)
    )

    out_dir = report_stage_dir(run_dir)
    review_json = out_dir / "final_review.json"
    review_md = out_dir / "final_review.md"
    review_md_clean = out_dir / "final_review_clean.md"
    review_audit = out_dir / "final_review_audit.json"
    pdf_path = out_dir / "final_review.pdf"

    md_ok = copy_file_if_exists(final_md, review_md)
    audit_ok = copy_file_if_exists(final_audit, review_audit)
    pdf_ok = copy_file_if_exists(final_pdf, pdf_path)

    settings = get_settings()
    reference_check_payload = _load_reference_check_payload(run_dir)
    reference_check_markdown = ""
    reference_check_appended = False
    pdf_render_error = ""

    if md_ok:
        if final_md is not None and final_md.exists():
            _absolutize_markdown_image_refs(
                markdown_path=review_md,
                source_base_dirs=[final_md.parent, bridge.job_dir],
            )
        # Always snapshot the clean (no-refcheck) version for the teaser sub-stage.
        shutil.copy2(review_md, review_md_clean)

        # When execution was skipped, strip the Evaluation Status column so the
        # report and the teaser figure do not display all-Inconclusive placeholders.
        exec_payload_early = read_json_file(execution_stage_dir(run_dir) / "execution.json")
        if exec_payload_early.get("status") == "skipped":
            for md_path in (review_md, review_md_clean):
                stripped = _strip_experiment_eval_status(
                    md_path.read_text(encoding="utf-8", errors="ignore")
                )
                md_path.write_text(stripped, encoding="utf-8")

        if reference_check_payload.get("enabled"):
            reference_check_markdown = _append_reference_check_section(
                markdown_path=review_md,
                reference_check=reference_check_payload,
                max_issues=max(1, int(settings.reference_check_report_max_issues)),
            )
            reference_check_appended = bool(reference_check_markdown.strip())

        source_name = bridge.paper_pdf.name if bridge.paper_pdf else "paper.pdf"
        rendered_pdf_ok, pdf_render_error = _render_review_pdf(
            markdown_path=review_md,
            pdf_path=pdf_path,
            workspace_title=bridge.paper_key,
            source_pdf_name=source_name,
        )
        if reference_check_appended:
            pdf_ok = rendered_pdf_ok
            if not pdf_ok and pdf_path.exists():
                try:
                    pdf_path.unlink()
                except OSError:
                    pass
        else:
            pdf_ok = rendered_pdf_ok or pdf_ok

    execution_payload = read_json_file(execution_stage_dir(run_dir) / "execution.json")

    write_json_file(
        review_json,
        {
            "paper_key": bridge.paper_key,
            "run_id": run_dir.name,
            "job_id": bridge.job_id,
            "status": bridge.own_payload.get("status"),
            "message": bridge.own_payload.get("message"),
            "error": bridge.own_payload.get("error"),
            "usage": bridge.own_payload.get("usage") or {},
            "metadata": metadata,
            "execution": execution_payload,
            "reference_check": reference_check_payload,
            "reference_check_markdown": reference_check_markdown,
            "final_markdown": _read_text(review_md) if md_ok else "",
            "final_audit_path": str(final_audit_raw)
            if (final_audit is not None and final_audit.exists())
            else "",
            "final_audit": read_json_file(review_audit) if audit_ok else {},
            "final_markdown_path": final_md_raw if (final_md is not None and final_md.exists()) else "",
            "final_pdf_path": final_pdf_raw if (final_pdf is not None and final_pdf.exists()) else "",
        },
    )

    # ``main`` is the canonical user-facing artifact (the rendered review
    # markdown). Only populate keys for files that actually exist on disk so
    # callers don't dereference paths to nothing. ``json`` is always written
    # because we just produced ``review_json`` above. Tie the overall stage
    # status to ``md_ok`` so the contract ``status == "ok" ⟹ outputs["main"]
    # exists`` holds.
    outputs: dict[str, str] = {"json": str(review_json)}
    if md_ok:
        outputs["main"] = str(review_md)
        outputs["markdown"] = str(review_md)
    if review_md_clean.exists():
        outputs["markdown_clean"] = str(review_md_clean)
    if audit_ok:
        outputs["audit_json"] = str(review_audit)
    if pdf_ok:
        outputs["pdf"] = str(pdf_path)
    error = ""
    if not md_ok:
        if final_md is None:
            error = "agent runner produced no final_markdown_path"
        elif not final_md.exists():
            error = f"final review markdown not found at {final_md}"
        else:
            error = f"failed to copy final review markdown from {final_md} to {review_md}"

    extra: dict[str, Any] = {}
    if pdf_render_error:
        # Markdown is the canonical artifact, so we keep status="ok" when it's
        # written even if the PDF render failed; surface the cause in extra so
        # the run summary records why no PDF exists.
        extra["pdf_render_error"] = pdf_render_error

    return StageResult(
        status="ok" if md_ok else "failed",
        outputs=outputs,
        extra=extra,
        error=error,
    )


if __name__ == "__main__":
    raise SystemExit("Internal stage module. Use scripts/execute_review_pipeline.py.")
