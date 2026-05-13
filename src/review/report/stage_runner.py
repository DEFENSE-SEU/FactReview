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

from agent_runtime.runner import augment_claims_with_assessment_status
from common.config import get_settings
from common.pipeline_context import (
    claim_extract_stage_dir,
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
from review.report.claim_audit import audit_review_markdown
from review.report.pdf_renderer import build_review_report_pdf
from schemas.stage import StageResult
from util.fs import copy_file_if_exists


_DETAILED_CLAIMS_HEADING = "**Detailed extracted claims:**"


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


def _markdown_cell(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text.replace("|", "&#124;") or "Not found in manuscript"


def _claim_location_text(claim: dict[str, Any]) -> str:
    location = claim.get("location")
    if not isinstance(location, dict):
        return ""
    parts: list[str] = []
    section_id = str(location.get("section_id") or "").strip()
    if section_id:
        parts.append(section_id)
    page = location.get("page")
    if page is not None:
        parts.append(f"page {page}")
    char_start = location.get("char_start")
    char_end = location.get("char_end")
    if char_start is not None or char_end is not None:
        start = "" if char_start is None else str(char_start)
        end = "" if char_end is None else str(char_end)
        parts.append(f"chars {start}-{end}")
    return "; ".join(parts)


def _claim_table_row_count(claims_body: str) -> int:
    lines = claims_body.splitlines()
    header_idx = -1
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped.startswith("|")
            and "claim" in stripped.lower()
            and "evidence" in stripped.lower()
            and "location" in stripped.lower()
        ):
            header_idx = idx
            break
    if header_idx < 0:
        return 0
    count = 0
    for line in lines[header_idx + 2 :]:
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            break
        count += 1
    return count


def _format_detailed_claims_block(claims: list[dict[str, Any]]) -> str:
    rows = [
        _DETAILED_CLAIMS_HEADING,
        "Full detailed claim-extraction layer used for downstream review; teaser figures use only the compact core-claim layer.",
        "",
        "| ID | Claim | Type | Importance | Location | Evidence targets |",
        "|---|---|---|---|---|---|",
    ]
    for idx, claim in enumerate(claims, start=1):
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("id") or f"claim_{idx:02d}")
        targets = claim.get("evidence_targets")
        rows.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(claim_id),
                    _markdown_cell(claim.get("text")),
                    _markdown_cell(claim.get("type")),
                    _markdown_cell(claim.get("importance")),
                    _markdown_cell(_claim_location_text(claim)),
                    _markdown_cell(", ".join(str(t) for t in targets) if isinstance(targets, list) else ""),
                ]
            )
            + " |"
        )
    return "\n".join(rows).rstrip()


def _append_detailed_claims_block(markdown_text: str, facts_payload: dict[str, Any]) -> str:
    claims = facts_payload.get("claims")
    if not isinstance(claims, list) or not claims:
        return markdown_text
    if _DETAILED_CLAIMS_HEADING in markdown_text:
        return markdown_text

    sec = re.search(
        r"(?ims)(^##\s+(?:\*\*)?3\.\s+Claims(?:\*\*)?\s*$\n)(?P<body>.*?)(?=^##\s+|\Z)",
        markdown_text,
    )
    if not sec:
        return markdown_text
    body = sec.group("body")
    if _claim_table_row_count(body) >= len(claims):
        return markdown_text

    lines = body.splitlines()
    insert_idx = -1
    header_idx = -1
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped.startswith("|")
            and "claim" in stripped.lower()
            and "evidence" in stripped.lower()
            and "location" in stripped.lower()
        ):
            header_idx = idx
            break
    if header_idx >= 0:
        insert_idx = header_idx + 2
        while insert_idx < len(lines):
            stripped = lines[insert_idx].strip()
            if not (stripped.startswith("|") and stripped.endswith("|")):
                break
            insert_idx += 1
    if insert_idx < 0:
        insert_idx = len(lines)

    block = _format_detailed_claims_block([claim for claim in claims if isinstance(claim, dict)])
    updated_lines = lines[:insert_idx] + ["", block, ""] + lines[insert_idx:]
    updated_body = "\n".join(updated_lines).rstrip() + "\n\n"
    return markdown_text[: sec.start("body")] + updated_body + markdown_text[sec.end("body") :]


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

    # Re-normalize the claims table after the runtime job. The first pass inside
    # the agent runtime ran before execution, so this stage can add real
    # execution alignment when available and keep markdown-table escaping stable
    # before the mandatory claim audit.
    if md_ok:
        exec_json = read_json_file(execution_stage_dir(run_dir) / "execution.json")
        exec_alignment = exec_json.get("alignment") if isinstance(exec_json, dict) else {}
        current_md = review_md.read_text(encoding="utf-8", errors="ignore")
        augmented_md = augment_claims_with_assessment_status(
            current_md,
            summary=exec_json.get("summary") or {},
            alignment=exec_alignment if isinstance(exec_alignment, dict) else {},
        )
        if augmented_md != current_md:
            review_md.write_text(augmented_md, encoding="utf-8")

    settings = get_settings()
    reference_check_payload = _load_reference_check_payload(run_dir)
    claim_extract_payload = read_json_file(claim_extract_stage_dir(run_dir) / "facts.json")
    reference_check_markdown = ""
    reference_check_appended = False
    pdf_render_error = ""
    claim_audit_payload: dict[str, Any] = {}

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
        if exec_json.get("status") == "skipped":
            for md_path in (review_md, review_md_clean):
                stripped = _strip_experiment_eval_status(
                    md_path.read_text(encoding="utf-8", errors="ignore")
                )
                md_path.write_text(stripped, encoding="utf-8")

        # Post-report claim audit. review_md and review_md_clean are identical
        # at this point (refcheck is appended later only to review_md), so run
        # the mandatory LLM audit once and copy the audited canonical markdown
        # into the clean teaser source.
        claim_audit_payload = {}
        if review_md.exists():
            source_text = review_md.read_text(encoding="utf-8", errors="ignore")
            audited_text, outcome = audit_review_markdown(
                source_text,
                execution_alignment=exec_alignment if isinstance(exec_alignment, dict) else None,
            )
            if audited_text != source_text:
                review_md.write_text(audited_text, encoding="utf-8")
            current_text = review_md.read_text(encoding="utf-8", errors="ignore")
            detailed_text = _append_detailed_claims_block(current_text, claim_extract_payload)
            if detailed_text != current_text:
                review_md.write_text(detailed_text, encoding="utf-8")
            shutil.copy2(review_md, review_md_clean)
            claim_audit_payload = {
                "claim_results": [
                    {
                        "original_status": c.original_status,
                        "final_status": c.final_status,
                        "llm_verdict": c.llm_verdict,
                        "llm_reason": c.llm_reason,
                        "agent_self_verdict": c.agent_self_verdict,
                        "agent_self_reason": c.agent_self_reason,
                        "notes": c.notes,
                    }
                    for c in outcome.claim_results
                ],
                "axis_self_selection_ratio": outcome.axis_self_selection_ratio,
                "ablation_components_missing": outcome.ablation_components_missing,
                "extra_weaknesses": outcome.extra_weaknesses,
                "llm_raw": outcome.llm_raw,
            }

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

    if claim_audit_payload:
        write_json_file(out_dir / "claim_audit.json", claim_audit_payload)

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
            "claim_audit": claim_audit_payload,
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
