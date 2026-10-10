"""Deterministic delivery from the last validated review after a writer failure."""

import hashlib
import json
from pathlib import Path

from review.delivery import checked_delivery
from review.report.v2 import (
    _checked_report,
    render_markdown,
    validate_publication_language,
    write_static_review,
)
from schemas.review import DeliveryCheck, FinalReview


def interrupted_report_history(directory: Path):
    """Index owned export files without adopting their interrupted contents."""
    names = {
        "json": "final_review.json", "markdown": "final_review.md",
        "appendix_markdown": "technical_appendix.md", "bundle_markdown": "review_bundle.md",
        "pdf": "final_review.pdf", "appendix_pdf": "technical_appendix.pdf",
        "bundle_pdf": "review_bundle.pdf", "manifest": "report_manifest.json",
    }
    result = {}
    for key, name in names.items():
        path = directory / name
        if path.is_file() and path.resolve().is_relative_to(directory.resolve()):
            data = path.read_bytes()
            result[key] = {
                "path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data), "role": "interrupted_writer_history",
            }
    return result


def write_recovery_review(review, output_dir: Path, *, render_narrative=True, **context):
    """Keep the failed export intact and save a separate partial JSON/Markdown."""
    output_dir.mkdir(parents=True, exist_ok=False)
    result = checked_delivery(review)
    markdown = (
        "# FactReview — Partial review\n\n"
        "Narrative rendering is unavailable. The last validated claim, evidence, "
        "status and execution records are retained in final_review.json.\n"
    )
    try:
        if render_narrative:
            result = _checked_report(result)
            validate_publication_language([result.model_dump(), context.get("issues") or []])
            markdown = render_markdown(result, _checked=True, **context)
    except Exception as exc:
        result = checked_delivery(result, additional_checks=[DeliveryCheck(
            stage="report", component="markdown_recovery", state="failed",
            reason=f"Recovery narrative rendering raised {type(exc).__name__}; retained records are in final_review.json.",
        )])
    result.review_markdown = markdown
    artifact, text = output_dir / "final_review.json", output_dir / "final_review.md"
    artifact.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    text.write_text(markdown, encoding="utf-8")
    return result, {"json": str(artifact), "markdown": str(text)}


def _scientific_records(review):
    return review.model_dump(mode="json", exclude={
        "review_markdown", "run_status", "incomplete_stages", "delivery_checks",
    })


def _minimal_package(review, output_dir, prior_delivery, *, history=None, errors=None):
    result, outputs = write_recovery_review(review, output_dir, render_narrative=False)
    manifest = {
        "version": "static-recovery-fallback-v1", "model_calls": 0,
        "prior_delivery": prior_delivery, "interrupted_static_history": history or {},
        "pdf_snapshots": {}, "pdf_targets": {},
        "records_equal_saved_json": result.model_dump(mode="json") == json.loads(Path(outputs["json"]).read_text("utf-8")),
        "artifacts": {key: {"name": Path(value).name, "role": "current_delivery",
                            "sha256": hashlib.sha256(Path(value).read_bytes()).hexdigest()}
                      for key, value in outputs.items()},
        "render_errors": errors or {},
    }
    path = output_dir / "report_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["manifest"] = str(path)
    return result, outputs


def finalize_teaser_review(review, output_dir: Path, *, prior_outputs, report_succeeded,
                           presentation="full", **context):
    """Export only canonical healthy PDFs after a late teaser failure.

    Prior files remain in their original directory. An interrupted whole writer
    cannot supply eligible PDFs, even when it happened to write readable files.
    """
    names = {"pdf": "final_review.pdf", "appendix_pdf": "technical_appendix.pdf",
             "bundle_pdf": "review_bundle.pdf"}
    prior_delivery = {}
    for key, value in prior_outputs.items():
        if not key.startswith("report_") or key.endswith("_error"):
            continue
        path = Path(value)
        if path.is_file():
            data = path.read_bytes()
            prior_delivery[key.removeprefix("report_")] = {
                "path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data), "role": "prior_delivery_history",
            }
    if not report_succeeded:
        # Earlier writer recovery has already retained the last validated
        # records. Never re-enter that failed formatter or adopt its PDFs.
        return _minimal_package(review, output_dir, prior_delivery)
    eligible = []
    checks = []
    source_json = prior_outputs.get("report_json")
    origin = Path(source_json).parent if source_json else None
    source_ok = False
    source_manifest = None
    manifest_provided = "report_manifest" in prior_outputs
    if not source_json and any("report_" + key in prior_outputs for key in names):
        checks.append(DeliveryCheck(stage="report", component="pdf_delivery_source", state="failed",
                                    reason="Prior canonical PDFs lack their validated report JSON; no PDF was adopted."))
    if report_succeeded and source_json:
        try:
            saved = FinalReview.model_validate_json(Path(source_json).read_text("utf-8"))
            source_ok = _scientific_records(saved) == _scientific_records(review)
            if not source_ok:
                raise ValueError("Prior scientific records do not match validated review")
            if manifest_provided:
                source_manifest = json.loads(Path(prior_outputs["report_manifest"]).read_text("utf-8"))
                if not isinstance(source_manifest, dict) or not isinstance(source_manifest.get("artifacts"), dict):
                    raise ValueError("Provided source manifest must contain an artifacts object")
            elif presentation != "full":
                raise ValueError("Layered PDF delivery requires its source manifest")
        except Exception as exc:
            source_ok = False
            checks.append(DeliveryCheck(stage="report", component="pdf_delivery_source", state="failed",
                                        reason=f"Prior delivery validation raised {type(exc).__name__}."))
    if report_succeeded and source_ok:
        for key, name in names.items():
            value = prior_outputs.get("report_" + key)
            if not value or (presentation == "full" and key != "pdf"):
                continue
            path = Path(value)
            metadata = prior_delivery.get(key)
            valid = (metadata is not None and path.name == name and path.parent.resolve() == origin.resolve())
            if manifest_provided:
                entry = source_manifest["artifacts"].get(key)
                valid = valid and isinstance(entry, dict) and entry.get("role") == "current_delivery" and (
                    entry.get("sha256") == metadata["sha256"] and entry.get("name") == name
                )
            if valid:
                eligible.append(key)
            else:
                checks.append(DeliveryCheck(stage="report", component=key + ".source", state="failed",
                                            reason="Prior canonical PDF is absent, changed or lacks a current-delivery role."))
    review = checked_delivery(review, additional_checks=checks)
    original = _scientific_records(review)
    try:
        outputs = write_static_review(review, output_dir, pdf_keys=eligible,
                                      prior_delivery=prior_delivery, presentation=presentation, **context)
        result = FinalReview.model_validate_json(Path(outputs["json"]).read_text("utf-8"))
        if _scientific_records(result) != original:
            raise ValueError("Static delivery changed scientific records")
        return result, outputs
    except Exception as exc:
        checks = [DeliveryCheck(stage="report", component="static_delivery_writer", state="failed",
                                reason=f"Static delivery raised {type(exc).__name__}; validated records retained separately.")]
        if eligible:
            checks.append(DeliveryCheck(stage="report", component="pdf_delivery_finalization", state="incomplete",
                                        reason="Static PDF package finalization failed; prior files remain unchanged."))
        result = checked_delivery(review, additional_checks=checks)
        history = interrupted_report_history(output_dir)
        # A late package failure can occur after a healthy correction has
        # retained its first-pass bytes. Keep that owned history discoverable.
        for key, metadata in interrupted_report_history(output_dir / "pdf_history").items():
            history["history_" + key] = metadata
        return _minimal_package(result, output_dir / "fallback", prior_delivery, history=history,
                                errors={"static_delivery_error": type(exc).__name__})
