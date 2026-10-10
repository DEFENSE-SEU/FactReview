"""Deterministic delivery from the last validated review after a writer failure."""

import hashlib
from pathlib import Path

from review.delivery import checked_delivery
from review.report.v2 import _checked_report, render_markdown, validate_publication_language
from schemas.review import DeliveryCheck


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


def write_recovery_review(review, output_dir: Path, **context):
    """Keep the failed export intact and save a separate partial JSON/Markdown."""
    output_dir.mkdir(parents=True, exist_ok=False)
    result = checked_delivery(review)
    try:
        result = _checked_report(result)
        validate_publication_language([result.model_dump(), context.get("issues") or []])
        markdown = render_markdown(result, _checked=True, **context)
    except Exception as exc:
        result = checked_delivery(result, additional_checks=[DeliveryCheck(
            stage="report", component="markdown_recovery", state="failed",
            reason=f"Recovery narrative rendering raised {type(exc).__name__}; retained records are in final_review.json.",
        )])
        markdown = (
            "# FactReview — Partial review\n\n"
            "Narrative rendering is unavailable. The last validated claim, evidence, "
            "status and execution records are retained in final_review.json.\n"
        )
    result.review_markdown = markdown
    artifact, text = output_dir / "final_review.json", output_dir / "final_review.md"
    artifact.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    text.write_text(markdown, encoding="utf-8")
    return result, {"json": str(artifact), "markdown": str(text)}
