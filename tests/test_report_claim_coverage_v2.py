"""Coverage presentation retains the assessed scientific records."""

import json
from pathlib import Path

import pytest

from review.report import compact, v2
from schemas.claim import Claim, Condition
from schemas.review import FinalReview


def review():
    return FinalReview(
        paper_key="coverage fixture",
        run_id="offline",
        claims=[
            Claim(
                id="c-safe",
                text="Original retained claim.",
                loc={"page": 1},
                conditions=[Condition(id="c1", description="Original full condition")],
                needs=["Code"],
                status="unverified",
            )
        ],
    )


@pytest.mark.parametrize("status", ["complete", "partial", "failed", "not_run"])
def test_full_and_layered_keep_coverage_visible_without_reassessment(tmp_path, status):
    original = review()
    snapshot = original.model_dump(mode="json", exclude={"review_markdown"})
    coverage = {
        "status": status,
        "initial_claims": 1,
        "final_claims": 1,
        "windows_total": 3,
        "windows_reviewed": 3 if status == "complete" else 1,
        "windows_unreviewed": 0 if status == "complete" else 2,
        "unresolved_observations": 0 if status == "complete" else 2,
        "blocked_claim_ids": [] if status == "complete" else ["c-blocked"],
        "audit_path": "screening/coverage.json",
    }
    for presentation in ("full", "layered"):
        outputs = v2.write_review(
            original,
            tmp_path / presentation,
            presentation=presentation,
            render_pdf=False,
            claim_coverage=coverage,
        )
        markdown = Path(outputs["markdown"]).read_text("utf-8")
        assert "### Claim extraction coverage" in markdown
        assert f"Coverage check status: **{v2._text(status)}**" in markdown
        assert f"Source windows reviewed: {coverage['windows_reviewed']} / 3" in markdown
        assert f"unreviewed: {coverage['windows_unreviewed']}" in markdown
        assert f"Unresolved observations: {coverage['unresolved_observations']}" in markdown
        assert "Initial claims: 1; final claims: 1" in markdown
        assert "Model-based coverage checks cannot prove exhaustive claim extraction." in markdown
        assert "screening/coverage\\.json" in markdown
        if status != "complete":
            assert "c\\-blocked" in markdown
        saved = json.loads(Path(outputs["json"]).read_text("utf-8"))
        saved.pop("review_markdown")
        assert saved == snapshot
        if presentation == "layered":
            manifest = json.loads(Path(outputs["manifest"]).read_text("utf-8"))
            assert manifest["delivery_context"]["claim_coverage"] == coverage
            assert "### Claim extraction coverage" in Path(outputs["appendix_markdown"]).read_text("utf-8")
    assert original.model_dump(mode="json", exclude={"review_markdown"}) == snapshot


def test_coverage_reaches_full_and_layered_pdf_inputs(tmp_path, monkeypatch):
    captured = []

    def full_pdf(**kwargs):
        captured.append(kwargs["final_report_markdown"])
        return b"mock PDF"

    def layered_pdf(checked, markdown, context, targets, title):
        assert context["claim_coverage"]["status"] == "failed"
        captured.append(markdown)
        return b"mock PDF"

    monkeypatch.setattr("review.report.pdf_renderer.build_review_report_pdf", full_pdf)
    monkeypatch.setattr(compact, "_build_pdf", layered_pdf)
    for presentation in ("full", "layered"):
        outputs = v2.write_review(
            review(),
            tmp_path / presentation,
            presentation=presentation,
            claim_coverage={"status": "failed"},
        )
        assert "pdf" in outputs
    assert len(captured) == 4
    assert all("Coverage check status: **failed**" in text for text in captured)
    assert all("Source windows reviewed: unavailable / unavailable" in text for text in captured)


def test_unspecified_coverage_keeps_legacy_presentation(tmp_path):
    original = review()
    assert v2.render_markdown(original) == v2.render_markdown(original, claim_coverage=None)
    outputs = v2.write_review(original, tmp_path, presentation="layered", render_pdf=False)
    manifest = json.loads(Path(outputs["manifest"]).read_text("utf-8"))
    assert "claim_coverage" not in manifest["delivery_context"]
    assert "### Claim extraction coverage" not in Path(outputs["markdown"]).read_text("utf-8")
