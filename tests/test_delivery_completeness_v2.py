"""Bounded delivery controls; external services are blocked by imported fixtures."""

import hashlib
import json
from pathlib import Path

import pytest

from review.delivery import checked_delivery
from review.report.v2 import write_review
from review.teaser.v2 import write_teaser
from schemas.claim import Claim, Condition
from schemas.limitations import VerificationLimitation
from schemas.review import FinalReview
from tests.test_pipeline_v2 import offline_boundaries as offline_boundaries
from tests.test_pipeline_v2 import run_tiny
from tests.test_pipeline_v2 import tiny_inputs as tiny_inputs
from tests.test_report_advice_v2 import generate
from tests.test_report_advice_v2 import review as advice_review


def seed():
    return FinalReview(paper_key="fixture", run_id="delivery", claims=[Claim(
        id="c", text="An unverified claim", loc={"page": 1},
        conditions=[Condition(id="a", description="A claim")], needs=["Experiments"],
    )])


def test_structured_nested_failures_preserve_science_and_stable_stages():
    original = seed()
    original.claims[0].verification_limitations.append(VerificationLimitation(
        claim_id="c", condition_ids=["a"], stage="Literature", kind="branch_failed", reason="Service failed.",
    ))
    result = checked_delivery(original, claim_coverage={"status": "partial"},
                              figure_context_coverage={"failed": 1}, writing_coverage={"total": 2, "checked": 1})
    assert result.run_status == "partial" and result.incomplete_stages == ["screening", "verification"]
    assert {c.component for c in result.delivery_checks} == {"claim_coverage", "figure_context_coverage", "writing_coverage", "Literature.branch_failed"}
    assert result.claims == original.claims and original.run_status == "completed"
    assert checked_delivery(result).delivery_checks == result.delivery_checks


def test_author_limits_scientific_outcomes_and_legacy_unknown_do_not_fail_delivery():
    original = seed()
    original.execution_requested = False
    original.ledger = [{"state": "blocked", "reason": "Author weights absent"},
                       {"approval": "declined", "returncode": 1}]
    original.claims[0].verification_limitations.append(VerificationLimitation(
        claim_id="c", condition_ids=["a"], stage="Experiments", kind="plan_rejected", reason="No author resources.",
    ))
    result = checked_delivery(original, stages={"execution": "skipped"},
                              claim_coverage={"status": "not_run"}, figure_context_coverage={"total": 1, "not_requested": 1})
    assert result.run_status == "completed" and result.incomplete_stages == []
    assert result.claims[0].status == "unverified"


def test_rejected_candidate_and_null_legacy_counters_are_not_unfinished_reviews():
    result = checked_delivery(seed(), claim_coverage={
        "status": "complete", "windows_total": None, "windows_reviewed": None,
        "original_claim_reviews_required": 2, "original_claim_reviews_completed": 2,
        "candidate_claim_checks_required": 1, "candidate_claim_checks_passed": 0,
    })
    assert result.run_status == "completed"
    result = checked_delivery(seed(), claim_coverage={"original_claim_reviews_required": 2, "original_claim_reviews_completed": 1})
    assert result.incomplete_stages == ["screening"]


@pytest.mark.parametrize("presentation", ["full", "layered"])
def test_post_revalidation_stale_advice_updates_all_text_artifacts(tmp_path, monkeypatch, presentation):
    from llm.client import LLMConfig

    monkeypatch.setattr("review.report.advice.resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None))
    original = advice_review(tmp_path)
    original.claims = [original.claims[0]]
    result = generate(original, tmp_path).review
    result.advice_requested = True
    assert result.claims[0].advice.state == "generated"
    (tmp_path / "source.md").write_text("Changed after generation.", encoding="utf-8")
    output = write_review(result, tmp_path / "render", render_pdf=False, presentation=presentation)
    saved = FinalReview.model_validate_json(Path(output["json"]).read_text("utf-8"))
    assert saved.claims[0].advice.state == "unavailable"
    assert saved.run_status == "partial" and saved.incomplete_stages == ["report"]
    assert "advice" in Path(output["markdown"]).read_text("utf-8")
    teaser = write_teaser(saved, tmp_path / "teaser")
    payload = json.loads(Path(teaser["json"]).read_text("utf-8"))
    assert payload["run_status"] == saved.run_status and payload["delivery_checks"] == [c.model_dump(mode="json") for c in saved.delivery_checks]
    assert result.claims[0].advice.state == "generated"


def test_pipeline_nested_partial_keeps_operational_stages_and_outputs_consistent(tiny_inputs, monkeypatch):
    summary, _, _, _ = run_tiny(tiny_inputs, monkeypatch, render_pdf=False)
    assert set(summary["stages"].values()) == {"ok"}
    saved = json.loads(Path(summary["outputs"]["report_json"]).read_text("utf-8"))
    teaser = json.loads(Path(summary["outputs"]["teaser_json"]).read_text("utf-8"))
    snapshot = json.loads(Path(summary["outputs"]["assessment_snapshot"]).read_text("utf-8"))
    assert snapshot["run_status"] == "partial"
    assert saved["run_status"] == teaser["run_status"] == summary["run_status"] == "partial"
    assert saved["incomplete_stages"] == teaser["incomplete_stages"] == summary["incomplete_stages"]
    assert saved["delivery_checks"] == teaser["delivery_checks"] == summary["delivery_checks"]
    assert "Partial review" in Path(summary["outputs"]["report_markdown"]).read_text("utf-8")


def test_requested_full_pdf_failure_is_recorded_without_retry(tmp_path, monkeypatch):
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("Fixture export failed")

    monkeypatch.setattr("review.report.pdf_renderer.build_review_report_pdf", fail)
    output = write_review(seed(), tmp_path / "failed")
    saved = json.loads(Path(output["json"]).read_text("utf-8"))
    assert len(calls) == 1 and "pdf_error" in output and "pdf" not in output
    assert saved["run_status"] == "partial" and saved["incomplete_stages"] == ["report"]
    assert "Fixture export failed" in Path(output["markdown"]).read_text("utf-8")
    other = write_review(seed(), tmp_path / "disabled", render_pdf=False)
    assert len(calls) == 1 and json.loads(Path(other["json"]).read_text("utf-8"))["run_status"] == "completed"


def test_layered_late_failure_preserves_healthy_pdfs_and_marks_pending_finalization(tmp_path, monkeypatch):
    calls = []

    def build(review, markdown, context, targets, title):
        calls.append(title)
        if len(calls) == 3:
            raise RuntimeError("Fixture bundle failed")
        return b"fixture PDF bytes"

    monkeypatch.setattr("review.report.compact._build_pdf", build)
    original = seed()
    output = write_review(original, tmp_path / "layered", presentation="layered")
    saved = json.loads(Path(output["json"]).read_text("utf-8"))
    manifest = json.loads(Path(output["manifest"]).read_text("utf-8"))
    assert len(calls) == 3 and "pdf" in output and "appendix_pdf" in output and "bundle_pdf_error" in output
    assert saved["run_status"] == "partial" and saved["claims"] == original.model_dump(mode="json")["claims"]
    assert "pdf_delivery_finalization" in {c["component"] for c in saved["delivery_checks"]}
    assert manifest["records_equal_saved_json"]
    for key, artifact in manifest["artifacts"].items():
        assert artifact["sha256"] == hashlib.sha256(Path(output[key]).read_bytes()).hexdigest()
    assert all("report export incomplete" in Path(output[key]).read_text("utf-8") for key in ("appendix_markdown", "bundle_markdown"))
