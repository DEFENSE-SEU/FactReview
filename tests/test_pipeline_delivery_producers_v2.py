"""One offline handoff control for retained operation failures."""

import json
from pathlib import Path

import pipeline_v2
from schemas.review import DeliveryCheck
from tests.test_pipeline_v2 import offline_boundaries as offline_boundaries
from tests.test_pipeline_v2 import run_tiny
from tests.test_pipeline_v2 import tiny_inputs as tiny_inputs


def test_multiple_producer_failures_reach_canonical_report_teaser_and_summary(tiny_inputs, monkeypatch):
    original_screening = pipeline_v2.screen_paper
    original_verification = pipeline_v2.verify_claims
    references = [DeliveryCheck(
        stage="screening", component="reference_pdf", state="failed",
        reason=f"PDF confirmation failed. Audit: reference_validation.json page {page}.",
    ) for page in (1, 2)]
    global_check = DeliveryCheck(
        stage="verification", component="global_literature.search", state="failed",
        reason="Query failed. Audit: global-search-audit.json#/queries/0.",
    )

    def screening(*args, **kwargs):
        result = original_screening(*args, **kwargs)
        result.delivery_checks.extend(references)
        return result

    async def verification(*args, **kwargs):
        result = await original_verification(*args, **kwargs)
        result.delivery_checks.append(global_check)
        return result

    monkeypatch.setattr(pipeline_v2, "screen_paper", screening)
    monkeypatch.setattr(pipeline_v2, "verify_claims", verification)
    summary, _, _, _ = run_tiny(tiny_inputs, monkeypatch, render_pdf=False)
    saved = json.loads(Path(summary["outputs"]["report_json"]).read_text("utf-8"))
    teaser = json.loads(Path(summary["outputs"]["teaser_json"]).read_text("utf-8"))
    expected = [item.model_dump(mode="json") for item in [*references, global_check]]
    assert set(summary["stages"].values()) == {"ok"}
    assert summary["counts"] == {"supported": 4, "flawed": 0, "questioned": 0, "unverified": 0}
    assert saved["run_status"] == teaser["run_status"] == summary["run_status"] == "partial"
    assert saved["delivery_checks"] == teaser["delivery_checks"] == summary["delivery_checks"]
    assert all(item in saved["delivery_checks"] for item in expected)
    assert len([item for item in saved["delivery_checks"] if item["component"] == "reference_pdf"]) == 2
    assert {"screening", "verification"} <= set(summary["incomplete_stages"])
