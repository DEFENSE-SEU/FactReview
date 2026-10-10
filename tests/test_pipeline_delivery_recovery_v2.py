"""Late writer failures retain scientific records and an honest delivery artifact."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import pipeline_v2
from tests.test_pipeline_v2 import offline_boundaries as offline_boundaries
from tests.test_pipeline_v2 import run_tiny
from tests.test_pipeline_v2 import tiny_inputs as tiny_inputs


@pytest.mark.parametrize("failed", ["report", "teaser"])
def test_late_writer_failure_keeps_last_review_and_other_delivery_work(tiny_inputs, monkeypatch, failed):
    writer = Mock(side_effect=RuntimeError("Fixture writer failed"))
    monkeypatch.setattr(pipeline_v2, "write_review" if failed == "report" else "write_teaser", writer)
    summary, _, _, _ = run_tiny(tiny_inputs, monkeypatch, render_pdf=False)
    assert writer.call_count == 1
    assert summary["stages"][failed] == "failed"
    assert summary["stages"]["teaser" if failed == "report" else "report"] == "ok"
    review = json.loads(Path(summary["outputs"]["report_json"]).read_text("utf-8"))
    teaser = json.loads(Path(summary["outputs"]["teaser_json"]).read_text("utf-8"))
    assert summary["counts"] == {"supported": 4, "flawed": 0, "questioned": 0, "unverified": 0}
    assert review["run_status"] == teaser["run_status"] == summary["run_status"] == "partial"
    assert failed in review["incomplete_stages"]
    assert review["delivery_checks"] == teaser["delivery_checks"] == summary["delivery_checks"]
    assert "Partial review" in Path(summary["outputs"]["report_markdown"]).read_text("utf-8")
    if failed == "teaser":
        previous = json.loads(Path(summary["outputs"]["history_report_json"]).read_text("utf-8"))
        assert previous["claims"] == review["claims"]
        assert "teaser" not in previous["incomplete_stages"]
        assert "teaser_image" not in summary["outputs"]


def test_late_teaser_failure_preserves_successful_pdf_as_labeled_history(tiny_inputs, monkeypatch):
    writer = Mock(side_effect=RuntimeError("Fixture teaser failed"))
    monkeypatch.setattr(pipeline_v2, "write_teaser", writer)
    monkeypatch.setattr("review.report.pdf_renderer.build_review_report_pdf", lambda **kwargs: b"preserved mock PDF")
    summary, _, _, _ = run_tiny(tiny_inputs, monkeypatch, render_pdf=True)
    assert writer.call_count == 1
    assert Path(summary["outputs"]["history_report_pdf"]).read_bytes() == b"preserved mock PDF"
    assert "report_pdf" not in summary["outputs"]
    review = json.loads(Path(summary["outputs"]["report_json"]).read_text("utf-8"))
    assert {"report", "teaser"} <= set(review["incomplete_stages"])
    assert "pdf_delivery_finalization" in {item["component"] for item in review["delivery_checks"]}
