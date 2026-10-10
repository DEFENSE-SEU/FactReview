"""Unreadable report history cannot stop independent delivery or authorize PDFs."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import pipeline_v2
from review.delivery import checked_delivery
from review.recovery import finalize_teaser_review
from schemas.review import DeliveryCheck, FinalReview
from tests.test_pipeline_v2 import offline_boundaries as offline_boundaries
from tests.test_pipeline_v2 import run_tiny
from tests.test_pipeline_v2 import tiny_inputs as tiny_inputs
from tests.test_report_layered_v2 import fixture_review


def science(review):
    data = review.model_dump(mode="json", exclude={
        "review_markdown", "run_status", "incomplete_stages", "delivery_checks", "advice_requested",
    })
    for claim in data["claims"]:
        claim.pop("advice", None)
    return data


def unreadable(monkeypatch, path):
    read = Path.read_bytes
    attempts = []

    def checked(current):
        if current == path:
            attempts.append(current)
            raise PermissionError("Fixture history is locked")
        return read(current)

    monkeypatch.setattr(Path, "read_bytes", checked)
    return attempts


def test_locked_interrupted_report_preserves_original_error_and_delivers_teaser(tiny_inputs, monkeypatch):
    blocked = []

    def writer(review, directory, **kwargs):
        directory.mkdir(parents=True)
        path = directory / "final_review.pdf"
        path.write_bytes(b"interrupted fixture PDF")
        blocked.append((path, unreadable(monkeypatch, path)))
        raise RuntimeError("Fixture original report writer failed")

    monkeypatch.setattr(pipeline_v2, "write_review", Mock(side_effect=writer))
    summary, _, _, runner = run_tiny(tiny_inputs, monkeypatch, render_pdf=False)
    assert summary["stages"]["report"] == "failed" and summary["stages"]["teaser"] == "ok"
    assert summary["stage_errors"]["report"] == "RuntimeError: Fixture original report writer failed"
    assert runner.call_count == 1 and len(blocked[0][1]) == 1
    review = FinalReview.model_validate_json(Path(summary["outputs"]["report_json"]).read_text("utf-8"))
    assessed = FinalReview.model_validate_json(Path(summary["outputs"]["assessment_snapshot"]).read_text("utf-8"))
    assert science(review) == science(assessed)
    teaser = json.loads(Path(summary["outputs"]["teaser_json"]).read_text("utf-8"))
    assert review.run_status == teaser["run_status"] == summary["run_status"] == "partial"
    assert review.model_dump(mode="json")["delivery_checks"] == teaser["delivery_checks"] == summary["delivery_checks"]
    assert any(item.component == "report_history" for item in review.delivery_checks)
    assert "report_pdf" not in summary["outputs"] and "history_report_pdf" not in summary["outputs"]
    errors = json.loads(Path(summary["outputs"]["history_report_errors"]).read_text("utf-8"))
    assert errors == {"pdf": {"path": str(blocked[0][0]), "error_type": "PermissionError"}}
    assert blocked[0][0].stat().st_size == len(b"interrupted fixture PDF")


def test_locked_prior_pdf_is_recorded_and_never_eligible_for_static_export(tmp_path, monkeypatch):
    original = fixture_review()
    origin = tmp_path / "original"
    origin.mkdir()
    source = origin / "final_review.json"
    source.write_text(original.model_dump_json(), encoding="utf-8")
    pdf = origin / "final_review.pdf"
    pdf.write_bytes(b"prior fixture PDF")
    attempts = unreadable(monkeypatch, pdf)
    build = Mock(side_effect=AssertionError("Unreadable prior PDF is not eligible"))
    monkeypatch.setattr("review.report.pdf_renderer.build_review_report_pdf", build)
    partial = checked_delivery(original, additional_checks=[DeliveryCheck(
        stage="teaser", component="teaser_writer", state="failed", reason="Fixture teaser failed",
    )])
    review, outputs = finalize_teaser_review(partial, tmp_path / "final", report_succeeded=True,
        prior_outputs={"report_json": str(source), "report_pdf": str(pdf)}, presentation="full")
    assert science(review) == science(original) and source.read_text("utf-8") == original.model_dump_json()
    assert len(attempts) == 1 and build.call_count == 0 and "pdf" not in outputs
    assert {"report", "teaser"} <= set(review.incomplete_stages)
    assert any(item.component == "pdf.history" for item in review.delivery_checks)
    manifest = json.loads(Path(outputs["manifest"]).read_text("utf-8"))
    unavailable = manifest["prior_delivery"]["pdf"]
    assert unavailable == {"path": str(pdf), "role": "unavailable_prior_delivery_history", "error_type": "PermissionError"}
    assert manifest["pdf_snapshots"] == {} and "sha256" not in unavailable
    assert pdf.stat().st_size == len(b"prior fixture PDF")


def test_locked_nested_static_history_still_delivers_fallback_records(tmp_path, monkeypatch):
    original = fixture_review()
    source = tmp_path / "final_review.json"
    source.write_text(original.model_dump_json(), encoding="utf-8")
    blocked = []

    def failed_static(review, directory, **kwargs):
        directory.mkdir()
        (directory / "final_review.md").write_text("Interrupted fixture", encoding="utf-8")
        history = directory / "pdf_history"
        history.mkdir()
        path = history / "final_review.pdf"
        path.write_bytes(b"interrupted static fixture PDF")
        blocked.append((path, unreadable(monkeypatch, path)))
        raise RuntimeError("Fixture static package failed")

    writer = Mock(side_effect=failed_static)
    monkeypatch.setattr("review.recovery.write_static_review", writer)
    partial = checked_delivery(original, additional_checks=[DeliveryCheck(
        stage="teaser", component="teaser_writer", state="failed", reason="Fixture teaser failed",
    )])
    review, outputs = finalize_teaser_review(partial, tmp_path / "final", report_succeeded=True,
        prior_outputs={"report_json": str(source)}, presentation="full")
    assert science(review) == science(original) and writer.call_count == 1 and len(blocked[0][1]) == 1
    assert {"report", "teaser"} <= set(review.incomplete_stages) and "pdf" not in outputs
    manifest = json.loads(Path(outputs["manifest"]).read_text("utf-8"))
    assert manifest["render_errors"] == {
        "static_delivery_error": "RuntimeError",
        "history_errors": {"history_pdf": {"path": str(blocked[0][0]), "error_type": "PermissionError"}},
    }
    assert "history_pdf" not in manifest["interrupted_static_history"]
    assert "markdown" in manifest["interrupted_static_history"]
    assert any(item.component == "report_history" for item in review.delivery_checks)
    assert json.loads(Path(outputs["json"]).read_text("utf-8")) == review.model_dump(mode="json")
    assert blocked[0][0].stat().st_size == len(b"interrupted static fixture PDF")


@pytest.mark.parametrize("failed_key", ["json", "manifest"])
def test_unreadable_prior_metadata_is_not_read_again_for_validation(tmp_path, monkeypatch, failed_key):
    original = fixture_review()
    source = tmp_path / "final_review.json"
    source.write_text(original.model_dump_json(), encoding="utf-8")
    metadata = tmp_path / "report_manifest.json"
    metadata.write_text(json.dumps({"artifacts": {}}), encoding="utf-8")
    failed = {"json": source, "manifest": metadata}[failed_key]
    attempts = unreadable(monkeypatch, failed)
    read_text = Path.read_text
    repeated = []

    def checked(path, *args, **kwargs):
        if path == failed:
            repeated.append(path)
            raise AssertionError("Unavailable history must not be read again")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", checked)
    partial = checked_delivery(original, additional_checks=[DeliveryCheck(
        stage="teaser", component="teaser_writer", state="failed", reason="Fixture teaser failed",
    )])
    review, outputs = finalize_teaser_review(partial, tmp_path / "final", report_succeeded=True,
        prior_outputs={"report_json": str(source), "report_manifest": str(metadata)}, presentation="full")
    assert len(attempts) == 1 and not repeated and science(review) == science(original)
    assert "pdf" not in outputs and any(item.component == "pdf_delivery_source" for item in review.delivery_checks)
    manifest = json.loads(Path(outputs["manifest"]).read_text("utf-8"))
    assert manifest["prior_delivery"][failed_key]["error_type"] == "PermissionError"
    assert "sha256" not in manifest["prior_delivery"][failed_key]
