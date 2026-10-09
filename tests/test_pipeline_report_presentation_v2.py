"""Layout selection and individual PDF failures survive pipeline delivery."""

import json
import sys
from pathlib import Path

import pytest

import pipeline_full
import pipeline_v2
from tests.test_pipeline_v2 import offline_boundaries as offline_boundaries
from tests.test_pipeline_v2 import run_tiny
from tests.test_pipeline_v2 import tiny_inputs as tiny_inputs


@pytest.mark.parametrize("mode", ["full", "layered"])
def test_cli_accepts_explicit_report_layout(monkeypatch, mode):
    monkeypatch.setattr(sys, "argv", ["factreview", "paper.pdf", "--report-presentation", mode])
    assert pipeline_full.parse_args().report_presentation == mode


def test_invalid_layout_stops_before_external_work(tiny_inputs, monkeypatch):
    args, parser = tiny_inputs
    args.report_presentation = "invalid"
    summary, model, retrieval, runner = run_tiny(tiny_inputs, monkeypatch, render_pdf=False)
    assert "Unknown report presentation" in summary["stage_errors"]["materials"]
    assert parser.calls == [] and retrieval.queries == []
    assert model.calls == []
    runner.assert_not_called()


def test_each_pdf_error_is_diagnostic_and_never_an_output_path(tiny_inputs, monkeypatch):
    args, _ = tiny_inputs
    args.report_presentation = "layered"
    original = pipeline_v2.write_review
    seen = []

    def render(review, directory, **kwargs):
        seen.append(kwargs.pop("presentation"))
        outputs = original(review, directory, **kwargs)
        outputs.update(
            pdf_error="Injected primary PDF failure",
            appendix_pdf_error="Injected appendix PDF failure",
            bundle_pdf_error="Injected bundle PDF failure",
        )
        return outputs

    monkeypatch.setattr(pipeline_v2, "write_review", render)
    summary, _, _, _ = run_tiny(tiny_inputs, monkeypatch, render_pdf=False)
    assert seen == ["layered"] and summary["report_presentation"] == "layered"
    assert summary["stages"]["report"] == "ok" and summary["stages"]["teaser"] == "ok"
    for name in ("pdf", "appendix_pdf", "bundle_pdf"):
        assert summary["stage_errors"][f"report_{name}"].startswith("Injected")
        assert f"report_{name}_error" not in summary["outputs"]
        assert f"report_{name}" not in summary["outputs"]
    assert Path(summary["outputs"]["report_json"]).is_file()
    saved = json.loads((Path(summary["run_dir"]) / "full_pipeline_summary.json").read_text("utf-8"))
    assert saved["stage_errors"] == summary["stage_errors"]


def test_layered_report_and_teaser_use_the_same_assessed_claims(tiny_inputs, monkeypatch):
    args, parser = tiny_inputs
    args.report_presentation = "layered"
    summary, model, _, runner = run_tiny(tiny_inputs, monkeypatch, render_pdf=True)
    assert summary["stage_errors"] == {}
    assert set(summary["stages"]) == set(pipeline_v2.STAGES)
    assert set(summary["stages"].values()) == {"ok"}
    assert summary["counts"] == {"supported": 4, "flawed": 0, "questioned": 0, "unverified": 0}
    assert runner.call_count == 1 and len(parser.calls) == 1
    assert model.calls.count("report_generation") == 4
    for key in (
        "markdown",
        "json",
        "pdf",
        "appendix_markdown",
        "appendix_pdf",
        "bundle_markdown",
        "bundle_pdf",
        "manifest",
    ):
        assert Path(summary["outputs"]["report_" + key]).is_file()
    manifest = json.loads(Path(summary["outputs"]["report_manifest"]).read_text("utf-8"))
    assert manifest["records_equal_saved_json"] and manifest["checked_records_equal_input"]
    assert not manifest["render_errors"]
    saved = json.loads(Path(summary["outputs"]["report_json"]).read_text("utf-8"))
    teaser = json.loads(Path(summary["outputs"]["teaser_json"]).read_text("utf-8"))
    assert len(saved["claims"]) == 4
    assert all(c["advice"]["state"] == "generated" for c in saved["claims"])
    assert {c["id"]: c["status"] for c in saved["claims"]} == {c["id"]: c["status"] for c in teaser["claims"]}
    assert Path(summary["outputs"]["teaser_image"]).is_file()
