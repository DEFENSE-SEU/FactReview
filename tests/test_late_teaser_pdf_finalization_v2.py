"""Static late-delivery exports preserve original files and scientific records."""

import hashlib
import json
import re
from pathlib import Path
from unittest.mock import Mock

import pytest

import pipeline_v2
from review.delivery import checked_delivery
from review.report.v2 import write_review
from schemas.review import DeliveryCheck, FinalReview
from tests.test_pipeline_v2 import offline_boundaries as offline_boundaries
from tests.test_pipeline_v2 import run_tiny
from tests.test_pipeline_v2 import tiny_inputs as tiny_inputs
from tests.test_report_layered_v2 import fixture_review, pdf_navigation


def science(review):
    return review.model_dump(mode="json", exclude={"review_markdown", "run_status", "incomplete_stages", "delivery_checks"})


def partial(review):
    return checked_delivery(review, additional_checks=[DeliveryCheck(
        stage="teaser", component="teaser_writer", state="failed", reason="Fixture teaser failed",
    )])


def renderer(calls, failures=()):
    def build(**kwargs):
        title = kwargs.get("workspace_title", "")
        calls.append(title)
        if len(calls) in failures:
            raise RuntimeError("Fixture static export failed")
        targets = kwargs.get("navigation_targets", {})
        for page, anchor in enumerate(re.findall(r'id="([^"]+)"', kwargs["final_report_markdown"]), 1):
            targets[anchor] = page
        return json.dumps({"status": kwargs["status"], "markdown": kwargs["final_report_markdown"]}).encode()
    return build


def manifest_check(outputs, review):
    manifest = json.loads(Path(outputs["manifest"]).read_text("utf-8"))
    for key, item in manifest["artifacts"].items():
        assert hashlib.sha256(Path(outputs[key]).read_bytes()).hexdigest() == item["sha256"]
    for key, snapshot in manifest["pdf_snapshots"].items():
        if key.startswith("history_"):
            continue
        assert snapshot["delivery"]["run_status"] == review.run_status
        assert snapshot["delivery"]["incomplete_stages"] == review.incomplete_stages
        pdf = json.loads(Path(outputs[key]).read_bytes())
        assert "partial" in pdf["status"] and "teaser" in pdf["markdown"]
    return manifest


def test_pipeline_late_teaser_delivers_current_full_pdf(tiny_inputs, monkeypatch):
    writer = Mock(side_effect=RuntimeError("Fixture teaser failed"))
    calls = []
    monkeypatch.setattr(pipeline_v2, "write_teaser", writer)
    monkeypatch.setattr("review.report.pdf_renderer.build_review_report_pdf", renderer(calls))
    summary, _, _, _ = run_tiny(tiny_inputs, monkeypatch, render_pdf=True)
    assert writer.call_count == 1 and len(calls) == 2
    assert "report_pdf" in summary["outputs"]
    review = FinalReview.model_validate_json(Path(summary["outputs"]["report_json"]).read_text("utf-8"))
    original = FinalReview.model_validate_json(Path(summary["outputs"]["history_report_json"]).read_text("utf-8"))
    assert science(review) == science(original)
    assert "report" not in review.incomplete_stages
    assert "pdf_delivery_finalization" not in {x.component for x in review.delivery_checks}
    assert summary["delivery_checks"] == json.loads(Path(summary["outputs"]["teaser_json"]).read_text("utf-8"))["delivery_checks"]
    outputs = {key.removeprefix("report_"): value for key, value in summary["outputs"].items() if key.startswith("report_")}
    manifest = manifest_check(outputs, review)
    assert manifest["prior_delivery"]["pdf"]["path"] == summary["outputs"]["history_report_pdf"]


@pytest.mark.parametrize("failures,expected_calls", [((), 6), ((4,), 8), ((4, 7), 8)])
def test_layered_late_export_finalizes_only_healthy_set(tmp_path, monkeypatch, failures, expected_calls):
    from review.recovery import finalize_teaser_review

    calls = []
    monkeypatch.setattr("review.report.pdf_renderer.build_review_report_pdf", renderer(calls, failures))
    original = fixture_review()
    source = write_review(original, tmp_path / "original", presentation="layered")
    before = {path.relative_to(tmp_path / "original"): path.read_bytes() for path in (tmp_path / "original").rglob("*") if path.is_file()}
    saved = FinalReview.model_validate_json(Path(source["json"]).read_text("utf-8"))
    review, outputs = finalize_teaser_review(partial(saved), tmp_path / "final", report_succeeded=True,
        prior_outputs={"report_" + key: value for key, value in source.items()}, presentation="layered")
    assert len(calls) == expected_calls
    assert science(review) == science(saved)
    assert before == {path.relative_to(tmp_path / "original"): path.read_bytes() for path in (tmp_path / "original").rglob("*") if path.is_file()}
    manifest = manifest_check(outputs, review)
    assert manifest["records_equal_saved_json"] and manifest["checked_changes"]["advice_claim_indices"] == []
    assert set(manifest["prior_delivery"]) == set(source)
    assert manifest["pdf_targets"]["appendix"] == ({} if failures else manifest["pdf_snapshots"]["appendix_pdf"]["pages"])
    if failures:
        assert "appendix_pdf" not in outputs
        assert sum("Technical appendix" in title for title in calls[3:]) == 1
        assert "report" in review.incomplete_stages
        assert Path(outputs["history_pdf"]).read_bytes() != Path(source["pdf"]).read_bytes()
        assert ("pdf" not in outputs) == (7 in failures)
        assert any(x.component == "pdf_delivery_finalization" for x in review.delivery_checks)
    else:
        assert "report" not in review.incomplete_stages
        assert set(manifest["pdf_snapshots"]) == {"pdf", "appendix_pdf", "bundle_pdf"}


@pytest.mark.parametrize("fail_recovery", [False, True])
def test_failed_original_and_history_keys_never_exported(tmp_path, monkeypatch, fail_recovery):
    from review.recovery import finalize_teaser_review

    calls = []
    monkeypatch.setattr("review.report.pdf_renderer.build_review_report_pdf", renderer(calls, (1, 6) if fail_recovery else (1,)))
    source = write_review(fixture_review(), tmp_path / "original", presentation="layered")
    saved = FinalReview.model_validate_json(Path(source["json"]).read_text("utf-8"))
    prior = {"report_" + key: value for key, value in source.items() if not key.endswith("_error")}
    prior["report_history_appendix_pdf"] = source["pdf"]
    before = len(calls)
    review, outputs = finalize_teaser_review(partial(saved), tmp_path / "final", report_succeeded=True,
                                           prior_outputs=prior, presentation="layered")
    assert len(calls) - before == 2
    assert "appendix_pdf" not in outputs and science(review) == science(saved)
    assert all("Technical appendix" not in title for title in calls[before:])
    assert "history_pdf" not in outputs and "history_bundle_pdf" not in outputs
    if fail_recovery:
        assert "pdf" not in outputs and "bundle_pdf" in outputs
        manifest_check(outputs, review)


def test_interrupted_writer_pdfs_are_not_adopted(tmp_path, monkeypatch):
    from review.recovery import finalize_teaser_review

    build = Mock(side_effect=AssertionError("No eligible export"))
    formatter = Mock(side_effect=AssertionError("Failed formatter cannot repeat"))
    monkeypatch.setattr("review.report.pdf_renderer.build_review_report_pdf", build)
    monkeypatch.setattr("review.report.v2.render_markdown", formatter)
    old = tmp_path / "old.pdf"
    old.write_bytes(b"interrupted bytes")
    review, outputs = finalize_teaser_review(partial(fixture_review()), tmp_path / "final", report_succeeded=False,
                                            prior_outputs={"report_pdf": str(old), "history_report_pdf": str(old)}, presentation="layered")
    assert build.call_count == 0 and "pdf" not in outputs
    assert formatter.call_count == 0
    assert old.read_bytes() == b"interrupted bytes"
    assert science(review) == science(fixture_review())


def test_static_formatter_failure_uses_minimal_fallback_once(tmp_path, monkeypatch):
    from review.recovery import finalize_teaser_review

    calls = []
    monkeypatch.setattr("review.report.pdf_renderer.build_review_report_pdf", renderer(calls))
    source = write_review(fixture_review(), tmp_path / "original", presentation="layered")
    formatter = Mock(side_effect=RuntimeError("Fixture formatter failed"))
    monkeypatch.setattr("review.report.v2.render_markdown", formatter)
    saved = FinalReview.model_validate_json(Path(source["json"]).read_text("utf-8"))
    review, outputs = finalize_teaser_review(partial(saved), tmp_path / "final", report_succeeded=True,
        prior_outputs={"report_" + key: value for key, value in source.items()}, presentation="layered")
    assert formatter.call_count == 1
    assert len(calls) == 3
    assert science(review) == science(saved)
    assert "Narrative rendering is unavailable" in Path(outputs["markdown"]).read_text("utf-8")
    assert "report" in review.incomplete_stages


def test_changed_source_pdf_is_excluded_without_rewriting_it(tmp_path, monkeypatch):
    from review.recovery import finalize_teaser_review

    calls = []
    monkeypatch.setattr("review.report.pdf_renderer.build_review_report_pdf", renderer(calls))
    source = write_review(fixture_review(), tmp_path / "original", presentation="layered")
    Path(source["appendix_pdf"]).write_bytes(b"changed source bytes")
    saved = FinalReview.model_validate_json(Path(source["json"]).read_text("utf-8"))
    review, outputs = finalize_teaser_review(partial(saved), tmp_path / "final", report_succeeded=True,
        prior_outputs={"report_" + key: value for key, value in source.items()}, presentation="layered")
    assert len(calls) == 5 and "appendix_pdf" not in outputs
    assert Path(source["appendix_pdf"]).read_bytes() == b"changed source bytes"
    assert any(x.component == "appendix_pdf.source" for x in review.delivery_checks)
    assert science(review) == science(saved)
    manifest_check(outputs, review)


def test_static_real_pdf_navigation_uses_new_pages_without_reassessment(tmp_path, monkeypatch):
    from review.recovery import finalize_teaser_review

    source = write_review(fixture_review(), tmp_path / "original", presentation="layered")
    saved = FinalReview.model_validate_json(Path(source["json"]).read_text("utf-8"))
    checked = Mock(side_effect=AssertionError("Already validated records cannot be reassessed"))
    monkeypatch.setattr("review.report.v2._checked_report", checked)
    review, outputs = finalize_teaser_review(partial(saved), tmp_path / "final", report_succeeded=True,
        prior_outputs={"report_" + key: value for key, value in source.items()}, presentation="layered")
    assert checked.call_count == 0 and science(review) == science(saved)
    manifest = json.loads(Path(outputs["manifest"]).read_text("utf-8"))
    bundle, links = pdf_navigation(outputs["bundle_pdf"])
    main, _ = pdf_navigation(outputs["pdf"])
    appendix, _ = pdf_navigation(outputs["appendix_pdf"])
    boundary = manifest["pdf_targets"]["bundle"][manifest["records"][""]["appendix_anchor"]]
    assert any(a < boundary <= b for a, b in links)
    assert any(b < boundary <= a for a, b in links)
    for record in manifest["records"].values():
        assert record["main_pdf_page"] and record["appendix_pdf_page"]
        assert record["bundle_main_page"] and record["bundle_appendix_page"]
    main_text = "\n".join(page.extract_text() or "" for page in main.pages)
    claim = manifest["records"]["/claims/0/source_quote"]
    assert f"technical_appendix.pdf, page {claim['appendix_pdf_page']}" in main_text
    assert "teaser" in main_text
    assert all("teaser" in "\n".join(page.extract_text() or "" for page in pdf.pages)
               for pdf in (appendix, bundle))
    assert all(not any("/URI" in str(ref.get_object()) for ref in page.get("/Annots", []))
               for page in bundle.pages)
    for key, metadata in manifest["artifacts"].items():
        assert metadata["sha256"] == hashlib.sha256(Path(outputs[key]).read_bytes()).hexdigest()


def test_missing_validated_json_never_adopts_pdf(tmp_path, monkeypatch):
    from review.recovery import finalize_teaser_review

    build = Mock(side_effect=AssertionError("Unvalidated PDF cannot export"))
    monkeypatch.setattr("review.report.pdf_renderer.build_review_report_pdf", build)
    old = tmp_path / "final_review.pdf"
    old.write_bytes(b"unvalidated PDF")
    review, outputs = finalize_teaser_review(partial(fixture_review()), tmp_path / "final", report_succeeded=True,
                                            prior_outputs={"report_pdf": str(old)}, presentation="layered")
    assert build.call_count == 0 and "pdf" not in outputs
    assert any(x.component == "pdf_delivery_source" for x in review.delivery_checks)
    assert old.read_bytes() == b"unvalidated PDF" and science(review) == science(fixture_review())


def test_late_package_failure_retains_first_pass_and_corrected_history(tmp_path, monkeypatch):
    from review.recovery import finalize_teaser_review

    calls = []
    monkeypatch.setattr("review.report.pdf_renderer.build_review_report_pdf", renderer(calls, (4,)))
    source = write_review(fixture_review(), tmp_path / "original", presentation="layered")
    saved = FinalReview.model_validate_json(Path(source["json"]).read_text("utf-8"))
    formatter = Mock(side_effect=RuntimeError("Fixture late manifest failed"))
    monkeypatch.setattr("review.report.compact._paths", formatter)
    review, outputs = finalize_teaser_review(partial(saved), tmp_path / "final", report_succeeded=True,
        prior_outputs={"report_" + key: value for key, value in source.items()}, presentation="layered")
    assert formatter.call_count == 1 and len(calls) == 8
    assert "pdf" not in outputs and science(review) == science(saved)
    manifest = json.loads(Path(outputs["manifest"]).read_text("utf-8"))
    history = manifest["interrupted_static_history"]
    assert {"pdf", "bundle_pdf", "history_pdf", "history_bundle_pdf"} <= set(history)
    for item in history.values():
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]
    assert any(x.component == "pdf_delivery_finalization" for x in review.delivery_checks)
    assert "Narrative rendering is unavailable" in Path(outputs["markdown"]).read_text("utf-8")


@pytest.mark.parametrize("payload", [{}, [], None, {"version": "missing-artifacts"}, {"artifacts": []}])
def test_provided_manifest_requires_artifacts_object(tmp_path, monkeypatch, payload):
    from review.recovery import finalize_teaser_review

    calls = []
    monkeypatch.setattr("review.report.pdf_renderer.build_review_report_pdf", renderer(calls))
    source = write_review(fixture_review(), tmp_path / "original", presentation="layered")
    saved = FinalReview.model_validate_json(Path(source["json"]).read_text("utf-8"))
    Path(source["manifest"]).write_text(json.dumps(payload), encoding="utf-8")
    original = {Path(value): Path(value).read_bytes() for value in source.values()}
    review, outputs = finalize_teaser_review(partial(saved), tmp_path / "final", report_succeeded=True,
        prior_outputs={"report_" + key: value for key, value in source.items()}, presentation="layered")
    assert len(calls) == 3 and not {"pdf", "appendix_pdf", "bundle_pdf"} & outputs.keys()
    assert any(item.component == "pdf_delivery_source" for item in review.delivery_checks)
    assert science(review) == science(saved)
    assert all(path.read_bytes() == data for path, data in original.items())


@pytest.mark.parametrize("invalid", ["missing", "role", "hash"])
def test_provided_manifest_entry_never_falls_back_to_legacy(tmp_path, monkeypatch, invalid):
    from review.recovery import finalize_teaser_review

    calls = []
    monkeypatch.setattr("review.report.pdf_renderer.build_review_report_pdf", renderer(calls))
    source = write_review(fixture_review(), tmp_path / "original", presentation="full")
    saved = FinalReview.model_validate_json(Path(source["json"]).read_text("utf-8"))
    entry = {"name": "final_review.pdf", "role": "current_delivery",
             "sha256": hashlib.sha256(Path(source["pdf"]).read_bytes()).hexdigest()}
    if invalid == "role":
        entry["role"] = "initial_pdf_history"
    elif invalid == "hash":
        entry["sha256"] = "0" * 64
    manifest = tmp_path / "original" / "report_manifest.json"
    manifest.write_text(json.dumps({"artifacts": {} if invalid == "missing" else {"pdf": entry}}), encoding="utf-8")
    source["manifest"] = str(manifest)
    original = {Path(value): Path(value).read_bytes() for value in source.values()}
    review, outputs = finalize_teaser_review(partial(saved), tmp_path / "final", report_succeeded=True,
        prior_outputs={"report_" + key: value for key, value in source.items()}, presentation="full")
    assert len(calls) == 1 and "pdf" not in outputs
    assert any(item.component == "pdf.source" for item in review.delivery_checks)
    assert science(review) == science(saved)
    assert all(path.read_bytes() == data for path, data in original.items())


def test_layered_missing_manifest_cannot_use_full_legacy_branch(tmp_path, monkeypatch):
    from review.recovery import finalize_teaser_review

    calls = []
    monkeypatch.setattr("review.report.pdf_renderer.build_review_report_pdf", renderer(calls))
    source = write_review(fixture_review(), tmp_path / "original", presentation="layered")
    saved = FinalReview.model_validate_json(Path(source["json"]).read_text("utf-8"))
    review, outputs = finalize_teaser_review(partial(saved), tmp_path / "final", report_succeeded=True,
        prior_outputs={"report_" + key: value for key, value in source.items() if key != "manifest"}, presentation="layered")
    assert len(calls) == 3 and not {"pdf", "appendix_pdf", "bundle_pdf"} & outputs.keys()
    assert any(item.component == "pdf_delivery_source" for item in review.delivery_checks)
    assert science(review) == science(saved)


def test_whole_report_formatter_failure_is_not_reentered(tiny_inputs, monkeypatch):
    snapshots = []
    def failing(review, **kwargs):
        snapshots.append(science(review))
        raise RuntimeError("Fixture report formatter failed")
    monkeypatch.setattr("review.report.v2.render_markdown", failing)
    monkeypatch.setattr("review.recovery.render_markdown", failing)
    summary, _, _, _ = run_tiny(tiny_inputs, monkeypatch, render_pdf=False)
    review = FinalReview.model_validate_json(Path(summary["outputs"]["report_json"]).read_text("utf-8"))
    teaser = json.loads(Path(summary["outputs"]["teaser_json"]).read_text("utf-8"))
    assert snapshots == [science(review)]
    assert summary["stages"]["report"] == "failed" and summary["stages"]["teaser"] == "ok"
    assert teaser["delivery_checks"] == summary["delivery_checks"] == review.model_dump(mode="json")["delivery_checks"]
    assert "Narrative rendering is unavailable" in Path(summary["outputs"]["report_markdown"]).read_text("utf-8")
