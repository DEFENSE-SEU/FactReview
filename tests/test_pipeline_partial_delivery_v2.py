"""A failed dependency stays failed while independent completed work is delivered."""

import json
from pathlib import Path

import pipeline_v2
from screening.checks import paper_finding
from tests import test_pipeline_v2 as pipeline_fixtures
from tests.test_pipeline_v2 import ModelBoundary, run_tiny

offline_boundaries = pipeline_fixtures.offline_boundaries
tiny_inputs = pipeline_fixtures.tiny_inputs


def _review(summary):
    return json.loads(Path(summary["outputs"]["report_json"]).read_text(encoding="utf-8"))


def test_extraction_failure_delivers_independent_screening_without_downstream_services(
    tiny_inputs, monkeypatch
):
    boundary = ModelBoundary()

    def call(**kwargs):
        if kwargs["module"] == "screening.claims":
            return {"status": "error", "error": "extraction unavailable"}
        return boundary(**kwargs)

    monkeypatch.setattr(
        "screening.stage.check_writing",
        lambda materials, **kwargs: [
            paper_finding(
                materials,
                materials.blocks[0],
                quote=materials.blocks[0].text,
                kind="writing",
                level="clarity_issue",
                text="Preserved writing observation",
            )
        ],
    )
    summary, _, retrieval, runner = run_tiny(tiny_inputs, monkeypatch, call=call, render_pdf=False)
    assert summary["stages"]["screening"] == "failed"
    assert all(summary["stages"][name] == "skipped" for name in ("verification", "execution", "assessment"))
    assert summary["stages"]["report"] == summary["stages"]["teaser"] == "ok"
    assert not retrieval.queries and runner.call_count == 0
    screening = json.loads(Path(summary["outputs"]["screening"]).read_text(encoding="utf-8"))
    assert screening["claim_extraction_status"] == "failed"
    assert screening["figure_coverage"]["checked"] == 1
    review = _review(summary)
    assert review["run_status"] == "partial" and review["incomplete_stages"] == ["screening"]
    assert review["claims"] == []
    assert any(item["text"] == "Preserved writing observation" for item in review["findings"])
    assert "Partial review" in review["review_markdown"]
    teaser = json.loads(Path(summary["outputs"]["teaser_json"]).read_text(encoding="utf-8"))
    assert teaser["run_status"] == "partial"
    assert "Partial review" in Path(summary["outputs"]["teaser_image"]).read_text(encoding="utf-8")
    assert summary["run_stats"]["modules"]["analysis"]["status"] == "failed"


def test_execution_exception_preserves_verified_snapshot_and_discards_callback_mutations(
    tiny_inputs, monkeypatch
):
    captured = []

    def fail_execution(plans, claims, *args, **kwargs):
        captured.extend(claim.model_dump(mode="json") for claim in claims)
        claims[0].notes.append("uncommitted callback mutation")
        claims[0].evidence.clear()
        raise RuntimeError("execution orchestrator failed")

    monkeypatch.setattr(pipeline_v2, "execute_plans", fail_execution)
    summary, _, _, runner = run_tiny(tiny_inputs, monkeypatch, render_pdf=False)
    assert summary["stages"]["execution"] == "failed"
    assert summary["stages"]["assessment"] == summary["stages"]["report"] == "ok"
    review = _review(summary)
    assert review["run_status"] == "partial" and review["incomplete_stages"] == ["execution"]
    assert len(review["claims"]) == len(captured) == 4
    for before in captured:
        after = next(c for c in review["claims"] if c["id"] == before["id"])
        assert after["evidence"] == before["evidence"]
        assert "uncommitted callback mutation" not in after["notes"]
    assert "execution orchestrator failed" in str(review)
    assert not review["ledger"] and runner.call_count == 0
    assert "outcomes and cleanup state are unknown" in review["review_markdown"]
    assert "Execution was not run" not in review["review_markdown"]
    assert summary["run_stats"]["modules"]["execution"]["status"] == "failed"


def test_verification_exception_keeps_extracted_claims_with_explicit_failure(tiny_inputs, monkeypatch):
    async def fail_verification(claims, *args, **kwargs):
        claims[0].text = "uncommitted mutation"
        raise RuntimeError("verification orchestrator failed")

    monkeypatch.setattr(pipeline_v2, "verify_claims", fail_verification)
    summary, _, retrieval, runner = run_tiny(tiny_inputs, monkeypatch, render_pdf=False)
    assert summary["stages"]["verification"] == "failed"
    assert summary["stages"]["execution"] == "skipped"
    review = _review(summary)
    assert review["run_status"] == "partial" and review["incomplete_stages"] == ["verification"]
    assert len(review["claims"]) == 4
    assert all(c["status"] == "unverified" and not c["evidence"] for c in review["claims"])
    assert "uncommitted mutation" not in str(review)
    assert "verification orchestrator failed" in str(review)
    assert not retrieval.queries and runner.call_count == 0
    assert summary["run_stats"]["modules"]["analysis"]["status"] == "failed"


def test_invalid_execution_configuration_still_delivers_l1_and_l2(tiny_inputs, monkeypatch):
    tiny_inputs[0].max_attempts = 99
    summary, _, _, runner = run_tiny(tiny_inputs, monkeypatch, render_pdf=False)
    assert summary["stages"]["execution"] == "failed"
    assert _review(summary)["run_status"] == "partial"
    assert len(_review(summary)["claims"]) == 4
    assert runner.call_count == 0


def test_complete_host_ledger_survives_late_stage_failure_without_restoring_execution_evidence(
    tiny_inputs, monkeypatch
):
    execute = pipeline_v2.execute_plans

    def fail_after_saved_ledger(*args, **kwargs):
        result = execute(*args, **kwargs)
        assert result.ledger and any(e.source == "execution" for c in result.claims for e in c.evidence)
        raise RuntimeError("failure after saving host ledger")

    monkeypatch.setattr(pipeline_v2, "execute_plans", fail_after_saved_ledger)
    summary, _, _, runner = run_tiny(tiny_inputs, monkeypatch, render_pdf=False)
    review = _review(summary)
    assert summary["stages"]["execution"] == "failed" and runner.call_count == 1
    assert review["run_status"] == "partial" and len(review["ledger"]) == 1
    assert review["ledger"][0]["attempts"]
    assert all(e["source"] != "execution" for c in review["claims"] for e in c["evidence"])
    assert "audit only" in review["review_markdown"]
    assert "Execution was not run" not in review["review_markdown"]


def test_historical_materials_without_table_crops_report_unavailable_visual_coverage(
    tiny_inputs, monkeypatch
):
    parse = pipeline_v2.parse_materials

    async def old_materials(**kwargs):
        materials = await parse(**kwargs)
        assert materials.tables
        materials.tables.clear()
        return materials

    monkeypatch.setattr(pipeline_v2, "parse_materials", old_materials)
    summary, model, _, _ = run_tiny(tiny_inputs, monkeypatch, render_pdf=False)
    assert summary["table_coverage"] == {"total": 1, "checked": 0, "failed": 0, "unavailable": 1}
    assert "screening_tables.visual" not in model.calls
    assert "rebuild shared materials" in _review(summary)["review_markdown"]
