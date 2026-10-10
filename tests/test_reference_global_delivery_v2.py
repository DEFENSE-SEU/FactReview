"""Explicit operation failures survive reference/global Literature recovery."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from schemas.review import DeliveryCheck
from screening.references import check_bibliography
from screening.stage import screen_paper
from tests.test_dispatch_v2 import claim, materials
from tests.test_literature_service_responsibility_v2 import offline as offline
from tests.test_reference_pdf_validation import decision
from tests.test_reference_pdf_validation import reference_input as reference_input
from verification.contracts import BranchResult
from verification.dispatch import verify_claims
from verification.literature import verify_literature


@pytest.mark.parametrize("mode", ["failed", "protocol", "exception"])
def test_reference_checker_records_one_structured_failure(reference_input, tmp_path, mode):
    material, _ = reference_input
    sink = []

    def checker(**kwargs):
        if mode == "exception":
            raise TimeoutError("private-fixture-token")
        return {"ok": False, "error_message": "Unavailable"} if mode == "failed" else {"ok": True}

    if mode == "exception":
        with pytest.raises(TimeoutError):
            check_bibliography(material, tmp_path / "reference", checker=checker, delivery_checks=sink)
    else:
        finding, issues = check_bibliography(material, tmp_path / "reference", checker=checker, delivery_checks=sink)
        assert not finding and issues
    assert len(sink) == 1
    assert sink[0].stage == "screening" and sink[0].component == "reference_checker"
    assert sink[0].claim_id is None and sink[0].responsibility == "system"
    assert str(tmp_path / "reference") in sink[0].reason and "private-fixture-token" not in sink[0].reason


@pytest.mark.parametrize("mode", ["unavailable", "failed", "parser_artifact", "uncertain", "identity"])
def test_reference_pdf_operation_and_scientific_uncertainty_are_distinct(reference_input, tmp_path, mode):
    material, backend = reference_input
    if mode == "unavailable":
        material.pages = []
    if mode == "identity":
        backend["issues"][0]["verified_url"] = "https://doi.org/10.9999/other"

    def model(**kwargs):
        if mode == "failed":
            return {"status": "error", "error": "Fixture failure"}
        return {"results": [decision(classification=mode if mode in {"parser_artifact", "uncertain"} else "manuscript_error")]}

    sink = []
    findings, issues = check_bibliography(material, tmp_path / "reference", checker=lambda **kwargs: backend,
                                        call=model, delivery_checks=sink)
    assert not findings and issues
    assert len(sink) == (1 if mode in {"unavailable", "failed"} else 0)
    if sink:
        assert sink[0].component == "reference_pdf" and sink[0].state == mode
        assert "reference_validation.json" in sink[0].reason


def test_empty_bibliography_is_unrequested_and_sink_is_not_shared(reference_input, tmp_path):
    material, _ = reference_input
    material.bibliography = []
    sink = []
    checker = Mock(side_effect=AssertionError("Empty bibliography cannot call checker"))
    assert check_bibliography(material, tmp_path, checker=checker, delivery_checks=sink)[0] == []
    assert not sink and not checker.called


@pytest.mark.parametrize("raises", [False, True])
def test_screening_preserves_reference_failure_once_and_continues(reference_input, tmp_path, monkeypatch, raises):
    material, _ = reference_input
    monkeypatch.setattr("screening.stage.extract_claims", lambda *args, **kwargs: [])
    monkeypatch.setattr("screening.stage.review_claim_coverage", lambda *args, **kwargs: SimpleNamespace(
        claims=[], coverage={"status": "complete"}, blocked_claim_ids=[], issues=[],
    ))
    monkeypatch.setattr("screening.stage.check_writing", lambda *args, **kwargs: [])
    monkeypatch.setattr("screening.stage.check_tables", lambda *args, **kwargs: [])
    monkeypatch.setattr("screening.stage.check_figures", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr("screening.stage.check_visual_tables", lambda *args, **kwargs: ([], []))

    def checker(**kwargs):
        if raises:
            raise TimeoutError("Fixture unavailable")
        return {"ok": False, "error_message": "Fixture unavailable"}

    result = screen_paper(material, tmp_path / "screening", reference_checker=checker)
    assert result.claim_extraction_status == "ok" and len(result.delivery_checks) == 1
    assert result.delivery_checks[0].component == "reference_checker"
    saved = json.loads((tmp_path / "screening/screening.json").read_text("utf-8"))
    assert saved["delivery_checks"] == [c.model_dump(mode="json") for c in result.delivery_checks]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["search", "protocol", "read", "comparison", "bounded_scope"])
async def test_global_literature_explicit_failures_keep_audit_and_redaction(tmp_path, mode):
    material = materials(tmp_path)
    material.title = "Graph neural networks for link prediction"
    material.abstract = "Graph neural networks and relation embedding."
    token = "private-fixture-token"
    paper = {"id": "2001.00001", "arxiv_id": "2001.00001", "title": "Prior composition", "published": "2020-01-01"}
    search = {"success": True, "provider": "fixture", "complete": mode != "bounded_scope", "papers": [paper] if mode in {"read", "comparison"} else []}
    if mode == "search":
        search = {"success": False, "error": token, "papers": []}
    elif mode == "protocol":
        search = []
    searcher = SimpleNamespace(search=AsyncMock(return_value=search), lookup_metadata=None,
                              search_cfg=SimpleNamespace(provider="fixture", api_key=token, base_url=None))
    reader = SimpleNamespace(read_cfg=SimpleNamespace(provider="fixture", api_key=token, base_url=None),
                             read_papers=AsyncMock(return_value={"success": True, "items": [{"id": paper["id"], "success": True, "paper": paper, "evidence": [{"text": "A prior graph composition.", "page": 2}]}]}))
    if mode == "read":
        reader.read_papers.side_effect = TimeoutError(token)
    model = Mock(return_value={"status": "ok", "comparisons": []} if mode != "comparison" else {})
    result = await verify_literature(None, material, submission_deadline="2021-01-01", searcher=searcher,
                                     reader=reader, call=model, output_dir=tmp_path / "audit")
    assert not result.verification_limitations
    checks = result.delivery_checks
    assert bool(checks) is (mode != "bounded_scope")
    expected = {"search": "search", "protocol": "search", "read": "read_papers", "comparison": "comparison"}
    if checks:
        assert all(c.stage == "verification" and c.claim_id is None for c in checks)
        assert all(c.component == "global_literature." + expected[mode] for c in checks)
        assert all("global-search-audit.json#/" in c.reason and token not in c.reason for c in checks)
    audit = json.loads((tmp_path / "audit/global-search-audit.json").read_text("utf-8"))
    assert token not in json.dumps(audit)
    assert bool([e for e in audit["context_events"] if e["system_limited"]]) is bool(checks)


@pytest.mark.asyncio
async def test_dispatch_propagates_global_checks_and_rejects_foreign_scopes(tmp_path):
    item = DeliveryCheck(stage="verification", component="global_literature.search", state="failed", reason="Audit: global-search-audit.json#/queries/0")
    c = claim(["Code"])
    result = await verify_claims([c], materials(tmp_path), tmp_path / "valid", branches={"Code": lambda *args: {}},
                                global_literature=lambda *args: BranchResult(delivery_checks=[item]))
    assert result.delivery_checks == [item] and not result.claims[0].verification_limitations
    for index, change in enumerate(({"stage": "report"}, {"claim_id": "foreign"}, {"component": "other"})):
        foreign = item.model_copy(update=change)
        result = await verify_claims([c], materials(tmp_path), tmp_path / f"foreign-{index}",
                                    branches={"Code": lambda *args: {}}, global_literature=lambda *args: BranchResult(delivery_checks=[foreign]))
        assert len(result.delivery_checks) == 1 and result.delivery_checks[0].component == "global_literature"
        assert result.delivery_checks[0].stage == "verification" and result.delivery_checks[0].claim_id is None
    result = await verify_claims([c], materials(tmp_path), tmp_path / "branch", branches={"Code": lambda *args: BranchResult(delivery_checks=[item])})
    assert not result.delivery_checks and result.claims[0].verification_limitations[0].kind == "branch_failed"


@pytest.mark.asyncio
async def test_global_exception_is_structured_but_free_text_does_not_decide(tmp_path):
    def failed(*args):
        raise TimeoutError("private-fixture-token")

    result = await verify_claims([], materials(tmp_path), tmp_path / "failed", branches={}, global_literature=failed)
    assert len(result.delivery_checks) == 1 and "private-fixture-token" not in result.delivery_checks[0].reason
    assert "verification.json" in result.delivery_checks[0].reason
    result = await verify_claims([], materials(tmp_path), tmp_path / "healthy", branches={},
                                global_literature=lambda *args: {"issues": ["Global literature search failed: arbitrary scientific text"]})
    assert not result.delivery_checks
