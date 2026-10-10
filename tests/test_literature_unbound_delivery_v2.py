"""Requested operation failures retain delivery scope without inventing claim conditions."""

import json

import pytest

from assessment import assess_claims
from review.delivery import checked_delivery
from schemas.review import DeliveryCheck, FinalReview
from tests.test_literature_service_responsibility_v2 import (
    boundaries,
    comparison,
    inputs,
    paper,
    read_response,
)
from tests.test_literature_service_responsibility_v2 import offline as offline
from verification.contracts import BranchResult
from verification.dispatch import verify_claims
from verification.literature import verify_literature


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["service", "protocol", "identity", "healthy", "abstract"])
async def test_unbound_uncited_read_retains_supported_claim_and_operation_delivery(tmp_path, mode):
    claim, materials = inputs(tmp_path)
    original = claim.model_dump(mode="json")
    extra = "2001.00003"
    searcher, reader, call = boundaries(rows=[comparison("2001.00001", "c1"), comparison()])
    searcher.search.return_value["papers"] = [paper(extra)]

    async def read(items):
        pid = items[0]["id"]
        result = read_response(pid)
        if pid != extra:
            return result
        if mode == "service":
            raise TimeoutError("Unbound requested paper unavailable")
        if mode == "protocol":
            return {"success": True, "items": []}
        if mode == "identity":
            result["items"][0]["paper"]["arxiv_id"] = "2002.00001"
        if mode == "abstract":
            result["items"][0].pop("evidence")
        return result

    reader.read_papers.side_effect = read

    async def branch(current, shared):
        return await verify_literature(current, shared, submission_deadline="2021-01-31",
            searcher=searcher, reader=reader, call=call, output_dir=tmp_path / "audit")

    verification = await verify_claims([claim], materials, tmp_path / "verification", branches={"Literature": branch})
    assert claim.model_dump(mode="json") == original
    assert searcher.search.await_count == 3 and reader.read_papers.await_count == 3 and call.call_count == 1
    assessed = assess_claims(verification.claims)[0]
    assert assessed.status.value == "supported" and not assessed.questions and not assessed.verification_limitations
    audit = json.loads((tmp_path / "audit/claim-search-audit.json").read_text("utf-8"))
    assert audit["novelty_condition_ids"] == []
    review = checked_delivery(FinalReview(paper_key="unbound", run_id="fixture",
        claims=[assessed], delivery_checks=verification.delivery_checks))
    failed = mode in {"service", "protocol", "identity"}
    assert review.run_status == ("partial" if failed else "completed")
    assert review.incomplete_stages == (["verification"] if failed else [])
    assert len(review.delivery_checks) == int(failed)
    if failed:
        item = review.delivery_checks[0]
        assert item.claim_id == claim.id and item.component == "literature.read_papers" and item.state == "failed"
        assert extra in item.reason and "claim-search-audit.json#/reads/" in item.reason
        assert any(event["identifier"] == extra and event["system_limited"] and event["condition_ids"] == []
                   for event in audit["context_events"])


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [{"stage": "report"}, {"claim_id": "foreign"},
                                   {"component": "global_literature.read_papers"}, {"state": "unavailable"},
                                   pytest.param({"responsibility": "author"}, id="foreign-responsibility")])
async def test_claim_operation_delivery_rejects_foreign_scope(tmp_path, change):
    claim, materials = inputs(tmp_path)
    item = DeliveryCheck(stage="verification", component="literature.read_papers", state="failed",
                         claim_id=claim.id, reason="Audit: claim-search-audit.json#/reads/0")
    item = item.model_copy(update=change)
    result = await verify_claims([claim], materials, tmp_path / "verification",
                                branches={"Literature": lambda *_: BranchResult(delivery_checks=[item])})
    assert not result.delivery_checks
    assert result.claims[0].verification_limitations[0].kind == "branch_failed"


@pytest.mark.asyncio
async def test_global_preconstructed_operation_cannot_attribute_system_failure_to_author(tmp_path):
    claim, materials = inputs(tmp_path)
    item = DeliveryCheck(stage="verification", component="global_literature.read_papers", state="failed",
                         reason="Audit: global-search-audit.json#/reads/0")
    item = item.model_copy(update={"responsibility": "author"})
    result = await verify_claims([claim], materials, tmp_path / "verification",
        branches={"Literature": lambda *_: {}}, global_literature=lambda *_: BranchResult(delivery_checks=[item]))
    assert len(result.delivery_checks) == 1
    assert result.delivery_checks[0].responsibility == "system"
    assert result.delivery_checks[0].component == "global_literature"
    assert result.claims[0].verification_limitations == []
