"""Retained concern scope and source audits survive advice and report delivery."""

import hashlib
from pathlib import Path

import pytest

from llm.client import LLMConfig
from review.report.advice import advice_input, generate_advice, theory_source_integrity
from review.report.v2 import _theory_derivations, write_review
from schemas.claim import Claim
from schemas.review import FinalReview
from tests.test_theory_concerns_v2 import assessed, division_case, run_case


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    cfg = LLMConfig("mock", "concern-delivery", None, None)
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: cfg)
    monkeypatch.setattr("review.report.advice.resolve_llm_config", lambda: cfg)

    def blocked(*args, **kwargs):
        pytest.fail("Delivery tests require mocked external boundaries")

    monkeypatch.setattr("screening.checks.llm_json", blocked)
    monkeypatch.setattr("review.report.advice.llm_json", blocked)
    monkeypatch.setattr("requests.sessions.Session.request", blocked)
    monkeypatch.setattr("httpx.Client.send", blocked)
    monkeypatch.setattr("httpx.AsyncClient.send", blocked)
    monkeypatch.setattr("subprocess.run", blocked)


def prepared(tmp_path, disposition="answerable_concern"):
    claim, materials, response, decision = division_case(tmp_path)
    decision["disposition"] = disposition
    branch, _ = run_case(
        tmp_path,
        claim,
        materials,
        response,
        {"schema_version": "theory-concern-v1", "items": [decision]},
    )
    return assessed(claim, branch)


@pytest.mark.parametrize("presentation", ["full", "layered"])
def test_scope_target_trace_and_resolution_visible_without_changing_records(tmp_path, presentation):
    claim = prepared(tmp_path)
    before = claim.model_dump(mode="json")
    review = claim.theory_derivations[0].concern_reviews[0]
    result = write_review(
        FinalReview(paper_key="theory-scope", run_id="offline", claims=[claim]),
        tmp_path / "report",
        render_pdf=False,
        presentation=presentation,
    )
    text = "\n".join(Path(path).read_text("utf-8") for path in result.values() if str(path).endswith(".md"))
    assert "Concern review for c1: validated" in text
    assert "answerable\\_concern" in text and "Target source:" in text
    assert "For all real x, x/x = 1" in text
    assert "Trace steps: s1" in text and "gap indices (zero-based): 0" in text
    assert "Ask whether the intended statement restricts x to nonzero values" in text
    assert "Concern audit:" in text and "concern\\-" in text
    assert review.target_sources and claim.model_dump(mode="json") == before


def test_empty_review_list_preserves_historical_trace_display(tmp_path):
    claim = prepared(tmp_path)
    claim.theory_derivations[0].concern_reviews = []
    original = claim.model_dump(mode="json")
    for record in original["theory_derivations"]:
        record.pop("concern_reviews")
    legacy = Claim.model_validate(original)
    assert _theory_derivations(claim) == _theory_derivations(legacy)
    assert "Concern review" not in "\n".join(_theory_derivations(legacy))


def test_concern_audit_fragment_resolves_to_hashed_local_file(tmp_path):
    claim = prepared(tmp_path)
    review = claim.theory_derivations[0].concern_reviews[0]
    path = Path(review.audit_pointer.split("#", 1)[0]).resolve()
    data = advice_input(claim, [])
    assert data["source_files"][str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert all("#/reviews/" not in p for p in data["source_files"])


@pytest.mark.parametrize("origin", ["review_hash", "target_source"])
def test_review_specific_original_source_hash_is_revalidated(tmp_path, origin):
    claim = prepared(tmp_path)
    review = claim.theory_derivations[0].concern_reviews[0]
    source = tmp_path / "scope-target.md"
    source.write_text("original condition source", encoding="utf-8")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    if origin == "review_hash":
        review.source_hashes[str(source)] = expected
    else:
        review.target_sources[0].locator = str(source)
        review.target_sources[0].artifact_sha256 = expected
    assert str(source.resolve()) in theory_source_integrity(claim)
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Theory source artifact changed"):
        theory_source_integrity(claim)


@pytest.mark.parametrize("mutation", ["unchanged", "bytes", "delete"])
def test_scope_audit_change_during_advice_cannot_publish_stale_recommendation(tmp_path, mutation):
    claim = prepared(tmp_path)
    audit = Path(claim.theory_derivations[0].concern_reviews[0].audit_pointer.split("#", 1)[0])
    status_before = claim.status

    def model(**kwargs):
        if mutation == "bytes":
            audit.write_text(audit.read_text("utf-8") + "\n", encoding="utf-8")
        elif mutation == "delete":
            audit.unlink()
        return {
            "status": "ok",
            "claim_id": claim.id,
            "items": [
                {
                    "condition_ids": ["c1"],
                    "basis_refs": ["/evidence/0"],
                    "action": "author_question",
                    "text": "Does the claimed domain exclude zero?",
                }
            ],
        }

    result = generate_advice(
        FinalReview(paper_key="theory-scope", run_id="offline", claims=[claim]),
        tmp_path / "advice",
        call=model,
    )
    output = result.review.claims[0]
    assert output.status == status_before
    if mutation == "unchanged":
        assert output.advice.state == "generated" and len(output.advice.items) == 1
    else:
        assert output.advice.state == "unavailable" and not output.advice.items
        assert "changed" in output.advice.failure_reason.lower()
