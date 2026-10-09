"""Service outages stay unavailable, with concise diagnostics and healthy neighbors."""

import copy
import json
from pathlib import Path

import pytest

from llm.client import LLMConfig
from review.report import advice
from schemas.review import FinalReview
from tests.test_advice_request_v2 import claim


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("This test must mock all external boundaries")

    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr(advice, "llm_json", forbidden)
    monkeypatch.setattr(
        advice,
        "resolve_llm_config",
        lambda: LLMConfig("mock", "fixture", "https://fixture.invalid", "secret-fixture-key"),
    )


@pytest.mark.parametrize("message", ["ReadTimeout: service timed out", "SSLError: TLS UNEXPECTED_EOF"])
def test_provider_error_diagnostic_keeps_failure_and_continues_neighbor(tmp_path, message):
    review = FinalReview(
        paper_key="fixture",
        run_id="mock",
        claims=[claim(), claim().model_copy(update={"id": "healthy"})],
    )
    original = review.model_dump(mode="json")
    failure = {"status": "error", "error": message + " secret-fixture-key", "provider": "mock"}
    failure_before = copy.deepcopy(failure)
    calls = []

    def call(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return failure
        return {
            "status": "ok",
            "claim_id": "healthy",
            "items": [
                {
                    "text": "Please provide evidence.",
                    "condition_ids": ["a"],
                    "basis_refs": ["/coverage_gaps/a"],
                }
            ],
        }

    result = advice.generate_advice(review, tmp_path, call=call)
    assert len(calls) == 2 and result.counts == {"generated": 1, "unavailable": 1}
    failed, healthy = result.review.claims
    assert failed.advice.items == [] and healthy.advice.items
    assert message in failed.advice.failure_reason
    assert "Advice provider request failed" in failed.advice.failure_reason
    assert "validation errors" not in failed.advice.failure_reason
    audit = json.loads(Path(failed.advice.audit_pointer).read_text("utf-8"))
    assert audit["failure_kind"] == "provider_error" and audit["attempted"] is True
    assert audit["response"]["error"] == message + " [redacted]"
    assert "secret-fixture-key" not in json.dumps(audit)
    assert "secret-fixture-key" not in result.review.model_dump_json()
    assert failure == failure_before and review.model_dump(mode="json") == original


@pytest.mark.parametrize(
    "raw", [{"status": "error"}, {"status": "error", "error": []}, {"status": "ok", "items": []}]
)
def test_malformed_output_still_uses_original_schema_validation(tmp_path, raw):
    result = advice.generate_advice(
        FinalReview(paper_key="fixture", run_id="mock", claims=[claim()]), tmp_path, call=lambda **kw: raw
    )
    checked = result.review.claims[0].advice
    audit = json.loads(Path(checked.audit_pointer).read_text("utf-8"))
    assert checked.state == "unavailable" and checked.items == []
    assert "ValidationError" in checked.failure_reason
    assert "failure_kind" not in audit


def test_source_mutation_takes_precedence_over_service_failure(tmp_path):
    review = FinalReview(paper_key="fixture", run_id="mock", claims=[claim()])

    def call(**kwargs):
        review.claims[0].text = "Mutated caller"
        return {"status": "error", "error": "Injected service failure"}

    result = advice.generate_advice(review, tmp_path, call=call)
    checked = result.review.claims[0].advice
    audit = json.loads(Path(checked.audit_pointer).read_text("utf-8"))
    assert checked.state == "unavailable" and checked.items == []
    assert "Caller assessment changed" in checked.failure_reason
    assert "failure_kind" not in audit
