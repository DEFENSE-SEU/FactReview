"""Fresh Theory positives require the requested trace; historical records stay readable."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from assessment import assess_claim
from llm.client import LLMConfig
from schemas.claim import Claim, ClaimLocation, Condition, TheoryDerivationRecord
from schemas.materials import MaterialBlock, SharedMaterials
from screening import checks
from tests.test_theory_derivations_v2 import paper
from verification.theory import verify_theory


def test_fresh_legacy_theory_is_unverified_and_complete_ordinary_trace_still_supports(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("External network, retrieval, Docker and subprocess calls are forbidden")

    for boundary in ("httpx.Client.send", "httpx.AsyncClient.send",
                     "requests.sessions.Session.request", "subprocess.run", "subprocess.Popen"):
        monkeypatch.setattr(boundary, forbidden)
    monkeypatch.setattr(checks, "resolve_llm_config", lambda: LLMConfig("mock", "trace-control", None, None))
    text = "For every input x, the error E(x) = 0."
    path = tmp_path / "assertion.md"
    path.write_text(text, encoding="utf-8")
    loc = ClaimLocation(page=1, section="Theory", char_start=0, char_end=len(text))
    claim = Claim(id="c001", text=text, loc=loc, source_block_id="main:1", source_quote=text,
                  conditions=[Condition(id="c001.k1", description="For every input x")], needs=["Theory"])
    materials = SharedMaterials(paper_key="theory-legacy-control", source_pdf=str(tmp_path / "assertion.pdf"),
                               markdown=text, markdown_path=str(path), content_list_path="", provider="fixture",
                               blocks=[MaterialBlock(id="main:1", text=text, loc=loc)])
    legacy = {"items": [{"block_id": "main:1", "quote": text, "covered": ["c001.k1"],
                         "fully_supported_conditions": ["c001.k1"], "kind": "derivation", "direction": "support",
                         "detail": "The equation matches the claimed zero-error condition.",
                         "step_quote": "E(x) = 0", "main_block_id": None}], "appendix_block_ids": []}
    positive_dir = tmp_path / "ordinary"
    positive_dir.mkdir()
    ordinary, ordinary_materials, complete = paper(positive_dir)
    original = (claim.model_dump(mode="json"), materials.model_dump(mode="json"), copy.deepcopy(legacy))
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    responses = [legacy, complete]
    calls = []

    def model(**kwargs):
        payload = json.loads(kwargs["prompt"])
        calls.append(payload)
        assert kwargs["module"] == "verification.theory"
        assert payload["output_schema"]["properties"]["schema_version"]["const"] == "theory-derivation-v1"
        return copy.deepcopy(responses[len(calls) - 1])

    monkeypatch.setattr(checks, "llm_json", model)
    result = verify_theory(claim, materials, output_dir=tmp_path / "legacy-audit", visual_recheck_rounds=0)
    assessed = assess_claim(claim.model_copy(update={"evidence": result.evidence}, deep=True))
    assert result.evidence and all(not evidence.sufficient for evidence in result.evidence)
    assert assessed.status == "unverified"
    assert result.evidence[0].pointer.quote == text and result.evidence[0].covered == ["c001.k1"]
    record = result.theory_derivations[0]
    assert record.state == "legacy_unavailable" and record.schema_version is None and record.trace is None
    assert record.transport == "live" and record.adopted
    assert "legacy_trace_unavailable" in record.issues[0]
    assert any("versioned derivation trace" in issue for issue in result.issues)
    assert "versioned derivation trace" in result.evidence[0].note
    retained = json.loads(Path(record.audit_pointer).read_text(encoding="utf-8"))
    assert retained["response"] == legacy and retained["status"] == "returned"
    assert retained["request"]["claim"] == original[0]
    assert TheoryDerivationRecord.model_validate(record.model_dump(mode="json")) == record
    assert (claim.model_dump(mode="json"), materials.model_dump(mode="json"), legacy) == original
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hash

    supported = verify_theory(ordinary, ordinary_materials, output_dir=tmp_path / "complete-audit",
                              visual_recheck_rounds=0)
    positive = assess_claim(ordinary.model_copy(update={"evidence": supported.evidence}, deep=True))
    assert positive.status == "supported" and supported.evidence[0].sufficient
    trace = supported.theory_derivations[0]
    assert trace.schema_version == "theory-derivation-v1" and trace.state == "validated"
    assert trace.trace.outcome == "completed" and len(trace.trace.steps) == 2
    assert json.loads(Path(trace.audit_pointer).read_text(encoding="utf-8"))["response"] == complete
    assert len(calls) == 2 and ordinary.evidence == []
    (tmp_path / "public-theory-guard-result.json").write_text(json.dumps({
        "legacy_status": assessed.status, "legacy_sufficient": [e.sufficient for e in result.evidence],
        "legacy_trace_state": record.state, "complete_status": positive.status,
        "complete_trace_steps": len(trace.trace.steps), "external_mock_calls": len(calls), "real_external_calls": 0,
    }, indent=2) + "\n", encoding="utf-8")
