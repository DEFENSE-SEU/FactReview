"""One public producer-contract control; all scientific choices are explicit mocks."""
import copy
import json

import pytest

from screening import claim_coverage as m
from tests.test_claim_coverage_v2 import CFG, validation
from tests.test_claim_coverage_v3 import case, review
from tests.test_claim_coverage_wire_v5 import compact_response, unpack


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Output contract control forbids external services and processes")
    monkeypatch.setattr(m, "resolve_llm_config", lambda: CFG)
    monkeypatch.setattr(m, "llm_json", forbidden)
    for name in ("requests.sessions.Session.request", "httpx.Client.send",
                 "httpx.AsyncClient.send", "subprocess.run", "subprocess.Popen"):
        monkeypatch.setattr(name, forbidden)


def test_public_original_review_contract_covers_every_required_identity(tmp_path):
    paper, original = case()
    # Split existing mock material into two explicitly supplied original claims.
    # No extraction or scientific-quality result is inferred from this fixture.
    claims = [original[0].model_copy(deep=True, update={"id": "claim_001",
              "text": "The method reaches accuracy 90.", "conditions": [original[0].conditions[0]]}),
              original[0].model_copy(deep=True, update={"id": "claim_002",
              "text": "The method uses batch size 16.", "conditions": [original[0].conditions[1]]})]
    before = paper.model_dump(mode="json"), [c.model_dump(mode="json") for c in claims]
    calls = []
    def call(**kw):
        shown, encoded = kw["prompt"].removeprefix("OUTPUT_SCHEMA:\n").split("\nDATA_JSON:\n", 1)
        schema, packed = json.loads(shown), json.loads(encoded)
        payload = unpack(packed)
        if kw["module"] == "screening.claims.coverage":
            parts = [review({**payload, "current_claims": [c]}) for c in payload["current_claims"]]
            raw = copy.deepcopy(parts[0])
            raw["claim_reviews"] = [r for part in parts for r in part["claim_reviews"]]
            response = compact_response(raw)
        else:
            assert kw["module"] == "screening.claims.coverage_validation"
            response = compact_response(validation(payload))
        calls.append({"module": kw["module"], "schema": schema, "payload": packed, "response": response})
        return response
    result = m.review_claim_coverage(paper, claims, call=call, output_dir=tmp_path / "review")
    assert [r["module"] for r in calls] == ["screening.claims.coverage", "screening.claims.coverage_validation"]
    third = calls[1]
    required = third["payload"]["required_original_claim_reviews"]
    assert len(required) == 2
    contract = third["schema"]["properties"]["original_claim_reviews"]
    assert contract.get("minItems") == contract.get("maxItems") == len(required)
    positions = contract["prefixItems"]
    assert len(positions) == len(required)
    for position, required_row in zip(positions, required, strict=True):
        assert position["allOf"][0] == contract["items"]
        bound = position["allOf"][1]["properties"]
        assert bound["claim_id"]["const"] == required_row["claim_id"]
        assert bound["claim_digest"]["const"] == required_row["digest"]
    for key in ("context_id", "window_id"):
        assert third["schema"]["properties"][key]["const"] == third["payload"][key]
    assert "const" not in calls[0]["schema"]["properties"]["context_id"]
    assert result.coverage["status"] == "complete" and not result.coverage["revisions"]
    assert result.coverage["original_claim_reviews_completed"] == result.coverage["original_claim_reviews_required"] == 2
    assert result.coverage["budget"]["validation_calls"] == 1
    audit = json.loads((tmp_path / "review" / "coverage.json").read_text(encoding="utf-8"))
    assert audit["attempts"][1]["input"] == third["payload"]
    assert before == (paper.model_dump(mode="json"), [c.model_dump(mode="json") for c in claims])
    (tmp_path / "captured.json").write_text(json.dumps(third, ensure_ascii=False), encoding="utf-8")
