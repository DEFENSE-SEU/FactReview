"""Public first-pass metadata contract; scientific judgments are explicit mocks."""
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
        pytest.fail("First-pass contract control forbids external services and processes")
    monkeypatch.setattr(m, "resolve_llm_config", lambda: CFG)
    monkeypatch.setattr(m, "llm_json", forbidden)
    for name in ("requests.sessions.Session.request", "httpx.Client.send",
                 "httpx.AsyncClient.send", "subprocess.run", "subprocess.Popen"):
        monkeypatch.setattr(name, forbidden)


def test_public_first_pass_required_identity_and_missing_rows_stay_partial(tmp_path):
    for mode in ("complete", "short"):
        paper, original = case()
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
                if mode == "short":
                    raw["claim_reviews"] = raw["claim_reviews"][:1]
                response = compact_response(raw)
            else:
                assert kw["module"] == "screening.claims.coverage_validation"
                response = compact_response(validation(payload))
            calls.append({"module": kw["module"], "schema": schema, "payload": packed, "response": copy.deepcopy(response)})
            return response
        output = tmp_path / mode
        result = m.review_claim_coverage(paper, claims, call=call, output_dir=output)
        assert [r["module"] for r in calls] == ["screening.claims.coverage", "screening.claims.coverage_validation"]
        first = calls[0]
        required = first["payload"]["required_claim_checks"]
        assert len(required) == 2
        contract = first["schema"]["properties"]["claim_reviews"]
        assert contract.get("minItems") == contract.get("maxItems") == len(required)
        assert len(contract["prefixItems"]) == len(required)
        for position, row in zip(contract["prefixItems"], required, strict=True):
            assert position["allOf"][0] == contract["items"]
            assert position["allOf"][1]["properties"]["claim_id"]["const"] == row["claim_id"]
            assert position["allOf"][1]["properties"]["claim_digest"]["const"] == row["digest"]
        for key in ("context_id", "window_id"):
            assert first["schema"]["properties"][key]["const"] == first["payload"][key]
        baseline = m.CoverageReviewV5.model_json_schema()
        assert first["schema"]["$defs"] == baseline["$defs"]
        assert contract["items"] == baseline["properties"]["claim_reviews"]["items"]
        assert before == (paper.model_dump(mode="json"), [c.model_dump(mode="json") for c in claims])
        assert not result.coverage["revisions"]
        audit = json.loads((output / "coverage.json").read_text(encoding="utf-8"))
        assert audit["attempts"][0]["input"] == first["payload"]
        assert audit["attempts"][0]["response"] == first["response"]
        assert len(first["response"]["claim_reviews"]) == (2 if mode == "complete" else 1)
        assert result.coverage["original_claim_reviews_completed"] == result.coverage["original_claim_reviews_required"] == 2
        assert result.coverage["claim_checks_completed"] == (2 if mode == "complete" else 1)
        assert result.coverage["status"] == ("complete" if mode == "complete" else "partial")
        assert result.coverage["claim_checks_unreviewed"] == (0 if mode == "complete" else 1)
        (output / "captured-first.json").write_text(json.dumps(first, ensure_ascii=False), encoding="utf-8")
