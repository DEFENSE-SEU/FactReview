"""Source repair retains every claim and the exact original-grounding contract."""

import json

import pytest

from common import run_stats
from llm.client import LLMConfig
from schemas.claim import ClaimLocation
from schemas.materials import MaterialBlock, SharedMaterials
from screening.claims import ClaimExtractionError, extract_claims


@pytest.fixture
def paper(tmp_path, monkeypatch):
    monkeypatch.setattr("screening.claims.resolve_llm_config", lambda: LLMConfig("mock", "test", None, None))
    text = "The result improves.\n<table><tr><td>Method</td><td>0.75</td></tr></table>"
    path = tmp_path / "paper.md"
    path.write_text(text, encoding="utf-8")
    return SharedMaterials(
        paper_key="table",
        source_pdf="paper.pdf",
        markdown=text,
        markdown_path=str(path),
        content_list_path="content.json",
        provider="fixture",
        blocks=[
            MaterialBlock(id="b1", text=text, loc=ClaimLocation(page=1, char_start=0, char_end=len(text)))
        ],
    )


def candidate(quote, text="The reported score is 0.75."):
    return {
        "text": text,
        "source_block_id": "b1",
        "source_quote": quote,
        "conditions": [{"id": "score", "description": "Reported score"}],
        "needs": ["Experiments"],
        "importance": "core",
    }


def test_repairs_html_quote_without_changing_claim_or_discarding_valid_candidate(paper, tmp_path):
    table = "<tr><td>Method</td><td>0.75</td></tr>"
    raw = {
        "status": "ok",
        "claims": [candidate("Method 0.75"), candidate("The result improves.", "The result improves.")],
    }
    modules = []

    def model(**request):
        modules.append(request["module"])
        if request["module"] == "screening.claims":
            return raw
        context = json.loads(request["prompt"])
        assert [row["index"] for row in context["candidates"]] == [1]
        assert context["candidates"][0]["claim"]["text"] == raw["claims"][0]["text"]
        assert table in context["blocks"][0]["text"]
        return {"repairs": [{"index": 1, "source_block_id": "b1", "source_quote": table}]}

    with run_stats.run_scope(tmp_path / "run_stats.json"):
        claims = extract_claims(paper, call=model, max_source_repairs=3)
    assert modules == ["screening.claims", "screening.claims.source_repair"]
    assert len(claims) == 2 and [c.text for c in claims] == [row["text"] for row in raw["claims"]]
    assert claims[0].source_quote == table
    assert paper.markdown[claims[0].loc.char_start : claims[0].loc.char_end] == table
    assert raw["claims"][0]["source_quote"] == "Method 0.75"
    audit = json.loads(next((tmp_path / "claim_extraction").glob("*.json")).read_text(encoding="utf-8"))
    assert audit["status"] == "ok" and len(audit["attempts"]) == 2
    assert audit["attempts"][0]["source_errors"] and audit["attempts"][1]["source_errors"] == []


@pytest.mark.parametrize(
    "repairs",
    [
        [],
        [{"index": 2, "source_block_id": "b1", "source_quote": "The result improves."}],
        [{"index": 1, "source_block_id": "b1", "source_quote": "The result improves."}] * 2,
    ],
)
def test_repair_cannot_drop_duplicate_or_replace_candidate_identity(paper, repairs):
    def model(**request):
        if request["module"] == "screening.claims":
            return {"status": "ok", "claims": [candidate("Absent quote")]}
        return {"repairs": repairs}

    with pytest.raises(ClaimExtractionError, match="exactly once"):
        extract_claims(paper, call=model, max_source_repairs=3)


def test_failed_repairs_are_capped_and_original_error_and_responses_survive(paper, tmp_path):
    modules = []

    def model(**request):
        modules.append(request["module"])
        if request["module"] == "screening.claims":
            return {"status": "ok", "claims": [candidate("Method 0.75")]}
        return {"repairs": [{"index": 1, "source_block_id": "b1", "source_quote": "Method 0.75"}]}

    with (
        run_stats.run_scope(tmp_path / "run_stats.json"),
        pytest.raises(ClaimExtractionError, match="does not occur"),
    ):
        extract_claims(paper, call=model, max_source_repairs=3)
    assert len(modules) == 4
    audit = json.loads(next((tmp_path / "claim_extraction").glob("*.json")).read_text(encoding="utf-8"))
    assert audit["status"] == "failed" and len(audit["attempts"]) == 4
    assert all(row["source_errors"] for row in audit["attempts"])


def test_direct_strict_default_still_rejects_inexact_source_without_retry(paper):
    modules = []

    def model(**request):
        modules.append(request["module"])
        return {"status": "ok", "claims": [candidate("Method 0.75")]}

    with pytest.raises(ClaimExtractionError, match="does not occur"):
        extract_claims(paper, call=model)
    assert modules == ["screening.claims"]


@pytest.mark.parametrize("budget", [-1, 4, True, 1.5])
def test_source_repair_budget_cannot_exceed_confirmed_cap(paper, budget):
    with pytest.raises(ValueError, match="0 to 3"):
        extract_claims(paper, call=lambda **_: pytest.fail("invalid budget"), max_source_repairs=budget)


def test_repair_cannot_jump_to_a_block_outside_its_supplied_context(paper):
    paper.blocks.append(MaterialBlock(id="hidden", text="An unrelated result.", loc=ClaimLocation(page=2)))

    def model(**request):
        if request["module"] == "screening.claims":
            return {"status": "ok", "claims": [candidate("Missing source.")]}
        assert [b["id"] for b in json.loads(request["prompt"])["blocks"]] == ["b1"]
        return {
            "repairs": [{"index": 1, "source_block_id": "hidden", "source_quote": "An unrelated result."}]
        }

    with pytest.raises(ClaimExtractionError, match="outside the provided"):
        extract_claims(paper, call=model, max_source_repairs=3)


@pytest.mark.parametrize("error", ["response", "exception"])
def test_repair_error_preserves_extraction_exception_contract(paper, error):
    def model(**request):
        if request["module"] == "screening.claims":
            return {"status": "ok", "claims": [candidate("Missing source.")]}
        if error == "exception":
            raise TimeoutError("Fixture timed out")
        return {"status": "error", "error": "Fixture unavailable"}

    with pytest.raises(ClaimExtractionError, match="source repair 1 failed"):
        extract_claims(paper, call=model, max_source_repairs=3)
