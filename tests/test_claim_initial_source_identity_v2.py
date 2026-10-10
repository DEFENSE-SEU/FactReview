"""The initial extraction response must retain its original block identity."""

import copy
import json
import subprocess

import pytest

from llm.client import LLMConfig
from schemas.claim import ClaimLocation
from schemas.materials import MaterialBlock, SharedMaterials
from screening import claims as module


def test_initial_page_mutation_cannot_become_a_grounded_claim(monkeypatch, tmp_path):
    text = (
        "On MiniSet test, model ExactMatch has accuracy 0.75. Accuracy is a fraction, "
        "computed by exact equality of each prediction and label. Three of the four "
        "fixed test predictions are correct. Table 1 reports the result and Figure 1 "
        "shows the corresponding counts. No repeated-run uncertainty or "
        "population-performance conclusion is claimed."
    )
    table = (
        "Table 1. MiniSet test result. Accuracy is a fraction; bold type identifies the "
        "measured accuracy, with no ranking against other models.\n"
        "<table><tr><td>Model</td><td>Test accuracy</td><td>Examples</td></tr>"
        "<tr><td>ExactMatch</td><td>0.75</td><td>4</td></tr></table>"
    )
    materials = SharedMaterials(
        paper_key="initial-source-identity",
        source_pdf="paper.pdf",
        markdown=text + "\n\n" + table,
        markdown_path="paper.md",
        content_list_path="",
        provider="fixture",
        blocks=[
            MaterialBlock(
                id="block_10",
                text=text,
                loc=ClaimLocation(page=1, char_start=0, char_end=len(text)),
            ),
            MaterialBlock(id="block_11", text=table, loc=ClaimLocation(page=1)),
        ],
    )
    candidate = {
        "text": (
            "On MiniSet test, model ExactMatch has accuracy 0.75, where accuracy is a "
            "fraction computed by exact equality of each prediction and label, over "
            "four fixed test predictions, with no repeated-run uncertainty or "
            "population-performance conclusion claimed."
        ),
        "source_block_id": "block_10",
        "source_quote": text,
        "source_refs": [{"source_block_id": "block_11", "source_quote": table, "covered": ["c1"]}],
        "conditions": [
            {
                "id": "c1",
                "dataset": "MiniSet test",
                "metric": "accuracy",
                "settings": {
                    "model": "ExactMatch",
                    "accuracy": 0.75,
                    "examples": 4,
                    "computation": "fraction computed by exact equality of each prediction and label",
                    "uncertainty": "no repeated-run uncertainty claimed",
                    "scope": "no population-performance conclusion claimed",
                    "ranking": "no ranking against other models",
                },
                "description": "Reported MiniSet test accuracy for ExactMatch on four fixed test predictions.",
            }
        ],
        "needs": ["Experiments"],
        "importance": "core",
    }

    def forbidden(*args, **kwargs):
        pytest.fail("External operation reached by offline extraction control")

    monkeypatch.setattr(module, "resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None))
    monkeypatch.setattr(module, "llm_json", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    saved = materials.model_dump(mode="json")
    raw = {"status": "ok", "claims": [candidate]}
    original_response = copy.deepcopy(raw)
    current = materials.model_copy(deep=True)
    calls = []
    stats = tmp_path / "rejected" / "run_stats.json"
    monkeypatch.setattr(module.run_stats, "stats_path", lambda: stats)

    def call(**request):
        calls.append(request["module"])
        data = json.loads(request["prompt"].split("\nPAPER_DATA_JSON:\n", 1)[1])
        assert request["module"] == "screening.claims"
        assert data["blocks"][0]["loc"]["page"] == 1
        assert data["blocks"][0]["text"] == text
        current.blocks[0].loc.page = 9
        return copy.deepcopy(raw)

    with pytest.raises(module.ClaimExtractionError, match="Original materials changed"):
        module.extract_claims(current, call=call, max_source_repairs=0)
    assert calls == ["screening.claims"]
    rejected = json.loads(next((stats.parent / "claim_extraction").glob("*.json")).read_bytes())
    assert rejected["status"] == "failed"
    assert len(rejected["attempts"]) == 1
    assert rejected["attempts"][0]["response"] == original_response
    assert current.blocks[0].loc.page == 9
    assert materials.model_dump(mode="json") == saved
    assert raw == original_response

    stats = tmp_path / "healthy" / "run_stats.json"
    calls.clear()

    def healthy(**request):
        calls.append(request["module"])
        return copy.deepcopy(raw)

    result = module.extract_claims(materials, call=healthy, max_source_repairs=0)
    assert calls == ["screening.claims"]
    assert len(result) == 1 and result[0].loc.page == 1
    assert result[0].source_quote == text
    assert materials.markdown[result[0].loc.char_start : result[0].loc.char_end] == text
    assert result[0].source_refs[0].source_quote == table
    assert result[0].conditions[0].model_dump(mode="json") == candidate["conditions"][0]
    assert result[0].needs == candidate["needs"]
    assert result[0].text == candidate["text"] and result[0].status == "unverified"
    assert materials.model_dump(mode="json") == saved and raw == original_response
    healthy_audit = json.loads(next((stats.parent / "claim_extraction").glob("*.json")).read_bytes())
    assert healthy_audit["status"] == "ok" and len(healthy_audit["attempts"]) == 1
