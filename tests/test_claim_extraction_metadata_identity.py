"""The title and abstract actually supplied to extraction retain their identity."""

import copy
import json

import pytest

from llm.client import LLMConfig
from schemas.claim import ClaimLocation
from schemas.materials import MaterialBlock, SharedMaterials
from screening import claims as module


def test_supplied_title_and_abstract_mutation_rejects_the_original_response(tmp_path, monkeypatch):
    text = "The method uses Adam as its optimizer."
    paper = SharedMaterials(
        paper_key="metadata-identity", title="Original Method", abstract=text,
        source_pdf="paper.pdf", markdown=text, markdown_path="paper.md", content_list_path="",
        provider="fixture", blocks=[MaterialBlock(id="b1", text=text, loc=ClaimLocation(page=1))],
    )
    raw = {"status": "ok", "claims": [{
        "text": text, "source_block_id": "b1", "source_quote": text, "source_refs": [],
        "conditions": [{"id": "c1", "settings": {"optimizer": "Adam"}, "description": "Optimizer selection"}],
        "needs": ["Code"], "importance": "core",
    }]}
    original = paper.model_dump(mode="json")

    def forbidden(*args, **kwargs):
        pytest.fail("External call forbidden by offline extraction control")

    monkeypatch.setattr(module, "resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None))
    monkeypatch.setattr(module, "llm_json", forbidden)
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)
    for field in ("title", "abstract"):
        current = paper.model_copy(deep=True)
        stats = tmp_path / field / "run_stats.json"
        monkeypatch.setattr(module.run_stats, "stats_path", lambda stats=stats: stats)
        calls = []

        def call(**request):
            supplied = json.loads(request["prompt"].split("PAPER_DATA_JSON:\n", 1)[1])
            assert supplied[field] == original[field]
            calls.append(request["module"])
            setattr(current, field, original[field] + " changed")
            return copy.deepcopy(raw)

        with pytest.raises(module.ClaimExtractionError, match="Original materials changed"):
            module.extract_claims(current, call=call, max_source_repairs=0)
        audit = json.loads(next((stats.parent / "claim_extraction").glob("*.json")).read_text(encoding="utf-8"))
        assert audit["status"] == "failed" and audit["attempts"][0]["response"] == raw
        assert calls == ["screening.claims"]
        assert paper.model_dump(mode="json") == original

    monkeypatch.setattr(module.run_stats, "stats_path", lambda: tmp_path / "healthy" / "run_stats.json")
    healthy = module.extract_claims(paper, call=lambda **_: copy.deepcopy(raw), max_source_repairs=0)
    assert len(healthy) == 1 and healthy[0].text == text
    assert healthy[0].source_quote == text and healthy[0].conditions[0].settings == {"optimizer": "Adam"}
    assert paper.model_dump(mode="json") == original
