"""Purpose routing and source/condition-bound reading; every external boundary mocked."""
import json
from unittest.mock import AsyncMock

import pytest

from llm.client import LLMConfig
from tests.test_literature_service_responsibility_v2 import boundaries, comparison, inputs
from verification import literature
from verification.dispatch import verify_claims


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("All external boundaries must be mocked")
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr("httpx.AsyncClient.send", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("verification.literature.llm_json", forbidden)
    monkeypatch.setattr("verification.literature._default_adapter", forbidden)
    monkeypatch.setattr("verification.literature.resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None))


def audit(path, name="claim"):
    return json.loads((path / f"{name}-search-audit.json").read_text("utf-8"))


def compare(**kwargs):
    payload = json.loads(kwargs["prompt"].split("DATA_JSON:\n", 1)[1])
    return {"status": "ok", "comparisons": [comparison(row["paper_id"], cid) for row in payload["sources"]
                            for cid in row["citation_condition_ids"]]}


async def test_default_dispatch_citation_has_no_expansion_and_global_stays_active(tmp_path, monkeypatch):
    claim, material = inputs(tmp_path)
    searcher, reader, _ = boundaries()
    real = literature.verify_literature
    received = []
    async def wrapped(c, m, **kwargs):
        received.append((c.id if c else None, kwargs.get("expand_uncited")))
        return await real(c, m, searcher=searcher, reader=reader, **kwargs)
    monkeypatch.setattr(literature, "verify_literature", wrapped)
    result = await verify_claims([claim], material, tmp_path / "audit", submission_deadline="2022-01-01", call=compare)
    assert received == [("claim", False), (None, False)]
    assert searcher.search.await_count == 3
    assert reader.read_papers.await_count == 2
    assert {cid for e in result.claims[0].evidence for cid in e.covered} == {"c1", "c2"}
    assert audit(tmp_path / "audit")["queries"] == []
    assert len(audit(tmp_path / "audit", "global")["queries"]) == 3
    assert audit(tmp_path / "audit")["retrieval_routing"]["purpose"] == "citation_only"


async def test_reader_question_binds_only_actual_citation_conditions_and_sources(tmp_path):
    claim, material = inputs(tmp_path)
    claim.conditions[0].settings = {"qualifiers": ["only assumption A", "no population claim"]}
    searcher, reader, _ = boundaries()
    await literature.verify_literature(claim, material, searcher=searcher, reader=reader,
        call=compare, submission_deadline="2022-01-01", output_dir=tmp_path / "audit", expand_uncited=False)
    for request in reader.read_papers.await_args_list:
        item = request.kwargs["items"][0]
        payload = json.loads(item["question"].split("DATA_JSON:\n", 1)[1])
        cid = "c1" if item["id"] == "2001.00001" else "c2"
        assert payload["claim"]["id"] == claim.id and payload["claim"]["text"] == claim.text
        assert payload["claim"]["loc"] == claim.loc.model_dump(mode="json")
        assert payload["claim"]["source_block_id"] == claim.source_block_id
        assert [row["id"] for row in payload["conditions"]] == [cid]
        assert len(payload["source_excerpts"]) == 1
        assert payload["source_excerpts"][0]["covered"] == [cid]
        assert payload["source_excerpts"][0]["loc"]["page"] == int(cid[-1])
        if cid == "c1":
            assert payload["conditions"][0]["settings"]["qualifiers"] == claim.conditions[0].settings["qualifiers"]
    assert searcher.search.await_count == 0


async def test_compat_direct_expansion_and_novelty_false_still_search(tmp_path):
    claim, material = inputs(tmp_path)
    searcher, reader, _ = boundaries()
    await literature.verify_literature(claim, material, searcher=searcher, reader=reader,
        call=compare, submission_deadline="2022-01-01", output_dir=tmp_path / "compat")
    assert searcher.search.await_count == 3
    assert audit(tmp_path / "compat")["retrieval_routing"]["purpose"] == "compat_uncited"
    claim.text = "This is the first method to prove the graph bound."
    claim.conditions[0].description = "Novelty of the first method under assumption A"
    searcher.search.reset_mock()
    await literature.verify_literature(claim, material, searcher=searcher, reader=reader,
        call=compare, submission_deadline="2022-01-01", output_dir=tmp_path / "novelty", expand_uncited=False)
    assert searcher.search.await_count == 3
    assert audit(tmp_path / "novelty")["novelty_condition_ids"] == ["c1"]
    assert audit(tmp_path / "novelty")["retrieval_routing"]["purpose"] == "claim_novelty"


async def test_global_verified_method_seed_and_reader_target(tmp_path):
    claim, material = inputs(tmp_path)
    material.title, material.abstract = "A study", "A short description."
    block = material.blocks[0]
    block.loc.section = "section_2: 2. Methods"
    # Source coordinates and actual block content are preserved.
    target = {"source_block_id": block.id, "source_quote": block.text,
              "loc": block.loc.model_dump(mode="json"), "covered": ["c1"], "claim_id": claim.id}
    searcher, reader, _ = boundaries()
    searcher.search.return_value["papers"] = [{"id": "2001.00001", "arxiv_id": "2001.00001",
        "title": "Independent bounds", "published": "2020-01-01", "abstract": "Independent graph assumptions."}]
    await literature.verify_literature(None, material, searcher=searcher, reader=reader,
        call=lambda **kw: {"status": "ok", "comparisons": []}, manuscript_targets=[target],
        submission_deadline="2022-01-01", output_dir=tmp_path / "global", expand_uncited=False)
    assert searcher.search.await_count == 3
    assert all("graph" in request.kwargs["query"] for request in searcher.search.await_args_list)
    payload = json.loads(reader.read_papers.await_args.kwargs["items"][0]["question"].split("DATA_JSON:\n", 1)[1])
    assert payload["claim"] is None and payload["conditions"] == []
    assert payload["manuscript_targets"][0]["source_quote"] == block.text
    assert payload["manuscript_targets"][0]["loc"]["section"] == block.loc.section
    row = audit(tmp_path / "global", "global")
    assert row["global_target_scope"]["targets"][0]["seed_available"] is True


async def test_global_background_and_unvalidated_target_never_seed(tmp_path):
    _claim, material = inputs(tmp_path)
    material.title, material.abstract = "A study", "A short description."
    block = material.blocks[0]
    block.loc.section = "section_2: 2. Related Work"
    target = {"source_block_id": block.id, "source_quote": block.text,
              "loc": block.loc.model_dump(mode="json"), "covered": ["c1"]}
    forged = {**target, "loc": {**target["loc"], "section": "Methods"}}
    searcher, reader, _ = boundaries()
    call = AsyncMock(return_value={"comparisons": []})
    result = await literature.verify_literature(None, material, searcher=searcher, reader=reader,
        call=call, manuscript_targets=[target, forged], submission_deadline="2022-01-01",
        output_dir=tmp_path / "audit", expand_uncited=False)
    assert searcher.search.await_count == reader.read_papers.await_count == call.await_count == 0
    assert len(result.delivery_checks) == 1
    check = result.delivery_checks[0]
    assert check.component == "global_literature.search" and check.state == "unavailable" and check.responsibility == "system"
    row = audit(tmp_path / "audit", "global")
    assert len(row["manuscript_targets"]) == 1 and row["omission_target_unavailable"]
    assert row["global_target_scope"]["targets"][0]["seed_available"] is False
    assert result.findings == result.questions == result.evidence == []
