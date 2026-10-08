"""Single-domain retrieval exercises the real dispatcher and literature policy."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from assessment import assess_claim
from llm.client import LLMConfig
from schemas.claim import Claim, ClaimLocation, Condition
from schemas.materials import MaterialBlock, SharedMaterials
from verification.dispatch import verify_claims
from verification.literature import literature_queries, verify_literature


@pytest.fixture(autouse=True)
def mock_model_config(monkeypatch):
    monkeypatch.setattr(
        "verification.literature.resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None)
    )
    monkeypatch.setattr("verification.literature.llm_json", Mock(side_effect=AssertionError("Unmocked LLM")))


def inputs(tmp_path, topic="image classification"):
    text = f"Our novel {topic} method uses a new mechanism."
    loc = ClaimLocation(page=1, char_start=0, char_end=len(text))
    materials = SharedMaterials(
        paper_key="single-domain",
        title=f"A robust {topic} model",
        abstract=f"We study {topic}.",
        source_pdf=str(tmp_path / "paper.pdf"),
        markdown=text,
        markdown_path=str(tmp_path / "paper.md"),
        content_list_path=str(tmp_path / "content.json"),
        provider="fixture",
        blocks=[MaterialBlock(id="b1", text=text, loc=loc)],
    )
    claim = Claim(
        id="c1",
        text=text,
        loc=loc,
        source_block_id="b1",
        source_quote=text,
        conditions=[Condition(id="novelty", description=f"Novel mechanism for {topic}")],
        needs=["Literature"],
    )
    return materials, claim


def boundaries(papers=(), **changes):
    adapter = Mock()
    adapter.search = AsyncMock(
        return_value={
            "success": True,
            "provider": "fixture",
            "papers": list(papers),
            **changes,
        }
    )
    adapter.read_papers = AsyncMock(
        return_value={
            "success": True,
            "items": [
                {
                    "id": "1901.00001",
                    "success": True,
                    "evidence": [{"page": 2, "text": "This method uses a closely related mechanism."}],
                }
            ],
        }
    )
    return adapter


def read_audit(materials, claim_id="c1"):
    root = Path(materials.markdown_path).parent
    return json.loads(next(root.rglob(f"{claim_id}-search-audit.json")).read_text(encoding="utf-8"))


@pytest.mark.parametrize("topic", ["image classification", "machine translation", "privacy"])
def test_one_recognized_topic_produces_three_distinct_intents(tmp_path, topic):
    materials, claim = inputs(tmp_path, topic)
    assert literature_queries(None, materials) == [
        f"{topic} mechanism",
        f"{topic} target setting",
        f"{topic} evaluation protocol baseline",
    ]
    assert literature_queries(claim, materials) == literature_queries(None, materials)


@pytest.mark.asyncio
@pytest.mark.parametrize("topic", ["image classification", "machine translation"])
async def test_dispatch_runs_single_domain_claim_and_global_related_work(tmp_path, monkeypatch, topic):
    materials, claim = inputs(tmp_path, topic)
    prior = {
        "id": "1901.00001",
        "arxiv_id": "1901.00001",
        "title": "A prior formulation of a related mechanism",
        "published": "2019-01-01",
    }
    adapter = boundaries([prior], complete=False)
    monkeypatch.setattr("verification.literature._default_adapter", lambda: adapter)
    calls = []

    def model(**kwargs):
        payload = json.loads(kwargs["prompt"].split("\nDATA_JSON:\n")[1])
        calls.append(payload["claim"])
        is_claim = payload["claim"] is not None
        return {
            "status": "ok",
            "comparisons": [
                {
                    "paper_id": prior["id"],
                    "purpose": "novelty" if is_claim else "related_work",
                    "relation": "partial",
                    "quote": "This method uses a closely related mechanism.",
                    "covered": ["novelty"] if is_claim else [],
                    "mechanism": "The source shares one operator with the submission.",
                    "setting": f"Both address {topic}.",
                    "protocol": "Their evaluation settings differ.",
                    "note": "A concrete related mechanism should be discussed.",
                }
            ],
        }

    result = await verify_claims(
        [claim], materials, tmp_path / "verification", submission_deadline="2021-01-01", call=model
    )
    assert adapter.search.await_count == 6
    queries = [call.kwargs["query"] for call in adapter.search.await_args_list]
    expected = literature_queries(claim, materials)
    assert queries[:3] == queries[3:] == expected
    assert adapter.read_papers.await_count == 2
    assert [c is None for c in calls] == [False, True]
    assert assess_claim(result.claims[0]).status == "questioned"
    assert any(finding.kind == "related_work" for finding in result.findings)
    for identifier in ("c1", "global"):
        audit = read_audit(materials, identifier)
        assert audit["query_terms"] == [topic]
        assert audit["query_policy"] == "closed_technical_vocabulary"
        assert len(audit["query_intents"]) == 3
        assert not audit["adequate_for_no_close_prior_work"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope",
    [
        {},
        {"complete": False},
        {"complete": True, "truncated": True},
        {"complete": True, "has_more": True},
        {"complete": True, "error": "partial service failure"},
        {"complete": True, "success": False},
        {"complete": True, "question_results": [{"success": False}]},
    ],
)
async def test_single_topic_empty_results_do_not_certify_search_scope(tmp_path, scope):
    materials, claim = inputs(tmp_path)
    adapter = boundaries(**scope)
    call = Mock()
    result = await verify_literature(
        claim, materials, submission_deadline="2021-01-01", searcher=adapter, reader=adapter, call=call
    )
    assert adapter.search.await_count == 3
    assert not result.evidence and assess_claim(claim).status == "unverified"
    assert any("Search scope is incomplete" in issue for issue in result.issues)
    adapter.read_papers.assert_not_awaited()
    call.assert_not_called()
    audit = read_audit(materials)
    assert not audit["adequate_for_no_close_prior_work"]
    assert json.loads(audit["search_scope"])["query_terms"] == ["image classification"]


@pytest.mark.asyncio
async def test_explicitly_complete_single_topic_scope_retains_scope_bounded_support(tmp_path):
    materials, claim = inputs(tmp_path)
    adapter = boundaries(complete=True)
    result = await verify_literature(
        claim, materials, submission_deadline="2021-01-01", searcher=adapter, reader=adapter, call=Mock()
    )
    assert len(result.evidence) == 1 and result.evidence[0].sufficient
    scope = json.loads(result.evidence[0].pointer.quote)
    assert scope["query_terms"] == ["image classification"]
    assert scope["supported_novelty_conditions"] == ["novelty"]
    assert all(query["complete"] is True for query in scope["queries"])


@pytest.mark.asyncio
@pytest.mark.parametrize("global_search", [False, True])
async def test_unknown_domain_remains_explicit_without_author_search(tmp_path, global_search):
    materials, claim = inputs(tmp_path, "quux")
    materials.title = "Ada Lovelace quux work author:John Smith"
    adapter = boundaries(complete=True)
    result = await verify_literature(
        None if global_search else claim,
        materials,
        submission_deadline="2021-01-01",
        searcher=adapter,
        reader=adapter,
        call=Mock(),
    )
    assert not result.evidence
    assert any("no recognized technical domain terms" in issue for issue in result.issues)
    adapter.search.assert_not_awaited()
    adapter.read_papers.assert_not_awaited()


def test_single_topic_queries_exclude_identity_and_search_operators(tmp_path):
    materials, claim = inputs(tmp_path)
    materials.title += " by Ada Lovelace author:John Smith site:openreview.net"
    assert literature_queries(claim, materials) == [
        "image classification mechanism",
        "image classification target setting",
        "image classification evaluation protocol baseline",
    ]
