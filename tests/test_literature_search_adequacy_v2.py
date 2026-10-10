"""Scientific scope controls use mocked retrieval/read/model boundaries only."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from llm.client import LLMConfig
from tests.literature_scope_fixtures import native_response, scientific_response
from tests.test_literature_v2 import boundaries, comparison
from tests.test_literature_v2 import claim as claim
from tests.test_literature_v2 import materials as materials
from tests.test_literature_v2 import paper as paper
from verification.literature import verify_literature


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("All external boundaries must be mocked")
    monkeypatch.setattr("verification.literature.resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None))
    monkeypatch.setattr("verification.literature.llm_json", forbidden)
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("httpx.AsyncClient.send", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)


def observed(papers, *, legacy=False, exhausted=True):
    searcher, reader = boundaries(papers)
    async def search(*, query, cutoff_date):
        return {**native_response(query, papers, cutoff_date, exhausted=exhausted), **({"complete": True} if legacy else {})}
    searcher.search.side_effect = search
    return searcher, reader


def answer(rows=(), *, mutate=None, states=None):
    def call(**kwargs):
        payload = json.loads(kwargs["prompt"].split("\nDATA_JSON:\n", 1)[1])
        response = scientific_response(payload, rows, states=states)
        if mutate:
            mutate(response)
        return response
    return Mock(side_effect=call)


def audit_for(materials):
    return json.loads(next(Path(materials.markdown_path).parent.rglob("*-search-audit.json")).read_text("utf-8"))


def absence(result):
    return [item for item in result.evidence if item.pointer.key == "search_scope"]


@pytest.mark.asyncio
async def test_legacy_complete_alone_never_authorizes_absence(claim, materials, paper):
    searcher, reader = boundaries([paper])
    searcher.search.side_effect = None
    searcher.search.return_value.pop("search_coverage", None)
    searcher.search.return_value["complete"] = True
    result = await verify_literature(claim, materials, submission_deadline="2021-01-31",
        searcher=searcher, reader=reader, call=answer([comparison(paper)]))
    assert absence(result) == []


@pytest.mark.asyncio
async def test_true_native_zero_hits_requires_one_scientific_scope_call(claim, materials):
    searcher, reader = observed([])
    call = answer()
    result = await verify_literature(claim, materials, submission_deadline="2021-01-31",
                                    searcher=searcher, reader=reader, call=call)
    assert call.call_count == 1 and reader.read_papers.await_count == 0
    assert absence(result)[0].covered == ["cond1"] and absence(result)[0].sufficient
    audit = audit_for(materials)
    assert audit["scientific_search_scope"]["raw_zero_hits"] is True
    assert audit["search_adequacy"]["valid"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["digest", "missing"])
async def test_native_scope_bad_scientific_protocol_never_authorizes_absence(claim, materials, paper, bad):
    searcher, reader = observed([paper], legacy=True)
    def mutate(response):
        if bad == "missing":
            response.pop("search_adequacy", None)
        else:
            response.setdefault("search_adequacy", {})["scope_digest"] = "0" * 64
    result = await verify_literature(claim, materials, submission_deadline="2021-01-31",
        searcher=searcher, reader=reader, call=answer([comparison(paper)], mutate=mutate))
    assert absence(result) == []
    assert any(row["category"] == "comparison_protocol_failure" for row in audit_for(materials)["context_events"])


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["condition", "query", "claim", "source", "duplicate", "pagination_fact"])
async def test_scientific_judgment_is_closed_to_current_scope(claim, materials, paper, bad):
    searcher, reader = observed([paper])
    def mutate(response):
        scope = response["search_adequacy"]
        if bad == "condition":
            scope["conditions"][0]["condition_id"] = "outside"
        elif bad == "query":
            scope["conditions"][0]["queries"][0]["intent"] = "author search"
        elif bad == "claim":
            scope["claim_id"] = "other"
        elif bad == "source":
            scope["source_ids"] = []
        elif bad == "duplicate":
            scope["conditions"].append(scope["conditions"][0])
        else:
            scope["exhausted"] = True
    result = await verify_literature(claim, materials, submission_deadline="2021-01-31",
        searcher=searcher, reader=reader, call=answer([comparison(paper)], mutate=mutate))
    assert absence(result) == []
    assert not audit_for(materials)["search_adequacy"]["valid"]
    assert result.questions == []


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["inadequate", "unresolved"])
async def test_scientific_insufficiency_has_no_operational_or_author_failure(claim, materials, paper, state):
    searcher, reader = observed([paper])
    result = await verify_literature(claim, materials, submission_deadline="2021-01-31",
        searcher=searcher, reader=reader, call=answer([comparison(paper)], states={"cond1": state}))
    assert absence(result) == [] and result.questions == [] and result.verification_limitations == []
    assert audit_for(materials)["search_adequacy"]["valid"]
    assert any(state in issue for issue in result.issues)


@pytest.mark.asyncio
async def test_budget_truncation_keeps_real_positive_comparison_evidence(claim, materials, paper):
    searcher, reader = observed([paper], exhausted=False)
    call = answer([comparison(paper, relation="same")])
    result = await verify_literature(claim, materials, submission_deadline="2021-01-31",
        searcher=searcher, reader=reader, call=call)
    assert call.call_count == 1 and reader.read_papers.await_count == 1
    assert len(result.evidence) == 1 and result.evidence[0].direction == "flaw" and result.evidence[0].sufficient
    assert absence(result) == [] and result.verification_limitations == []


@pytest.mark.asyncio
async def test_only_one_condition_may_be_scientifically_adequate(claim, materials, paper):
    from schemas.claim import Condition
    claim.conditions.append(Condition(id="cond2", description="novel mechanism in a different setting"))
    searcher, reader = observed([paper])
    result = await verify_literature(claim, materials, submission_deadline="2021-01-31",
        searcher=searcher, reader=reader,
        call=answer([comparison(paper, covered=["cond1", "cond2"])], states={"cond2": "unresolved"}))
    assert len(absence(result)) == 1 and absence(result)[0].covered == ["cond1"]
    assert result.verification_limitations == [] and not audit_for(materials)["adequate_for_no_close_prior_work"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["filtered", "self", "unknown_date", "unread", "abstract", "mixed_failure", "normalization"])
async def test_source_and_raw_scope_gates_remain_conservative(claim, materials, paper, kind):
    if kind == "filtered":
        paper["published"] = "2022-01-01"
    elif kind == "self":
        paper["title"] = materials.title
    elif kind == "unknown_date":
        paper.pop("published")
        paper.pop("arxiv_id")
        paper["id"] = "doi:10.1000/unknown"
        paper["url"] = "https://doi.org/10.1000/unknown"
    searcher, reader = observed([paper])
    if kind == "unread":
        reader.read_papers.side_effect = RuntimeError("offline reader unavailable")
    elif kind == "abstract":
        reader.read_papers.side_effect = None
        reader.read_papers.return_value = {"success": True, "items": [{"id": paper["id"], "success": True, "paper": paper}]}
    elif kind in {"mixed_failure", "normalization"}:
        original = searcher.search.side_effect
        async def search(*, query, cutoff_date):
            response = await original(query=query, cutoff_date=cutoff_date)
            if searcher.search.await_count == 2:
                if kind == "mixed_failure":
                    response["success"] = False
                    response["error"] = "offline query unavailable"
                else:
                    response["search_coverage"]["queries"][0]["pages"][0]["normalization_dropped_count"] = 1
            return response
        searcher.search.side_effect = search
    quote = paper["abstract"] if kind in {"abstract", "unread"} else None
    call = answer([comparison(paper, quote=quote)])
    result = await verify_literature(claim, materials, submission_deadline="2021-01-31",
        searcher=searcher, reader=reader, call=call)
    assert absence(result) == []
    assert not audit_for(materials)["adequate_for_no_close_prior_work"]
    if kind in {"filtered", "self"}:
        assert call.call_count == 0 and reader.read_papers.await_count == 0
        assert not audit_for(materials)["scientific_search_scope"]["raw_zero_hits"]
    if kind == "mixed_failure":
        assert call.call_count == 1 and reader.read_papers.await_count == 1
        assert any(row["operation"] == "search" for row in audit_for(materials)["context_events"])


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["arxiv", "semantic_scholar", "openalex"])
async def test_native_provider_zero_gate_is_count_bound(claim, materials, provider):
    searcher, reader = observed([])
    async def search(*, query, cutoff_date):
        return native_response(query, [], cutoff_date, provider=provider)
    searcher.search.side_effect = search
    call = answer()
    result = await verify_literature(claim, materials, submission_deadline="2021-01-31",
        searcher=searcher, reader=reader, call=call)
    assert call.call_count == 1 and len(absence(result)) == 1
    assert audit_for(materials)["scientific_search_scope"]["native_exhausted"]


@pytest.mark.asyncio
async def test_concern_in_one_condition_preserves_independent_scope_support(claim, materials, paper):
    from schemas.claim import Condition
    claim.conditions.append(Condition(id="cond2", description="novel mechanism in a different setting"))
    searcher, reader = observed([paper])
    result = await verify_literature(claim, materials, submission_deadline="2021-01-31",
        searcher=searcher, reader=reader,
        call=answer([comparison(paper, covered=["cond1"]), comparison(paper, relation="partial", covered=["cond2"])]))
    assert len(absence(result)) == 1 and absence(result)[0].covered == ["cond1"]
    assert any(item.concern and item.covered == ["cond2"] for item in result.evidence)
    decisions = audit_for(materials)["condition_scope_decisions"]
    assert decisions["cond1"]["supported"] and not decisions["cond2"]["supported"]


@pytest.mark.parametrize("provider", ["arxiv", "semantic_scholar", "openalex"])
def test_native_multi_page_chain_requires_observed_continuity(provider, paper):
    import copy
    import hashlib

    from util.cutoff_date import parse_submission_deadline
    from verification.literature_search_scope import _native_query
    cutoff = parse_submission_deadline("2021-01-31")
    other = {**paper, "id": "2001.00002", "title": "A distinct compositional study"}
    response = native_response("graph mechanism", [paper, other], cutoff, provider=provider)
    scope = response["search_coverage"]["queries"][0]
    scope["limits"].update(page_size=1, max_pages=2)
    first = scope["pages"][0]
    first.update(limit=1, raw_count=1, normalized_count=1, exhausted=False)
    cursor = hashlib.sha256(b"second").hexdigest()
    first["next_cursor_sha256"] = cursor if provider == "openalex" else None
    first["next_offset"] = None if provider == "openalex" else 1
    second = copy.deepcopy(first)
    second.update(index=1, offset=1, exhausted=True, next_offset=None, next_cursor_sha256=None,
                  request_cursor_sha256=cursor if provider == "openalex" else None)
    scope["pages"].append(second)
    assert _native_query("graph mechanism", response, cutoff.to_metadata())[0]
    second["offset"] = 0
    assert not _native_query("graph mechanism", response, cutoff.to_metadata())[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["service", "protocol", "identity", "abstract"])
async def test_uncited_novelty_read_failure_responsibility_is_condition_scoped(claim, materials, paper, mode):
    from schemas.claim import Condition
    claim.conditions.append(Condition(id="performance", metric="accuracy", description="Reported 90% accuracy"))
    claim_before = claim.model_dump(mode="json")
    material_before = materials.model_dump(mode="json")
    searcher, reader = observed([paper])
    if mode == "service":
        reader.read_papers.side_effect = RuntimeError("offline read unavailable")
    else:
        reader.read_papers.side_effect = None
        metadata = {**paper, **({"arxiv_id": "2002.00002"} if mode == "identity" else {})}
        reader.read_papers.return_value = {"success": True, "items": [] if mode == "protocol" else [
            {"id": paper["id"], "success": True, "paper": metadata}
        ]}
    call = answer([comparison(paper, quote=paper["abstract"])])
    result = await verify_literature(claim, materials, submission_deadline="2021-01-31",
        searcher=searcher, reader=reader, call=call)
    audit = audit_for(materials)
    assert claim.model_dump(mode="json") == claim_before and materials.model_dump(mode="json") == material_before
    assert searcher.search.await_count == 3 and reader.read_papers.await_count == 1 and call.call_count == 1
    assert absence(result) == [] and result.questions == []
    if mode == "abstract":
        assert result.verification_limitations == []
        assert all(not event["system_limited"] for event in audit["context_events"])
        assert any(event["category"] == "abstract_only" for event in audit["context_events"])
    else:
        category = {"service": "service_failure", "protocol": "reader_protocol_failure", "identity": "identity_conflict"}[mode]
        event = next(event for event in audit["context_events"] if event["category"] == category)
        assert event["operation"] == "read_papers" and event["condition_ids"] == ["cond1"] and event["system_limited"]
        assert len(result.verification_limitations) == 1
        assert result.verification_limitations[0].claim_id == claim.id
        assert result.verification_limitations[0].condition_ids == ["cond1"]
