"""Six bounded literal/producer contracts; all external boundaries are mocks."""

import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from fact_generation.positioning.structured_query import INTENTS, LiteralQueryTerm, parse_structured_query
from llm.client import LLMConfig
from schemas.claim import Claim, ClaimLocation, Condition, EvidenceNeed
from schemas.materials import MaterialBlock, SharedMaterials
from tests.literature_scope_fixtures import native_response, scientific_response
from verification import literature
from verification.contracts import BranchResult
from verification.literature_search_concepts import (
    catalog_current,
    plan_concepts,
    scientific_claim_digest,
)
from verification.literature_search_plan import (
    build_literal_plan,
    condition_eligible,
    plan_closed,
    query_strings,
)

PHRASES = {"mechanism": "finite volume", "target setting": "heat transport", "evaluation protocol baseline": "convergence order"}


def planning_response(request, phrases=None):
    phrases = phrases or PHRASES
    concepts = []
    for source in request["sources"]:
        for role, phrase in phrases.items():
            if phrase in source["source_quote"]:
                concepts.append({"concept_id": f"s{len(concepts) + 1}", "source_id": source["source_id"],
                                 "phrase": phrase, "role_basis_quote": source["source_quote"], "role": role,
                                 "role_reason": f"The actual body describes {phrase} under this scientific role.",
                                 "entity_kind": "technical_concept", "unit_ids": list(source["unit_ids"])})
    return {"version": "literal-concept-proposal-v1", "input_digest": request["input_digest"], "concepts": concepts,
            "units": [{"unit_id": u["unit_id"], "global_body_role": "scientific_method" if u["purpose"] == "global_omission" else None,
                       "roles": {role: {"status": "present" if any(c["role"] == role and u["unit_id"] in c["unit_ids"] for c in concepts) else "not_stated",
                                        "concept_ids": [c["concept_id"] for c in concepts if c["role"] == role and u["unit_id"] in c["unit_ids"]],
                                        "reason": "Actual source role or explicitly not stated."} for role in INTENTS}} for u in request["units"]]}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("External/model/retrieval/Docker/process boundaries must be mocked")
    for target in ("requests.sessions.Session.request", "httpx.Client.send", "httpx.AsyncClient.send",
                   "urllib.request.urlopen", "subprocess.run", "verification.literature.llm_json",
                   "verification.literature._default_adapter", "fact_generation.execution.v2.docker_runner"):
        monkeypatch.setattr(target, forbidden)
    monkeypatch.setattr(literature, "resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None))


def inputs(tmp_path, *, text=None):
    quote = text or "Our first novel finite volume scheme models heat transport with convergence order evaluation."
    prefix = "页 α. "
    body = prefix + quote
    md = tmp_path / "paper.md"
    md.write_text(body, encoding="utf-8")
    loc = ClaimLocation(page=1, char_start=0, char_end=len(body))
    block = MaterialBlock(id="body", text=body, loc=loc, kind="abstract")
    claim = Claim(id="claim", text=quote, source_block_id=block.id, source_quote=quote,
                  loc=ClaimLocation(page=1, char_start=len(prefix), char_end=len(body)),
                  conditions=[Condition(id="c1", description="Novel mechanism under the original stated conditions")], needs=["Literature"])
    materials = SharedMaterials(paper_key="literal", title="Original unrelated submission",
                                abstract="Original manuscript identity context remains available.", source_pdf=str(tmp_path / "paper.pdf"),
                                markdown=body, markdown_path=str(md), content_list_path="", provider="mock", blocks=[block])
    targets = [{**s, "claim_id": claim.id} for s in literature._claim_source_excerpts(claim, materials)]
    return claim, materials, targets


def model(**kwargs):
    data = json.loads(kwargs["prompt"].split("\nCONCEPT_DATA_JSON:\n", 1)[1])
    return planning_response(data)


def catalog(claim, materials, targets, *, call=model):
    return asyncio.run(plan_concepts([claim], materials, targets, call=call))


def test_outside_vocabulary_exact_unicode_source_and_ambiguity(tmp_path):
    c, m, t = inputs(tmp_path)
    cat = catalog(c, m, t)
    plan = build_literal_plan(c, m, cat)
    assert len(plan["queries"]) == 3 and condition_eligible(plan, "c1") and plan_closed(plan)
    for concept in plan["concepts"]:
        span = concept["phrase_span"]
        assert m.blocks[0].text[span["start"]:span["end"]] == concept["phrase"]
        assert m.markdown[concept["loc"]["char_start"]:concept["loc"]["char_end"]] == concept["phrase"]
    assert any('ti:"finite volume"' in q for q in query_strings(plan))
    changed = c.model_copy(deep=True)
    changed.notes.append("Downstream audit enrichment")
    assert scientific_claim_digest(c) == scientific_claim_digest(changed) and catalog_current(cat, changed, m)
    changed.conditions[0].description += " Additional scientific qualifier"
    assert not catalog_current(cat, changed, m)
    def bad(**kw):
        return {**model(**kw), "concepts": [{**v, "phrase": "invented phrase"} for v in model(**kw)["concepts"]]}
    assert catalog(c, m, t, call=bad)["failures"]
    c2, m2, t2 = inputs(tmp_path, text="Our novel finite volume heat transport convergence order and finite volume scheme.")
    ambiguous = catalog(c2, m2, t2)
    assert any(r["reason"] == "invalid_source_or_participation" for r in ambiguous["rejected"])
    assert not condition_eligible(build_literal_plan(c2, m2, ambiguous), "c1")


def test_closed_digest_units_and_local_source_participation(tmp_path):
    c, m, t = inputs(tmp_path)
    for mutate in (lambda r: r.update(input_digest="0" * 64), lambda r: r["units"].pop(),
                   lambda r: r["units"].append(copy.deepcopy(r["units"][0]))):
        def bad(**kw):
            response = model(**kw)
            mutate(response)
            return response
        assert catalog(c, m, t, call=bad)["state"] == "failed"
    def foreign(**kw):
        r = model(**kw)
        r["concepts"][0]["unit_ids"].append("foreign")
        return r
    cat = catalog(c, m, t, call=foreign)
    assert cat["failures"] and not condition_eligible(build_literal_plan(c, m, cat), "c1")
    assert any(c["role"] == "target setting" for c in build_literal_plan(c, m, cat)["concepts"])
    second = "Our novel second study models heat transport."
    start = len(m.markdown) + 1
    m.markdown += "\n" + second
    m.blocks.append(MaterialBlock(id="second", text=second, loc=ClaimLocation(page=2, char_start=start, char_end=start + len(second))))
    from pathlib import Path
    Path(m.markdown_path).write_text(m.markdown, encoding="utf-8")
    c.conditions.append(Condition(id="c2", description="Novel second method"))
    c = Claim.model_validate({**c.model_dump(), "source_refs": [
        {"source_block_id": "body", "source_quote": c.source_quote, "loc": c.loc, "covered": ["c1"]},
        {"source_block_id": "second", "source_quote": second, "loc": m.blocks[1].loc, "covered": ["c2"]}]})
    # Build real typed references, retaining each condition's original scope.
    t = [{**s, "claim_id": c.id} for s in literature._claim_source_excerpts(c, m)]
    plan = build_literal_plan(c, m, catalog(c, m, t))
    setting = next(q for q in plan["queries"] if q["intent"] == "target setting")
    assert setting["condition_ids"] == ["c1", "c2"] and len(setting["groups"]) == 1
    assert condition_eligible(plan, "c1") and not condition_eligible(plan, "c2")
    witnesses = {x["concept_id"]: x for x in plan["concepts"]}
    assert all("c2" in witnesses[sid]["condition_ids"] for sid in setting["participation"]["c2"]["target setting"])


def test_literal_syntax_identity_and_missing_mechanism_positive(tmp_path):
    for phrase in ('ti:evil', 'https://evil', 'x" OR all:"x', 'Smith AND Jones', '(operator)'):
        with pytest.raises(ValueError):
            LiteralQueryTerm(concept_id="s1", source_concept_ids=["s1"], phrase=phrase, role="mechanism")
    c, m, t = inputs(tmp_path, text="Our first novel study measures heat transport.")
    cat = catalog(c, m, t)
    plan = build_literal_plan(c, m, cat)
    assert len(plan["queries"]) == 1 and plan["queries"][0]["intent"] == "target setting"
    assert not condition_eligible(plan, "c1") and 'heat transport' in query_strings(plan)[0]
    from tests.test_literature_service_responsibility_v2 import paper, read_response
    class PartialSearch:
        search_cfg = SimpleNamespace(provider="arxiv")
        async def search_structured(self, *, query, cutoff_date):
            return native_response(query.compile(start=0, limit=8).expression, [paper("2001.00002")], cutoff_date, exhausted=False)
    def comparison(**kw):
        data = json.loads(kw["prompt"].split("\nDATA_JSON:\n", 1)[1])
        return scientific_response(data, [{"paper_id": "2001.00002", "purpose": "novelty", "relation": "partial",
                                          "quote": "The bound holds under the stated assumptions.", "covered": ["c1"],
                                          "mechanism": "Bound mechanisms need investigation.", "setting": "Thermal transport overlaps.",
                                          "protocol": "Only partial evidence.", "note": "A source concern remains actionable under this bounded scope."}])
    reader = SimpleNamespace(read_papers=AsyncMock(return_value=read_response("2001.00002")))
    result = asyncio.run(literature.verify_literature(c, m, submission_deadline="2021-01-01", searcher=PartialSearch(),
                      reader=reader, call=comparison, output_dir=tmp_path, search_policy="literal", concept_catalog=cat))
    assert reader.read_papers.await_count == 1 and len(result.evidence) == 1 and result.evidence[0].concern
    assert not result.evidence[0].sufficient and not result.verification_limitations
    def identity(**kw):
        r = model(**kw)
        for concept in r["concepts"]:
            concept["entity_kind"] = "author_or_citation_identity"
        return r
    assert not build_literal_plan(c, m, catalog(c, m, t, call=identity))["queries"]
    m.blocks[0].kind = "heading"
    calls = []
    cat = catalog(c, m, t, call=lambda **kw: calls.append(kw))
    assert calls == [] and cat["concepts"] == []


def test_shared_future_allows_other_peer_and_global_same_catalog(tmp_path, monkeypatch):
    from verification import dispatch
    c, m, _t = inputs(tmp_path)
    c.needs.append(EvidenceNeed.THEORY)
    peer_done = asyncio.Event()
    calls, seen = [], []
    async def planner(**kw):
        calls.append(kw["module"])
        await asyncio.wait_for(peer_done.wait(), timeout=2)
        return model(**kw)
    async def theory(*args, **kwargs):
        peer_done.set()
        return BranchResult()
    async def lit(claim, materials, **kwargs):
        cat = kwargs["concept_catalog"]
        seen.append((claim.id if claim else None, cat["digest"]))
        return BranchResult()
    monkeypatch.setattr("verification.theory.verify_theory", theory)
    monkeypatch.setattr(literature, "verify_literature", lit)
    asyncio.run(dispatch.verify_claims([c], m, tmp_path / "out", call=planner))
    assert calls == ["verification.literature.planning"]
    assert len(seen) == 2 and seen[0][1] == seen[1][1] and {s[0] for s in seen} == {"claim", None}
    # Without novelty, the global target still uses the shared producer; no claim-specific task.
    c.conditions[0].description = "Reported method"
    c.text = "We describe the method."
    calls.clear()
    seen.clear()
    async def citation_lit(claim, materials, **kwargs):
        seen.append((claim.id if claim else None, kwargs["concept_catalog"] is not None))
        return BranchResult()
    monkeypatch.setattr(literature, "verify_literature", citation_lit)
    asyncio.run(dispatch.verify_claims([c], m, tmp_path / "citation", call=model))
    assert seen == [("claim", False), (None, True)]


def test_real_literal_transport_and_partial_candidates_scope(tmp_path, monkeypatch):
    c, m, t = inputs(tmp_path)
    cat = catalog(c, m, t)
    queries = []
    class Search:
        search_cfg = SimpleNamespace(provider="arxiv")
        async def search_structured(self, *, query, cutoff_date):
            wire = query.compile(start=0, limit=8)
            queries.append(query)
            response = native_response(wire.expression, [], cutoff_date)
            scope = response["search_coverage"]["queries"][0]
            scope.update(query_mode=query.version, plan_digest=query.plan_digest, query_digest=wire.digest,
                         endpoint="https://export.arxiv.org/api/query")
            scope["pages"][0]["request"] = {"url": wire.url, "params": wire.params}
            return response
    def comparison(**kw):
        data = json.loads(kw["prompt"].split("\nDATA_JSON:\n", 1)[1])
        return scientific_response(data)
    result = asyncio.run(literature.verify_literature(c, m, submission_deadline="2021-01-01", searcher=Search(),
                    reader=AsyncMock(), call=comparison, output_dir=tmp_path, search_policy="literal", concept_catalog=cat))
    assert len(queries) == 3 and all(q.version == "structured-arxiv-literal-query-v2" for q in queries)
    assert len(result.evidence) == 1 and result.evidence[0].sufficient is True
    audit = json.loads((tmp_path / "claim-search-audit.json").read_text(encoding="utf-8"))
    assert audit["scientific_search_scope"]["native_exhausted"] and audit["condition_scope_decisions"]["c1"]["supported"]
    assert parse_structured_query(queries[0].model_dump(mode="json")).compile(start=8, limit=2).params["start"] == 8
    degraded = SimpleNamespace(search=AsyncMock(side_effect=lambda *, query, cutoff_date: native_response(query, [], cutoff_date)))
    result = asyncio.run(literature.verify_literature(c, m, submission_deadline="2021-01-01", searcher=degraded,
                    reader=AsyncMock(), call=comparison, output_dir=tmp_path / "degraded", search_policy="literal", concept_catalog=cat))
    assert not result.evidence
    # Exercise the real adapter's wire-version dispatch and native paging with HTTP mocked.
    import httpx

    from fact_generation.positioning.paper_search import (
        PaperReadConfig,
        PaperSearchAdapter,
        PaperSearchConfig,
    )
    from tests.test_literature_grounded_search_plan_v2 import feed, slot
    observed = []
    async def get(self, url, **kwargs):
        from urllib.parse import parse_qs, urlsplit
        req = httpx.Request("GET", url, **kwargs)
        observed.append(str(req.url))
        params = parse_qs(urlsplit(str(req.url)).query)
        return httpx.Response(200, text=feed(2, int(params["start"][0]), int(params["max_results"][0]), entry=True), request=req)
    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    monkeypatch.setattr("fact_generation.positioning.paper_search.ARXIV_REQUESTS.slot", slot)
    monkeypatch.setattr("fact_generation.positioning.paper_search.ARXIV_REQUESTS.observe_retry_after", lambda *a: None)
    adapter = PaperSearchAdapter(PaperSearchConfig(True, "arxiv", None, None, "", 30, "", 5, page_size=1, max_pages=2, max_results=2),
                                 PaperReadConfig(None, None, "", 30))
    actual = asyncio.run(adapter.search_structured(query=queries[0], cutoff_date=literature.parse_submission_deadline("2021-01-01")))
    scope = actual["search_coverage"]["queries"][0]
    assert scope["query_mode"] == "structured-arxiv-literal-query-v2" and scope["exhausted"] is True
    assert [p["request"]["url"] for p in scope["pages"]] == observed
    assert all(p["request"] == {"url": queries[0].compile(start=p["offset"], limit=p["limit"]).url,
                               "params": queries[0].compile(start=p["offset"], limit=p["limit"]).params} for p in scope["pages"])


def test_planning_failure_and_normal_unstated_responsibility(tmp_path):
    c, m, t = inputs(tmp_path)
    def failed(**kw):
        raise RuntimeError("mock model service failed")
    cat = catalog(c, m, t, call=failed)
    search = AsyncMock()
    result = asyncio.run(literature.verify_literature(c, m, submission_deadline="2021-01-01", searcher=search,
                    reader=AsyncMock(), call=model, output_dir=tmp_path, search_policy="literal", concept_catalog=cat))
    assert result.verification_limitations and not search.called and not result.evidence
    assert result.verification_limitations[0].responsibility == "system"
    def unstated(**kw):
        request = json.loads(kw["prompt"].split("\nCONCEPT_DATA_JSON:\n", 1)[1])
        return planning_response(request, {r: "absent" for r in INTENTS})
    cat = catalog(c, m, t, call=unstated)
    result = asyncio.run(literature.verify_literature(c, m, submission_deadline="2021-01-01", searcher=search,
                    reader=AsyncMock(), call=model, output_dir=tmp_path / "unstated", search_policy="literal", concept_catalog=cat))
    assert not result.verification_limitations and not result.delivery_checks and not result.evidence
