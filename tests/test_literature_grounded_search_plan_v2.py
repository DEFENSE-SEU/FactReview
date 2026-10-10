"""Six bounded contracts; retrieval, model, transport and execution stay mocked."""

import copy
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from fact_generation.positioning.paper_search import PaperReadConfig, PaperSearchAdapter, PaperSearchConfig
from llm.client import LLMConfig
from schemas.claim import Claim, ClaimLocation, Condition
from schemas.materials import MaterialBlock, SharedMaterials
from tests.literature_scope_fixtures import native_response, scientific_response
from tests.test_literature_service_responsibility_v2 import read_response
from verification import literature


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("External boundaries must be explicit mocks")

    for target in (
        "requests.sessions.Session.request",
        "httpx.Client.send",
        "httpx.AsyncClient.send",
        "subprocess.run",
        "verification.literature.llm_json",
        "verification.literature._default_adapter",
    ):
        monkeypatch.setattr(target, forbidden)
    monkeypatch.setattr(literature, "resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None))


def inputs(tmp_path, *, second=False, incomplete=False):
    texts = ["The first method uses pseudo-labels for semi-supervised learning and evaluation by accuracy."]
    if incomplete:
        texts[0] = "The first method uses pseudo-labels for semi-supervised learning."
    if second:
        texts.append("The first graph neural network method applies to node classification.")
    md = "\n".join(texts)
    p = tmp_path / "paper.md"
    p.write_text(md, encoding="utf-8")
    blocks, start = [], 0
    for i, text in enumerate(texts):
        blocks.append(
            MaterialBlock(
                id=f"b{i}",
                text=text,
                loc=ClaimLocation(page=i + 1, char_start=start, char_end=start + len(text)),
            )
        )
        start += len(text) + 1
    conditions = [Condition(id=f"c{i + 1}", description="Novelty: " + text) for i, text in enumerate(texts)]
    refs = [
        {"source_block_id": b.id, "source_quote": b.text, "loc": b.loc, "covered": [conditions[i].id]}
        for i, b in enumerate(blocks)
    ]
    claim = Claim(
        id="claim",
        text="The first method introduces a novel approach.",
        source_block_id=blocks[0].id,
        source_quote=blocks[0].text,
        loc=blocks[0].loc,
        source_refs=refs,
        conditions=conditions,
        needs=["Literature"],
    )
    materials = SharedMaterials(
        paper_key="fixture",
        title="Our unrelated manuscript",
        abstract="Original submission scope.",
        source_pdf=str(tmp_path / "absent.pdf"),
        markdown=md,
        markdown_path=str(p),
        content_list_path="",
        provider="fixture",
        blocks=blocks,
    )
    return claim, materials


def plan_for(claim, materials):
    from verification.literature_search_plan import build_grounded_plan

    return build_grounded_plan(claim, materials, novelty_ids={c.id for c in claim.conditions})


def typed(row):
    from fact_generation.positioning.structured_query import StructuredPaperQuery

    return StructuredPaperQuery.model_validate(row)


def adapter(*, pages=2, results=2):
    return PaperSearchAdapter(
        PaperSearchConfig(
            True, "arxiv", None, None, "", 30, "", 5, page_size=1, max_pages=pages, max_results=results
        ),
        PaperReadConfig(None, None, "", 30),
    )


@asynccontextmanager
async def slot():
    yield


def feed(total, start, limit, *, entry=False, date="2020-01-01", malformed=False):
    if malformed:
        return "<broken"
    item = (
        f"<entry><id>http://arxiv.org/abs/2001.0000{start + 1}</id><title>Independent source {start}</title><published>{date}T00:00:00Z</published><summary>Independent scientific result.</summary></entry>"
        if entry
        else ""
    )
    return f'<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"><opensearch:totalResults>{total}</opensearch:totalResults><opensearch:startIndex>{start}</opensearch:startIndex><opensearch:itemsPerPage>{limit}</opensearch:itemsPerPage>{item}</feed>'


async def test_located_concepts_and_safe_unknown_terms(tmp_path):
    claim, material = inputs(tmp_path)
    material.blocks[0].text = material.blocks[0].text.replace("pseudo-labels", "pseudo\u2011labels")
    material.markdown = material.blocks[0].text
    material.blocks[0].loc.char_end = len(material.markdown)
    claim.source_quote = material.markdown
    claim.source_refs[0].source_quote = material.markdown
    claim.source_refs[0].loc = material.blocks[0].loc
    claim.loc = material.blocks[0].loc
    Path = __import__("pathlib").Path
    Path(material.markdown_path).write_text(material.markdown, encoding="utf-8")
    plan = plan_for(claim, material)
    assert {c["family"] for c in plan["concepts"]} == {"pseudo_label", "semi_supervised", "accuracy"}
    for c in plan["concepts"]:
        span = c["block_local_span"]
        assert material.blocks[0].text[span["start"] : span["end"]] == c["quote"]
        assert c["condition_ids"] == ["c1"] and c["loc"]["page"] == 1 and len(c["term_sha256"]) == 64
    claim.conditions[0].settings["dataset"] = "Imaginary author AND au:Smith https://unsafe.invalid"
    assert plan_for(claim, material)["concepts"] == plan["concepts"]
    material.blocks[0].kind = "abstract"
    assert plan_for(claim, material)["concepts"] == plan["concepts"]
    material.blocks[0].kind = "metadata"
    assert plan_for(claim, material)["concepts"] == []
    material.blocks[0].kind = "text"
    forged = copy.deepcopy(plan["queries"][0])
    forged["groups"][0][0]["phrase"] = "Smith OR au:Smith"
    with pytest.raises(ValueError):
        typed(forged)
    claim.source_quote = "invented pseudo-label"
    assert plan_for(claim, material)["conditions"]["c1"]["eligible"] is False
    duplicate, duplicated = inputs(tmp_path)
    duplicate.loc = duplicate.loc.model_copy(deep=True)
    duplicate.source_refs[0].loc = duplicate.loc.model_copy(deep=True)
    duplicated.blocks[0].text += " " + duplicated.blocks[0].text
    duplicated.markdown = duplicated.blocks[0].text
    duplicated.blocks[0].loc.char_end = len(duplicated.markdown)
    repeated = plan_for(duplicate, duplicated)
    assert repeated["conditions"]["c1"]["eligible"] is False
    assert any("ambiguous" in reason for reason in repeated["source_errors"])


async def test_actual_roles_partial_and_distinct_query_ast(tmp_path):
    claim, material = inputs(tmp_path, second=True)
    plan = plan_for(claim, material)
    assert len(plan["queries"]) == 3 and plan["conditions"]["c1"]["eligible"] is True
    assert plan["conditions"]["c2"]["eligible"] is False
    assert "evaluation protocol baseline" in plan["conditions"]["c2"]["missing_roles"]
    assert len({typed(q).compile(start=0, limit=1).expression for q in plan["queries"]}) == 3
    claim, material = inputs(tmp_path, incomplete=True)
    partial = plan_for(claim, material)
    assert len(partial["queries"]) == 2 and partial["conditions"]["c1"]["eligible"] is False
    assert [q["intent"] for q in partial["queries"]] == ["mechanism", "target setting"]
    shared, sources = inputs(tmp_path, second=True)
    b = sources.blocks[1]
    b.text = "The first method uses pseudo-labels for node classification."
    b.loc.char_end = b.loc.char_start + len(b.text)
    sources.markdown = sources.blocks[0].text + "\n" + b.text
    shared.source_refs[1].source_quote = b.text
    shared.source_refs[1].loc = b.loc
    shared.conditions[1].description = "Novelty: " + b.text
    shared_plan = plan_for(shared, sources)
    concepts = {c["concept_id"]: c for c in shared_plan["concepts"]}
    mechanism_query = shared_plan["queries"][0]
    for cid in ("c1", "c2"):
        witnesses = mechanism_query["participation"][cid]["mechanism"]
        assert witnesses and all(cid in concepts[sid]["condition_ids"] for sid in witnesses)
        assert all(concepts[sid]["source_block_id"] == ("b0" if cid == "c1" else "b1") for sid in witnesses)
    assert len(mechanism_query["groups"][0][0]["source_concept_ids"]) == 2
    from verification.literature_search_plan import condition_eligible, plan_digest

    corrupted = copy.deepcopy(shared_plan)
    corrupted["queries"][0]["participation"]["c2"]["mechanism"] = mechanism_query["participation"]["c1"][
        "mechanism"
    ]
    corrupted["digest"] = plan_digest(corrupted)
    for q in corrupted["queries"]:
        q["plan_digest"] = corrupted["digest"]
    assert condition_eligible(corrupted, "c2") is False


async def test_typed_http_and_legacy_wire_keep_distinct_contracts(tmp_path, monkeypatch):
    claim, material = inputs(tmp_path)
    query = typed(plan_for(claim, material)["queries"][1])
    observed = []

    async def get(self, url, **kwargs):
        request = httpx.Request("GET", url, **kwargs)
        observed.append(str(request.url))
        from urllib.parse import parse_qs, urlsplit

        params = parse_qs(urlsplit(str(request.url)).query)
        start = int(params["start"][0])
        limit = int(params["max_results"][0])
        return httpx.Response(200, text=feed(2, start, limit, entry=True), request=request)

    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    monkeypatch.setattr("fact_generation.positioning.paper_search.ARXIV_REQUESTS.slot", slot)
    monkeypatch.setattr(
        "fact_generation.positioning.paper_search.ARXIV_REQUESTS.observe_retry_after", lambda *a: None
    )
    a = adapter()
    result = await a.search_structured(
        query=query, cutoff_date=literature.parse_submission_deadline("2022-01-01")
    )
    scope = result["search_coverage"]["queries"][0]
    assert scope["translated_query"] == query.compile(start=0, limit=1).expression
    assert scope["plan_digest"] == query.plan_digest and scope["query_mode"] == "structured-arxiv-query-v1"
    assert scope["exhausted"] is True and scope["raw_count"] == 2
    assert [p["request"]["url"] for p in scope["pages"]] == observed
    assert all("sortBy=relevance" in u and "sortOrder=descending" in u for u in observed)
    observed.clear()
    legacy = await adapter(pages=1, results=1).search(query='ti:"pseudo-label" AND abs:"semi-supervised"')
    assert (
        "sortBy" not in observed[0]
        and legacy["search_coverage"]["queries"][0]["translated_query"]
        == "ti pseudo-label abs semi-supervised"
    )


@pytest.mark.parametrize("failure", [False, True])
async def test_typed_budget_or_later_failure_keeps_observed_rows(tmp_path, monkeypatch, failure):
    claim, material = inputs(tmp_path)
    query = typed(plan_for(claim, material)["queries"][0])

    async def get(self, url, **kwargs):
        from urllib.parse import parse_qs, urlsplit

        request = httpx.Request("GET", url, **kwargs)
        p = parse_qs(urlsplit(str(request.url)).query)
        start = int(p["start"][0])
        limit = int(p["max_results"][0])
        return httpx.Response(
            200, text=feed(2, start, limit, entry=True, malformed=bool(start and failure)), request=request
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    monkeypatch.setattr("fact_generation.positioning.paper_search.ARXIV_REQUESTS.slot", slot)
    monkeypatch.setattr(
        "fact_generation.positioning.paper_search.ARXIV_REQUESTS.observe_retry_after", lambda *a: None
    )
    result = await adapter(pages=2 if failure else 1).search_structured(
        query=query, cutoff_date=literature.parse_submission_deadline("2022-01-01")
    )
    s = result["search_coverage"]["queries"][0]
    assert len(result["papers"]) == 1 and s["raw_count"] == 1 and s["exhausted"] is not True
    assert s["stop_reason"] == ("request_failed" if failure else "request_budget")
    assert result["success"] is not failure and result["partial"] is failure


def structured_response(query, cutoff):
    compiled = query.compile(start=0, limit=8)
    r = native_response(compiled.expression, [], cutoff)
    s = r["search_coverage"]["queries"][0]
    s.update(
        query_mode=query.version,
        plan_digest=query.plan_digest,
        query_digest=compiled.digest,
        endpoint="https://export.arxiv.org/api/query",
    )
    s["pages"][0]["request"] = {"url": compiled.url, "params": compiled.params}
    r["question_results"][0]["search_coverage"] = copy.deepcopy(s)
    return r


@pytest.mark.parametrize("wrong_digest", [False, True])
async def test_grounded_absence_remains_condition_and_scientific_bound(tmp_path, wrong_digest):
    claim, material = inputs(tmp_path, second=True)

    async def search_structured(*, query, cutoff_date):
        return structured_response(query, cutoff_date)

    searcher = SimpleNamespace(
        search_cfg=SimpleNamespace(provider="arxiv"),
        search_structured=AsyncMock(side_effect=search_structured),
    )

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"].split("DATA_JSON:\n", 1)[1])
        r = scientific_response(payload)
        if wrong_digest:
            r["search_adequacy"]["scope_digest"] = "0" * 64
        return r

    reader = SimpleNamespace(read_papers=AsyncMock())
    result = await literature.verify_literature(
        claim,
        material,
        submission_deadline="2022-01-01",
        searcher=searcher,
        reader=reader,
        call=call,
        output_dir=tmp_path / "audit",
        search_policy="grounded",
    )
    assert searcher.search_structured.await_count == 3 and reader.read_papers.await_count == 0
    assert {
        cid for e in result.evidence if e.sufficient and e.direction == "support" for cid in e.covered
    } == (set() if wrong_digest else {"c1"})
    saved = json.loads((tmp_path / "audit/claim-search-audit.json").read_text("utf-8"))
    assert saved["scientific_search_scope"]["version"] == "literature-search-scope-v2"
    assert saved["condition_scope_decisions"]["c2"]["grounded_plan_eligible"] is False


async def test_capacity_degradation_and_explicit_structured_failure(tmp_path):
    claim, material = inputs(tmp_path)
    paper = {
        "id": "2001.00001",
        "arxiv_id": "2001.00001",
        "title": "Independent prior result",
        "published": "2020-01-01",
    }
    searcher = SimpleNamespace(
        search_cfg=SimpleNamespace(provider="openalex"),
        search=AsyncMock(return_value={"success": True, "papers": [paper], "complete": True}),
    )
    reader = SimpleNamespace(read_papers=AsyncMock(side_effect=read_response))

    def comparison(**kwargs):
        payload = json.loads(kwargs["prompt"].split("DATA_JSON:\n", 1)[1])
        r = scientific_response(
            payload,
            [
                {
                    "paper_id": "2001.00001",
                    "purpose": "novelty",
                    "relation": "same",
                    "quote": "The bound holds under the stated assumptions.",
                    "covered": ["c1"],
                    "mechanism": "Same concrete mechanism",
                    "setting": "Same target setting",
                    "protocol": "Same evaluation protocol",
                    "note": "Grounded same prior method.",
                }
            ],
        )
        return r

    # read_response has a positional id signature; keep actual reader protocol mock explicit.
    reader.read_papers.side_effect = lambda items: read_response(items[0]["id"])
    result = await literature.verify_literature(
        claim,
        material,
        searcher=searcher,
        reader=reader,
        call=comparison,
        submission_deadline="2022-01-01",
        output_dir=tmp_path / "degraded",
        search_policy="grounded",
    )
    assert any(e.direction == "flaw" and e.sufficient for e in result.evidence)
    assert not any(e.direction == "support" and e.sufficient for e in result.evidence)
    assert result.delivery_checks == [] and result.verification_limitations == []
    saved = json.loads((tmp_path / "degraded/claim-search-audit.json").read_text("utf-8"))
    assert saved["retrieval_routing"]["transport"] == "degraded_legacy"
    failed = SimpleNamespace(
        search_cfg=SimpleNamespace(provider="arxiv"),
        search_structured=AsyncMock(side_effect=RuntimeError("mock transport failure")),
    )
    result = await literature.verify_literature(
        claim,
        material,
        searcher=failed,
        reader=reader,
        call=comparison,
        submission_deadline="2022-01-01",
        output_dir=tmp_path / "failure",
        search_policy="grounded",
    )
    assert result.evidence == [] and result.verification_limitations
    assert all(v.condition_ids == ["c1"] for v in result.verification_limitations)
