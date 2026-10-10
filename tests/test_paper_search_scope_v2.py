"""Paging transport evidence stays independent of scientific adequacy."""

import json
import xml.etree.ElementTree as ET

import httpx
import pytest

from tests.test_paper_search_delivery_v2 import FAKE_KEY, adapter, response
from util.cutoff_date import parse_cutoff


def page(provider, index, *, total=2, empty=False):
    title = f"Fixture paper {index}"
    if provider == "arxiv":
        entry = "" if empty else f"<entry><id>https://arxiv.org/abs/1901.0000{index + 1}</id><title>{title}</title><published>2019-01-01</published></entry>"
        return (f'<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
                f'<opensearch:totalResults>{total}</opensearch:totalResults>'
                f'<opensearch:startIndex>{index}</opensearch:startIndex>'
                f'<opensearch:itemsPerPage>1</opensearch:itemsPerPage>{entry}</feed>')
    paper = {"title": title, "display_name": title, "paperId": f"S{index}", "id": f"W{index}",
             "publicationDate": "2019-01-01", "publication_date": "2019-01-01"}
    if provider == "semantic_scholar":
        return {"data": [] if empty else [paper], "total": total, "offset": index,
                **({"next": index + 1} if index + 1 < total else {})}
    return {"results": [] if empty else [paper],
            "meta": {"count": total, "next_cursor": f"cursor-{index + 1}" if index + 1 < total else None}}


def page_index(request):
    params = request.url.params
    cursor = params.get("cursor", "*")
    return int(params.get("offset") or params.get("start") or (cursor.rsplit("-", 1)[-1] if cursor != "*" else 0))


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["arxiv", "semantic_scholar", "openalex"])
@pytest.mark.parametrize("max_pages,exhausted,stop", [(1, False, "request_budget"), (2, True, "provider_exhausted")])
async def test_observed_pages_prove_exhaustion_or_show_budget_boundary(monkeypatch, provider, max_pages, exhausted, stop):
    requests = []

    def handler(request):
        requests.append(request)
        return response(provider, page(provider, page_index(request)))

    obj = adapter(monkeypatch, provider, handler)
    obj.search_cfg.page_size = 1
    obj.search_cfg.max_pages = max_pages
    obj.search_cfg.max_results = 3
    result = await obj.search(query="graph convolution")
    scope = result["search_coverage"]["queries"][0]
    assert len(requests) == max_pages and result["success"] is True
    assert len(result["papers"]) == max_pages
    assert scope["exhausted"] is exhausted and scope["stop_reason"] == stop
    assert scope["raw_count"] == max_pages and len(scope["pages"]) == max_pages
    assert all(row["provider_total"] == 2 and row["response_sha256"] for row in scope["pages"])
    assert FAKE_KEY not in json.dumps(scope)
    assert result.get("complete") is not True


@pytest.mark.asyncio
async def test_later_page_failure_retains_delivered_page_and_does_not_disable_next_query(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        if page_index(request) == 1 and request.url.params.get("query") == "first":
            return httpx.Response(503, text=FAKE_KEY)
        return response("semantic_scholar", page("semantic_scholar", page_index(request)))

    obj = adapter(monkeypatch, "semantic_scholar", handler)
    obj.search_cfg.page_size = 1
    obj.search_cfg.max_pages = 2
    obj.search_cfg.max_results = 3
    result = await obj.search(query="first")
    scope = result["search_coverage"]["queries"][0]
    assert result["success"] is False and result["partial"] is True
    assert len(result["papers"]) == 1 and scope["raw_count"] == 1
    assert scope["stop_reason"] == "request_failed" and scope["pages"][-1]["status"] == "failed"
    assert FAKE_KEY not in json.dumps(result)
    later = await obj.search(query="next")
    assert later["success"] is True and len(calls) == 4
    assert (await obj.get_search_runtime_state()).started


@pytest.mark.asyncio
async def test_remote_complete_boolean_remains_a_declaration_without_page_evidence(monkeypatch):
    obj = adapter(monkeypatch, "remote", lambda request: response("remote", {
        "success": True, "complete": True, "provider": "remote", "papers": [],
    }))
    result = await obj.search(query="graph convolution")
    scope = result["search_coverage"]["queries"][0]
    assert result["success"] is True and result["legacy_complete_declared"] is True
    assert result["question_results"][0]["legacy_complete_declared"] is True
    assert result.get("complete") is not True
    assert scope["exhausted"] is None and scope["stop_reason"] == "provider_unknown"


@pytest.mark.asyncio
async def test_cutoff_filter_does_not_change_raw_paging_exhaustion(monkeypatch):
    obj = adapter(monkeypatch, "semantic_scholar", lambda request: response("semantic_scholar", {
        "total": 30, "offset": 0, "next": 1,
        "data": [{"paperId": "S1", "title": "Future paper", "publicationDate": "2024-01-01"}],
    }))
    obj.search_cfg.page_size = 1
    obj.search_cfg.max_results = 1
    result = await obj.search(query="graph convolution", cutoff_date=parse_cutoff("2020-01-01"))
    scope = result["search_coverage"]["queries"][0]
    assert result["success"] is True and result["papers"] == []
    assert scope["raw_count"] == 1 and scope["filtered_out_count"] == 1 and scope["retained_count"] == 0
    assert scope["exhausted"] is False and scope["stop_reason"] == "result_budget"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["arxiv", "semantic_scholar", "openalex"])
async def test_provider_overrun_is_partial_and_cannot_exceed_candidate_budget(monkeypatch, provider):
    payload = page(provider, 0)
    extra = page(provider, 1)
    if provider == "arxiv":
        feed = ET.fromstring(payload)
        feed.append(ET.fromstring(extra).find("{http://www.w3.org/2005/Atom}entry"))
        payload = ET.tostring(feed, encoding="unicode")
    else:
        key = "data" if provider == "semantic_scholar" else "results"
        payload[key].extend(extra[key])
    obj = adapter(monkeypatch, provider, lambda request: response(provider, payload))
    obj.search_cfg.page_size = 1
    obj.search_cfg.max_results = 1
    result = await obj.search(query="graph convolution")
    scope = result["search_coverage"]["queries"][0]
    assert result["success"] is False and result["partial"] is True
    assert len(result["papers"]) == 1 and scope["raw_count"] == 2
    assert scope["pages"][0]["budget_dropped_count"] == 1
    assert scope["exhausted"] is None and scope["stop_reason"] == "protocol_failed"


@pytest.mark.asyncio
async def test_semantic_scholar_relevance_cap_is_an_incomplete_scope(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        offset = int(request.url.params["offset"])
        return response("semantic_scholar", {
            "total": 1_500, "offset": offset, "next": offset + 100,
            "data": [{"paperId": f"S{index}", "title": f"Paper {index}"}
                     for index in range(offset, offset + 100)],
        })

    obj = adapter(monkeypatch, "semantic_scholar", handler)
    obj.search_cfg.page_size = 100
    obj.search_cfg.max_pages = 20
    obj.search_cfg.max_results = 2_000
    result = await obj.search(query="graph convolution")
    scope = result["search_coverage"]["queries"][0]
    assert result["success"] is True and len(requests) == 10 and len(result["papers"]) == 1_000
    assert scope["exhausted"] is False and scope["stop_reason"] == "provider_limit"
    assert scope["limits"]["provider_result_cap"] == 1_000 and result.get("complete") is not True


@pytest.mark.asyncio
@pytest.mark.parametrize("total,success,exhausted,stop", [
    (0, True, True, "provider_exhausted"),
    (10, False, None, "protocol_failed"),
    (None, True, None, "provider_unknown"),
])
async def test_openalex_terminal_count_must_match_consumed_results(monkeypatch, total, success, exhausted, stop):
    meta = {"next_cursor": None, **({"count": total} if total is not None else {})}
    obj = adapter(monkeypatch, "openalex", lambda request: response("openalex", {"results": [], "meta": meta}))
    result = await obj.search(query="graph convolution")
    scope = result["search_coverage"]["queries"][0]
    assert result["success"] is success and result["papers"] == []
    assert scope["exhausted"] is exhausted and scope["stop_reason"] == stop


@pytest.mark.asyncio
async def test_later_normalization_failure_preserves_actual_received_count(monkeypatch):
    def handler(request):
        offset = page_index(request)
        payload = page("semantic_scholar", offset)
        payload["data"][0]["citationCount"] = 0 if not offset else "invalid number"
        return response("semantic_scholar", payload)

    obj = adapter(monkeypatch, "semantic_scholar", handler)
    obj.search_cfg.page_size = 1
    obj.search_cfg.max_pages = 2
    obj.search_cfg.max_results = 3
    result = await obj.search(query="graph convolution")
    scope = result["search_coverage"]["queries"][0]
    assert result["success"] is False and result["partial"] is True and len(result["papers"]) == 1
    assert scope["raw_count"] == 2 == sum(row.get("raw_count", 0) for row in scope["pages"])
    assert scope["stop_reason"] == "protocol_failed" and scope["exhausted"] is None
