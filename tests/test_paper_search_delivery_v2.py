"""Provider delivery contracts use real wire shapes and an in-memory HTTP transport."""

import json
from contextlib import asynccontextmanager
from unittest.mock import Mock

import httpx
import pytest

from fact_generation.positioning import paper_search as search_module
from tests.test_paper_search_adapter import _adapter

PROVIDERS = ("arxiv", "semantic_scholar", "openalex", "remote")
FAKE_KEY = "fixture-secret-only"
EMPTY_ATOM = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'


def wire(provider, *, empty=False):
    if provider == "arxiv":
        return EMPTY_ATOM if empty else '<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2401.12345</id><title>Fixture paper</title></entry></feed>'
    paper = {"title": "Fixture paper", "paperId": "S1", "id": "W1", "display_name": "Fixture paper"}
    return {"semantic_scholar": {"data": [] if empty else [paper]},
            "openalex": {"results": [] if empty else [paper]},
            "remote": [] if empty else [paper]}[provider]


def adapter(monkeypatch, provider, handler):
    result = _adapter()
    cfg = result.search_cfg
    cfg.enabled = True
    cfg.provider = provider
    cfg.base_url = "https://retrieval.fixture"
    cfg.health_endpoint = ""
    cfg.api_key = cfg.openalex_api_key = cfg.semantic_scholar_api_key = FAKE_KEY
    cfg.openalex_base_url = cfg.semantic_scholar_base_url = "https://retrieval.fixture"
    original_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(search_module.httpx, "AsyncClient", lambda **kw: original_client(transport=transport, **kw))

    @asynccontextmanager
    async def slot():
        yield

    monkeypatch.setattr(search_module.ARXIV_REQUESTS, "slot", slot)
    monkeypatch.setattr(search_module.ARXIV_REQUESTS, "observe_retry_after", Mock())
    return result


def response(provider, data):
    return httpx.Response(200, text=data) if provider == "arxiv" else httpx.Response(200, json=data)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_valid_empty_wire_is_a_successful_query_without_completeness(monkeypatch, provider):
    obj = adapter(monkeypatch, provider, lambda request: response(provider, wire(provider, empty=True)))
    result = await obj.search(query="empty")
    assert result["success"] is True and result["papers"] == []
    assert result["question_results"][0]["success"] is True
    assert result.get("complete") is not True


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_query_failure_preserves_batch_success_and_next_call(monkeypatch, provider):
    calls = []

    def handler(request):
        calls.append(request)
        body = json.loads(request.content) if request.method == "POST" else {}
        q = (body.get("query") or " ".join(body.get("question_list") or [])) if body else str(request.url)
        if "broken" in q:
            return httpx.Response(503, text="temporary unavailable")
        return response(provider, wire(provider))

    obj = adapter(monkeypatch, provider, handler)
    result = await obj.search(question_list=["first", "broken", "last"])
    assert result["success"] is False and result["partial"] is True
    assert len(result["papers"]) == 1
    assert [row["success"] for row in result["question_results"]] == [True, False, True]
    assert all(FAKE_KEY not in str(row) for row in result["question_results"])
    next_result = await obj.search(query="after")
    assert next_result["success"] is True and len(calls) == 4
    assert (await obj.get_search_runtime_state()).started is True
    assert result.get("complete") is not True


@pytest.mark.asyncio
@pytest.mark.parametrize("provider,data", [
    ("arxiv", '<html xmlns="http://www.w3.org/1999/xhtml"></html>'),
    ("arxiv", '<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/api/errors#bad</id><title>Error</title></entry></feed>'),
    ("semantic_scholar", {}), ("semantic_scholar", {"data": {"bad": "shape"}}),
    ("semantic_scholar", {"data": [], "error": FAKE_KEY}),
    ("openalex", {}), ("openalex", {"results": [None]}),
    ("openalex", {"results": [], "error": FAKE_KEY}),
    ("remote", {"success": True}), ("remote", [None]),
    ("remote", {"success": True, "papers": [], "error": FAKE_KEY}),
])
async def test_malformed_success_payload_is_explicit_failure(monkeypatch, provider, data):
    obj = adapter(monkeypatch, provider, lambda request: response(provider, data))
    result = await obj.search(query="bad shape")
    assert result["success"] is False and result["papers"] == []
    assert result["question_results"][0]["success"] is False
    assert FAKE_KEY not in json.dumps(result)


@pytest.mark.asyncio
async def test_remote_error_and_health_urls_cannot_echo_credentials(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={
            "success": False, "status": "unhealthy" if "/health" in str(request.url) else "failed",
            "error": "Bearer " + FAKE_KEY, "message": "https://host/?api_key=" + FAKE_KEY,
        })

    obj = adapter(monkeypatch, "remote", handler)
    result = await obj.search(query="failed")
    assert FAKE_KEY not in json.dumps(result)
    obj.search_cfg.health_endpoint = "/health?api_key=" + FAKE_KEY
    obj.search_cfg.base_url = "https://user:" + FAKE_KEY + "@retrieval.fixture"
    state = await obj.get_search_runtime_state(force_refresh=True)
    assert state.started is False and FAKE_KEY not in json.dumps(state.to_dict())
