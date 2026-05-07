"""Tests for the publication-date cutoff path in SemanticScholarAdapter.

These exercise both the server-side ``year=`` query parameter and the
client-side double-check filter. The HTTP layer is mocked so the tests stay
hermetic.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from fact_generation.positioning.semantic_scholar import (
    SemanticScholarAdapter,
    SemanticScholarConfig,
)
from util.cutoff_date import parse_cutoff


def _make_adapter() -> SemanticScholarAdapter:
    return SemanticScholarAdapter(
        SemanticScholarConfig(
            enabled=True,
            base_url="https://api.semanticscholar.org/graph/v1",
            api_key=None,
            timeout_seconds=10,
            top_k=8,
        )
    )


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    def __init__(self, payload: dict[str, Any], capture: dict[str, Any]) -> None:
        self._payload = payload
        self._capture = capture

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def get(self, url: str, params: dict[str, str], headers: dict[str, str]) -> _FakeResponse:
        self._capture["url"] = url
        self._capture["params"] = dict(params)
        self._capture["headers"] = dict(headers)
        return _FakeResponse(self._payload)


def _patch_httpx(monkeypatch: pytest.MonkeyPatch, *, payload: dict[str, Any]) -> dict[str, Any]:
    capture: dict[str, Any] = {}

    def factory(*args: object, **kwargs: object) -> _FakeClient:
        return _FakeClient(payload, capture)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return capture


def test_cutoff_sends_year_param_and_filters_client_side(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "data": [
            {"title": "Old paper", "year": 2018, "citationCount": 50, "venue": "X"},
            # Server year filter is year-only; suppose S2 erroneously returns
            # one row past cutoff. Client-side filter must catch it.
            {"title": "Future paper", "year": 2025, "citationCount": 10, "venue": "Y"},
            {"title": "Borderline", "year": 2022, "citationCount": 30, "venue": "Z"},
            {"title": "No-year paper", "year": None, "citationCount": 5, "venue": "Q"},
        ]
    }
    capture = _patch_httpx(monkeypatch, payload=payload)
    adapter = _make_adapter()
    cutoff = parse_cutoff("2022")

    result = asyncio.run(
        adapter.search_related(query="positioning analysis", cutoff_date=cutoff)
    )

    assert capture["params"]["year"] == "-2022"
    assert result["success"] is True
    titles = [p["title"] for p in result["papers"]]
    assert "Future paper" not in titles
    assert "Borderline" in titles
    assert "Old paper" in titles
    # Papers with no year are kept (we don't silently drop missing metadata).
    assert "No-year paper" in titles
    assert result["filtered_out_count"] == 1
    assert result["cutoff_date"] == {
        "value": "2022",
        "precision": "year",
        "source": "user",
    }


def test_no_cutoff_omits_year_param_and_filtered_count(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "data": [
            {"title": "Old paper", "year": 2018, "citationCount": 50, "venue": "X"},
            {"title": "Recent paper", "year": 2025, "citationCount": 10, "venue": "Y"},
        ]
    }
    capture = _patch_httpx(monkeypatch, payload=payload)
    adapter = _make_adapter()

    result = asyncio.run(adapter.search_related(query="positioning analysis"))

    assert "year" not in capture["params"]
    assert result["cutoff_date"] is None
    titles = [p["title"] for p in result["papers"]]
    assert "Recent paper" in titles
    assert "Old paper" in titles
    assert "filtered_out_count" not in result
