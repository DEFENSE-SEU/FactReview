"""Test B.4 — Semantic Scholar search backend."""

from __future__ import annotations

from typing import Any

import pytest

from refcopilot.cache.disk_cache import DiskCache
from refcopilot.models import Backend, Reference, SourceFormat
from refcopilot.ratelimit.semantic_scholar import SemanticScholarRateLimiter
from refcopilot.search.semantic_scholar import SemanticScholarBackend


class _Resp:
    def __init__(self, status_code: int, payload: Any, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self) -> Any:
        return self._payload


def _attention_payload(paper_id: str = "abc123") -> dict[str, Any]:
    return {
        "paperId": paper_id,
        "title": "Attention Is All You Need",
        "authors": [
            {"name": "Ashish Vaswani"},
            {"name": "Noam Shazeer"},
        ],
        "year": 2017,
        "venue": "NIPS 2017",
        "publicationVenue": {"name": "Conference on Neural Information Processing Systems"},
        "journal": {"name": ""},
        "externalIds": {
            "DOI": "10.5555/3295222.3295349",
            "ArXiv": "1706.03762",
        },
        "url": "https://www.semanticscholar.org/paper/abc123",
    }


def _make_backend(http_get, *, tmp_path) -> SemanticScholarBackend:
    return SemanticScholarBackend(
        api_key="test-key",
        cache=DiskCache(tmp_path),
        rate_limiter=SemanticScholarRateLimiter(base_interval_seconds=0.0, jitter=0.0),
        http_get=http_get,
    )


def test_endpoint_priority_doi_first(tmp_path):
    calls: list[str] = []

    def http_get(url, params, headers):
        calls.append(url)
        return _Resp(200, _attention_payload())

    backend = _make_backend(http_get, tmp_path=tmp_path)
    ref = Reference(
        raw="x",
        source_format=SourceFormat.BIBTEX,
        title="Attention Is All You Need",
        doi="10.5555/3295222.3295349",
        arxiv_id="1706.03762",
    )
    out = backend.lookup(ref)
    assert len(out) == 1
    assert any("/paper/DOI:" in c for c in calls)
    # Should NOT have called search/match or search since DOI hit
    assert not any("/paper/search" in c for c in calls)


def test_endpoint_priority_arxiv_second(tmp_path):
    calls: list[str] = []

    def http_get(url, params, headers):
        calls.append(url)
        return _Resp(200, _attention_payload())

    backend = _make_backend(http_get, tmp_path=tmp_path)
    ref = Reference(
        raw="x",
        source_format=SourceFormat.BIBTEX,
        title="Attention Is All You Need",
        arxiv_id="1706.03762",
    )
    backend.lookup(ref)
    assert any("/paper/ARXIV:1706.03762" in c for c in calls)


def test_endpoint_search_match_third(tmp_path):
    calls: list[str] = []

    def http_get(url, params, headers):
        calls.append(url)
        if "/paper/search/match" in url:
            return _Resp(200, {"data": [_attention_payload()]})
        if "/paper/search" in url and "/match" not in url:
            return _Resp(200, {"data": [_attention_payload()]})
        return _Resp(200, _attention_payload())

    backend = _make_backend(http_get, tmp_path=tmp_path)
    ref = Reference(
        raw="x",
        source_format=SourceFormat.BIBTEX,
        title="Attention Is All You Need",
    )
    backend.lookup(ref)
    # The first call when only title exists should be search/match.
    assert "/paper/search/match" in calls[0]


def test_search_match_failure_falls_back_to_search(tmp_path):
    calls: list[str] = []

    def http_get(url, params, headers):
        calls.append(url)
        if "/paper/search/match" in url:
            return _Resp(200, {"data": []})
        return _Resp(200, {"data": [_attention_payload()]})

    backend = _make_backend(http_get, tmp_path=tmp_path)
    ref = Reference(
        raw="x", source_format=SourceFormat.BIBTEX, title="Attention Is All You Need"
    )
    out = backend.lookup(ref)
    assert len(out) >= 1
    assert any("/paper/search" in c and "/match" not in c for c in calls)


def test_429_with_retry_after(tmp_path):
    invocations = {"n": 0}

    def http_get(url, params, headers):
        invocations["n"] += 1
        if invocations["n"] == 1:
            return _Resp(429, {}, headers={"Retry-After": "0"})
        return _Resp(200, _attention_payload())

    backend = _make_backend(http_get, tmp_path=tmp_path)
    rec = backend._fetch_by_id("DOI", "10.5555/foo")
    assert rec is not None
    assert invocations["n"] == 2


def test_429_exhausted_returns_none(tmp_path):
    def http_get(url, params, headers):
        return _Resp(429, {}, headers={"Retry-After": "0"})

    backend = _make_backend(http_get, tmp_path=tmp_path)
    rec = backend._fetch_by_id("DOI", "10.5555/foo")
    assert rec is None
    assert backend._failed is True


def test_payload_to_record(tmp_path):
    def http_get(url, params, headers):
        return _Resp(200, _attention_payload())

    backend = _make_backend(http_get, tmp_path=tmp_path)
    rec = backend._fetch_by_id("DOI", "10.5555/3295222.3295349")
    assert rec.backend == Backend.SEMANTIC_SCHOLAR
    assert rec.doi == "10.5555/3295222.3295349"
    assert rec.arxiv_id == "1706.03762"
    assert rec.year == 2017
    assert rec.publication_venue == "Conference on Neural Information Processing Systems"
    assert rec.s2_paper_id == "abc123"


def test_short_circuits_after_failure(tmp_path):
    invocations = {"n": 0}

    def http_get(url, params, headers):
        invocations["n"] += 1
        return _Resp(500, {})

    backend = _make_backend(http_get, tmp_path=tmp_path)
    backend.rate_limiter = SemanticScholarRateLimiter(base_interval_seconds=0.0, max_retries=1, jitter=0.0)
    backend._fetch_by_id("DOI", "10.x/y")
    assert backend._failed is True
    n_after_first = invocations["n"]

    out = backend.lookup(
        Reference(raw="r", source_format=SourceFormat.BIBTEX, title="Some Title")
    )
    assert out == []
    assert invocations["n"] == n_after_first
