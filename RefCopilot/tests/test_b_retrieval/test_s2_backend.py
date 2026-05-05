"""Semantic Scholar search backend."""

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


def test_429_exhausted_returns_none_without_poisoning_backend(tmp_path):
    """429 retries-exhausted means 'slow down' — the next reference may well
    succeed, so we must NOT permanently fail the backend."""
    def http_get(url, params, headers):
        return _Resp(429, {}, headers={"Retry-After": "0"})

    backend = _make_backend(http_get, tmp_path=tmp_path)
    rec = backend._fetch_by_id("DOI", "10.5555/foo")
    assert rec is None
    # Backend stays available for the next reference.
    assert backend._failed is False
    # But the most recent call was flagged as transient so callers skip caching.
    assert backend._last_was_transient is True


def test_persistent_5xx_poisons_backend(tmp_path):
    """Network errors / 5xx exhaustion DO short-circuit — the service is down."""
    def http_get(url, params, headers):
        return _Resp(503, {})

    backend = _make_backend(http_get, tmp_path=tmp_path)
    backend.rate_limiter = SemanticScholarRateLimiter(
        base_interval_seconds=0.0, max_retries=1, jitter=0.0
    )
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


def _payload_with_title(title: str, paper_id: str = "xyz") -> dict[str, Any]:
    return {
        "paperId": paper_id,
        "title": title,
        "authors": [],
        "year": 2025,
        "venue": "Some Venue",
        "publicationVenue": None,
        "journal": None,
        "externalIds": {},
        "url": f"https://www.semanticscholar.org/paper/{paper_id}",
    }


def test_search_match_drops_dissimilar_top_hit(tmp_path):
    """If /paper/search/match returns a topical neighbour rather than the
    cited paper, drop it instead of treating it as a confident match."""
    def http_get(url, params, headers):
        if "/paper/search/match" in url:
            return _Resp(
                200,
                {"data": [_payload_with_title("Populations cholerae Vibrio Boundaries")]},
            )
        # Relevance fallback returns nothing relevant either.
        return _Resp(200, {"data": []})

    backend = _make_backend(http_get, tmp_path=tmp_path)
    out = backend.lookup(
        Reference(
            raw="r",
            source_format=SourceFormat.BIBTEX,
            title="AssetOpsBench: CODS 2025 Competition Release",
        )
    )
    assert out == []


def test_search_relevance_filters_unrelated_results(tmp_path):
    """Relevance ranking can return wildly off-topic papers near the top —
    the similarity gate must drop them."""
    def http_get(url, params, headers):
        if "/paper/search/match" in url:
            return _Resp(200, {"data": []})
        # Two unrelated papers, one barely-related.
        return _Resp(
            200,
            {
                "data": [
                    _payload_with_title(
                        "DockerGate: Automated Seccomp Policy Generation for Docker Images",
                        "noise1",
                    ),
                    _payload_with_title(
                        "Security Vulnerabilities in Docker Images: A Cross-Tag Study",
                        "noise2",
                    ),
                ]
            },
        )

    backend = _make_backend(http_get, tmp_path=tmp_path)
    out = backend.lookup(
        Reference(
            raw="r",
            source_format=SourceFormat.BIBTEX,
            title="AssetOpsBench Docker images",
        )
    )
    # Both results are topically near "docker images" but neither is the
    # AssetOpsBench paper — must be filtered out.
    assert out == []


def test_search_keeps_real_typo_variant(tmp_path):
    """Casing/punctuation drift between cited and canonical title must pass."""
    def http_get(url, params, headers):
        if "/paper/search/match" in url:
            return _Resp(
                200,
                {
                    "data": [
                        _payload_with_title(
                            "MathArena: Evaluating LLMs on Uncontaminated Math Competitions",
                            "matharena",
                        )
                    ]
                },
            )
        return _Resp(200, {"data": []})

    backend = _make_backend(http_get, tmp_path=tmp_path)
    out = backend.lookup(
        Reference(
            raw="r",
            source_format=SourceFormat.BIBTEX,
            title="Math-arena: Evaluating llms on uncontaminated math competitions",
        )
    )
    assert len(out) == 1
    assert out[0].record_id == "matharena"


def test_transient_error_does_not_poison_search_cache(tmp_path):
    """If the relevance-search endpoint hits 429 retries until exhausted, the
    backend must NOT cache the empty result — otherwise the next run would
    keep returning [] without re-trying the live API."""
    def http_get(url, params, headers):
        if "/paper/search/match" in url:
            return _Resp(404, {})
        # /paper/search → always 429 → eventually exhausted → returns None
        return _Resp(429, {}, headers={"Retry-After": "0"})

    backend = SemanticScholarBackend(
        api_key="test-key",
        cache=DiskCache(tmp_path),
        rate_limiter=SemanticScholarRateLimiter(
            base_interval_seconds=0.0, max_retries=1, jitter=0.0
        ),
        http_get=http_get,
    )
    out = backend.lookup(
        Reference(raw="r", source_format=SourceFormat.BIBTEX, title="Some Real Paper")
    )
    assert out == []
    # Cache must be empty for the relevance key so a future run retries.
    cached = DiskCache(tmp_path).get_api(
        "semantic_scholar", "search_some real paper_"
    )
    assert cached is None


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
