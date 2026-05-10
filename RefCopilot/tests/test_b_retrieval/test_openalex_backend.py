"""OpenAlex search backend."""

from __future__ import annotations

import pytest

from refcopilot.cache.disk_cache import DiskCache
from refcopilot.models import Backend, Reference, SourceFormat
from refcopilot.ratelimit.openalex import OpenAlexRateLimiter
from refcopilot.search.openalex import OpenAlexBackend


_WORK_PAYLOAD = {
    "id": "https://openalex.org/W2741809807",
    "doi": "https://doi.org/10.7717/peerj.4375",
    "title": "The state of OA: a large-scale analysis of the prevalence and impact of Open Access articles",
    "display_name": "The state of OA: a large-scale analysis of the prevalence and impact of Open Access articles",
    "publication_year": 2018,
    "type": "article",
    "authorships": [
        {"author": {"display_name": "Heather Piwowar"}},
        {"author": {"display_name": "Jason Priem"}},
        {"author": {"display_name": "Vincent Larivière"}},
    ],
    "primary_location": {
        "source": {"display_name": "PeerJ", "type": "journal"}
    },
    "ids": {
        "openalex": "https://openalex.org/W2741809807",
        "doi": "https://doi.org/10.7717/peerj.4375",
    },
}


_SEARCH_PAYLOAD = {
    "meta": {"count": 2, "per_page": 5},
    "results": [
        _WORK_PAYLOAD,
        {
            # Similar title but year=2010 — dropped when caller asks for year=2018.
            "id": "https://openalex.org/W9999",
            "doi": None,
            "title": "The state of OA: an early survey",
            "display_name": "The state of OA: an early survey",
            "publication_year": 2010,
            "type": "article",
            "authorships": [{"author": {"display_name": "Older Author"}}],
            "primary_location": {"source": {"display_name": "Some Journal", "type": "journal"}},
            "ids": {"openalex": "https://openalex.org/W9999"},
        },
    ],
}


class _Resp:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.headers: dict[str, str] = {}

    def json(self):
        return self._payload


def _mock_http(payload, status: int = 200):
    calls: list[dict] = []

    def get(url, params):
        calls.append({"url": url, "params": params})
        return _Resp(status, payload)

    return get, calls


def _make_backend(tmp_path, http_get, *, max_retries: int = 3) -> OpenAlexBackend:
    return OpenAlexBackend(
        api_key="test-key",
        cache=DiskCache(tmp_path),
        rate_limiter=OpenAlexRateLimiter(
            base_interval_seconds=0.0, jitter=0.0, max_retries=max_retries
        ),
        http_get=http_get,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_empty_api_key_rejected(tmp_path):
    with pytest.raises(ValueError):
        OpenAlexBackend(api_key="")
    with pytest.raises(ValueError):
        OpenAlexBackend(api_key="   ")


def test_api_key_injected_into_query_params(tmp_path):
    get, calls = _mock_http(_WORK_PAYLOAD)
    backend = _make_backend(tmp_path, get)
    backend.lookup_by_doi("10.7717/peerj.4375")
    assert calls[0]["params"]["api_key"] == "test-key"


# ---------------------------------------------------------------------------
# Lookup by DOI
# ---------------------------------------------------------------------------


def test_lookup_by_doi_parses_fields(tmp_path):
    get, calls = _mock_http(_WORK_PAYLOAD)
    backend = _make_backend(tmp_path, get)
    rec = backend.lookup_by_doi("10.7717/peerj.4375")
    assert rec is not None
    assert rec.backend == Backend.OPENALEX
    assert rec.record_id == "W2741809807"
    assert rec.title.startswith("The state of OA")
    assert rec.authors == ["Heather Piwowar", "Jason Priem", "Vincent Larivière"]
    assert rec.year == 2018
    assert rec.venue == "PeerJ"
    assert rec.publication_venue == "PeerJ"
    assert rec.doi == "10.7717/peerj.4375"
    assert rec.url == "https://openalex.org/W2741809807"
    # URL is /works/doi:<bare_doi>
    assert calls[0]["url"].endswith("/works/doi:10.7717/peerj.4375")


def test_lookup_by_doi_strips_url_prefix(tmp_path):
    get, calls = _mock_http(_WORK_PAYLOAD)
    backend = _make_backend(tmp_path, get)
    backend.lookup_by_doi("https://doi.org/10.7717/PeerJ.4375")
    # Lower-cased, prefix stripped
    assert calls[0]["url"].endswith("/works/doi:10.7717/peerj.4375")


def test_lookup_by_doi_returns_none_on_404(tmp_path):
    def get(url, params):
        return _Resp(404, None)

    backend = _make_backend(tmp_path, get)
    assert backend.lookup_by_doi("10.0000/missing") is None


# ---------------------------------------------------------------------------
# Title search
# ---------------------------------------------------------------------------


def test_search_by_title_uses_title_search_filter(tmp_path):
    get, calls = _mock_http(_SEARCH_PAYLOAD)
    backend = _make_backend(tmp_path, get)
    results = backend.search_by_title("The state of OA", max_results=5)
    assert len(results) == 2
    assert results[0].record_id == "W2741809807"
    assert calls[0]["url"].endswith("/works")
    assert calls[0]["params"]["filter"] == "title.search:The state of OA"
    assert calls[0]["params"]["per-page"] == "5"


def test_search_strips_filter_special_chars(tmp_path):
    """Commas and pipes in titles must not break OpenAlex filter syntax."""
    get, calls = _mock_http({"results": []})
    backend = _make_backend(tmp_path, get)
    backend.search_by_title("Title, with: commas | and pipes")
    assert calls[0]["params"]["filter"] == "title.search:Title  with: commas   and pipes"


def test_search_filters_by_year_with_one_year_tolerance(tmp_path):
    get, _ = _mock_http(_SEARCH_PAYLOAD)
    backend = _make_backend(tmp_path, get)
    results = backend.search_by_title("The state of OA", year=2018)
    # 2018 keeps the W2741809807 hit; 2010 drops the noise.
    assert len(results) == 1
    assert results[0].record_id == "W2741809807"


def test_search_drops_topically_unrelated_results(tmp_path):
    """OpenAlex relevance ranking can return topical neighbours; the title-
    similarity gate must filter them out."""
    payload = {
        "meta": {"count": 1},
        "results": [
            {
                "id": "https://openalex.org/W123",
                "title": "A roadmap on computational methods in optical imaging",
                "display_name": "A roadmap on computational methods in optical imaging",
                "publication_year": 2024,
                "type": "article",
                "authorships": [{"author": {"display_name": "X. Author"}}],
                "primary_location": {"source": {"display_name": "Applied Physics B", "type": "journal"}},
            }
        ],
    }
    get, _ = _mock_http(payload)
    backend = _make_backend(tmp_path, get)
    results = backend.search_by_title(
        "Workshop on deepfake detection, localization, and interpretability"
    )
    assert results == []


def test_lookup_falls_back_to_title_search_when_no_doi(tmp_path):
    get, calls = _mock_http(_SEARCH_PAYLOAD)
    backend = _make_backend(tmp_path, get)
    ref = Reference(
        raw="x",
        source_format=SourceFormat.BIBTEX,
        title="The state of OA",
        year=2018,
    )
    out = backend.lookup(ref)
    assert len(out) >= 1
    assert calls[0]["url"].endswith("/works")
    assert calls[0]["params"]["filter"].startswith("title.search:")


def test_lookup_prefers_doi_over_title(tmp_path):
    get, calls = _mock_http(_WORK_PAYLOAD)
    backend = _make_backend(tmp_path, get)
    ref = Reference(
        raw="x",
        source_format=SourceFormat.BIBTEX,
        title="The state of OA",
        doi="10.7717/peerj.4375",
    )
    out = backend.lookup(ref)
    assert len(out) == 1
    # Single call hits the DOI endpoint, not search.
    assert len(calls) == 1
    assert calls[0]["url"].endswith("/works/doi:10.7717/peerj.4375")


def test_lookup_returns_empty_when_no_doi_or_title(tmp_path):
    get, calls = _mock_http({"results": []})
    backend = _make_backend(tmp_path, get)
    ref = Reference(raw="x", source_format=SourceFormat.BIBTEX)
    assert backend.lookup(ref) == []
    assert calls == []


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_cache_hit_avoids_http(tmp_path):
    get, calls = _mock_http(_WORK_PAYLOAD)
    backend = _make_backend(tmp_path, get)
    backend.lookup_by_doi("10.7717/peerj.4375")
    assert len(calls) == 1
    backend.lookup_by_doi("10.7717/peerj.4375")
    assert len(calls) == 1, "second call should be served from cache"


def test_cache_remembers_misses(tmp_path):
    def get(url, params):
        return _Resp(404, None)

    calls_made = []

    def counting_get(url, params):
        calls_made.append(1)
        return get(url, params)

    backend = _make_backend(tmp_path, counting_get)
    assert backend.lookup_by_doi("10.0000/missing") is None
    assert len(calls_made) == 1
    assert backend.lookup_by_doi("10.0000/missing") is None
    assert len(calls_made) == 1, "negative result should also be cached"


def test_cache_stores_raw_api_payload(tmp_path):
    """Cached value must be the verbatim API payload so parsing/filtering
    changes don't require cache invalidation."""
    get, _ = _mock_http(_WORK_PAYLOAD)
    cache = DiskCache(tmp_path)
    backend = OpenAlexBackend(
        api_key="test-key",
        cache=cache,
        rate_limiter=OpenAlexRateLimiter(base_interval_seconds=0.0, jitter=0.0),
        http_get=get,
    )
    backend.lookup_by_doi("10.7717/peerj.4375")
    cached = cache.get_api("openalex", "doi_10.7717_peerj.4375")
    assert cached == _WORK_PAYLOAD


# ---------------------------------------------------------------------------
# 429 / transient-error handling
# ---------------------------------------------------------------------------


def test_429_retries_then_succeeds(tmp_path):
    calls: list[dict] = []
    responses = [
        _Resp(429, None),
        _Resp(429, None),
        _Resp(200, _WORK_PAYLOAD),
    ]
    responses[0].headers = {"Retry-After": "0"}
    responses[1].headers = {"Retry-After": "0"}

    def get(url, params):
        calls.append({"url": url, "params": params})
        return responses[len(calls) - 1]

    backend = _make_backend(tmp_path, get)
    rec = backend.lookup_by_doi("10.7717/peerj.4375")
    assert rec is not None
    assert rec.record_id == "W2741809807"
    assert len(calls) == 3


def test_transient_error_does_not_poison_cache(tmp_path):
    def get(url, params):
        return _Resp(429, None)

    backend = _make_backend(tmp_path, get, max_retries=1)
    assert backend.search_by_title("any title here") == []
    cached = DiskCache(tmp_path).get_api("openalex", "title_any title here_")
    assert cached is None


# ---------------------------------------------------------------------------
# Source-type venue filtering
# ---------------------------------------------------------------------------


def test_repository_source_type_drops_venue(tmp_path):
    """Works hosted only on a 'repository' (e.g. an arXiv-style preprint
    server) shouldn't claim that as a publication venue."""
    payload = {
        "id": "https://openalex.org/W42",
        "doi": "https://doi.org/10.48550/arXiv.1706.03762",
        "title": "Attention Is All You Need",
        "display_name": "Attention Is All You Need",
        "publication_year": 2017,
        "type": "preprint",
        "authorships": [{"author": {"display_name": "A. Vaswani"}}],
        "primary_location": {
            "source": {"display_name": "arXiv (Cornell University)", "type": "repository"}
        },
    }
    get, _ = _mock_http(payload)
    backend = _make_backend(tmp_path, get)
    rec = backend.lookup_by_doi("10.48550/arxiv.1706.03762")
    assert rec is not None
    # Title/authors/year still come through; venue suppressed.
    assert rec.title == "Attention Is All You Need"
    assert rec.year == 2017
    assert rec.venue is None
    assert rec.publication_venue is None


# ---------------------------------------------------------------------------
# Retraction flag
# ---------------------------------------------------------------------------


def test_lookup_captures_is_retracted_true(tmp_path):
    payload = {**_WORK_PAYLOAD, "is_retracted": True}
    get, _ = _mock_http(payload)
    backend = _make_backend(tmp_path, get)
    rec = backend.lookup_by_doi("10.7717/peerj.4375")
    assert rec is not None
    assert rec.is_retracted is True


def test_lookup_is_retracted_defaults_false_when_absent(tmp_path):
    # _WORK_PAYLOAD has no is_retracted key — should coerce to False.
    get, _ = _mock_http(_WORK_PAYLOAD)
    backend = _make_backend(tmp_path, get)
    rec = backend.lookup_by_doi("10.7717/peerj.4375")
    assert rec is not None
    assert rec.is_retracted is False


def test_lookup_is_retracted_coerces_none_to_false(tmp_path):
    payload = {**_WORK_PAYLOAD, "is_retracted": None}
    get, _ = _mock_http(payload)
    backend = _make_backend(tmp_path, get)
    rec = backend.lookup_by_doi("10.7717/peerj.4375")
    assert rec is not None
    assert rec.is_retracted is False


def test_select_fields_request_includes_is_retracted(tmp_path):
    get, calls = _mock_http(_WORK_PAYLOAD)
    backend = _make_backend(tmp_path, get)
    backend.lookup_by_doi("10.7717/peerj.4375")
    assert "is_retracted" in calls[0]["params"]["select"]
