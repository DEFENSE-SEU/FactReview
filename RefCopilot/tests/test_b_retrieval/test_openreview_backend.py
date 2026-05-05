"""OpenReview search backend."""

from __future__ import annotations

from refcopilot.cache.disk_cache import DiskCache
from refcopilot.models import Backend, Reference, SourceFormat
from refcopilot.ratelimit.openreview import OpenReviewRateLimiter
from refcopilot.search.openreview import OpenReviewBackend, _extract_forum_id


_NOTE_PAYLOAD = {
    "notes": [
        {
            "id": "BeFjjyzWOJ",
            "forum": "BeFjjyzWOJ",
            "cdate": 1746980299278,  # ~2025-05
            "content": {
                "title": {
                    "value": (
                        "A Technical Report on “Erasing the Invisible”: "
                        "The 2024 NeurIPS Competition on Stress Testing Image Watermarks"
                    )
                },
                "authors": {
                    "value": [
                        "Mucong Ding",
                        "Bang An",
                        "Tahseen Rabbani",
                    ]
                },
                "venue": {"value": "NeurIPS 2025 Datasets and Benchmarks Track poster"},
                "venueid": {"value": "NeurIPS.cc/2025/Datasets_and_Benchmarks_Track"},
            },
        }
    ]
}


_SEARCH_PAYLOAD = {
    "notes": [
        {
            "id": "BeFjjyzWOJ",
            "forum": "BeFjjyzWOJ",
            "content": {
                "title": {
                    "value": "A Technical Report on Erasing the Invisible"
                },
                "authors": {"value": ["Mucong Ding", "Bang An"]},
                "venue": {"value": "NeurIPS 2025 Datasets and Benchmarks Track poster"},
                "venueid": {"value": "NeurIPS.cc/2025/Datasets_and_Benchmarks_Track"},
            },
        },
        {
            # Similar title (passes the similarity gate) but year=2019, so the
            # year-tolerance filter drops it when caller specifies year=2025.
            "id": "OldVariant",
            "forum": "OldVariant",
            "content": {
                "title": {"value": "Erasing the Invisible: an early study"},
                "authors": {"value": ["Other Author"]},
                "venue": {"value": "ICLR 2019"},
                "venueid": {"value": "ICLR.cc/2019/Conference"},
            },
        },
    ]
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


# ---------------------------------------------------------------------------
# forum_id extraction
# ---------------------------------------------------------------------------


def test_extract_forum_id_from_forum_url():
    assert (
        _extract_forum_id("https://openreview.net/forum?id=BeFjjyzWOJ")
        == "BeFjjyzWOJ"
    )


def test_extract_forum_id_from_pdf_url():
    assert (
        _extract_forum_id("https://openreview.net/pdf?id=BeFjjyzWOJ&type=preview")
        == "BeFjjyzWOJ"
    )


def test_extract_forum_id_returns_none_for_other_hosts():
    assert _extract_forum_id("https://arxiv.org/abs/1706.03762") is None
    assert _extract_forum_id("") is None
    assert _extract_forum_id(None) is None


# ---------------------------------------------------------------------------
# Lookup by forum ID
# ---------------------------------------------------------------------------


def test_lookup_by_id_parses_fields(tmp_path):
    get, calls = _mock_http(_NOTE_PAYLOAD)
    backend = OpenReviewBackend(
        cache=DiskCache(tmp_path),
        rate_limiter=OpenReviewRateLimiter(min_interval_seconds=0.0),
        http_get=get,
    )
    rec = backend.lookup_by_id("BeFjjyzWOJ")
    assert rec is not None
    assert rec.backend == Backend.OPENREVIEW
    assert rec.record_id == "BeFjjyzWOJ"
    assert "Erasing the Invisible" in rec.title
    assert "Mucong Ding" in rec.authors
    assert rec.year == 2025  # parsed from venueid
    assert rec.venue == "NeurIPS 2025 Datasets and Benchmarks Track poster"
    assert rec.url == "https://openreview.net/forum?id=BeFjjyzWOJ"
    assert calls[0]["params"]["id"] == "BeFjjyzWOJ"


def test_lookup_by_id_returns_none_when_empty(tmp_path):
    get, _ = _mock_http({"notes": []})
    backend = OpenReviewBackend(
        cache=DiskCache(tmp_path),
        rate_limiter=OpenReviewRateLimiter(min_interval_seconds=0.0),
        http_get=get,
    )
    assert backend.lookup_by_id("DOES_NOT_EXIST") is None


def test_lookup_via_reference_url_uses_forum_id(tmp_path):
    get, calls = _mock_http(_NOTE_PAYLOAD)
    backend = OpenReviewBackend(
        cache=DiskCache(tmp_path),
        rate_limiter=OpenReviewRateLimiter(min_interval_seconds=0.0),
        http_get=get,
    )
    ref = Reference(
        raw="x",
        source_format=SourceFormat.BIBTEX,
        title="Erasing the Invisible",
        url="https://openreview.net/forum?id=BeFjjyzWOJ",
    )
    out = backend.lookup(ref)
    assert len(out) == 1
    assert out[0].record_id == "BeFjjyzWOJ"
    # URL path should hit /notes?id=..., not /notes/search
    assert calls[0]["url"].endswith("/notes")


# ---------------------------------------------------------------------------
# Title search
# ---------------------------------------------------------------------------


def test_search_by_title(tmp_path):
    get, calls = _mock_http(_SEARCH_PAYLOAD)
    backend = OpenReviewBackend(
        cache=DiskCache(tmp_path),
        rate_limiter=OpenReviewRateLimiter(min_interval_seconds=0.0),
        http_get=get,
    )
    results = backend.search_by_title("Erasing the Invisible", max_results=5)
    assert len(results) == 2
    assert results[0].record_id == "BeFjjyzWOJ"
    assert calls[0]["params"]["term"] == "Erasing the Invisible"
    assert calls[0]["url"].endswith("/notes/search")


def test_search_filters_by_year_with_one_year_tolerance(tmp_path):
    get, _ = _mock_http(_SEARCH_PAYLOAD)
    backend = OpenReviewBackend(
        cache=DiskCache(tmp_path),
        rate_limiter=OpenReviewRateLimiter(min_interval_seconds=0.0),
        http_get=get,
    )
    # year=2025 keeps the NeurIPS 2025 hit but drops the ICLR 2019 noise
    results = backend.search_by_title("Erasing the Invisible", year=2025)
    assert len(results) == 1
    assert results[0].record_id == "BeFjjyzWOJ"


def test_lookup_falls_back_to_title_search_when_no_url(tmp_path):
    get, calls = _mock_http(_SEARCH_PAYLOAD)
    backend = OpenReviewBackend(
        cache=DiskCache(tmp_path),
        rate_limiter=OpenReviewRateLimiter(min_interval_seconds=0.0),
        http_get=get,
    )
    ref = Reference(
        raw="x",
        source_format=SourceFormat.BIBTEX,
        title="Erasing the Invisible",
        year=2025,
    )
    out = backend.lookup(ref)
    assert len(out) >= 1
    assert calls[0]["url"].endswith("/notes/search")


def test_lookup_returns_empty_when_no_title_or_url(tmp_path):
    get, calls = _mock_http({"notes": []})
    backend = OpenReviewBackend(
        cache=DiskCache(tmp_path),
        rate_limiter=OpenReviewRateLimiter(min_interval_seconds=0.0),
        http_get=get,
    )
    ref = Reference(raw="x", source_format=SourceFormat.BIBTEX)
    assert backend.lookup(ref) == []
    assert calls == []


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_cache_hit_avoids_http(tmp_path):
    get, calls = _mock_http(_NOTE_PAYLOAD)
    backend = OpenReviewBackend(
        cache=DiskCache(tmp_path),
        rate_limiter=OpenReviewRateLimiter(min_interval_seconds=0.0),
        http_get=get,
    )
    backend.lookup_by_id("BeFjjyzWOJ")
    assert len(calls) == 1
    backend.lookup_by_id("BeFjjyzWOJ")
    assert len(calls) == 1, "second call should be served from cache"


def test_cache_remembers_misses(tmp_path):
    get, calls = _mock_http({"notes": []})
    backend = OpenReviewBackend(
        cache=DiskCache(tmp_path),
        rate_limiter=OpenReviewRateLimiter(min_interval_seconds=0.0),
        http_get=get,
    )
    assert backend.lookup_by_id("MISSING") is None
    assert len(calls) == 1
    assert backend.lookup_by_id("MISSING") is None
    assert len(calls) == 1, "negative result should also be cached"


# ---------------------------------------------------------------------------
# 429 / transient-error handling
# ---------------------------------------------------------------------------


def test_429_retries_then_succeeds(tmp_path):
    """First two requests rate-limited, third succeeds — backend retries."""
    calls: list[dict] = []
    responses = [
        _Resp(429, None),
        _Resp(429, None),
        _Resp(200, _NOTE_PAYLOAD),
    ]
    responses[0].headers = {"Retry-After": "0"}
    responses[1].headers = {"Retry-After": "0"}

    def get(url, params):
        calls.append({"url": url, "params": params})
        return responses[len(calls) - 1]

    backend = OpenReviewBackend(
        cache=DiskCache(tmp_path),
        rate_limiter=OpenReviewRateLimiter(min_interval_seconds=0.0, max_retries=3),
        http_get=get,
    )
    rec = backend.lookup_by_id("BeFjjyzWOJ")
    assert rec is not None
    assert rec.record_id == "BeFjjyzWOJ"
    assert len(calls) == 3


def test_transient_error_does_not_poison_cache(tmp_path):
    """When all retries fail, do NOT cache the empty result — let next run try again."""
    def get(url, params):
        return _Resp(429, None)

    backend = OpenReviewBackend(
        cache=DiskCache(tmp_path),
        rate_limiter=OpenReviewRateLimiter(min_interval_seconds=0.0, max_retries=1),
        http_get=get,
    )
    assert backend.search_by_title("erasing the invisible") == []
    # The cache should remain empty so the next run can retry the same query.
    cached = DiskCache(tmp_path).get_api(
        "openreview", "title_erasing the invisible_"
    )
    assert cached is None


# ---------------------------------------------------------------------------
# Unpublished-submission filtering
# ---------------------------------------------------------------------------


_REJECTED_PAYLOAD = {
    "notes": [
        {
            "id": "ld6JUQbhes",
            "forum": "ld6JUQbhes",
            "content": {
                "title": {"value": "AssetOpsBench: Benchmarking AI Agents"},
                "authors": {"value": ["Dhaval Patel", "Shuxin Lin"]},
                "venue": {"value": "Submitted to ICLR 2026"},
                "venueid": {"value": "ICLR.cc/2026/Conference/Rejected_Submission"},
            },
        }
    ]
}


def test_rejected_submission_drops_venue(tmp_path):
    """A 'Submitted to X' / Rejected_Submission record is not a citable venue."""
    get, _ = _mock_http(_REJECTED_PAYLOAD)
    backend = OpenReviewBackend(
        cache=DiskCache(tmp_path),
        rate_limiter=OpenReviewRateLimiter(min_interval_seconds=0.0),
        http_get=get,
    )
    rec = backend.lookup_by_id("ld6JUQbhes")
    assert rec is not None
    # Title/authors/year/url still come through — only venue is suppressed.
    assert rec.title.startswith("AssetOpsBench")
    assert "Dhaval Patel" in rec.authors
    assert rec.year == 2026
    assert rec.url == "https://openreview.net/forum?id=ld6JUQbhes"
    assert rec.venue is None
    assert rec.publication_venue is None


def test_pending_submission_drops_venue(tmp_path):
    """venueid ending in plain '/Submission' (under review) also drops the venue."""
    payload = {
        "notes": [
            {
                "id": "abc",
                "forum": "abc",
                "content": {
                    "title": {"value": "Pending paper"},
                    "authors": {"value": ["A. Author"]},
                    "venue": {"value": "Submitted to ICLR 2026"},
                    "venueid": {"value": "ICLR.cc/2026/Conference/Submission"},
                },
            }
        ]
    }
    get, _ = _mock_http(payload)
    backend = OpenReviewBackend(
        cache=DiskCache(tmp_path),
        rate_limiter=OpenReviewRateLimiter(min_interval_seconds=0.0),
        http_get=get,
    )
    rec = backend.lookup_by_id("abc")
    assert rec is not None
    assert rec.venue is None


def test_withdrawn_submission_drops_venue(tmp_path):
    payload = {
        "notes": [
            {
                "id": "wxyz",
                "forum": "wxyz",
                "content": {
                    "title": {"value": "Withdrawn paper"},
                    "authors": {"value": ["A. Author"]},
                    # venue text might even be empty for withdrawn submissions —
                    # the venueid alone is enough to know it's not published.
                    "venueid": {"value": "ICLR.cc/2026/Conference/Withdrawn_Submission"},
                },
            }
        ]
    }
    get, _ = _mock_http(payload)
    backend = OpenReviewBackend(
        cache=DiskCache(tmp_path),
        rate_limiter=OpenReviewRateLimiter(min_interval_seconds=0.0),
        http_get=get,
    )
    rec = backend.lookup_by_id("wxyz")
    assert rec is not None
    assert rec.venue is None


def test_published_venue_is_kept(tmp_path):
    """The accepted-paper case must NOT be filtered."""
    get, _ = _mock_http(_NOTE_PAYLOAD)  # NeurIPS 2025 accepted
    backend = OpenReviewBackend(
        cache=DiskCache(tmp_path),
        rate_limiter=OpenReviewRateLimiter(min_interval_seconds=0.0),
        http_get=get,
    )
    rec = backend.lookup_by_id("BeFjjyzWOJ")
    assert rec is not None
    assert rec.venue == "NeurIPS 2025 Datasets and Benchmarks Track poster"


# ---------------------------------------------------------------------------
# Search title-similarity gate
# ---------------------------------------------------------------------------


def test_search_drops_topically_unrelated_results(tmp_path):
    """OpenReview ranks by topical relevance; an unrelated paper that shares
    one or two words with the cited title must NOT be returned as a match."""
    payload = {
        "notes": [
            # Same topical field as the query (deepfake detection) but a
            # different paper — must be dropped because content-token overlap
            # is below the similarity threshold once stopwords are removed.
            {
                "id": "noise1",
                "forum": "noise1",
                "content": {
                    "title": {"value": "Roadmap on computational methods in optical imaging and holography"},
                    "authors": {"value": ["X. Author"]},
                    "venue": {"value": "Applied Physics B"},
                    "venueid": {"value": "AppliedPhysicsB.cc/2024/Journal"},
                },
            },
            {
                "id": "noise2",
                "forum": "noise2",
                "content": {
                    "title": {"value": "DDL: A Dataset for Interpretable Deepfake Detection and Localization"},
                    "authors": {"value": ["Y. Author"]},
                    "venue": {"value": "CoRR 2025"},
                    "venueid": {"value": "CoRR.cc/2025"},
                },
            },
        ]
    }
    get, _ = _mock_http(payload)
    backend = OpenReviewBackend(
        cache=DiskCache(tmp_path),
        rate_limiter=OpenReviewRateLimiter(min_interval_seconds=0.0),
        http_get=get,
    )
    results = backend.search_by_title(
        "Workshop on deepfake detection, localization, and interpretability (ijcai 2025)"
    )
    assert results == []


def test_cache_stores_raw_api_payload_not_filtered_records(tmp_path):
    """The on-disk cache must hold the unmodified API response. This lets us
    change parsing / filtering / similarity thresholds without invalidating
    cache — the next read just re-derives records from the raw notes."""
    payload = {
        "notes": [
            {
                "id": "abc",
                "forum": "abc",
                "content": {
                    "title": {"value": "Some Paper Title"},
                    "authors": {"value": ["A. Author"]},
                    "venue": {"value": "NeurIPS 2025"},
                    "venueid": {"value": "NeurIPS.cc/2025/Conference"},
                },
            }
        ]
    }
    get, _ = _mock_http(payload)
    cache = DiskCache(tmp_path)
    backend = OpenReviewBackend(
        cache=cache,
        rate_limiter=OpenReviewRateLimiter(min_interval_seconds=0.0),
        http_get=get,
    )
    backend.lookup_by_id("abc")
    cached = cache.get_api("openreview", "id_abc")
    # Cached value is the verbatim API payload — note shape preserved with
    # ``content.<field>.value`` wrappers — not a flattened ExternalRecord.
    assert cached == payload


def test_search_keeps_typo_and_casing_variants(tmp_path):
    """Casing / punctuation variants must still pass the similarity gate."""
    payload = {
        "notes": [
            {
                "id": "matharena",
                "forum": "matharena",
                "content": {
                    "title": {"value": "MathArena: Evaluating LLMs on Uncontaminated Math Competitions"},
                    "authors": {"value": ["Mislav Balunovic"]},
                    "venue": {"value": "NeurIPS 2025 poster"},
                    "venueid": {"value": "NeurIPS.cc/2025/Conference"},
                },
            }
        ]
    }
    get, _ = _mock_http(payload)
    backend = OpenReviewBackend(
        cache=DiskCache(tmp_path),
        rate_limiter=OpenReviewRateLimiter(min_interval_seconds=0.0),
        http_get=get,
    )
    results = backend.search_by_title(
        "Math-arena: Evaluating llms on uncontaminated math competitions"
    )
    assert len(results) == 1
    assert results[0].record_id == "matharena"
