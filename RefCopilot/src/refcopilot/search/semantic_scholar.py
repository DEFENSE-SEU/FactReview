"""Semantic Scholar backend.

Endpoint priority (first match wins):
  1. /paper/DOI:{doi}
  2. /paper/ARXIV:{arxiv_id}
  3. /paper/search/match    (exact title match)
  4. /paper/search          (relevance ranking, fallback)

429 handling:
  - Inspect `Retry-After` header (preferred over default backoff)
  - Up to N retries with exponential backoff + jitter
  - On final failure, return [] and short-circuit subsequent calls in this run
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Protocol

import httpx

from refcopilot.cache.disk_cache import DiskCache
from refcopilot.models import Backend, ExternalRecord, Reference
from refcopilot.ratelimit.semantic_scholar import (
    SemanticScholarRateLimiter,
    parse_retry_after,
)
from refcopilot.verify.text_match import title_similarity
from refcopilot.verify.thresholds import SEARCH_RESULT_MIN_TITLE_SIM

logger = logging.getLogger(__name__)


_DEFAULT_BASE = "https://api.semanticscholar.org/graph/v1"
_FIELDS = (
    "title,authors,year,externalIds,url,abstract,openAccessPdf,"
    "isOpenAccess,venue,publicationVenue,journal"
)


class _Response(Protocol):
    """Subset of :class:`httpx.Response` used by this module."""

    status_code: int
    headers: dict[str, str]

    def json(self) -> Any: ...


HttpFn = Callable[[str, dict[str, str], dict[str, str]], _Response]


class SemanticScholarBackend:
    name = Backend.SEMANTIC_SCHOLAR.value

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE,
        cache: DiskCache | None = None,
        rate_limiter: SemanticScholarRateLimiter | None = None,
        http_get: HttpFn | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.api_key = (api_key or "").strip() or None
        self.base_url = base_url.rstrip("/")
        self.cache = cache
        self.rate_limiter = rate_limiter or SemanticScholarRateLimiter()
        self._http_get = http_get
        self.timeout = timeout
        # Permanent failure (network down, persistent 5xx). Once set, all
        # subsequent lookups short-circuit to []. NOT set by 429 alone.
        self._failed = False
        # True when the most recent ``_get_json`` returned None for a transient
        # reason (429 exhaustion, network error). Callers use this to decide
        # whether to cache an empty result — we don't want to cache "not
        # found" if the underlying call simply failed transiently.
        self._last_was_transient = False

    def lookup(self, ref: Reference) -> list[ExternalRecord]:
        if self._failed:
            return []

        if ref.doi:
            rec = self._fetch_by_id("DOI", ref.doi)
            if rec:
                return [rec]

        if ref.arxiv_id:
            rec = self._fetch_by_id("ARXIV", ref.arxiv_id)
            if rec:
                return [rec]

        if ref.title:
            rec = self._search_match(ref.title, year=ref.year)
            if rec:
                return [rec]
            return self._search_relevance(ref.title, year=ref.year)

        return []

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    def _fetch_by_id(self, prefix: str, identifier: str) -> ExternalRecord | None:
        cache_key = f"{prefix.lower()}_{identifier.replace('/', '_')}"
        payload = self._cached_or_fetch_payload(
            cache_key,
            f"/paper/{prefix}:{identifier}",
            {"fields": _FIELDS},
        )
        return _payload_to_record(payload) if payload else None

    def _search_match(self, title: str, *, year: int | None) -> ExternalRecord | None:
        cache_key = f"match_{_safe(title)[:80]}_{year or ''}"
        payload = self._cached_or_fetch_payload(
            cache_key,
            "/paper/search/match",
            {"query": title, "fields": _FIELDS},
        )
        if not payload:
            return None
        candidates = payload.get("data") or []
        if not candidates:
            return None
        rec = _payload_to_record(candidates[0])
        # ``/paper/search/match`` occasionally returns a topical neighbour
        # rather than the actual paper. Apply the similarity gate at READ
        # time so threshold changes don't invalidate the cache.
        if rec is None or title_similarity(title, rec.title) < SEARCH_RESULT_MIN_TITLE_SIM:
            return None
        return rec

    def _search_relevance(self, title: str, *, year: int | None) -> list[ExternalRecord]:
        cache_key = f"search_{_safe(title)[:80]}_{year or ''}"
        payload = self._cached_or_fetch_payload(
            cache_key,
            "/paper/search",
            {"query": title, "fields": _FIELDS, "limit": "5"},
        )
        if not payload:
            return []
        records: list[ExternalRecord] = []
        for item in (payload.get("data") or [])[:5]:
            rec = _payload_to_record(item)
            if rec is None:
                continue
            # Drop topical neighbours that share a few words but aren't the
            # cited paper (e.g. an unrelated docker-security paper ranking
            # high for "AssetOpsBench Docker images").
            if title_similarity(title, rec.title) < SEARCH_RESULT_MIN_TITLE_SIM:
                continue
            records.append(rec)
        return records

    def _cached_or_fetch_payload(
        self, cache_key: str, path: str, params: dict[str, str]
    ) -> dict[str, Any] | None:
        """Return the raw API response (cache hit or fresh fetch).

        The cache stores unmodified API payloads so that filtering / parsing
        changes never need a cache invalidation. Returns None for
        confirmed-empty (404 / empty data array) and for transient errors;
        only the former is cached.
        """
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached or None
        payload = self._get_json(path, params)
        if payload is None:
            if not self._last_was_transient:
                self._cache_set(cache_key, {})
            return None
        self._cache_set(cache_key, payload)
        return payload

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _get_json(self, path: str, params: dict[str, str]) -> Any | None:
        url = f"{self.base_url}{path}"
        headers: dict[str, str] = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        self._last_was_transient = False
        attempt = 0
        while attempt <= self.rate_limiter.max_retries:
            self.rate_limiter.acquire()
            try:
                if self._http_get is not None:
                    resp = self._http_get(url, params, headers)
                else:
                    resp = httpx.get(url, params=params, headers=headers, timeout=self.timeout)
            except Exception as exc:
                logger.warning("s2 request failed: %s", exc)
                attempt += 1
                if attempt > self.rate_limiter.max_retries:
                    self._failed = True
                    self._last_was_transient = True
                    return None
                time.sleep(self.rate_limiter.backoff_for_attempt(attempt))
                continue

            if resp.status_code == 200:
                try:
                    return resp.json()
                except Exception as exc:
                    logger.warning("s2 json decode failed: %s", exc)
                    return None

            if resp.status_code == 404:
                return None

            if resp.status_code == 429:
                retry_after = parse_retry_after(resp.headers.get("Retry-After"))
                wait = self.rate_limiter.backoff_for_attempt(attempt, retry_after_seconds=retry_after)
                logger.info("s2 429; sleeping %.2fs", wait)
                time.sleep(wait)
                attempt += 1
                continue

            logger.warning("s2 unexpected status %d for %s", resp.status_code, path)
            attempt += 1
            if attempt > self.rate_limiter.max_retries:
                self._failed = True
                self._last_was_transient = True
                return None
            time.sleep(self.rate_limiter.backoff_for_attempt(attempt))

        # Fell out of the loop without an explicit return — the only way that
        # happens is exhausting the 429 retry budget. Treat as transient:
        # don't poison the backend (the next reference may succeed) and let
        # callers know not to cache the empty result.
        self._last_was_transient = True
        return None

    def _cache_get(self, key: str) -> Any | None:
        if not self.cache:
            return None
        return self.cache.get_api(self.name, key)

    def _cache_set(self, key: str, value: Any) -> None:
        if not self.cache:
            return
        self.cache.set_api(self.name, key, value)


def _payload_to_record(payload: Any) -> ExternalRecord | None:
    if not isinstance(payload, dict):
        return None

    paper_id = (payload.get("paperId") or "").strip()
    if not paper_id and not payload.get("title"):
        return None

    authors_field = payload.get("authors") or []
    authors = [
        str(a.get("name", "")).strip()
        for a in authors_field
        if isinstance(a, dict) and a.get("name")
    ]

    external_ids = payload.get("externalIds") or {}
    doi = (external_ids.get("DOI") or "").strip().lower() or None
    arxiv_id = (external_ids.get("ArXiv") or "").strip() or None

    journal_name = _name_or_string(payload.get("journal"))
    pub_venue_name = _name_or_string(payload.get("publicationVenue"))
    venue = (payload.get("venue") or "").strip() or None

    return ExternalRecord(
        backend=Backend.SEMANTIC_SCHOLAR,
        record_id=paper_id,
        title=str(payload.get("title") or "").strip(),
        authors=authors,
        year=payload.get("year") if isinstance(payload.get("year"), int) else None,
        venue=venue,
        publication_venue=pub_venue_name,
        journal=journal_name,
        doi=doi,
        arxiv_id=arxiv_id,
        s2_paper_id=paper_id,
        url=str(payload.get("url") or "").strip(),
        raw={"externalIds": external_ids, "abstract": str(payload.get("abstract") or "")[:1000]},
    )


def _safe(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in (text or ""))


def _name_or_string(value) -> str | None:
    """S2 sometimes returns nested objects like {"name": "..."} and sometimes
    plain strings for the same field; normalize either to an Optional[str]."""
    if value is None:
        return None
    if isinstance(value, dict):
        name = value.get("name")
        return str(name).strip() or None if name else None
    if isinstance(value, str):
        return value.strip() or None
    return None
