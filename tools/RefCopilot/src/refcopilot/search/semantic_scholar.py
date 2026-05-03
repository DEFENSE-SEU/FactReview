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
from typing import Any, Callable

import httpx

from refcopilot.cache.disk_cache import DiskCache
from refcopilot.models import Backend, ExternalRecord, Reference
from refcopilot.ratelimit.semantic_scholar import (
    SemanticScholarRateLimiter,
    parse_retry_after,
)

logger = logging.getLogger(__name__)


_DEFAULT_BASE = "https://api.semanticscholar.org/graph/v1"
_FIELDS = (
    "title,authors,year,externalIds,url,abstract,openAccessPdf,"
    "isOpenAccess,venue,publicationVenue,journal"
)


HttpFn = Callable[[str, dict[str, str], dict[str, str]], "_Response"]


class _Response:
    status_code: int
    headers: dict[str, str]

    def json(self) -> Any: ...


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
        self._failed = False

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
            rec = self._search_match(ref.title, year=ref.year, authors=ref.authors)
            if rec:
                return [rec]
            return self._search_relevance(ref.title, year=ref.year, authors=ref.authors)

        return []

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    def _fetch_by_id(self, prefix: str, identifier: str) -> ExternalRecord | None:
        cache_key = f"{prefix.lower()}_{identifier.replace('/', '_')}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return _payload_to_record(cached)

        path = f"/paper/{prefix}:{identifier}"
        payload = self._get_json(path, {"fields": _FIELDS})
        if not payload:
            return None
        self._cache_set(cache_key, payload)
        return _payload_to_record(payload)

    def _search_match(
        self, title: str, *, year: int | None, authors: list[str]
    ) -> ExternalRecord | None:
        cache_key = f"match_{_safe(title)[:80]}_{year or ''}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return _payload_to_record(cached) if cached else None

        params = {"query": title, "fields": _FIELDS}
        payload = self._get_json("/paper/search/match", params)
        if not payload:
            self._cache_set(cache_key, {})
            return None

        candidates = payload.get("data") or []
        if not candidates:
            self._cache_set(cache_key, {})
            return None

        best = candidates[0]
        self._cache_set(cache_key, best)
        return _payload_to_record(best)

    def _search_relevance(
        self, title: str, *, year: int | None, authors: list[str]
    ) -> list[ExternalRecord]:
        cache_key = f"search_{_safe(title)[:80]}_{year or ''}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return [_payload_to_record(p) for p in cached if p]

        params = {"query": title, "fields": _FIELDS, "limit": "5"}
        payload = self._get_json("/paper/search", params)
        records: list[ExternalRecord] = []
        if payload:
            for item in (payload.get("data") or [])[:5]:
                rec = _payload_to_record(item)
                if rec:
                    records.append(rec)
        self._cache_set(cache_key, [r.model_dump() for r in records])
        return records

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _get_json(self, path: str, params: dict[str, str]) -> Any | None:
        url = f"{self.base_url}{path}"
        headers: dict[str, str] = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

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
                return None
            time.sleep(self.rate_limiter.backoff_for_attempt(attempt))

        self._failed = True
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
