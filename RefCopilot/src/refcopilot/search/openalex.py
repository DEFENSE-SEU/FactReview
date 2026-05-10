"""OpenAlex backend — broad scholarly index covering 270M+ works.

Useful as a cross-validation signal alongside arXiv, Semantic Scholar, and
OpenReview, especially for published journal articles, books, and non-CS
venues that the others sometimes miss.

Endpoint priority (first match wins):
  1. /works/doi:{doi}      — direct lookup by DOI (URN form)
  2. /works?search={title}  — relevance-ranked title search

Auth is a free API key passed as ``?api_key=...``. The key is required at
construction time; the pipeline only instantiates this backend when the user
has set ``OPENALEX_API_KEY``.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Protocol

import httpx

from refcopilot.cache.disk_cache import DiskCache
from refcopilot.models import Backend, ExternalRecord, Reference
from refcopilot.ratelimit.openalex import OpenAlexRateLimiter
from refcopilot.ratelimit.semantic_scholar import parse_retry_after
from refcopilot.verify.text_match import title_similarity
from refcopilot.verify.thresholds import SEARCH_RESULT_MIN_TITLE_SIM

logger = logging.getLogger(__name__)


_DEFAULT_BASE = "https://api.openalex.org"
_SELECT_FIELDS = (
    "id,doi,title,display_name,publication_year,authorships,"
    "primary_location,ids,type,is_retracted"
)


class _Response(Protocol):
    status_code: int
    headers: dict[str, str]

    def json(self) -> Any: ...


HttpFn = Callable[[str, dict[str, str]], _Response]


class OpenAlexBackend:
    name = Backend.OPENALEX.value

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = _DEFAULT_BASE,
        cache: DiskCache | None = None,
        rate_limiter: OpenAlexRateLimiter | None = None,
        http_get: HttpFn | None = None,
        timeout: float = 20.0,
    ) -> None:
        clean_key = (api_key or "").strip()
        if not clean_key:
            raise ValueError("OpenAlexBackend requires a non-empty api_key")
        self.api_key = clean_key
        self.base_url = base_url.rstrip("/")
        self.cache = cache
        self.rate_limiter = rate_limiter or OpenAlexRateLimiter()
        self._http_get = http_get
        self.timeout = timeout
        # Permanent failure (network down, persistent 5xx). Once set, all
        # subsequent lookups short-circuit to []. NOT set by 429 alone.
        self._failed = False
        # True when the most recent ``_get_json`` returned None for a transient
        # reason (429 exhaustion, network error). Used by callers to skip
        # caching empty results that should be retried next run.
        self._last_was_transient = False

    def lookup(self, ref: Reference) -> list[ExternalRecord]:
        if self._failed:
            return []
        if ref.doi:
            rec = self.lookup_by_doi(ref.doi)
            if rec:
                return [rec]
        if ref.title:
            return self.search_by_title(ref.title, year=ref.year, max_results=5)
        return []

    def lookup_by_doi(self, doi: str) -> ExternalRecord | None:
        clean_doi = _normalize_doi(doi)
        if not clean_doi:
            return None
        cache_key = f"doi_{clean_doi.replace('/', '_')}"
        payload = self._cached_or_fetch(
            cache_key, f"/works/doi:{clean_doi}", {"select": _SELECT_FIELDS}
        )
        if not payload:
            return None
        return _work_to_record(payload)

    def search_by_title(
        self, title: str, *, year: int | None = None, max_results: int = 5
    ) -> list[ExternalRecord]:
        clean = re.sub(r"\s+", " ", title.strip())
        cache_key = f"title_{clean[:80]}_{year or ''}"
        # ``filter=title.search:...`` restricts to the title field, which is
        # what we actually want for citation matching. The plain ``search=``
        # parameter ranks across title + abstract + other fields and produces
        # bizarre top hits for noisy queries (e.g. an unrelated paper from a
        # different field outranking the BERT NAACL paper for a "BERT ..."
        # query).
        payload = self._cached_or_fetch(
            cache_key,
            "/works",
            {
                "filter": f"title.search:{_filter_safe(clean)}",
                "per-page": str(max_results),
                "select": _SELECT_FIELDS,
            },
        )
        if not payload:
            return []
        results = (payload.get("results") or [])[:max_results]
        records: list[ExternalRecord] = []
        for work in results:
            rec = _work_to_record(work)
            if rec is None:
                continue
            # OpenAlex's search isn't strictly year-filtered; drop candidates
            # whose year is wildly off (allow ±1 for venues that span calendar
            # years, mirroring OpenReview).
            if year and rec.year and rec.year not in {year, year - 1, year + 1}:
                continue
            # Relevance-ranked search can return topical neighbours that share
            # a few keywords; require a minimum title-token overlap.
            if title_similarity(clean, rec.title) < SEARCH_RESULT_MIN_TITLE_SIM:
                continue
            records.append(rec)
        return records

    def _cached_or_fetch(
        self, cache_key: str, path: str, params: dict[str, str]
    ) -> dict[str, Any] | None:
        """Return the raw API payload (cache hit or fresh fetch).

        Cache stores the unmodified API response so changes to filtering /
        parsing logic don't require cache invalidation. Returns None for
        confirmed-empty (404) and for transient errors; only the former is
        cached.
        """
        if self.cache:
            cached = self.cache.get_api(self.name, cache_key)
            if cached is not None:
                return cached or None
        payload = self._get_json(path, params)
        if payload is None:
            if not self._last_was_transient and self.cache:
                self.cache.set_api(self.name, cache_key, {})
            return None
        if self.cache:
            self.cache.set_api(self.name, cache_key, payload)
        return payload

    def _get_json(self, path: str, params: dict[str, str]) -> Any | None:
        url = f"{self.base_url}{path}"
        # api_key travels as a query param, not a header.
        full_params = {**params, "api_key": self.api_key}

        self._last_was_transient = False
        attempt = 0
        while attempt <= self.rate_limiter.max_retries:
            self.rate_limiter.acquire()
            try:
                if self._http_get is not None:
                    resp = self._http_get(url, full_params)
                else:
                    resp = httpx.get(url, params=full_params, timeout=self.timeout)
            except Exception as exc:
                logger.warning("openalex request failed: %s", exc)
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
                    logger.warning("openalex json decode failed: %s", exc)
                    return None

            if resp.status_code == 404:
                return None

            if resp.status_code == 429:
                retry_after = parse_retry_after(resp.headers.get("Retry-After"))
                wait = self.rate_limiter.backoff_for_attempt(
                    attempt, retry_after_seconds=retry_after
                )
                logger.info("openalex 429; sleeping %.2fs", wait)
                time.sleep(wait)
                attempt += 1
                continue

            logger.warning(
                "openalex unexpected status %d for %s", resp.status_code, path
            )
            attempt += 1
            if attempt > self.rate_limiter.max_retries:
                self._failed = True
                self._last_was_transient = True
                return None
            time.sleep(self.rate_limiter.backoff_for_attempt(attempt))

        # Fell out of the loop without returning — only after 429 exhaustion.
        # Treat as transient: don't poison the backend.
        self._last_was_transient = True
        return None


_FILTER_SPECIAL_CHARS = re.compile(r"[,|]")


def _filter_safe(text: str) -> str:
    """Strip characters that have special meaning in OpenAlex filter syntax.

    Commas separate filters and pipes denote OR; either inside a value would
    confuse the parser. Title-content matching doesn't need them.
    """
    return _FILTER_SPECIAL_CHARS.sub(" ", text).strip()


_DOI_URL_PREFIXES = ("https://doi.org/", "http://doi.org/", "doi:")


def _normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    s = doi.strip()
    for prefix in _DOI_URL_PREFIXES:
        if s.lower().startswith(prefix):
            s = s[len(prefix):]
            break
    return s.strip().lower() or None


# OpenAlex returns work IDs as full URLs like ``https://openalex.org/W12345``.
_OPENALEX_ID_RE = re.compile(r"/(W\d+)$", re.IGNORECASE)


def _bare_openalex_id(url: str | None) -> str | None:
    if not url:
        return None
    m = _OPENALEX_ID_RE.search(url.strip())
    return m.group(1) if m else url.strip() or None


# Source ``type`` values that aren't real publication venues — for these we
# leave ``venue`` empty so the merger falls back to a more authoritative source.
_NON_VENUE_SOURCE_TYPES = {"repository", "ebook platform"}


def _work_to_record(work: dict[str, Any]) -> ExternalRecord | None:
    if not isinstance(work, dict):
        return None

    work_url = str(work.get("id") or "").strip()
    record_id = _bare_openalex_id(work_url) or ""
    if not record_id:
        return None

    title = str(work.get("title") or work.get("display_name") or "").strip()
    if not title:
        return None

    authorships = work.get("authorships") or []
    authors: list[str] = []
    for entry in authorships:
        if not isinstance(entry, dict):
            continue
        author = entry.get("author") or {}
        name = str(author.get("display_name") or "").strip()
        if name:
            authors.append(name)

    year = work.get("publication_year")
    if not isinstance(year, int):
        year = None

    doi = _normalize_doi(work.get("doi"))

    venue: str | None = None
    primary = work.get("primary_location") or {}
    source = primary.get("source") if isinstance(primary, dict) else None
    if isinstance(source, dict):
        source_type = str(source.get("type") or "").lower()
        if source_type not in _NON_VENUE_SOURCE_TYPES:
            venue = (str(source.get("display_name") or "").strip() or None)

    return ExternalRecord(
        backend=Backend.OPENALEX,
        record_id=record_id,
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        publication_venue=venue,
        doi=doi,
        is_retracted=bool(work.get("is_retracted")),
        url=work_url or f"https://openalex.org/{record_id}",
        raw={"openalex_id": record_id, "type": str(work.get("type") or "")},
    )
