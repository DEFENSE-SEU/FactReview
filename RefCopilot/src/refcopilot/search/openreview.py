"""OpenReview backend — covers conference papers (NeurIPS/ICLR/COLM/...) that
are exclusively hosted on openreview.net and never make it to arXiv.

Two lookup paths:
  1. **URL-based**: when the cited reference includes an ``openreview.net``
     URL, extract the forum ID and fetch ``/notes?id=<forum_id>`` directly.
  2. **Title search**: otherwise, hit ``/notes/search?term=<title>`` and
     return the top candidates ranked by OpenReview's own scorer.

Both endpoints work without authentication.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Protocol
from urllib.parse import parse_qs, urlparse

import httpx

from refcopilot.cache.disk_cache import DiskCache
from refcopilot.models import Backend, ExternalRecord, Reference
from refcopilot.ratelimit.openreview import OpenReviewRateLimiter
from refcopilot.ratelimit.semantic_scholar import parse_retry_after
from refcopilot.verify.text_match import title_similarity
from refcopilot.verify.thresholds import SEARCH_RESULT_MIN_TITLE_SIM

logger = logging.getLogger(__name__)


_DEFAULT_BASE = "https://api2.openreview.net"


class _Response(Protocol):
    status_code: int
    headers: dict[str, str]

    def json(self) -> Any: ...


HttpFn = Callable[[str, dict[str, str]], _Response]


class OpenReviewBackend:
    name = Backend.OPENREVIEW.value

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE,
        cache: DiskCache | None = None,
        rate_limiter: OpenReviewRateLimiter | None = None,
        http_get: HttpFn | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache = cache
        self.rate_limiter = rate_limiter or OpenReviewRateLimiter()
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
        forum_id = _extract_forum_id(ref.url)
        if forum_id:
            rec = self.lookup_by_id(forum_id)
            if rec:
                return [rec]
        if ref.title:
            return self.search_by_title(ref.title, year=ref.year, max_results=5)
        return []

    def lookup_by_id(self, forum_id: str) -> ExternalRecord | None:
        clean_id = forum_id.strip()
        payload = self._cached_or_fetch(
            f"id_{clean_id}", "/notes", {"id": clean_id}
        )
        if not payload:
            return None
        notes = payload.get("notes") or []
        if not notes:
            return None
        return _note_to_record(notes[0])

    def search_by_title(
        self, title: str, *, year: int | None = None, max_results: int = 5
    ) -> list[ExternalRecord]:
        clean = re.sub(r"\s+", " ", title.strip())
        cache_key = f"title_{clean[:80]}_{year or ''}"
        payload = self._cached_or_fetch(
            cache_key, "/notes/search", {"term": clean, "limit": str(max_results)}
        )
        if not payload:
            return []
        notes = (payload.get("notes") or [])[:max_results]
        records: list[ExternalRecord] = []
        for note in notes:
            rec = _note_to_record(note)
            if rec is None:
                continue
            # OpenReview's title search isn't year-filtered server-side; drop
            # candidates whose year is wildly off (allow ±1 for venues that
            # span calendar years).
            if year and rec.year and rec.year not in {year, year - 1, year + 1}:
                continue
            # OpenReview ranks by topical relevance, so a query for a workshop
            # website can return a paper that shares one or two topic words
            # but isn't the same work. Require a minimum content-token overlap
            # with the query before treating the candidate as a real match.
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
        confirmed-empty (404 or empty array) and for transient errors;
        only the former is cached.
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

        self._last_was_transient = False
        attempt = 0
        while attempt <= self.rate_limiter.max_retries:
            self.rate_limiter.acquire()
            try:
                if self._http_get is not None:
                    resp = self._http_get(url, params)
                else:
                    resp = httpx.get(url, params=params, timeout=self.timeout)
            except Exception as exc:
                logger.warning("openreview request failed: %s", exc)
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
                    logger.warning("openreview json decode failed: %s", exc)
                    return None

            if resp.status_code == 404:
                return None

            if resp.status_code == 429:
                retry_after = parse_retry_after(resp.headers.get("Retry-After"))
                wait = self.rate_limiter.backoff_for_attempt(
                    attempt, retry_after_seconds=retry_after
                )
                logger.info("openreview 429; sleeping %.2fs", wait)
                time.sleep(wait)
                attempt += 1
                continue

            logger.warning(
                "openreview unexpected status %d for %s", resp.status_code, path
            )
            attempt += 1
            if attempt > self.rate_limiter.max_retries:
                self._failed = True
                self._last_was_transient = True
                return None
            time.sleep(self.rate_limiter.backoff_for_attempt(attempt))

        # Fell out of the loop without returning — only happens after 429
        # exhaustion. Treat as transient: don't poison the backend.
        self._last_was_transient = True
        return None


_FORUM_HOSTS = {"openreview.net", "www.openreview.net"}


def _extract_forum_id(url: str | None) -> str | None:
    if not url:
        return None
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if parsed.netloc.lower() not in _FORUM_HOSTS:
        return None
    qs = parse_qs(parsed.query or "")
    candidates = qs.get("id") or qs.get("noteId")
    if not candidates:
        return None
    forum_id = candidates[0].strip()
    return forum_id or None


def _content_value(content: dict[str, Any], key: str) -> Any:
    """OpenReview API v2 wraps every content field as ``{"value": ...}``."""
    val = content.get(key)
    if isinstance(val, dict) and "value" in val:
        return val["value"]
    return val


def _year_from_venueid(venueid: str | None) -> int | None:
    if not venueid:
        return None
    m = re.search(r"\b(19|20)\d{2}\b", venueid)
    return int(m.group(0)) if m else None


def _year_from_cdate(cdate: Any) -> int | None:
    """``cdate`` is epoch milliseconds when present."""
    if not isinstance(cdate, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(cdate / 1000, tz=timezone.utc).year
    except (OverflowError, OSError, ValueError):
        return None


def _note_to_record(note: dict[str, Any]) -> ExternalRecord | None:
    if not isinstance(note, dict):
        return None
    forum_id = (note.get("id") or note.get("forum") or "").strip()
    if not forum_id:
        return None

    content = note.get("content") or {}
    title = str(_content_value(content, "title") or "").strip()
    if not title:
        return None

    authors_raw = _content_value(content, "authors") or []
    authors = [str(a).strip() for a in authors_raw if a]

    venue_raw = (_content_value(content, "venue") or "").strip() or None
    venueid = (_content_value(content, "venueid") or "").strip() or None

    # Unpublished submissions ("Submitted to ICLR 2026", or venueid paths
    # like .../Rejected_Submission, .../Withdrawn_Submission, .../Submission)
    # aren't real publication venues — leave venue empty so the merger falls
    # back to the arXiv preprint instead of citing a non-publication.
    venue = None if _is_unpublished(venue_raw, venueid) else venue_raw

    year = _year_from_venueid(venueid) or _year_from_cdate(note.get("cdate"))

    return ExternalRecord(
        backend=Backend.OPENREVIEW,
        record_id=forum_id,
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        publication_venue=venue,
        url=f"https://openreview.net/forum?id={forum_id}",
        raw={"venueid": venueid or ""},
    )


_UNPUBLISHED_VENUEID_MARKERS = (
    "/submission",
    "/rejected_submission",
    "/withdrawn_submission",
    "/desk_rejected_submission",
    "/active_submission",
)


def _is_unpublished(venue: str | None, venueid: str | None) -> bool:
    """Heuristic for OpenReview entries that aren't actual publications.

    OpenReview returns the same record shape for accepted papers and for
    submissions that are pending / rejected / withdrawn. The user-facing
    ``venue`` field for the latter looks like ``"Submitted to ICLR 2026"``,
    and the ``venueid`` path ends in ``Submission`` / ``Rejected_Submission``
    / ``Withdrawn_Submission``. None of those are citable venues.
    """
    if venue:
        if venue.lower().lstrip().startswith("submitted to "):
            return True
    if venueid:
        path = venueid.lower()
        for marker in _UNPUBLISHED_VENUEID_MARKERS:
            if path.endswith(marker):
                return True
    return False
