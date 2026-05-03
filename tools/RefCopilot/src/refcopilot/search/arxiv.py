"""arXiv backend — Atom feed search + version + withdrawn detection.

Uses raw HTTP against ``https://export.arxiv.org/api/query`` for both id-based
and title-based lookups, and scrapes per-version metadata from
``abs/{id}v{n}``. The ``http_get`` constructor argument is injectable so unit
tests can pass a mock.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Callable, Protocol

import httpx

from refcopilot.cache.disk_cache import DiskCache
from refcopilot.models import Backend, ExternalRecord, Reference
from refcopilot.ratelimit.arxiv import ArxivRateLimiter

logger = logging.getLogger(__name__)


_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

_ARXIV_API = "https://export.arxiv.org/api/query"


class _HttpResponse(Protocol):
    """Subset of :class:`httpx.Response` used by this module."""

    status_code: int
    text: str
    headers: dict[str, str]


HttpGetFn = Callable[[str, dict[str, str] | None], _HttpResponse]


class ArxivBackend:
    name = Backend.ARXIV.value

    def __init__(
        self,
        *,
        cache: DiskCache | None = None,
        rate_limiter: ArxivRateLimiter | None = None,
        http_get: HttpGetFn | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.cache = cache
        self.rate_limiter = rate_limiter or ArxivRateLimiter()
        self._http_get = http_get
        self.timeout = timeout

    def lookup(self, ref: Reference) -> list[ExternalRecord]:
        if ref.arxiv_id:
            rec = self.lookup_by_id(ref.arxiv_id)
            return [rec] if rec else []
        if ref.title:
            return self.search_by_title(ref.title, year=ref.year, max_results=5)
        return []

    def lookup_by_id(self, arxiv_id: str) -> ExternalRecord | None:
        clean_id = re.sub(r"v\d+$", "", str(arxiv_id), flags=re.IGNORECASE).strip()
        cache_key = f"id_{clean_id}"
        if self.cache:
            cached = self.cache.get_api(self.name, cache_key)
            if cached is not None:
                return ExternalRecord(**cached)

        params = {"id_list": clean_id, "max_results": "1"}
        feed = self._call_api(params)
        records = _parse_feed(feed)
        if not records:
            return None
        record = records[0]
        if self.cache:
            self.cache.set_api(self.name, cache_key, record.model_dump())
        return record

    def search_by_title(self, title: str, *, year: int | None = None, max_results: int = 5) -> list[ExternalRecord]:
        clean = re.sub(r"\s+", " ", title.strip())
        cache_key = f"title_{clean[:80]}_{year or ''}"
        if self.cache:
            cached = self.cache.get_api(self.name, cache_key)
            if cached is not None:
                return [ExternalRecord(**r) for r in cached]

        query = f'ti:"{clean}"'
        if year:
            query += f' AND submittedDate:[{year}01010000 TO {year}12312359]'
        params = {"search_query": query, "max_results": str(max_results)}
        feed = self._call_api(params)
        records = _parse_feed(feed)
        if self.cache:
            self.cache.set_api(self.name, cache_key, [r.model_dump() for r in records])
        return records

    def _call_api(self, params: dict[str, str]) -> str:
        self.rate_limiter.acquire()
        if self._http_get is not None:
            resp = self._http_get(_ARXIV_API, params)
        else:
            resp = httpx.get(_ARXIV_API, params=params, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"arxiv api returned {resp.status_code}")
        return resp.text


def _parse_feed(xml_text: str) -> list[ExternalRecord]:
    if not xml_text:
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("arxiv feed parse error: %s", exc)
        return []

    out: list[ExternalRecord] = []
    for entry in root.findall("atom:entry", _NS):
        record = _entry_to_record(entry)
        if record:
            out.append(record)
    return out


def _entry_to_record(entry: ET.Element) -> ExternalRecord | None:
    eid_text = (entry.findtext("atom:id", default="", namespaces=_NS) or "").strip()
    m = re.search(r"abs/(?P<id>\d{4}\.\d{4,5})(?:v(?P<v>\d+))?", eid_text)
    if not m:
        return None
    arxiv_id = m.group("id")
    version = int(m.group("v")) if m.group("v") else None

    title = re.sub(r"\s+", " ", (entry.findtext("atom:title", default="", namespaces=_NS) or "").strip())

    authors: list[str] = []
    for author in entry.findall("atom:author", _NS):
        name = (author.findtext("atom:name", default="", namespaces=_NS) or "").strip()
        if name:
            authors.append(name)

    published = entry.findtext("atom:published", default="", namespaces=_NS) or ""
    year_match = re.match(r"(\d{4})", published.strip())
    year = int(year_match.group(1)) if year_match else None

    summary = (entry.findtext("atom:summary", default="", namespaces=_NS) or "").strip()
    withdrawn = "this paper has been withdrawn" in summary.lower()

    journal_ref = (entry.findtext("arxiv:journal_ref", default="", namespaces=_NS) or "").strip()
    venue: str | None = journal_ref or None

    doi = (entry.findtext("arxiv:doi", default="", namespaces=_NS) or "").strip().lower() or None

    record = ExternalRecord(
        backend=Backend.ARXIV,
        record_id=arxiv_id,
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        publication_venue=venue,
        journal=venue,
        doi=doi,
        arxiv_id=arxiv_id,
        latest_arxiv_version=version,
        arxiv_versions=[version] if version else [],
        withdrawn=withdrawn,
        url=f"https://arxiv.org/abs/{arxiv_id}{f'v{version}' if version else ''}",
        raw={"summary": summary[:1000], "journal_ref": journal_ref},
    )
    return record
