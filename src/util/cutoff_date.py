"""Publication-date cutoff for positioning analysis.

Positioning compares the paper under review against prior work; including
papers that came out *after* the manuscript would produce anachronistic
novelty/comparison conclusions. This module centralises:

- Parsing user-supplied cutoff strings (``YYYY``, ``YYYY-MM``, ``YYYY-MM-DD``).
- Deriving a cutoff from an arXiv URL/ID (``2210.xxxxx`` -> 2022-10).
- A predicate + splitter for filtering retrieved papers by ``year`` and/or
  ``published`` fields.

The cutoff is **inclusive** on the upper bound (papers from the cutoff month
itself are kept). When precision is coarse (year-only), the upper bound is
end-of-year; when month-only, end-of-month.
"""

from __future__ import annotations

import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from urllib.parse import urlparse

# arXiv ID variants:
#   New scheme (2007-04 onward): "2210.12345" or "2210.12345v3"
#   Old scheme (pre-2007):       "hep-th/0701005" / "math.GT/0309136"
_ARXIV_NEW_ID = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}(?:v\d+)?$")
_ARXIV_OLD_ID = re.compile(r"^[a-z\-]+(?:\.[A-Za-z\-]+)?/(\d{2})(\d{2})\d+(?:v\d+)?$")


@dataclass(frozen=True)
class CutoffDate:
    """Inclusive upper bound on paper publication date."""

    year: int
    month: int
    day: int
    precision: str  # "year" | "month" | "day"

    def to_string(self) -> str:
        if self.precision == "year":
            return f"{self.year:04d}"
        if self.precision == "month":
            return f"{self.year:04d}-{self.month:02d}"
        return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"

    def to_date(self) -> date:
        return date(self.year, self.month, self.day)

    def s2_year_param(self) -> str:
        """Format compatible with the Semantic Scholar ``year`` filter.

        S2 accepts ``year=2019``, ``year=2016-2020``, ``year=-2015`` (<=2015),
        and ``year=2010-`` (>=2010). We use the ``-YYYY`` form to mean
        "publication year <= YYYY" since S2's year field is year-only.
        """
        return f"-{self.year}"

    def to_metadata(self) -> dict[str, str]:
        return {"value": self.to_string(), "precision": self.precision}


def parse_cutoff(token: str | None) -> CutoffDate | None:
    """Parse ``YYYY``, ``YYYY-MM`` or ``YYYY-MM-DD``. Return ``None`` for empty
    input. Raise ``ValueError`` for non-empty input that is not a valid date so
    the user gets a clear error rather than a silent fallback."""
    text = str(token or "").strip()
    if not text:
        return None
    parts = text.split("-")
    if len(parts) > 3:
        raise ValueError(f"invalid cutoff date: {token!r} (expected YYYY[-MM[-DD]])")
    try:
        year = int(parts[0])
    except ValueError as exc:
        raise ValueError(f"invalid cutoff date year: {token!r}") from exc
    if not (1900 <= year <= 2100):
        raise ValueError(f"cutoff year out of range: {token!r}")

    if len(parts) == 1:
        return CutoffDate(year=year, month=12, day=31, precision="year")

    try:
        month = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"invalid cutoff date month: {token!r}") from exc
    if not (1 <= month <= 12):
        raise ValueError(f"cutoff month out of range: {token!r}")

    if len(parts) == 2:
        last_day = monthrange(year, month)[1]
        return CutoffDate(year=year, month=month, day=last_day, precision="month")

    try:
        day = int(parts[2])
        date(year, month, day)
    except ValueError as exc:
        raise ValueError(f"invalid cutoff date day: {token!r}") from exc
    return CutoffDate(year=year, month=month, day=day, precision="day")


def derive_cutoff_from_source(source: str | None) -> CutoffDate | None:
    """Best-effort: pull a date from an arXiv URL or bare arXiv ID.

    Returns ``None`` for non-arXiv inputs (PDFs, opaque URLs, etc.).
    """
    text = str(source or "").strip()
    if not text:
        return None

    arxiv_id = _arxiv_id_from_token(text)
    if not arxiv_id:
        return None

    m = _ARXIV_NEW_ID.match(arxiv_id) or _ARXIV_OLD_ID.match(arxiv_id)
    if not m:
        return None
    yy = int(m.group(1))
    month = int(m.group(2))
    if not (1 <= month <= 12):
        return None
    # arXiv started in 1991. The new scheme starts in April 2007 (`0704`).
    # Years 91..99 -> 1991..1999; 00..50 -> 2000..2050.
    year = 1900 + yy if yy >= 91 else 2000 + yy
    last_day = monthrange(year, month)[1]
    return CutoffDate(year=year, month=month, day=last_day, precision="month")


def is_after_cutoff(
    *, paper_year: object | None, paper_published: str | None, cutoff: CutoffDate
) -> bool:
    """Return ``True`` iff the paper is strictly later than ``cutoff``.

    Preference order:
    1. If ``paper_published`` is a parseable ISO-like date, use day precision.
    2. Otherwise fall back to ``paper_year`` for year-precision compare.
    3. If neither is available, return ``False`` (keep the paper rather than
       silently dropping records with missing metadata).
    """
    pub_date = _parse_iso_date(paper_published)
    if pub_date is not None:
        return pub_date > cutoff.to_date()

    year_int = _coerce_year(paper_year)
    if year_int is None:
        return False
    return year_int > cutoff.year


def filter_papers(
    papers: list[dict] | None, cutoff: CutoffDate | None
) -> tuple[list[dict], list[dict]]:
    """Split papers into ``(kept, dropped)`` by ``cutoff``.

    Non-dict entries are silently skipped. When ``cutoff`` is ``None`` every
    paper is kept (no filtering).
    """
    if cutoff is None:
        return [p for p in (papers or []) if isinstance(p, dict)], []

    kept: list[dict] = []
    dropped: list[dict] = []
    for row in papers or []:
        if not isinstance(row, dict):
            continue
        published = (
            str(row.get("published") or row.get("updated") or "").strip() or None
        )
        if is_after_cutoff(
            paper_year=row.get("year"),
            paper_published=published,
            cutoff=cutoff,
        ):
            dropped.append(row)
        else:
            kept.append(row)
    return kept, dropped


def _arxiv_id_from_token(token: str) -> str:
    parsed = urlparse(token)
    if parsed.scheme and parsed.netloc and not parsed.netloc.lower().endswith("arxiv.org"):
        return ""
    raw = parsed.path if parsed.scheme else token
    # Tolerate scheme-less URLs like "arxiv.org/abs/2210.12345" — urlparse leaves
    # the entire string in .path, so strip a leading "arxiv.org/" if present.
    lowered = raw.lower().lstrip("/")
    if lowered.startswith("arxiv.org/"):
        raw = raw[raw.lower().index("arxiv.org/") + len("arxiv.org/"):]
    path = raw.strip("/")
    for prefix in ("abs/", "pdf/"):
        if path.startswith(prefix):
            path = path[len(prefix):]
            break
    if path.lower().endswith(".pdf"):
        path = path[:-4]
    return path


def _parse_iso_date(token: str | None) -> date | None:
    text = str(token or "").strip()
    if not text:
        return None
    head = text[:10]
    if len(head) < 10 or head[4] != "-" or head[7] != "-":
        return None
    try:
        return date(int(head[:4]), int(head[5:7]), int(head[8:10]))
    except ValueError:
        return None


def _coerce_year(value: object | None) -> int | None:
    if value is None:
        return None
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    if not (1900 <= year <= 2100):
        return None
    return year
