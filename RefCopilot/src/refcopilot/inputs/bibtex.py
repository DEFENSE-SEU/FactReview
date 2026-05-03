"""BibTeX → list[Reference] using pybtex.

pybtex handles standard BibTeX; we tolerate occasional malformed entries by
parsing in a lenient mode and skipping bad blocks.
"""

from __future__ import annotations

import logging
import re
from io import StringIO
from pathlib import Path

from pybtex.database.input.bibtex import Parser as BibtexParser

from refcopilot.models import Reference, SourceFormat

logger = logging.getLogger(__name__)


_ARXIV_FROM_EPRINT = re.compile(r"(?P<id>\d{4}\.\d{4,5})(v(?P<v>\d+))?$", re.IGNORECASE)


def parse_file(path: str | Path) -> list[Reference]:
    text = Path(path).read_text(encoding="utf-8")
    return parse_string(text)


def parse_string(text: str) -> list[Reference]:
    parser = BibtexParser()
    try:
        bib_data = parser.parse_stream(StringIO(text))
    except Exception as exc:
        logger.warning("pybtex failed to parse bibtex: %s; falling back to per-entry retry", exc)
        return _parse_lenient(text)

    return [_entry_to_reference(key, entry, text) for key, entry in bib_data.entries.items()]


def _parse_lenient(text: str) -> list[Reference]:
    """Best-effort: split into @-blocks and parse one at a time."""
    references: list[Reference] = []
    for raw_entry in _split_at_blocks(text):
        try:
            parser = BibtexParser()
            bib_data = parser.parse_stream(StringIO(raw_entry))
            for key, entry in bib_data.entries.items():
                references.append(_entry_to_reference(key, entry, raw_entry))
        except Exception as exc:
            logger.debug("skipped malformed entry: %s", exc)
    return references


def _split_at_blocks(text: str) -> list[str]:
    """Split a BibTeX stream into individual @-blocks via brace depth counting."""
    blocks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        at = text.find("@", i)
        if at < 0:
            break
        brace = text.find("{", at)
        if brace < 0:
            break
        depth = 1
        j = brace + 1
        while j < n and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        blocks.append(text[at:j])
        i = j
    return blocks


def _entry_to_reference(bibkey: str, entry, raw_text: str) -> Reference:
    fields = {k.lower(): v for k, v in entry.fields.items()}
    persons = entry.persons.get("author", []) + entry.persons.get("editor", [])
    authors = [_format_person(p) for p in persons]

    title = _strip_braces(fields.get("title"))
    year = _to_int(fields.get("year"))
    venue = (
        _strip_braces(fields.get("booktitle"))
        or _strip_braces(fields.get("journal"))
        or _strip_braces(fields.get("school"))
        or _strip_braces(fields.get("institution"))
    )
    doi = _normalize_doi(fields.get("doi"))
    url = (fields.get("url") or "").strip() or None

    arxiv_id, arxiv_version = _extract_arxiv(fields)

    return Reference(
        raw=_entry_block_text(bibkey, raw_text),
        source_format=SourceFormat.BIBTEX,
        bibkey=bibkey,
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        doi=doi,
        arxiv_id=arxiv_id,
        arxiv_version=arxiv_version,
        url=url,
    )


def _format_person(p) -> str:
    first = " ".join(p.first_names + p.middle_names).strip()
    last = " ".join(p.prelast_names + p.last_names + p.lineage_names).strip()
    if first and last:
        return f"{first} {last}".strip()
    return last or first or " ".join(getattr(p, "_first_names", [])) or str(p)


def _strip_braces(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    while s.startswith("{") and s.endswith("}") and len(s) >= 2:
        s = s[1:-1].strip()
    return s or None


def _to_int(value) -> int | None:
    if value is None:
        return None
    s = str(value).strip()
    m = re.search(r"\d{4}", s)
    return int(m.group(0)) if m else None


def _normalize_doi(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.IGNORECASE)
    return s.lower() or None


def _extract_arxiv(fields: dict) -> tuple[str | None, int | None]:
    eprint = fields.get("eprint") or ""
    m = _ARXIV_FROM_EPRINT.search(str(eprint))
    if m:
        return m.group("id"), int(m.group("v")) if m.group("v") else None

    url = fields.get("url") or ""
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(?P<id>\d{4}\.\d{4,5})(?:v(?P<v>\d+))?", str(url), re.IGNORECASE)
    if m:
        return m.group("id"), int(m.group("v")) if m.group("v") else None

    return None, None


def _entry_block_text(bibkey: str, raw_text: str) -> str:
    """Return the original @{...} block for this bibkey if findable, else empty."""
    pat = re.compile(rf"@\w+\s*\{{\s*{re.escape(bibkey)}\b", re.IGNORECASE)
    m = pat.search(raw_text)
    if not m:
        return ""
    start = m.start()
    depth = 0
    i = raw_text.find("{", start)
    if i < 0:
        return ""
    j = i
    while j < len(raw_text):
        if raw_text[j] == "{":
            depth += 1
        elif raw_text[j] == "}":
            depth -= 1
            if depth == 0:
                return raw_text[start : j + 1]
        j += 1
    return raw_text[start:]
