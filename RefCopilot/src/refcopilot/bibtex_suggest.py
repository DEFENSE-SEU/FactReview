"""Generate corrected BibTeX entries from a verified :class:`MergedRecord`.

The output is a drop-in replacement that users can paste into their .bib file
to fix the warnings RefCopilot raised against the original citation. A leading
provenance comment lists which backend supplied each field so reviewers can
audit the suggestion before accepting it.
"""

from __future__ import annotations

import re

from refcopilot.models import Backend, MergedRecord, Reference, SourceFormat
from refcopilot.verify.thresholds import ARXIV_VENUE_ALIASES

_INPROCEEDINGS_HINTS = ("conference", "proceedings", "workshop", "symposium", "meeting")


def suggest_bibtex(reference: Reference, merged: MergedRecord | None) -> str:
    """Render a corrected ``@<type>{...}`` entry for *reference*.

    Returns an empty string when *merged* is ``None`` or carries no usable
    fields, so callers can branch on truthiness.
    """
    if merged is None:
        return ""

    entry_type = _entry_type(reference, merged)
    bibkey = reference.bibkey or _generate_bibkey(merged)

    fields: list[tuple[str, str]] = []
    if merged.title:
        fields.append(("title", _normalize_value(merged.title)))
    if merged.authors:
        fields.append(("author", " and ".join(_normalize_value(a) for a in merged.authors)))
    if merged.year:
        fields.append(("year", str(merged.year)))
    if merged.venue:
        venue_field = "booktitle" if entry_type == "inproceedings" else "journal"
        fields.append((venue_field, _normalize_value(merged.venue)))
    if merged.doi:
        fields.append(("doi", merged.doi))
    if merged.arxiv_id:
        fields.append(("eprint", merged.arxiv_id))
        fields.append(("archivePrefix", "arXiv"))
    if merged.url:
        fields.append(("url", merged.url))

    if not fields:
        return ""

    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields)
    head = _format_provenance_comment(merged)
    return f"{head}\n@{entry_type}{{{bibkey},\n{body},\n}}"


def _format_provenance_comment(merged: MergedRecord) -> str:
    by_backend: dict[Backend, list[str]] = {}
    for field, backend in merged.provenance.items():
        by_backend.setdefault(backend, []).append(field)

    if not by_backend:
        return "% Suggested by RefCopilot."

    lines = ["% Suggested by RefCopilot. Field provenance:"]
    for backend in sorted(by_backend, key=lambda b: b.value):
        url = _backend_url(backend, merged)
        suffix = f" — {url}" if url else ""
        lines.append(f"%   {backend.value}: {', '.join(sorted(by_backend[backend]))}{suffix}")
    return "\n".join(lines)


def _backend_url(backend: Backend, merged: MergedRecord) -> str | None:
    for src in merged.sources:
        if src.backend == backend and src.url:
            return src.url
    return None


def _entry_type(reference: Reference, merged: MergedRecord) -> str:
    if reference.source_format == SourceFormat.BIBTEX and reference.raw:
        m = re.match(r"\s*@(\w+)\s*\{", reference.raw)
        if m:
            return m.group(1).lower()

    venue = (merged.venue or "").strip().lower()
    if not venue or venue in ARXIV_VENUE_ALIASES:
        return "misc" if merged.arxiv_id else "article"
    if any(hint in venue for hint in _INPROCEEDINGS_HINTS):
        return "inproceedings"
    return "article"


def _generate_bibkey(merged: MergedRecord) -> str:
    surname = ""
    if merged.authors:
        parts = merged.authors[0].split()
        surname = parts[-1] if parts else merged.authors[0]
    year = str(merged.year) if merged.year else ""
    title_word = ""
    for word in (merged.title or "").split():
        cleaned = "".join(c for c in word if c.isalpha())
        if len(cleaned) > 3:
            title_word = cleaned
            break
    cleaned = "".join(c for c in f"{surname}{year}{title_word}" if c.isalnum())
    return cleaned.lower() or "ref"


def _normalize_value(text: str) -> str:
    return " ".join(str(text).split())
