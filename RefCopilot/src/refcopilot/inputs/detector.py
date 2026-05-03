"""Auto-detect input type from a string spec or file path."""

from __future__ import annotations

import re
from pathlib import Path

from refcopilot.models import SourceFormat

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_ARXIV_BARE_RE = re.compile(r"^(?:arxiv:)?\d{4}\.\d{4,5}(v\d+)?$", re.IGNORECASE)


def detect(spec: str) -> SourceFormat:
    s = (spec or "").strip()
    if not s:
        raise ValueError("empty input spec")

    if _URL_RE.match(s) or _ARXIV_BARE_RE.match(s):
        return SourceFormat.URL

    p = Path(s)
    if p.exists() and p.is_file():
        suffix = p.suffix.lower()
        if suffix == ".bib":
            return SourceFormat.BIBTEX
        if suffix == ".pdf":
            return SourceFormat.PDF
        # .tex / .txt / unknown → treat as plain text
        return SourceFormat.TEXT

    if "@" in s and "{" in s and "}" in s:
        return SourceFormat.BIBTEX

    return SourceFormat.TEXT
