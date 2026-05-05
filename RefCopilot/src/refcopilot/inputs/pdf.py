"""PDF → text + bibliography section extraction.

Tries pypdf first; falls back to pdfplumber if pypdf returns very little text.
Then locates the bibliography section using a set of header regexes (last
match after 20% of the document) and truncates at appendix-like markers.
"""

from __future__ import annotations

import logging
import re
from io import BytesIO
from pathlib import Path

logger = logging.getLogger(__name__)


# Header patterns: prefer the LAST match in the document and require it
# to occur after the first 20% of the text (to skip Table-of-Contents hits).
_BIB_HEADERS: list[re.Pattern[str]] = [
    re.compile(
        r"\n[\s\d\t\r\xa0]*\d+\.?\s+(?:REFERENCES|References|Bibliography|BIBLIOGRAPHY)\s*:?\s*\n"
    ),
    re.compile(
        r"\n[\s\t\r\xa0\d]*(?:REFERENCES|References|Bibliography|BIBLIOGRAPHY)\s*:?\s*(?:\d*\s*)\n"
    ),
]

_APPENDIX_CUTS = [
    re.compile(r"\n\s*A(?:PPENDIX|ppendix|\.\s*\d|\s+[A-Z])"),
    re.compile(r"\n\s*ACKNOWLEDGMENTS?\b", re.IGNORECASE),
    re.compile(r"\n\s*ACKNOWLEDGEMENTS?\b", re.IGNORECASE),
    re.compile(r"\n\s*Supplementary\s+Material\b", re.IGNORECASE),
]


def extract_text(path: str | Path) -> str:
    pdf_bytes = Path(path).read_bytes()
    text = _extract_with_pypdf(pdf_bytes)
    if len(text) < 500:
        logger.info("pypdf produced %d chars; falling back to pdfplumber", len(text))
        text = _extract_with_pdfplumber(pdf_bytes) or text
    return text


def _extract_with_pypdf(pdf_bytes: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(pdf_bytes))
    pages = [(p.extract_text() or "").strip() for p in reader.pages]
    return "\n\n".join(pages)


def _extract_with_pdfplumber(pdf_bytes: bytes) -> str:
    import pdfplumber

    parts: list[str] = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = (page.extract_text() or "").strip()
            if t:
                parts.append(t)
    return "\n\n".join(parts)


def find_bibliography(text: str) -> str:
    """Return the bibliography substring, or empty string if not found."""
    if not text:
        return ""

    n = len(text)
    earliest_acceptable = int(n * 0.2)
    best: re.Match[str] | None = None
    for pattern in _BIB_HEADERS:
        for m in pattern.finditer(text):
            if m.start() < earliest_acceptable:
                continue
            if best is None or m.start() > best.start():
                best = m
    if best is None:
        return ""

    section = text[best.end():]
    for cut in _APPENDIX_CUTS:
        cm = cut.search(section)
        if cm:
            section = section[: cm.start()]
    return section.strip()


def extract_bibliography(path: str | Path) -> str:
    text = extract_text(path)
    bib = find_bibliography(text)
    if not bib:
        # No bibliography header matched. Fall back to the last 30% of the
        # document so the LLM extractor still has plausible candidates to
        # scan, and log it so callers can investigate noisy outputs.
        logger.warning(
            "no bibliography header found in %s; using last 30%% of document text",
            path,
        )
        bib = text[int(len(text) * 0.7):]
    return bib
