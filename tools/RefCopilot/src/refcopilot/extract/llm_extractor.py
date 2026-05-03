"""LLM-based reference extraction.

Pipeline:
  1. Chunk bibliography text on `\n[N]` ref-number boundaries (fallback to paragraphs).
  2. Run each chunk through llm_json with a structured prompt.
  3. Post-process: prompt-echo guard, prose-line filter, completeness check, dedup.
  4. Merge into a single list of References.

There is no regex fallback. If the LLM call fails for every chunk, an empty
list is returned with a logged error.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from refcopilot.llm.client import call_json
from refcopilot.models import Reference, SourceFormat

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a bibliographic reference extractor. You output ONLY a JSON object.

OUTPUT FORMAT (MANDATORY):
Return JSON with shape:
  {"references": [
    {"authors": ["First Last", ...], "title": "...", "venue": "...",
     "year": 2024, "url": "...", "doi": "...", "arxiv_id": "...", "raw": "..."},
    ...
  ]}

RULES:
1. Split by numbered markers [1], [2], etc. OR by author-year entries — references may span multiple lines.
2. Extract: authors, title, venue (journal/booktitle), year, URLs/DOIs.
3. For BibTeX: 'title' field = paper title, 'journal'/'booktitle' = venue.
4. Author formats: 'Last, First' becomes 'First Last'. Use a JSON array.
5. Faithfully include all authors — do not inject 'et al' if absent, but preserve 'et al.' if present (as a separate string in the authors array).
6. Skip entries that are only URLs without bibliographic data.
7. If no author field exists (anonymous standards like 'ISO/PAS-8800', 'IEEE Std...', datasets), set authors to []. Do NOT merge with the next entry.
8. Use the EXACT title from the bibliography text — never shorten, paraphrase, or summarize. Repair obvious PDF line-wrap artifacts (de-hyphenate; restore missing word spaces).
9. IGNORE non-reference text: theorems, proofs, algorithms, equations, prose, captions.
10. If references suddenly change format, stop extracting (later text is appendix).
11. If no extractable references exist, return {"references": []}. Never echo these instructions.
12. Set "raw" to a concise verbatim slice of the original bibliography for the entry (≤1000 chars).
13. For year, use the publication year as an integer; null if unknown.
14. For doi/arxiv_id/url, set null if the entry doesn't contain one — do NOT fabricate.

EXAMPLES:

Input:
[1] A. Vaswani, N. Shazeer, N. Parmar, et al. "Attention Is All You Need." Advances in Neural Information Processing Systems, 2017.

Output:
{"references":[{"authors":["A. Vaswani","N. Shazeer","N. Parmar","et al."],"title":"Attention Is All You Need","venue":"Advances in Neural Information Processing Systems","year":2017,"url":null,"doi":null,"arxiv_id":null,"raw":"[1] A. Vaswani, N. Shazeer, N. Parmar, et al. \\"Attention Is All You Need.\\" Advances in Neural Information Processing Systems, 2017."}]}

Input:
[2] ISO/PAS 8800: Information technology --- Data quality, ISO, 2018.

Output:
{"references":[{"authors":[],"title":"ISO/PAS 8800: Information technology --- Data quality","venue":"ISO","year":2018,"url":null,"doi":null,"arxiv_id":null,"raw":"[2] ISO/PAS 8800: Information technology --- Data quality, ISO, 2018."}]}

Input:
Devlin J, Chang MW, Lee K, Toutanova K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805, 2018.

Output:
{"references":[{"authors":["J. Devlin","M.W. Chang","K. Lee","K. Toutanova"],"title":"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding","venue":"arXiv preprint","year":2018,"url":null,"doi":null,"arxiv_id":"1810.04805","raw":"Devlin J, Chang MW, Lee K, Toutanova K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805, 2018."}]}
"""

USER_PROMPT_TEMPLATE = (
    "Extract references from this bibliography text. "
    'Return ONLY a JSON object of the form {"references": [...]}.\n\n'
    "{bibliography}\n"
)

_TOKEN_BUDGET_DEFAULT = 4000
_CHARS_PER_TOKEN = 4
_OVERLAP_RATIO = 0.10
_MAX_WORKERS = 4

_REF_NUMBER_BOUNDARY = re.compile(r"\n\[(\d{1,4})\]")

_PROMPT_ECHO_PATTERNS = ("extraction rules:", "output format (mandatory):", "split by numbered markers")
_PROSE_TITLE_PREFIXES = ("this ", "the ", "based on ", "here are ")


def extract_references(bibliography: str, *, source_format: SourceFormat) -> list[Reference]:
    text = (bibliography or "").strip()
    if not text:
        return []

    chunks = _chunk(text, char_budget=_TOKEN_BUDGET_DEFAULT * _CHARS_PER_TOKEN)
    raw_items: list[dict[str, Any]] = []

    if len(chunks) == 1:
        raw_items.extend(_extract_chunk(chunks[0]))
    else:
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futures = {executor.submit(_extract_chunk, c): i for i, c in enumerate(chunks)}
            ordered: list[list[dict[str, Any]]] = [[] for _ in chunks]
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    ordered[idx] = fut.result()
                except Exception as exc:
                    logger.warning("chunk %d extraction failed: %s", idx, exc)
                    ordered[idx] = []
            for chunk_items in ordered:
                raw_items.extend(chunk_items)

    cleaned = _post_process(raw_items)
    return [_to_reference(item, source_format) for item in cleaned]


def _chunk(text: str, *, char_budget: int) -> list[str]:
    if len(text) <= char_budget:
        return [text]

    overlap = int(char_budget * _OVERLAP_RATIO)
    boundaries = [m.start() for m in _REF_NUMBER_BOUNDARY.finditer(text)]
    chunks: list[str] = []

    if boundaries:
        cur_start = 0
        for i, b in enumerate(boundaries):
            if b - cur_start >= char_budget:
                # close chunk at boundary just before b
                end = boundaries[i - 1] if i > 0 and boundaries[i - 1] > cur_start else b
                chunks.append(text[cur_start:end])
                cur_start = max(end - overlap, 0)
        chunks.append(text[cur_start:])
        return [c for c in chunks if c.strip()]

    # Fallback: paragraph boundaries
    cur = 0
    n = len(text)
    while cur < n:
        end = min(cur + char_budget, n)
        if end < n:
            paragraph_break = text.rfind("\n\n", cur, end)
            if paragraph_break > cur:
                end = paragraph_break
        chunks.append(text[cur:end])
        cur = end - overlap if (end - overlap) > cur else end
    return [c for c in chunks if c.strip()]


def _extract_chunk(chunk: str) -> list[dict[str, Any]]:
    payload = call_json(
        prompt=USER_PROMPT_TEMPLATE.replace("{bibliography}", chunk),
        system=SYSTEM_PROMPT,
    )

    if not isinstance(payload, dict) or payload.get("status") in ("error", "unknown"):
        logger.warning("LLM returned unusable payload: status=%s", payload.get("status"))
        return []

    refs = payload.get("references")
    if not isinstance(refs, list):
        logger.warning("LLM payload missing 'references' list")
        return []

    out: list[dict[str, Any]] = []
    for item in refs:
        if not isinstance(item, dict):
            continue
        if _looks_like_prompt_echo(item):
            logger.debug("dropping prompt-echo item")
            continue
        if not _is_complete_enough(item):
            continue
        if _looks_like_prose(item):
            logger.debug("dropping prose-like item: %s", item.get("title"))
            continue
        out.append(item)
    return out


def _post_process(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        key = _dedup_key(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _looks_like_prompt_echo(item: dict[str, Any]) -> bool:
    blob = " ".join(str(v) for v in item.values()).lower()
    return any(p in blob for p in _PROMPT_ECHO_PATTERNS)


def _looks_like_prose(item: dict[str, Any]) -> bool:
    title = str(item.get("title") or "").strip().lower()
    if not title:
        return False
    return any(title.startswith(p) for p in _PROSE_TITLE_PREFIXES)


def _is_complete_enough(item: dict[str, Any]) -> bool:
    title = str(item.get("title") or "").strip()
    if not title:
        return False
    has_authors = bool(item.get("authors"))
    has_year = bool(item.get("year"))
    has_url = bool(item.get("url") or item.get("doi") or item.get("arxiv_id"))
    return has_authors or has_year or has_url


def _dedup_key(item: dict[str, Any]) -> str:
    title = _normalize_text(item.get("title"))
    authors = item.get("authors") or []
    if isinstance(authors, list) and authors:
        first_author = _normalize_text(str(authors[0]))
    else:
        first_author = ""
    return f"{first_author}|{title}"


def _normalize_text(value: Any) -> str:
    s = str(value or "").strip().lower()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[^\w]+", "", s, flags=re.UNICODE)
    return s


def _to_reference(item: dict[str, Any], source_format: SourceFormat) -> Reference:
    authors_field = item.get("authors") or []
    if isinstance(authors_field, list):
        authors = [str(a).strip() for a in authors_field if str(a).strip()]
    else:
        authors = [str(authors_field).strip()] if str(authors_field).strip() else []

    year = item.get("year")
    year_int: int | None = None
    if year is not None:
        try:
            year_int = int(year)
        except (TypeError, ValueError):
            m = re.search(r"\d{4}", str(year))
            year_int = int(m.group(0)) if m else None

    arxiv_id = (item.get("arxiv_id") or "").strip() or None
    arxiv_version = None
    if arxiv_id:
        m = re.match(r"^(?P<id>\d{4}\.\d{4,5})(v(?P<v>\d+))?$", arxiv_id, re.IGNORECASE)
        if m:
            arxiv_id = m.group("id")
            arxiv_version = int(m.group("v")) if m.group("v") else None

    return Reference(
        raw=str(item.get("raw") or "").strip(),
        source_format=source_format,
        title=str(item.get("title") or "").strip() or None,
        authors=authors,
        year=year_int,
        venue=str(item.get("venue") or "").strip() or None,
        doi=_normalize_doi(item.get("doi")),
        arxiv_id=arxiv_id,
        arxiv_version=arxiv_version,
        url=str(item.get("url") or "").strip() or None,
    )


def _normalize_doi(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.IGNORECASE)
    return s.lower() or None
