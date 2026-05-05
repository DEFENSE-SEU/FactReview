"""LLM-driven secondary verification for uncertain references.

When the heuristic :func:`refcopilot.verify.hallucination.pre_screen` returns
``UNCERTAIN`` or ``LIKELY``, this module asks the LLM whether the citation
plausibly refers to the same work as one of the retrieved candidates. The LLM
has broad academic knowledge from pretraining, which is a useful extra signal
for ambiguous cases.

The LLM returns a strict JSON object::

    {
      "verdict": "LIKELY" | "UNLIKELY" | "UNCERTAIN",
      "reason": "...",
      "suggestion": {            # optional, only when verdict == UNLIKELY
        "title": "...",
        "authors": ["..."],
        "year": 2025,
        "arxiv_id": "...",
        "doi": "..."
      }
    }

where the verdict mirrors :class:`refcopilot.models.HallucinationVerdict`
(``LIKELY`` = fake, ``UNLIKELY`` = real, ``UNCERTAIN`` = no opinion). The
optional ``suggestion`` block lets the pipeline re-run backend lookups with
a corrected title / arXiv id when the LLM thinks the cited paper is real
but the heuristic search came up empty.

If the LLM call fails or returns malformed output, the pre-screen verdict is
preserved.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel

from refcopilot.llm.client import call_json
from refcopilot.models import ExternalRecord, HallucinationVerdict, Reference

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are a careful bibliographic verifier. Given a citation and a list of "
    "retrieved candidate records from arXiv / Semantic Scholar, decide whether "
    "the citation plausibly refers to a real paper. Return JSON ONLY:\n"
    '{"verdict": "LIKELY" | "UNLIKELY" | "UNCERTAIN", "reason": "<1-2 sentences>", '
    '"suggestion": {"title": "...", "authors": ["..."], "year": 2025, "arxiv_id": "...", "doi": "..."}}\n'
    "Where:\n"
    "- LIKELY = the citation appears to be a hallucination (the paper does not exist as cited).\n"
    "- UNLIKELY = the citation refers to a real paper (one of the candidates, or another known work).\n"
    "- UNCERTAIN = you cannot confidently decide.\n"
    "Use your knowledge of the field. If candidates are empty but the cited "
    "title rings true, prefer UNCERTAIN over LIKELY.\n"
    "When (and only when) verdict == UNLIKELY AND the citation seems to point at a real "
    "paper that the candidates do not cover (e.g. cited title has a typo / different "
    "punctuation / older variant), include a `suggestion` object with the canonical "
    "metadata you believe is correct so the pipeline can re-search backends. Omit "
    "`suggestion` (or set it to null) otherwise. Do not invent identifiers; leave a "
    "field null/missing if you do not know it."
)


class LLMSuggestion(BaseModel):
    """LLM's best guess at the canonical citation metadata.

    All fields optional — the LLM may know only the title, or only an arxiv
    id, etc. We don't reuse :class:`Reference` because it requires ``raw``
    and ``source_format`` which the LLM has no business filling.
    """

    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    arxiv_id: str | None = None
    doi: str | None = None


class LLMVerification(BaseModel):
    verdict: HallucinationVerdict
    suggestion: LLMSuggestion | None = None


def verify(
    reference: Reference,
    matches: list[ExternalRecord],
    *,
    initial: HallucinationVerdict,
) -> LLMVerification:
    """Return the LLM-assisted final verdict (and optional suggestion)."""
    if initial == HallucinationVerdict.UNLIKELY:
        # Already confident the paper exists; don't waste an LLM call.
        return LLMVerification(verdict=initial)

    user_prompt = _build_prompt(reference, matches)
    try:
        payload: Any = call_json(prompt=user_prompt, system=_SYSTEM_PROMPT)
    except Exception as exc:
        logger.warning("LLM verifier call raised: %s", exc)
        return LLMVerification(verdict=initial)

    if not isinstance(payload, dict) or payload.get("status") in ("error", "unknown"):
        return LLMVerification(verdict=initial)

    raw_verdict = str(payload.get("verdict") or "").strip().upper()
    if raw_verdict not in {"LIKELY", "UNLIKELY", "UNCERTAIN"}:
        return LLMVerification(verdict=initial)

    verdict = HallucinationVerdict(raw_verdict)
    suggestion = _parse_suggestion(payload.get("suggestion")) if verdict == HallucinationVerdict.UNLIKELY else None
    return LLMVerification(verdict=verdict, suggestion=suggestion)


def _parse_suggestion(raw: Any) -> LLMSuggestion | None:
    if not isinstance(raw, dict):
        return None

    title = _opt_str(raw.get("title"))
    arxiv_id = _opt_str(raw.get("arxiv_id"))
    doi = _opt_str(raw.get("doi"))
    year_raw = raw.get("year")
    year: int | None = year_raw if isinstance(year_raw, int) else None
    authors_raw = raw.get("authors")
    authors: list[str] | None = None
    if isinstance(authors_raw, list):
        authors = [str(a).strip() for a in authors_raw if str(a).strip()]
        if not authors:
            authors = None

    # Suggestion is only useful if it tells us *something* to look up by.
    if not (title or arxiv_id or doi):
        return None

    return LLMSuggestion(
        title=title,
        authors=authors,
        year=year,
        arxiv_id=arxiv_id,
        doi=doi,
    )


def _opt_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    s = value.strip()
    return s or None


def _build_prompt(reference: Reference, matches: list[ExternalRecord]) -> str:
    citation = {
        "title": reference.title,
        "authors": reference.authors,
        "year": reference.year,
        "venue": reference.venue,
        "doi": reference.doi,
        "arxiv_id": reference.arxiv_id,
        "url": reference.url,
        "raw": (reference.raw or "")[:500],
    }
    candidates = [
        {
            "backend": m.backend.value,
            "title": m.title,
            "authors": m.authors[:6],
            "year": m.year,
            "venue": m.venue or m.publication_venue or m.journal,
            "doi": m.doi,
            "arxiv_id": m.arxiv_id,
            "url": m.url,
        }
        for m in matches[:5]
    ]
    return (
        "CITATION:\n"
        + json.dumps(citation, ensure_ascii=False, indent=2)
        + "\n\nCANDIDATES:\n"
        + json.dumps(candidates, ensure_ascii=False, indent=2)
        + '\n\nReturn JSON: {"verdict": "...", "reason": "...", "suggestion": {...} | null}'
    )
