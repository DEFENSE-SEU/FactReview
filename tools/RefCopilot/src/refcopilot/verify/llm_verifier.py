"""LLM-driven secondary verification for uncertain references.

Given a reference whose pre-screen verdict is UNCERTAIN or LIKELY (and we have
some retrieved candidates), ask the LLM to judge whether the citation
plausibly refers to the same work as one of the candidates. The LLM has access
to broad academic knowledge from pretraining; for ambiguous cases this is a
useful extra signal.

The LLM is asked to return a strict JSON object:
  {"verdict": "LIKELY" | "UNLIKELY" | "UNCERTAIN", "reason": "..."}

`verdict` here is the hallucination verdict (per `apply_hallucination_verdict`):
  LIKELY    → fake
  UNLIKELY  → real (promote unverified to verified)
  UNCERTAIN → no opinion

If the LLM call fails or returns malformed output, we keep the pre-screen
verdict.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from refcopilot.llm.client import call_json
from refcopilot.models import ExternalRecord, HallucinationVerdict, Reference

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are a careful bibliographic verifier. Given a citation and a list of "
    "retrieved candidate records from arXiv / Semantic Scholar, decide whether "
    "the citation plausibly refers to a real paper. Return JSON ONLY:\n"
    '{"verdict": "LIKELY" | "UNLIKELY" | "UNCERTAIN", "reason": "<1-2 sentences>"}\n'
    "Where:\n"
    "- LIKELY = the citation appears to be a hallucination (the paper does not exist as cited).\n"
    "- UNLIKELY = the citation refers to a real paper (one of the candidates, or another known work).\n"
    "- UNCERTAIN = you cannot confidently decide.\n"
    "Use your knowledge of the field. If candidates are empty but the cited "
    "title rings true, prefer UNCERTAIN over LIKELY."
)


def verify(
    reference: Reference,
    matches: list[ExternalRecord],
    *,
    initial: HallucinationVerdict,
) -> HallucinationVerdict:
    """Return the LLM-assisted final verdict, or `initial` on failure."""
    if initial == HallucinationVerdict.UNLIKELY:
        # Already confident the paper exists; don't waste an LLM call.
        return initial

    user_prompt = _build_prompt(reference, matches)
    try:
        payload: Any = call_json(prompt=user_prompt, system=_SYSTEM_PROMPT)
    except Exception as exc:
        logger.warning("LLM verifier call raised: %s", exc)
        return initial

    if not isinstance(payload, dict) or payload.get("status") in ("error", "unknown"):
        return initial

    raw_verdict = str(payload.get("verdict") or "").strip().upper()
    if raw_verdict in {"LIKELY", "UNLIKELY", "UNCERTAIN"}:
        return HallucinationVerdict(raw_verdict)
    return initial


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
        + '\n\nReturn JSON: {"verdict": "...", "reason": "..."}'
    )
