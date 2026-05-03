"""LLM-driven recheck for fake-flagged references.

When `verify.hallucination.to_issue` emits an error/fake issue, this module
asks the LLM whether the citation is actually a *legitimate non-academic*
reference (system card, technical report, vendor announcement, dataset card,
standard, documentation, white paper, etc.). If yes, the original error is
downgraded to a warning under the `non_academic` category.

The recheck is conservative: when the LLM call fails, returns malformed JSON,
or signals UNCERTAIN, the original error is preserved.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from refcopilot.llm.client import call_json
from refcopilot.models import (
    ExternalRecord,
    Issue,
    IssueCategory,
    Reference,
    Severity,
)

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are a citation classifier. A citation has been flagged because no "
    "matching academic record was found on arXiv or Semantic Scholar. Decide "
    "whether it is *actually* a legitimate non-academic reference rather than a "
    "fabrication.\n\n"
    "Legitimate non-academic categories include: system cards, model cards, "
    "technical reports, vendor blog posts or announcements (OpenAI / Anthropic / "
    "Google / Meta / etc.), dataset cards, software / API documentation, "
    "industry standards (ISO, IEEE, NIST), white papers, and government "
    "reports.\n\n"
    "Return JSON ONLY:\n"
    '  {"is_non_academic": true|false,\n'
    '   "citation_type": "system_card" | "blog_post" | "technical_report" | '
    '"dataset_card" | "documentation" | "standard" | "white_paper" | "other",\n'
    '   "reasoning": "<1 short sentence>"}\n\n'
    "Be conservative: if you cannot confidently identify a known non-academic "
    "source, return is_non_academic=false. Do not invent sources."
)


def recheck(
    reference: Reference,
    matches: list[ExternalRecord],
    original_issue: Issue,
) -> Issue:
    """If the LLM says this is a legitimate non-academic ref, return a warning;
    otherwise return the original error issue."""
    user_prompt = _build_prompt(reference, matches)
    try:
        payload: Any = call_json(prompt=user_prompt, system=_SYSTEM_PROMPT)
    except Exception as exc:
        logger.warning("non_academic recheck call raised: %s", exc)
        return original_issue

    if not isinstance(payload, dict) or payload.get("status") in ("error", "unknown"):
        return original_issue

    if not bool(payload.get("is_non_academic")):
        return original_issue

    citation_type = str(payload.get("citation_type") or "other").strip().lower()
    reasoning = str(payload.get("reasoning") or "").strip()

    return Issue(
        severity=Severity.WARNING,
        category=IssueCategory.NON_ACADEMIC,
        code=f"non_academic::{citation_type}",
        message=(
            f"Reference is a non-academic source ({citation_type}); "
            f"no arXiv / Semantic Scholar record is expected."
        ),
        suggestion=reasoning or None,
        confidence=0.8,
    )


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
            "authors": m.authors[:3],
            "year": m.year,
        }
        for m in matches[:3]
    ]
    return (
        "CITATION:\n"
        + json.dumps(citation, ensure_ascii=False, indent=2)
        + "\n\nCANDIDATES (poor matches that triggered the fake flag):\n"
        + json.dumps(candidates, ensure_ascii=False, indent=2)
        + '\n\nReturn JSON: {"is_non_academic": ..., "citation_type": ..., "reasoning": ...}'
    )
