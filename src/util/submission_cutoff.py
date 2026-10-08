"""Resolve the explicitly requested arXiv fallback with day-level provenance."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from util.cutoff_date import CutoffDate, parse_submission_deadline


def arxiv_identifier(source: str) -> str:
    token = source.strip()
    if "://" in token:
        parsed = urlsplit(token)
        if (
            parsed.scheme not in {"https", "http"}
            or parsed.hostname not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}
            or parsed.username
            or parsed.password
        ):
            raise ValueError("arXiv fallback requires an arXiv URL or explicit identifier")
        token = parsed.path.removeprefix("/")
    token = re.sub(r"^(?:arxiv:|abs/|pdf/)", "", token, flags=re.I).removesuffix(".pdf")
    if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z.-]+/\d{7})(?:v\d+)?", token, re.I):
        raise ValueError("arXiv fallback requires an arXiv URL or --arxiv-id for a local PDF")
    return re.sub(r"v\d+$", "", token, flags=re.I)


async def resolve_arxiv_first_submission(source: str, *, lookup=None) -> tuple[CutoffDate, dict]:
    """Use Atom's published date, never the revision date or the ID's month.

    https://info.arxiv.org/help/api/user-manual.html#_entry_metadata
    The caller must require explicit opt-in before invoking this function.
    """
    identifier = arxiv_identifier(source)
    if lookup is None:
        from fact_generation.positioning.paper_search import (
            PaperReadConfig,
            PaperSearchAdapter,
            PaperSearchConfig,
        )

        adapter = PaperSearchAdapter(
            PaperSearchConfig(True, "arxiv", None, None, "", 30, "", 10),
            PaperReadConfig(None, None, "", 30),
        )
        lookup = adapter.lookup_metadata
    response = await lookup(identifier=identifier)
    if not isinstance(response, dict) or response.get("success") is not True:
        raise ValueError("arXiv first-submission metadata could not be resolved")
    paper = response.get("paper")
    if not isinstance(paper, dict) or arxiv_identifier(str(paper.get("arxiv_id") or "")) != identifier:
        raise ValueError("arXiv first-submission metadata identity does not match the input")
    published = str(paper.get("published") or "").strip()
    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2}))?", published
    ):
        raise ValueError("arXiv first-submission metadata has no exact published date")
    cutoff = parse_submission_deadline(published[:10])
    return cutoff, {
        "source": "arxiv_first_submission",
        "value": cutoff.to_string(),
        "arxiv_id": identifier,
        "metadata_url": f"https://arxiv.org/abs/{identifier}",
        "published": published,
        "venue_deadline_known": False,
    }
