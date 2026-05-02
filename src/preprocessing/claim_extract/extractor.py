"""Top-level entry point for §3.1b fact extraction.

Takes an ingested :class:`Paper` and produces a list of
:class:`Claim` objects. Depending on :class:`ClaimExtractCfg.mode`,
the extractor runs:

  - ``heuristic`` : regex-only (deterministic, no API cost).
  - ``llm``       : LLM-only (strict JSON).
  - ``auto``      : LLM with heuristic as backfill. If the LLM call
                    fails or returns zero claims, we still return the
                    heuristic set so downstream stages have something
                    to work with.

After extraction, broad claims are run through the decomposer so the
returned list is guaranteed to be in its final §3.1b form.
"""

from __future__ import annotations

import importlib.resources as ir
import logging
from dataclasses import dataclass

from schemas.claim import Claim, ClaimLocation, ClaimType
from schemas.config import ClaimExtractCfg, LLMCfg
from schemas.paper import Paper, ReportedResult

from .decomposer import decompose_claims
from .heuristics import extract_claims_heuristic
from .results_parser import extract_reported_results

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractionResult:
    """Bundle returned by :func:`extract_facts`."""

    claims: list[Claim]
    reported_results: list[ReportedResult]
    backend: str  # "llm" | "heuristic" | "auto:llm+heuristic"


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE_NAME = "extract_claims.txt"


def _load_prompt_template() -> str:
    """Load the extraction prompt template bundled with this package."""
    return ir.files("preprocessing.claim_extract").joinpath(_PROMPT_TEMPLATE_NAME).read_text(encoding="utf-8")


def _render_sections_for_prompt(paper: Paper, *, max_chars: int = 18_000) -> str:
    """Format sections as a bullet list, truncated to keep prompts bounded."""
    lines: list[str] = []
    total = 0
    for s in paper.sections:
        body = (s.text or "").strip()
        chunk = f"[{s.id}] {s.title} (chars {s.char_start}-{s.char_end}):\n{body}"
        if total + len(chunk) > max_chars:
            lines.append(chunk[: max(0, max_chars - total)])
            lines.append("... [truncated]")
            break
        lines.append(chunk)
        total += len(chunk)
    return "\n\n".join(lines)


def _render_reported_summary(reported: list[ReportedResult], *, max_entries: int = 200) -> str:
    if not reported:
        return "(none extracted from tables)"
    rows = []
    for r in reported[:max_entries]:
        rows.append(
            f"- {r.table_id} r{r.row_index}c{r.col_index}: "
            f"method={r.method!r} metric={r.metric} value={r.value} "
            f"dataset={r.dataset!r} task={r.task!r}"
        )
    return "\n".join(rows)


def _call_llm_for_claims(
    paper: Paper,
    reported: list[ReportedResult],
    llm_cfg: LLMCfg,
) -> list[Claim] | None:
    """Run the LLM extraction pass, returning ``None`` on any failure."""
    try:
        from llm.client import llm_json, resolve_llm_config
    except Exception:
        logger.warning("LLM client not importable; skipping LLM extraction.")
        return None

    template = _load_prompt_template()
    prompt = template.format(
        title=(paper.metadata.title or paper.metadata.paper_key),
        paper_key=paper.metadata.paper_key,
        sections=_render_sections_for_prompt(paper),
        reported_summary=_render_reported_summary(reported),
    )

    cfg = resolve_llm_config(
        provider=llm_cfg.provider,
        model=llm_cfg.model,
        base_url=llm_cfg.base_url,
    )
    try:
        payload = llm_json(
            prompt=prompt,
            system="You are a careful reviewer extracting structured claims from a paper. Return strict JSON only.",
            cfg=cfg,
        )
    except Exception as exc:
        logger.warning("LLM extraction failed: %s", exc)
        return None

    raw_claims = (payload or {}).get("claims")
    if not isinstance(raw_claims, list):
        logger.warning("LLM extraction returned no 'claims' field; raw=%r", payload)
        return None

    return list(_parse_llm_claims(raw_claims))


def _parse_llm_claims(raw: list[dict]) -> list[Claim]:
    """Validate-and-coerce the LLM JSON into typed :class:`Claim` objects."""
    parsed: list[Claim] = []
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        try:
            ctype_raw = str(item.get("type", "")).strip().lower()
            ctype = ClaimType(ctype_raw) if ctype_raw else ClaimType.EMPIRICAL
        except ValueError:
            logger.warning("Unknown claim type %r; defaulting to empirical.", item.get("type"))
            ctype = ClaimType.EMPIRICAL

        loc_raw = item.get("location") or {}
        location = ClaimLocation(
            section_id=loc_raw.get("section_id"),
            char_start=loc_raw.get("char_start"),
            char_end=loc_raw.get("char_end"),
            page=loc_raw.get("page"),
        )
        claim = Claim(
            id=str(item.get("id") or f"claim_{i:02d}"),
            text=str(item.get("text", "")).strip(),
            type=ctype,
            scope=str(item.get("scope", "local")).strip() or "local",
            datasets=[str(x) for x in (item.get("datasets") or []) if x],
            baselines=[str(x) for x in (item.get("baselines") or []) if x],
            metrics=[str(x) for x in (item.get("metrics") or []) if x],
            location=location,
            evidence_targets=[str(x) for x in (item.get("evidence_targets") or []) if x],
        )
        if claim.text:
            parsed.append(claim)
    return parsed


# ---------------------------------------------------------------------------
# Merge logic (for mode="auto")
# ---------------------------------------------------------------------------


def _merge_claims(llm_claims: list[Claim], heuristic_claims: list[Claim]) -> list[Claim]:
    """Keep all LLM claims, append heuristic claims that add new information.

    A heuristic claim is kept when:
      - it adds a reproducibility claim the LLM missed (cheap and concrete), or
      - no LLM claim shares ≥60% of its tokens (very rough dedup).
    """
    if not heuristic_claims:
        return llm_claims
    if not llm_claims:
        return heuristic_claims

    out: list[Claim] = list(llm_claims)
    known_texts = [c.text.lower() for c in llm_claims]

    for hc in heuristic_claims:
        ht = hc.text.lower()
        if any(_jaccard(ht, kt) >= 0.6 for kt in known_texts):
            continue
        if hc.type == ClaimType.REPRODUCIBILITY or not any(
            c.type == hc.type and hc.location.section_id == c.location.section_id for c in llm_claims
        ):
            out.append(hc)
    # Re-number so ids stay stable & dense.
    for i, c in enumerate(out, start=1):
        if not c.id.startswith("claim_"):
            c.id = f"claim_{i:02d}"
    return out


def _jaccard(a: str, b: str) -> float:
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def extract_facts(
    paper: Paper,
    *,
    cfg: ClaimExtractCfg | None = None,
    llm_cfg: LLMCfg | None = None,
) -> ExtractionResult:
    """Run stage §3.1b and return claims + reported results.

    Parameters
    ----------
    paper
        The structured paper produced by :mod:`ingestion`.
    cfg
        Fact-extraction sub-config. Defaults to :class:`ClaimExtractCfg`
        with ``mode="auto"`` and decomposition enabled.
    llm_cfg
        LLM routing sub-config; only consulted when mode ∈ {auto, llm}.
    """
    cfg = cfg or ClaimExtractCfg()
    mode = (cfg.mode or "auto").lower()

    reported = extract_reported_results(paper)

    llm_claims: list[Claim] | None = None
    heuristic_claims: list[Claim] = []

    if mode in {"llm", "auto"} and llm_cfg is not None:
        llm_claims = _call_llm_for_claims(paper, reported, llm_cfg)

    if mode in {"heuristic", "auto"} or llm_claims is None:
        heuristic_claims = extract_claims_heuristic(paper)

    if mode == "llm":
        claims = llm_claims or []
        backend = "llm"
    elif mode == "heuristic":
        claims = heuristic_claims
        backend = "heuristic"
    else:  # auto
        if llm_claims is None:
            claims = heuristic_claims
            backend = "auto:heuristic-fallback"
        else:
            claims = _merge_claims(llm_claims, heuristic_claims)
            backend = "auto:llm+heuristic"

    if cfg.decompose_broad_claims:
        claims = decompose_claims(claims, reported)

    return ExtractionResult(claims=claims, reported_results=reported, backend=backend)
