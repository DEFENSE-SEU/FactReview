"""Top-level entry point for §3.1b fact extraction.

Takes an ingested :class:`Paper` and produces a list of
:class:`Claim` objects. Depending on :class:`ClaimExtractCfg.mode`,
the extractor runs:

  - ``heuristic`` : regex-only (deterministic, no API cost).
  - ``llm``       : LLM-only (strict JSON).
  - ``auto``      : strict LLM path. If the LLM call fails or returns an
                    unusable payload, extraction raises instead of falling
                    back to heuristic output.

The LLM prompt is responsible for keeping claims atomic. Legacy broad-claim
decomposition remains available only when explicitly enabled in config.
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

    core_claims: list[Claim]
    claims: list[Claim]
    reported_results: list[ReportedResult]
    backend: str  # "llm" | "heuristic" | "auto:llm"


class ClaimExtractionError(RuntimeError):
    """Raised when strict LLM claim extraction cannot produce usable claims."""


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
    claim_cfg: ClaimExtractCfg,
    llm_cfg: LLMCfg,
) -> tuple[list[Claim], list[Claim]]:
    """Run the LLM extraction pass, raising on any unusable result."""
    try:
        from llm.client import llm_json, resolve_llm_config
    except Exception as exc:
        raise ClaimExtractionError("LLM client is not importable for claim extraction.") from exc

    template = _load_prompt_template()
    prompt = template.format(
        title=(paper.metadata.title or paper.metadata.paper_key),
        paper_key=paper.metadata.paper_key,
        sections=_render_sections_for_prompt(paper, max_chars=claim_cfg.prompt_max_chars),
        reported_summary=_render_reported_summary(reported),
    )

    cfg = resolve_llm_config(
        provider=llm_cfg.provider,
        model=llm_cfg.model,
        base_url=llm_cfg.base_url,
        max_tokens=llm_cfg.max_tokens,
    )
    try:
        payload = llm_json(
            prompt=prompt,
            system="You are a careful reviewer extracting structured claims from a paper. Return strict JSON only.",
            cfg=cfg,
            module="analysis",
        )
    except Exception as exc:
        raise ClaimExtractionError(f"LLM claim extraction failed: {exc}") from exc

    raw_core_claims = (payload or {}).get("core_claims")
    if not isinstance(raw_core_claims, list):
        raise ClaimExtractionError(
            f"LLM claim extraction returned no list-valued 'core_claims' field: {payload!r}"
        )
    raw_claims = (payload or {}).get("claims")
    if not isinstance(raw_claims, list):
        raise ClaimExtractionError(f"LLM claim extraction returned no list-valued 'claims' field: {payload!r}")

    core_claims = list(_parse_llm_claims(raw_core_claims))
    claims = list(_parse_llm_claims(raw_claims))
    if not core_claims:
        raise ClaimExtractionError("LLM claim extraction returned zero usable core claims.")
    if not claims:
        raise ClaimExtractionError("LLM claim extraction returned zero usable claims.")
    return core_claims, claims


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
            importance=str(item.get("importance", "")).strip(),
            location=location,
            evidence_targets=[str(x) for x in (item.get("evidence_targets") or []) if x],
        )
        if claim.text:
            parsed.append(claim)
    return parsed


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
        with ``mode="auto"`` and strict LLM extraction.
    llm_cfg
        LLM routing sub-config; only consulted when mode ∈ {auto, llm}.
    """
    cfg = cfg or ClaimExtractCfg()
    mode = (cfg.mode or "auto").lower()

    reported = extract_reported_results(paper)

    if mode in {"llm", "auto"}:
        core_claims, claims = _call_llm_for_claims(paper, reported, cfg, llm_cfg or LLMCfg())
        backend = "llm" if mode == "llm" else "auto:llm"
    elif mode == "heuristic":
        claims = extract_claims_heuristic(paper)
        core_claims = claims[:3]
        backend = "heuristic"
    else:
        raise ValueError(f"Unknown claim extraction mode: {cfg.mode!r}")

    if cfg.decompose_broad_claims:
        claims = decompose_claims(claims, reported)

    return ExtractionResult(
        core_claims=core_claims,
        claims=claims,
        reported_results=reported,
        backend=backend,
    )
