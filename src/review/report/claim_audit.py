"""Post-hoc audits for the final review markdown.

The agent runner writes ``Status`` for each claim using a regex-based heuristic
(``_score_paper_evidence`` in ``agent_runtime/runner.py``). That heuristic can
only check whether the evidence text *mentions* a table/figure/number — it
cannot tell whether the cited numbers actually support the *verb* of the claim
(``leading``, ``outperforms``, ``state-of-the-art``…). It also misses
axis-self-selection in the technical positioning matrix and ablation coverage
gaps. This module adds those checks as a post-processing pass so they apply to
every paper without re-running the agent.

All audits are deterministic and string-based by default. ``apply_llm_claim_adjudication``
adds an optional LLM call that re-derives the status given the claim, evidence
and a manuscript excerpt; if the LLM is unavailable the deterministic result is
preserved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Status ordering, from most-supportive to most-skeptical. "Capping" toward a
# weaker status means: take whichever of {current, cap} sits later in this list.
_STATUS_ORDER = ["supported", "partially supported", "inconclusive", "in conflict"]
_STATUS_RANK = {label: i for i, label in enumerate(_STATUS_ORDER)}

# Words that turn a claim into a comparative/superlative statement. When any
# of these appear in the claim we require Δ ≥ 2σ vs. the strongest comparator
# before keeping ✓ Supported.
_SUPERLATIVE_PATTERN = re.compile(
    r"(?i)\b("
    r"state[-\s]?of[-\s]?the[-\s]?art|sota|"
    r"leading|leads?|"
    r"outperform(?:s|ed|ing)?|"
    r"surpass(?:es|ed|ing)?|"
    r"superior(?:ity)?|"
    r"best(?:[-\s]performing)?|"
    r"first|"
    r"highest|"
    r"top[-\s]?\d+|"
    r"new\s+state[-\s]?of[-\s]?the[-\s]?art|"
    r"achieves?\s+(?:the\s+)?(?:highest|best|leading|state[-\s]?of[-\s]?the[-\s]?art)|"
    # Hedged-comparative verbs that still encode "ours > theirs" claims.
    r"(?:can\s+)?improve(?:s|d|ment)?|"
    r"exceed(?:s|ed|ing)?|"
    r"beats?|"
    r"better\s+than|"
    r"gains?\s+over|"
    r"wins?\s+over|"
    r"ahead\s+of|"
    r"more\s+(?:accurate|effective|robust)\s+than"
    r")\b"
)

# Used when the claim itself is hedged ("X can improve Y"). If the *evidence*
# explicitly compares two configurations (X vs. Y, X compared with Y), we still
# treat the claim as comparative for the purposes of the significance cap.
_COMPARATIVE_EVIDENCE_PATTERN = re.compile(
    r"(?i)(?:\bversus\b|\bvs\.?\b|\bcompared\s+(?:with|to|against)\b|"
    r"\bover\b\s+(?:baseline|the\s+baseline|prior))"
)

# value ± sigma. Accepts ±, +/-, +-, and unicode minus.
_NUMBER_WITH_ERROR = re.compile(
    r"(?P<val>-?\d+(?:\.\d+)?)\s*(?:±|\+/-|\+-|±)\s*(?P<sigma>\d+(?:\.\d+)?)"
)
# bare numbers in 0–100 range, used as score candidates.
_BARE_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
# Numbers we want to mask before parsing scores: hyphen-prefixed numeric
# suffixes on a letter word (GPT-5.2, ResNet-50, T5-XL-1.1, R-GCN-2) and
# parameter-size suffixes (70B, 1.3B). We deliberately do not try to mask
# space-separated patterns like "Claude 4.5" because the same shape would
# also eat real scores like "at 59.0" or "BLEU 27.3"; any leaked low version
# digit is dominated by larger real comparators in max() downstream.
_VERSION_NAME_PATTERN = re.compile(
    r"(?<=[A-Za-z])-\d+(?:\.\d+)?(?:[-/]\d+(?:\.\d+)?)?\b"
)
_PARAM_SIZE_PATTERN = re.compile(r"\b\d+(?:\.\d+)?[BMK]\b")


@dataclass
class ClaimAuditResult:
    """Outcome of auditing a single claim row."""

    original_status: str
    final_status: str
    significance_cap: str | None = None
    superlative: bool = False
    delta_over_sigma: float | None = None
    paper_value: float | None = None
    paper_sigma: float | None = None
    comparator_value: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class ReportAuditOutcome:
    """Outcome of auditing the full report. Only carries audit metadata; the
    updated markdown is returned alongside by ``audit_review_markdown``."""

    claim_results: list[ClaimAuditResult] = field(default_factory=list)
    extra_weaknesses: list[str] = field(default_factory=list)
    axis_self_selection_ratio: float | None = None
    ablation_components_missing: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Status normalization helpers


def _strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", "", str(value or ""))


def _normalize_status(value: str) -> str:
    s = _strip_html(value).strip()
    s = s.replace("✓", "").replace("⚠", "").replace("✗", "").strip().lower()
    if not s:
        return ""
    if s.startswith("supported"):
        return "supported"
    if "partial" in s:
        return "partially supported"
    if "inconclusive" in s or s == "unclear":
        return "inconclusive"
    if "conflict" in s:
        return "in conflict"
    return s


def _format_status_html(label: str) -> str:
    if label == "supported":
        return '<span style="color: green;">✓ Supported</span>'
    if label == "partially supported":
        return '<span style="color: #E6B800;">⚠ Partially supported</span>'
    if label == "inconclusive":
        return '<span style="color: #E6B800;">⚠ Inconclusive</span>'
    if label == "in conflict":
        return '<span style="color: red;">✗ In conflict</span>'
    return label


def _cap_status(current: str, cap: str) -> str:
    """Return the more conservative of ``current`` and ``cap``."""
    cur = _normalize_status(current)
    cap_n = _normalize_status(cap)
    if cur not in _STATUS_RANK or cap_n not in _STATUS_RANK:
        return cur or cap_n
    return cur if _STATUS_RANK[cur] >= _STATUS_RANK[cap_n] else cap_n


# ---------------------------------------------------------------------------
# Numeric extraction helpers


def _mask_model_names(text: str) -> str:
    """Blank out hyphen-prefixed numeric suffixes (``GPT-5.2``, ``ResNet-50``,
    ``T5-XL-1.1``) and parameter-size suffixes (``70B``, ``1.3B``).

    The rule is intentionally narrow and structural — we only mask numbers that
    are immediately preceded by a hyphen attached to a letter, which is a
    near-universal version-id convention. Plain space-separated patterns like
    ``Claude 4.5`` are left alone because the same pattern would also eat real
    scores like ``at 59.0`` or ``BLEU 27.3``. Any leaked version digit is
    typically dominated by larger real comparator scores in max(), so the
    downstream comparator-selection step still works.
    """
    masked = _VERSION_NAME_PATTERN.sub(" ", text)
    masked = _PARAM_SIZE_PATTERN.sub(" ", masked)
    return masked


def _extract_value_sigma(text: str) -> tuple[float, float] | None:
    m = _NUMBER_WITH_ERROR.search(text or "")
    if not m:
        return None
    try:
        return float(m.group("val")), float(m.group("sigma"))
    except ValueError:
        return None


# Agent self-tag pattern: ``[verdict: <label>; reason: ...]`` appended to the
# Assessment cell. The agent prompt instructs the agent to emit this so its
# own evidence-aware verdict can be reconciled with the system audit.
_SELF_TAG_PATTERN = re.compile(
    r"(?i)\[verdict:\s*(?P<verdict>supported|partially[\s_-]?supported|partial|"
    r"inconclusive|unclear|in[\s_-]?conflict|conflict)"
    r"(?:\s*;\s*reason:\s*(?P<reason>[^\]]*))?\]"
)


def _extract_self_tag(assessment_text: str) -> tuple[str, str, str]:
    """Pull the agent's self-tag verdict from an Assessment cell.

    Returns ``(verdict_label, reason, cleaned_assessment)`` where verdict_label
    is normalized via ``_normalize_status`` (or ""), and cleaned_assessment is
    the assessment with the bracketed tag removed (so the user-visible cell
    stays clean).
    """
    raw = str(assessment_text or "")
    match = _SELF_TAG_PATTERN.search(raw)
    if not match:
        return "", "", raw
    verdict = _normalize_status(match.group("verdict") or "")
    reason = (match.group("reason") or "").strip()
    cleaned = (raw[: match.start()] + raw[match.end() :]).strip()
    return verdict, reason, cleaned


def _largest_versus_gap(text: str) -> tuple[float, float, float] | None:
    """Find paired ``A versus B`` numbers and return ``(paper, comparator, gap)``.

    We treat the *first* number in each pair as the paper-side value (since
    the standard wording is "<our system> at X versus <baseline> at Y") and
    return the pair with the largest |gap|. Numbers embedded in model-name
    tokens like "GPT-5.2" are masked out before scanning.
    """
    masked = _mask_model_names(text or "")
    pairs: list[tuple[float, float]] = []
    # Pattern 1: handle "<system> at 52.7 versus <other system> at 52.0".
    p1 = re.compile(
        r"(?i)at\s+(\d+(?:\.\d+)?)\s*(?:versus|vs\.?)\s+[^,;.]*?\s+at\s+(\d+(?:\.\d+)?)"
    )
    for m in p1.finditer(masked):
        try:
            a = float(m.group(1))
            b = float(m.group(2))
        except ValueError:
            continue
        if 0.0 < a <= 100.0 and 0.0 < b <= 100.0:
            pairs.append((a, b))
    if not pairs:
        return None
    # Choose the pair with the largest absolute gap (the strongest comparison
    # the claim could lean on).
    best = max(pairs, key=lambda p: abs(p[0] - p[1]))
    paper_val, comp_val = best
    return paper_val, comp_val, paper_val - comp_val


def _extract_score_candidates(
    text: str, *, exclude_value: float | None = None
) -> list[float]:
    """Return numbers in [0, 100] that are not embedded in model-name tokens.

    We also strip the matched ``value ± sigma`` substring so the paper's own
    value is not accidentally returned as a comparator.
    """
    masked = _mask_model_names(text or "")
    masked = _NUMBER_WITH_ERROR.sub("  ", masked)

    out: list[float] = []
    seen: set[str] = set()
    for m in _BARE_NUMBER.finditer(masked):
        token = m.group(0)
        if token in seen:
            continue
        try:
            v = float(token)
        except ValueError:
            continue
        if not (0.0 < v <= 100.0):
            continue
        if exclude_value is not None and abs(v - exclude_value) < 0.05:
            continue
        out.append(v)
        seen.add(token)
    return out


# ---------------------------------------------------------------------------
# Significance audit (suggestion 2)


def assess_significance_cap(
    *, claim: str, evidence: str
) -> ClaimAuditResult:
    """Decide a significance-based status cap for one claim.

    The deterministic rule:

    1. If the claim text contains no superlative ⇒ no cap (returned cap=None).
    2. Otherwise, parse the paper's reported ``value ± sigma`` from evidence.
    3. Find the strongest comparator value in the same evidence, excluding the
       paper's value itself and numbers that look like model-name versions.
    4. Compute ``Δ = paper_value − comparator_value`` (assuming higher-is-better).
       - If Δ < 1σ ⇒ cap to ⚠ Inconclusive.
       - If Δ < 2σ ⇒ cap to ⚠ Partially supported.
       - Else ⇒ no cap.
    """
    result = ClaimAuditResult(original_status="", final_status="")
    has_superlative = bool(_SUPERLATIVE_PATTERN.search(claim or ""))
    has_evidence_comparison = bool(_COMPARATIVE_EVIDENCE_PATTERN.search(evidence or ""))
    if not (has_superlative or has_evidence_comparison):
        return result
    result.superlative = has_superlative or has_evidence_comparison

    pair = _extract_value_sigma(evidence or "")
    if pair is None:
        # No σ in this evidence cell. If the evidence still has explicit
        # paired comparisons (e.g., "52.7 versus 52.0"), parse them to detect
        # whether the gap is at noise-level (≤2 absolute points), which is
        # typical noise for 100–1000 sample ML benchmarks.
        paired = _largest_versus_gap(evidence or "")
        if paired is not None:
            paper_val, comp_val, gap = paired
            result.paper_value = paper_val
            result.comparator_value = comp_val
            if gap < 0.0:
                result.significance_cap = "in conflict"
                result.notes.append(
                    f"Paper value {paper_val} is below the comparator {comp_val} "
                    f"in the cited 'X versus Y' pair (Δ={gap:.2f})."
                )
            else:
                # Without an error bar we cannot decide significance from the
                # gap magnitude alone — typical noise depends on benchmark size
                # and metric (BLEU σ ≈ 0.3, ImageNet σ ≈ 0.1, 100-row SWE-Bench
                # σ ≈ 1.9), so any single absolute threshold would be wrong on
                # most papers. Cap at Partially supported and let the LLM
                # adjudication pass decide whether to tighten further given the
                # benchmark context it can read.
                result.significance_cap = "partially supported"
                result.notes.append(
                    f"Largest 'X versus Y' gap is Δ={gap:.2f}; evidence reports "
                    f"no error bar so significance cannot be verified from "
                    f"deterministic checks alone."
                )
            return result
        # Otherwise: comparative wording with no parseable comparator.
        result.significance_cap = "partially supported"
        result.notes.append(
            "Claim uses comparative language but evidence reports no error bar "
            "or paired comparison; cannot verify gap is significant."
        )
        return result
    paper_val, paper_sigma = pair
    result.paper_value = paper_val
    result.paper_sigma = paper_sigma

    comparators = _extract_score_candidates(evidence or "", exclude_value=paper_val)
    if not comparators:
        result.significance_cap = "partially supported"
        result.notes.append(
            "Comparative claim but no comparator value parsed from evidence."
        )
        return result

    # Higher-is-better default. Most ML metrics in our pipeline (Resolve Rate,
    # accuracy, F1, BLEU, MRR…) follow this convention.
    best_comparator = max(comparators)
    result.comparator_value = best_comparator
    delta = paper_val - best_comparator

    if paper_sigma <= 0.0:
        # Treat zero-sigma as a non-numeric guard; only flag if we definitely
        # have a negative gap.
        if delta < 0.0:
            result.significance_cap = "in conflict"
            result.notes.append(
                f"Paper value {paper_val} is below comparator {best_comparator}."
            )
        return result

    ratio = delta / paper_sigma
    result.delta_over_sigma = ratio
    if delta < 0.0:
        result.significance_cap = "in conflict"
        result.notes.append(
            f"Paper value {paper_val} is below the strongest comparator "
            f"{best_comparator} (Δ={delta:.2f})."
        )
    elif ratio < 1.0:
        result.significance_cap = "inconclusive"
        result.notes.append(
            f"Δ={delta:.2f} vs. comparator {best_comparator}, σ={paper_sigma}; "
            f"Δ/σ={ratio:.2f} < 1 — gap is within one standard deviation."
        )
    elif ratio < 2.0:
        result.significance_cap = "partially supported"
        result.notes.append(
            f"Δ={delta:.2f} vs. comparator {best_comparator}, σ={paper_sigma}; "
            f"Δ/σ={ratio:.2f} < 2 — gap is below two standard deviations."
        )
    return result


# ---------------------------------------------------------------------------
# Markdown surgery: claims table


_CLAIMS_HEADER_PATTERN = re.compile(
    r"(?ims)^##\s+(?:\*\*)?3\.\s+Claims(?:\*\*)?\s*$\n(?P<body>.*?)(?=^##\s+|\Z)"
)


def _split_table_row(row: str) -> list[str]:
    s = row.strip()
    if not (s.startswith("|") and s.endswith("|")):
        return []
    return [c.strip() for c in s.strip("|").split("|")]


def _find_column_index(headers: list[str], *needles: str) -> int:
    lowered = [_strip_html(h).strip().lower() for h in headers]
    for needle in needles:
        for i, h in enumerate(lowered):
            if needle in h:
                return i
    return -1


def _apply_significance_to_claims_block(
    body: str,
) -> tuple[str, list[ClaimAuditResult]]:
    """Walk the claims table inside Section 3 and apply the significance cap.

    Returns the updated body and a list of per-claim audit results in the
    order they appear in the table.
    """
    # Use split("\n") rather than splitlines() so trailing-newline structure
    # round-trips correctly. splitlines() drops a trailing newline, which
    # collapses into the section boundary on re-join.
    lines = body.split("\n")
    header_idx = -1
    for i, ln in enumerate(lines):
        s = ln.strip()
        if (
            s.startswith("|")
            and "claim" in s.lower()
            and "evidence" in s.lower()
            and "status" in s.lower()
        ):
            header_idx = i
            break
    if header_idx < 0:
        return body, []
    if header_idx + 1 >= len(lines):
        return body, []

    headers = _split_table_row(lines[header_idx])
    sep_idx = header_idx + 1
    row_start = header_idx + 2

    claim_idx = _find_column_index(headers, "claim")
    evidence_idx = _find_column_index(headers, "evidence")
    assessment_idx = _find_column_index(headers, "assessment")
    status_idx = _find_column_index(headers, "status")

    if claim_idx < 0 or evidence_idx < 0 or status_idx < 0:
        return body, []

    audit_results: list[ClaimAuditResult] = []
    new_lines = list(lines)
    j = row_start
    while j < len(new_lines):
        s = new_lines[j].strip()
        if not (s.startswith("|") and s.endswith("|")):
            break
        cells = _split_table_row(new_lines[j])
        if len(cells) <= max(claim_idx, evidence_idx, status_idx):
            j += 1
            continue
        claim_text = _strip_html(cells[claim_idx])
        evidence_text = _strip_html(cells[evidence_idx])
        original_status = cells[status_idx]

        # Suggestion 3: if the agent embedded a [verdict: ...; reason: ...] tag
        # in its Assessment cell, parse it out and use it as a self-cap. The
        # bracketed tag is stripped from the visible Assessment so reviewers see
        # clean prose; the verdict is honored only if it is *more conservative*
        # than the system's other audits.
        agent_self_verdict = ""
        agent_self_reason = ""
        if assessment_idx >= 0 and assessment_idx < len(cells):
            agent_self_verdict, agent_self_reason, cleaned_assessment = _extract_self_tag(
                cells[assessment_idx]
            )
            if agent_self_verdict and cleaned_assessment != cells[assessment_idx]:
                cells[assessment_idx] = cleaned_assessment

        audit = assess_significance_cap(claim=claim_text, evidence=evidence_text)
        audit.original_status = _normalize_status(original_status)

        # Combine the system's significance cap with the agent's self-cap; take
        # the more conservative (so a generous regex verdict can be tightened
        # but never loosened by either source).
        cap_candidates: list[str] = []
        if audit.significance_cap:
            cap_candidates.append(audit.significance_cap)
        if agent_self_verdict:
            cap_candidates.append(agent_self_verdict)
            audit.notes.append(
                f"Agent self-tag verdict: {agent_self_verdict}"
                + (f" — {agent_self_reason}" if agent_self_reason else "")
            )

        if cap_candidates:
            new_status_label = audit.original_status
            for cap in cap_candidates:
                new_status_label = _cap_status(new_status_label, cap)
            audit.final_status = new_status_label
            if new_status_label != audit.original_status:
                cells[status_idx] = _format_status_html(new_status_label)
            new_lines[j] = "| " + " | ".join(cells) + " |"
        else:
            audit.final_status = audit.original_status
            # Even if no cap was applied, the assessment cell may have had a
            # self-tag stripped (no-op verdict); persist any cell edits.
            if assessment_idx >= 0 and assessment_idx < len(cells):
                new_lines[j] = "| " + " | ".join(cells) + " |"
        audit_results.append(audit)
        j += 1

    return "\n".join(new_lines), audit_results


def apply_significance_cap_to_markdown(
    markdown: str,
) -> tuple[str, list[ClaimAuditResult]]:
    text = str(markdown or "")
    sec = _CLAIMS_HEADER_PATTERN.search(text)
    if not sec:
        return text, []
    body = sec.group("body")
    new_body, audits = _apply_significance_to_claims_block(body)
    if new_body == body:
        return text, audits
    return text[: sec.start("body")] + new_body + text[sec.end("body") :], audits


# ---------------------------------------------------------------------------
# Technical Positioning axis self-selection audit (suggestion 4)


_TP_HEADER_PATTERN = re.compile(
    r"(?ims)^##\s+(?:\*\*)?2\.\s+Technical Positioning(?:\*\*)?\s*$\n(?P<body>.*?)(?=^##\s+|\Z)"
)


def _parse_first_table(body: str) -> tuple[list[str], list[list[str]]]:
    lines = body.splitlines()
    headers: list[str] = []
    rows: list[list[str]] = []
    i = 0
    while i + 1 < len(lines):
        h = lines[i].strip()
        sep = lines[i + 1].strip()
        if (
            h.startswith("|")
            and h.endswith("|")
            and re.fullmatch(r"\|[ :\-|]+\|", sep)
        ):
            headers = _split_table_row(lines[i])
            j = i + 2
            while j < len(lines):
                s = lines[j].strip()
                if not (s.startswith("|") and s.endswith("|")):
                    break
                rows.append(_split_table_row(lines[j]))
                j += 1
            return headers, rows
        i += 1
    return headers, rows


def audit_axis_self_selection(markdown: str) -> tuple[float | None, str | None]:
    """Detect axis-self-selection in the Technical Positioning table.

    Returns ``(ratio, weakness_bullet)``. ``ratio`` is the fraction of
    niche-dimension columns that are won only by the ``This Work`` row
    (everyone else is ×). When the ratio is ≥ 0.6 we emit a weakness bullet
    that reviewers can use to push back; otherwise the second element is None.
    """
    text = str(markdown or "")
    sec = _TP_HEADER_PATTERN.search(text)
    if not sec:
        return None, None
    headers, rows = _parse_first_table(sec.group("body"))
    if len(headers) < 3 or not rows:
        return None, None

    # Identify the niche-dimension columns. The agent prompt enforces:
    # column 0 = Research domain, column 1 = Method.
    niche_start = 2
    niche_cols = list(range(niche_start, len(headers)))
    if not niche_cols:
        return None, None

    # Identify the self row.
    self_row_idx = -1
    for i, row in enumerate(rows):
        if len(row) < 2:
            continue
        domain = _strip_html(row[0]).strip().lower()
        if "this work" in domain:
            self_row_idx = i
            break
    if self_row_idx < 0:
        return None, None

    self_row = rows[self_row_idx]
    other_rows = [r for i, r in enumerate(rows) if i != self_row_idx]
    if not other_rows:
        return None, None

    def _is_check(cell: str) -> bool:
        v = _strip_html(cell).strip()
        return v in {"√", "✓"}

    self_exclusive_wins = 0
    self_won = 0
    # Track each baseline row's exclusive wins so we can detect imbalance.
    baseline_exclusive_wins: list[int] = [0] * len(other_rows)
    for col in niche_cols:
        if col >= len(self_row):
            continue
        # Self-row's exclusive wins.
        if _is_check(self_row[col]):
            self_won += 1
            if all(col >= len(r) or not _is_check(r[col]) for r in other_rows):
                self_exclusive_wins += 1
        else:
            # Baseline-exclusive: exactly one baseline row has √ here AND the
            # self-row does not. (When the self-row also has √, it's not
            # exclusive to the baseline.)
            winners = [
                k
                for k, r in enumerate(other_rows)
                if col < len(r) and _is_check(r[col])
            ]
            if len(winners) == 1:
                baseline_exclusive_wins[winners[0]] += 1

    ratio = self_exclusive_wins / max(1, len(niche_cols))
    max_baseline_excl = max(baseline_exclusive_wins) if baseline_exclusive_wins else 0

    # Trigger when:
    # (a) self exclusive wins ≥ 3 absolute, AND
    # (b) ratio ≥ 0.4 of niche cols, AND
    # (c) self exclusive wins ≥ 2× any single baseline's exclusive wins (or
    #     baseline has 0 exclusive wins). This catches the classic pattern
    #     "I picked axes only I win on" while sparing matrices where one
    #     baseline also has its own winning axes.
    if (
        self_exclusive_wins < 3
        or ratio < 0.4
        or (max_baseline_excl > 0 and self_exclusive_wins < 2 * max_baseline_excl)
    ):
        return ratio, None

    bullet = (
        f"Comparison axes appear chosen to favor the proposed system "
        f"({self_exclusive_wins} of {len(niche_cols)} niche dimensions are "
        f"won only by This Work in the Technical Positioning table, while "
        f"the strongest baseline wins at most {max_baseline_excl} exclusively); "
        f"independent reviewers may want to verify that axes the matrix omits "
        f"would not flip the comparison."
    )
    return ratio, bullet


# ---------------------------------------------------------------------------
# Ablation coverage audit (suggestion 5)


_ABLATION_PATTERN = re.compile(
    r"(?ims)^###\s+(?:\*\*)?Ablation Result(?:\*\*)?\s*$\n(?P<body>.*?)(?=^##\s+|^###\s+|\Z)"
)


# Connector phrases that introduce a list of components in a method-design
# claim ("X with A, B, and C" / "system consisting of A, B, C" / "framework
# comprising A, B and C"). Domain-agnostic English list-introducers — we use
# this instead of a curated component vocabulary so the audit works across
# LLM, graph, CV, and biology papers. We deliberately exclude "provides" /
# "implements" / "uses" because they often introduce a single noun-phrase
# object rather than a list, which makes the captured group too greedy.
_COMPONENT_ENUMERATION_PATTERN = re.compile(
    r"""(?ix)
    \b(?:with|comprising|comprises|
       consisting\s+of|consists?\s+of|
       composed\s+of|composes\s+of|
       containing|contains?|
       made\s+up\s+of|including|includes?|
       incorporating|incorporates?|
       that\s+(?:includes?|combines?|comprises?))\b
    \s+
    (?P<list>[^.;]+?)
    (?:\.|;|$|\bto\b|\bfor\b|\bso\s+that\b|\bwhich\b)
    """
)

# Stop tokens we never want to keep as components: they're meta-words
# describing the paper itself (method/approach/system/...) rather than a
# component you would ablate. We deliberately do NOT include domain-specific
# words like "scaffold", "encoder", "convolution" — those are real components
# in some papers and the audit should let them through.
_COMPONENT_STOPWORDS = frozenset(
    {
        "method",
        "approach",
        "system",
        "framework",
        "model",
        "paper",
        "work",
        "pipeline",
        "architecture",
        "design",
        "the method",
        "this method",
        "our method",
        "this paper",
        "this work",
        "the paper",
        "the system",
        "the framework",
    }
)


def _split_enumeration(raw: str) -> list[str]:
    """Split ``A, B, and C`` / ``A; B; C`` / ``A and B`` into items."""
    s = raw.strip().rstrip(".")
    # Normalize " and " to a comma so we can split on a single delimiter.
    s = re.sub(r"(?i),?\s+and\s+", ", ", s)
    s = re.sub(r"\s*;\s*", ", ", s)
    parts = [p.strip().strip(".").strip() for p in s.split(",")]
    return [p for p in parts if p]


def _normalize_component(token: str) -> str:
    """Lower-case and trim a candidate component phrase. Strips leading
    determiners/adjectives that don't help with downstream matching."""
    t = token.lower().strip()
    t = re.sub(r"^(?:a|an|the)\s+", "", t)
    t = re.sub(r"\s+", " ", t).strip(" .,;:")
    return t


def _extract_components_from_claim(claim_text: str) -> list[str]:
    """Pull component noun phrases enumerated in the claim itself.

    We do NOT consult a curated component vocabulary; instead we search for
    English enumeration patterns ("X with A, B, and C", "X consisting of
    A, B, C") and split the matched list on commas / "and". This keeps the
    audit usable across LLM, graph, CV, and biology papers.
    """
    if not claim_text:
        return []
    deduped: list[str] = []
    for match in _COMPONENT_ENUMERATION_PATTERN.finditer(claim_text):
        list_text = match.group("list") or ""
        # An enumeration must contain at least one comma or " and "; a single
        # noun phrase like "with extensions" is too vague to call a list.
        if "," not in list_text and not re.search(
            r"(?i)\band\b", list_text
        ):
            continue
        for raw_item in _split_enumeration(list_text):
            item = _normalize_component(raw_item)
            if not item or len(item) < 3:
                continue
            if item in _COMPONENT_STOPWORDS:
                continue
            # Reject items that are obviously not components — an entire
            # sentence (>6 tokens) or a near-pure-numeric token like "27.3".
            # We keep items that *contain* digits inside a noun phrase
            # ("resnet-50 backbone", "v2 module") because those are legit
            # component names; we only reject when the alphabetic share is
            # too small for the item to be a meaningful name.
            if len(item.split()) > 6:
                continue
            letters = sum(1 for c in item if c.isalpha())
            if letters < 3:
                continue
            if item not in deduped:
                deduped.append(item)
    # Prefer longer multi-word tokens before short substrings of them.
    deduped.sort(key=len, reverse=True)
    pruned: list[str] = []
    for token in deduped:
        if not any(token in existing for existing in pruned):
            pruned.append(token)
    return pruned


def _component_appears_in_text(component: str, blob: str) -> bool:
    """Check whether ``component`` is referenced in ``blob`` (typically the
    ablation section). We accept several forms of the same notion:

    - Exact substring (``"context management"``).
    - Hyphen ↔ space variants (``"note-taking"`` ↔ ``"note taking"``).
    - Plural-aware match (``"extensions"`` ↔ ``"extension"``).
    - Head-noun fallback for multi-word components: a 2-word component like
      ``"resnet-50 backbone"`` is considered covered when the ablation text
      mentions ``"backbone"`` — that's the standard way ablation tables
      reference a component (they swap *one* specific instance for another,
      labeling the row by the head noun).
    """
    if not component:
        return False
    text = blob.lower()
    if component in text:
        return True
    variants = {
        component.replace("-", " "),
        component.replace(" ", "-"),
        component.rstrip("s"),
    }
    for variant in variants:
        if variant and variant in text:
            return True
    # Head-noun fallback for multi-word components (≥2 tokens). We only do
    # this when the head noun is at least 5 letters long so very generic
    # words like "head" or "set" don't cause false-positive coverage.
    tokens = component.replace("-", " ").split()
    if len(tokens) >= 2:
        head = tokens[-1]
        if len(head) >= 5 and re.search(rf"\b{re.escape(head)}\b", text):
            return True
    return False


def audit_ablation_coverage(markdown: str) -> tuple[list[str], str | None]:
    """Check whether components asserted in claims are actually ablated.

    Returns ``(missing_components, weakness_bullet)``. ``missing_components``
    lists the tokens (from the first claim row, which is typically the methodological
    one) that the ablation tables don't reference. The bullet is None when
    fewer than one component is missing.
    """
    text = str(markdown or "")
    claims_sec = _CLAIMS_HEADER_PATTERN.search(text)
    if not claims_sec:
        return [], None

    headers, rows = _parse_first_table(claims_sec.group("body"))
    if not headers or not rows:
        return [], None
    claim_idx = _find_column_index(headers, "claim")
    if claim_idx < 0:
        return [], None

    # Look at every claim row and collect components. We trigger the audit
    # whenever the claim itself enumerates ≥2 components (via the
    # connector+list pattern in ``_extract_components_from_claim``). This is
    # generic across LLM/CV/graph/biology papers — purely empirical claims
    # like "X achieves Y on benchmark Z" don't enumerate components and
    # therefore won't trigger.
    interesting_components: list[str] = []
    for row in rows:
        if claim_idx >= len(row):
            continue
        claim_text = _strip_html(row[claim_idx])
        components = _extract_components_from_claim(claim_text)
        # Require an actual enumeration (≥2 items) so a claim like
        # "method with X" doesn't get audited as if it had a list.
        if len(components) < 2:
            continue
        for c in components:
            if c not in interesting_components:
                interesting_components.append(c)

    if not interesting_components:
        return [], None

    ablation_match = _ABLATION_PATTERN.search(text)
    ablation_blob = ablation_match.group("body") if ablation_match else ""

    missing = [
        c for c in interesting_components if not _component_appears_in_text(c, ablation_blob)
    ]
    if len(missing) < 1:
        return [], None

    if len(missing) == len(interesting_components):
        bullet = (
            f"The method-design claim enumerates {len(interesting_components)} "
            f"components ({', '.join(interesting_components)}), but none of them "
            f"are evaluated in the ablation tables; the per-component contribution "
            f"of the design claim remains untested."
        )
    else:
        bullet = (
            f"The method-design claim enumerates {len(interesting_components)} "
            f"components ({', '.join(interesting_components)}), but the ablation "
            f"tables only exercise some of them; the following are not ablated: "
            f"{', '.join(missing)}."
        )
    return missing, bullet


# ---------------------------------------------------------------------------
# Weakness injection


_SUMMARY_HEADER_PATTERN = re.compile(
    r"(?ims)^##\s+(?:\*\*)?4\.\s+Summary(?:\*\*)?\s*$\n(?P<body>.*?)(?=^##\s+|\Z)"
)


def inject_weaknesses(markdown: str, bullets: list[str]) -> str:
    """Append extra weakness bullets to Section 4 / Weaknesses.

    The agent's section 4 puts ``Weaknesses:`` before a list. We insert the new
    bullets at the end of that list. Each bullet is prefixed ``[audit]`` so a
    reader can tell the system added it.
    """
    if not bullets:
        return markdown
    text = str(markdown or "")
    sec = _SUMMARY_HEADER_PATTERN.search(text)
    if not sec:
        return text

    body = sec.group("body")

    # Locate the Weaknesses label. The agent sometimes writes
    # "**Weaknesses:**" on its own line (canonical), but more often inlines
    # the first bullet on the same line ("**Weaknesses:** - First..."), so we
    # match by substring rather than by full-line anchor.
    label_match = re.search(r"(?i)\*{0,2}Weaknesses\*{0,2}\s*:?", body)
    if label_match is None:
        # Append a fresh Weaknesses block at the end of section 4.
        addition = "\n\n**Weaknesses:**\n" + "\n".join(
            f"- [audit] {b}" for b in bullets
        )
        new_body = body.rstrip() + addition + "\n"
        return text[: sec.start("body")] + new_body + text[sec.end("body") :]

    # Walk forward from the label to find the end of the weakness bullet
    # block — the last bullet line that still belongs to Weaknesses. We stop
    # at a blank line followed by a non-bullet, the next "**Strengths:**" /
    # "**Weaknesses:**" label, or the end of section 4.
    after_label = label_match.end()
    tail = body[after_label:]
    insertion_offset = len(tail)  # default: append to end of body
    in_bullet_block = False
    cursor = 0
    for raw_line in tail.split("\n"):
        # Track absolute offset including the trailing newline.
        line_len = len(raw_line) + 1  # for the "\n" we'll re-add
        stripped = raw_line.strip()
        is_bullet = stripped.startswith("- ") or stripped.startswith("* ")
        is_label = re.match(
            r"(?i)^\*{0,2}(?:Strengths|Weaknesses)\*{0,2}\s*:", stripped
        )
        if is_bullet:
            in_bullet_block = True
            cursor += line_len
            insertion_offset = cursor
            continue
        if in_bullet_block and not is_bullet and stripped == "":
            # Blank line — could be paragraph between bullets or end of block.
            cursor += line_len
            continue
        if in_bullet_block and (
            is_label or stripped.startswith("##") or not is_bullet
        ):
            # Hit something that is clearly outside the weakness list.
            break
        cursor += line_len

    # Compose the addition. We insert directly before ``insertion_offset``
    # within ``tail`` (i.e. at ``after_label + insertion_offset`` in body).
    addition = "".join(f"- [audit] {b}\n" for b in bullets)
    abs_pos = after_label + insertion_offset
    # Ensure the block we are appending after ends with a newline so the new
    # bullets render on their own line.
    prefix = "" if body[:abs_pos].endswith("\n") else "\n"
    new_body = body[:abs_pos] + prefix + addition + body[abs_pos:]
    return text[: sec.start("body")] + new_body + text[sec.end("body") :]


# ---------------------------------------------------------------------------
# Top-level audit entry point (deterministic-only)


def audit_review_markdown(markdown: str) -> tuple[str, ReportAuditOutcome]:
    """Apply all deterministic audits and return ``(updated_markdown, outcome)``."""
    outcome = ReportAuditOutcome()
    text, claim_audits = apply_significance_cap_to_markdown(markdown)
    outcome.claim_results = claim_audits

    bullets: list[str] = []

    # Synthesize per-claim audit notes into reviewer-readable bullets.
    for audit in claim_audits:
        if audit.significance_cap and audit.notes:
            note = audit.notes[-1]
            cap_label = audit.significance_cap.title()
            bullets.append(
                f"Status downgraded to {cap_label} on a comparative claim — {note}"
            )

    ratio, axis_bullet = audit_axis_self_selection(text)
    outcome.axis_self_selection_ratio = ratio
    if axis_bullet:
        bullets.append(axis_bullet)

    missing_components, ablation_bullet = audit_ablation_coverage(text)
    outcome.ablation_components_missing = missing_components
    if ablation_bullet:
        bullets.append(ablation_bullet)

    outcome.extra_weaknesses = bullets
    if bullets:
        text = inject_weaknesses(text, bullets)

    return text, outcome


# ---------------------------------------------------------------------------
# Optional LLM adjudication (suggestion 1)


_LLM_SYSTEM_PROMPT = (
    "You are an evidence-checking assistant for an academic-paper review system. "
    "Your job is to decide whether the claim is supported by the cited evidence "
    "with the rigor a careful peer reviewer would apply. Be conservative: if the "
    "evidence cites only the paper's own tables/figures, do not give ✓ Supported "
    "to comparative claims unless the gap is clearly larger than the reported "
    "uncertainty. Reply ONLY in JSON of the form "
    '{"verdict": "supported|partially_supported|inconclusive|in_conflict", '
    '"reason": "one short sentence"}.'
)


def _build_llm_prompt(*, claim: str, evidence: str, location: str) -> str:
    return (
        "Decide a status verdict for ONE claim against ONE evidence block.\n\n"
        "Decision rules:\n"
        "- 'supported' requires the evidence to directly justify the claim's verb.\n"
        "  For comparative claims (leading, outperforms, best, state-of-the-art),\n"
        "  the gap vs. the strongest comparator must be reasonably larger than the\n"
        "  reported error bar (>= 2 sigma if sigma is given).\n"
        "- 'partially_supported' applies when the evidence supports the qualitative\n"
        "  direction but the magnitude or scope is weaker than what the claim asserts,\n"
        "  or when only a subset of the claim's components is evidenced.\n"
        "- 'inconclusive' applies when the gap is within 1 sigma of the comparator,\n"
        "  the comparator is not actually evaluable from the evidence, or the\n"
        "  evidence is anecdotal (case study/appendix) rather than tabular.\n"
        "- 'in_conflict' applies when the evidence contradicts the claim's verb\n"
        "  (e.g., paper value lower than the strongest comparator).\n\n"
        f"Claim: {claim}\n"
        f"Evidence: {evidence}\n"
        f"Location: {location}\n\n"
        'Output only one JSON object: {"verdict": "...", "reason": "..."}'
    )


def _verdict_to_label(verdict: str) -> str:
    v = (verdict or "").strip().lower().replace("-", "_")
    mapping = {
        "supported": "supported",
        "partially_supported": "partially supported",
        "partially": "partially supported",
        "partial": "partially supported",
        "inconclusive": "inconclusive",
        "unclear": "inconclusive",
        "in_conflict": "in conflict",
        "conflict": "in conflict",
        "in conflict": "in conflict",
    }
    return mapping.get(v, "")


def apply_llm_claim_adjudication(
    markdown: str,
    *,
    llm_call: Any | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """Run an LLM adjudication pass on the claims table and reconcile statuses.

    The LLM returns an independent verdict per claim. The final status is the
    *more conservative* of the markdown's current status and the LLM verdict
    (so the deterministic significance cap from
    ``apply_significance_cap_to_markdown`` is never overridden upward).

    ``llm_call`` is an optional callable ``(prompt: str) -> dict[str, Any]`` used
    for testing. When None, we resolve the configured LLM via
    ``llm.client.resolve_llm_config`` and call ``llm_json``.
    """
    text = str(markdown or "")
    sec = _CLAIMS_HEADER_PATTERN.search(text)
    if not sec:
        return text, []

    body = sec.group("body")
    # Use split("\n") so trailing-newline structure round-trips after
    # "\n".join — splitlines() would drop a trailing newline and let the next
    # section heading collide with the last table row.
    lines = body.split("\n")
    header_idx = -1
    for i, ln in enumerate(lines):
        s = ln.strip()
        if (
            s.startswith("|")
            and "claim" in s.lower()
            and "evidence" in s.lower()
            and "status" in s.lower()
        ):
            header_idx = i
            break
    if header_idx < 0:
        return text, []

    headers = _split_table_row(lines[header_idx])
    claim_idx = _find_column_index(headers, "claim")
    evidence_idx = _find_column_index(headers, "evidence")
    location_idx = _find_column_index(headers, "location")
    status_idx = _find_column_index(headers, "status")
    if claim_idx < 0 or evidence_idx < 0 or status_idx < 0:
        return text, []

    if llm_call is None:
        try:
            from llm.client import llm_json, resolve_llm_config

            cfg = resolve_llm_config()
        except Exception:
            return text, []

        def _default_call(prompt: str) -> dict[str, Any]:
            return llm_json(prompt=prompt, system=_LLM_SYSTEM_PROMPT, cfg=cfg)

        llm_call = _default_call

    audits: list[dict[str, str]] = []
    new_lines = list(lines)
    j = header_idx + 2
    while j < len(new_lines):
        s = new_lines[j].strip()
        if not (s.startswith("|") and s.endswith("|")):
            break
        cells = _split_table_row(new_lines[j])
        if len(cells) <= max(claim_idx, evidence_idx, status_idx):
            j += 1
            continue
        claim_text = _strip_html(cells[claim_idx])
        evidence_text = _strip_html(cells[evidence_idx])
        location_text = (
            _strip_html(cells[location_idx]) if location_idx >= 0 and location_idx < len(cells) else ""
        )
        current_status = cells[status_idx]

        prompt = _build_llm_prompt(
            claim=claim_text, evidence=evidence_text, location=location_text
        )
        try:
            raw = llm_call(prompt) or {}
        except Exception as exc:  # pragma: no cover - defensive
            audits.append(
                {
                    "claim": claim_text,
                    "llm_status": "error",
                    "llm_reason": f"{type(exc).__name__}: {exc}",
                }
            )
            j += 1
            continue
        verdict = _verdict_to_label(str(raw.get("verdict") or ""))
        reason = str(raw.get("reason") or "").strip()
        audits.append(
            {
                "claim": claim_text,
                "llm_status": verdict or "unknown",
                "llm_reason": reason,
                "raw": str(raw)[:200],
            }
        )
        if verdict:
            new_status_label = _cap_status(current_status, verdict)
            if _normalize_status(current_status) != new_status_label:
                cells[status_idx] = _format_status_html(new_status_label)
                new_lines[j] = "| " + " | ".join(cells) + " |"
        j += 1

    new_body = "\n".join(new_lines)
    if new_body == body:
        return text, audits
    return text[: sec.start("body")] + new_body + text[sec.end("body") :], audits
