"""Post-hoc audits for the final review markdown.

The agent runner emits a ``Pending`` placeholder Status for non-experimental
claims (and for experimental claims that lack reproduction data). Experimental
claims with alignment data carry a deterministic numeric verdict from
``_status_from_paper_observed`` in the agent runtime. This module then runs a
single batched LLM call that:

1. Reads the claims table + ablation block from the report markdown.
2. Asks the LLM to decide a verdict for every claim row. The LLM reads the
   evidence/location text and infers metric direction (higher- vs lower-is-
   better) from context rather than relying on hard-coded rules.
3. Asks the LLM, in the same call, which components enumerated in
   methodological claims are not exercised by the ablation tables.

Axis self-selection in the technical-positioning matrix is a structural audit
(counting check marks across rows) and stays deterministic.

The LLM call is mandatory. If the configured LLM cannot be reached, the
caller is expected to surface the error rather than degrade silently.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Status ordering, from most-supportive to most-skeptical. "Capping" toward a
# weaker status means: take whichever of {current, cap} sits later in this list.
_STATUS_ORDER = ["supported", "partially supported", "inconclusive", "in conflict"]
_STATUS_RANK = {label: i for i, label in enumerate(_STATUS_ORDER)}


@dataclass
class ClaimAuditResult:
    """Outcome of auditing a single claim row."""

    original_status: str
    final_status: str
    llm_verdict: str = ""
    llm_reason: str = ""
    agent_self_verdict: str = ""
    agent_self_reason: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class ReportAuditOutcome:
    """Outcome of auditing the full report. The updated markdown is returned
    alongside by ``audit_review_markdown``."""

    claim_results: list[ClaimAuditResult] = field(default_factory=list)
    extra_weaknesses: list[str] = field(default_factory=list)
    axis_self_selection_ratio: float | None = None
    ablation_components_missing: list[str] = field(default_factory=list)
    llm_raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Status normalization helpers


def _strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", "", str(value or ""))


def _normalize_status(value: str) -> str:
    """Map any input status string into one of the canonical labels.

    ``pending`` and the empty string both normalize to ``""`` so that
    ``_cap_status`` treats them as "no status yet, replace with cap" rather
    than "more conservative than X".
    """
    s = _strip_html(value).strip()
    s = s.replace("✓", "").replace("⚠", "").replace("✗", "").strip().lower()
    if not s or s == "pending":
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
    """Return the more conservative of ``current`` and ``cap``.

    When one of the inputs is empty / unrecognized (e.g. ``Pending``), we
    return the other so the LLM verdict can fill in for an absent status.
    """
    cur = _normalize_status(current)
    cap_n = _normalize_status(cap)
    if cur not in _STATUS_RANK or cap_n not in _STATUS_RANK:
        return cur or cap_n
    return cur if _STATUS_RANK[cur] >= _STATUS_RANK[cap_n] else cap_n


# Agent self-tag pattern (free per-claim verdict the agent appends to the
# Assessment cell): [verdict: <label>; reason: <one short clause>].
_SELF_TAG_PATTERN = re.compile(
    r"(?i)\[verdict:\s*(?P<verdict>supported|partially[\s_-]?supported|partial|"
    r"inconclusive|unclear|in[\s_-]?conflict|conflict)"
    r"(?:\s*;\s*reason:\s*(?P<reason>[^\]]*))?\]"
)


def _extract_self_tag(assessment_text: str) -> tuple[str, str, str]:
    """Pull the agent's self-tag verdict out of an Assessment cell.

    Returns ``(verdict_label, reason, cleaned_assessment)``. ``verdict_label``
    is normalized via ``_normalize_status`` (or ""), and
    ``cleaned_assessment`` is the assessment with the bracketed tag removed
    so the user-visible cell stays clean.
    """
    raw = str(assessment_text or "")
    match = _SELF_TAG_PATTERN.search(raw)
    if not match:
        return "", "", raw
    verdict = _normalize_status(match.group("verdict") or "")
    reason = (match.group("reason") or "").strip()
    cleaned = (raw[: match.start()] + raw[match.end() :]).strip()
    return verdict, reason, cleaned


# ---------------------------------------------------------------------------
# Markdown structural helpers (table parsing, section anchors)


_CLAIMS_HEADER_PATTERN = re.compile(
    r"(?ims)^##\s+(?:\*\*)?3\.\s+Claims(?:\*\*)?\s*$\n(?P<body>.*?)(?=^##\s+|\Z)"
)
_TP_HEADER_PATTERN = re.compile(
    r"(?ims)^##\s+(?:\*\*)?2\.\s+Technical Positioning(?:\*\*)?\s*$\n(?P<body>.*?)(?=^##\s+|\Z)"
)
_ABLATION_PATTERN = re.compile(
    r"(?ims)^###\s+(?:\*\*)?Ablation Result(?:\*\*)?\s*$\n(?P<body>.*?)(?=^##\s+|^###\s+|\Z)"
)
_SUMMARY_HEADER_PATTERN = re.compile(
    r"(?ims)^##\s+(?:\*\*)?4\.\s+Summary(?:\*\*)?\s*$\n(?P<body>.*?)(?=^##\s+|\Z)"
)


def _split_table_row(row: str) -> list[str]:
    s = row.strip()
    if not (s.startswith("|") and s.endswith("|")):
        return []
    return [c.strip() for c in s.strip("|").split("|")]


def _cell_safe(value: str) -> str:
    # Keep literal pipe characters visible inside cells without letting them
    # become markdown table delimiters during status rewrites.
    return str(value or "").replace("|", "&#124;").strip()


def _normalize_claim_table_cells(cells: list[str], headers: list[str]) -> list[str]:
    expected = len(headers)
    if expected <= 0:
        return cells
    if len(cells) <= expected:
        return cells + [""] * max(0, expected - len(cells))

    header_signature = [_strip_html(h).replace("*", "").strip().lower() for h in headers]
    if header_signature == ["claim", "evidence", "assessment", "location"]:
        return [
            cells[0],
            " | ".join(cells[1:-2]),
            cells[-2],
            cells[-1],
        ]
    if header_signature == ["claim", "evidence", "assessment", "status", "location"]:
        return [
            cells[0],
            " | ".join(cells[1:-3]),
            cells[-3],
            cells[-2],
            cells[-1],
        ]
    return cells[: expected - 1] + [" | ".join(cells[expected - 1 :])]


def _find_column_index(headers: list[str], *needles: str) -> int:
    lowered = [_strip_html(h).strip().lower() for h in headers]
    for needle in needles:
        for i, h in enumerate(lowered):
            if needle in h:
                return i
    return -1


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


# ---------------------------------------------------------------------------
# Axis self-selection audit (deterministic, structural)


def audit_axis_self_selection(markdown: str) -> tuple[float | None, str | None]:
    """Detect axis-self-selection in the Technical Positioning table.

    Returns ``(ratio, weakness_bullet)``. ``ratio`` is the fraction of
    niche-dimension columns that are won only by the ``This Work`` row
    (everyone else is ×). When the matrix exhibits the classic
    "I picked axes only I win on" pattern, the second element is a weakness
    bullet reviewers can use to push back; otherwise it is None.

    Triggers when:
    - self exclusive wins >= 3 absolute, AND
    - ratio >= 0.4 of niche cols, AND
    - self exclusive wins >= 2x any single baseline's exclusive wins (or
      baseline has 0 exclusive wins).
    """
    text = str(markdown or "")
    sec = _TP_HEADER_PATTERN.search(text)
    if not sec:
        return None, None
    headers, rows = _parse_first_table(sec.group("body"))
    if len(headers) < 3 or not rows:
        return None, None

    # The agent prompt enforces: column 0 = Research domain, column 1 = Method.
    niche_start = 2
    niche_cols = list(range(niche_start, len(headers)))
    if not niche_cols:
        return None, None

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
    baseline_exclusive_wins: list[int] = [0] * len(other_rows)
    for col in niche_cols:
        if col >= len(self_row):
            continue
        if _is_check(self_row[col]):
            self_won += 1
            if all(col >= len(r) or not _is_check(r[col]) for r in other_rows):
                self_exclusive_wins += 1
        else:
            winners = [
                k
                for k, r in enumerate(other_rows)
                if col < len(r) and _is_check(r[col])
            ]
            if len(winners) == 1:
                baseline_exclusive_wins[winners[0]] += 1

    ratio = self_exclusive_wins / max(1, len(niche_cols))
    max_baseline_excl = max(baseline_exclusive_wins) if baseline_exclusive_wins else 0

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
# Weakness injection


def inject_weaknesses(markdown: str, bullets: list[str]) -> str:
    """Append extra weakness bullets to Section 4 / Weaknesses.

    The agent's section 4 puts ``Weaknesses:`` before a list. We insert the
    new bullets at the end of that list. Each bullet is prefixed ``[audit]``
    so a reader can tell the system added it.
    """
    if not bullets:
        return markdown
    text = str(markdown or "")
    sec = _SUMMARY_HEADER_PATTERN.search(text)
    if not sec:
        return text

    body = sec.group("body")

    # The agent sometimes writes "**Weaknesses:**" on its own line and
    # sometimes inlines the first bullet on the same line, so we match by
    # substring rather than full-line anchor.
    label_match = re.search(r"(?i)\*{0,2}Weaknesses\*{0,2}\s*:?", body)
    if label_match is None:
        addition = "\n\n**Weaknesses:**\n" + "\n".join(
            f"- [audit] {b}" for b in bullets
        )
        new_body = body.rstrip() + addition + "\n"
        return text[: sec.start("body")] + new_body + text[sec.end("body") :]

    # Walk forward from the label to find the end of the weakness bullet
    # block (the last bullet line that still belongs to Weaknesses).
    after_label = label_match.end()
    tail = body[after_label:]
    insertion_offset = len(tail)
    in_bullet_block = False
    cursor = 0
    for raw_line in tail.split("\n"):
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
            cursor += line_len
            continue
        if in_bullet_block and (
            is_label or stripped.startswith("##") or not is_bullet
        ):
            break
        cursor += line_len

    addition = "".join(f"- [audit] {b}\n" for b in bullets)
    abs_pos = after_label + insertion_offset
    prefix = "" if body[:abs_pos].endswith("\n") else "\n"
    new_body = body[:abs_pos] + prefix + addition + body[abs_pos:]
    return text[: sec.start("body")] + new_body + text[sec.end("body") :]


# ---------------------------------------------------------------------------
# Batched LLM adjudication (claim verdicts + ablation coverage)


_LLM_SYSTEM_PROMPT = (
    "You audit an academic-paper review report. For each claim row you are "
    "given, decide whether the cited evidence rigorously supports the claim. "
    "Then, looking across all methodological claims and the ablation block, "
    "list any enumerated components that the ablation tables do not cover. "
    "Be conservative like a careful peer reviewer: comparative claims "
    "(leading, outperforms, best, state-of-the-art) require the gap to be "
    "clearly larger than the reported uncertainty (>= 2 sigma if a sigma is "
    "reported, otherwise comparable to typical noise on the benchmark). "
    "Infer the metric direction (higher-is-better vs lower-is-better) from "
    "the evidence/location text rather than assuming. "
    "Reply ONLY in JSON of the form "
    '{"verdicts": [{"id": int, '
    '"verdict": "supported|partially_supported|inconclusive|in_conflict", '
    '"reason": "one short sentence"}], '
    '"ablation_missing_components": ["component name", ...]}.'
)


_VERDICT_LABEL_MAPPING: dict[str, str] = {
    "supported": "supported",
    "partially_supported": "partially supported",
    "partially supported": "partially supported",
    "partially": "partially supported",
    "partial": "partially supported",
    "inconclusive": "inconclusive",
    "unclear": "inconclusive",
    "in_conflict": "in conflict",
    "in conflict": "in conflict",
    "conflict": "in conflict",
}


def _verdict_to_label(verdict: str) -> str:
    v = (verdict or "").strip().lower().replace("-", "_")
    if v in _VERDICT_LABEL_MAPPING:
        return _VERDICT_LABEL_MAPPING[v]
    v_spaced = v.replace("_", " ")
    return _VERDICT_LABEL_MAPPING.get(v_spaced, "")


def _is_novelty_claim(claim: str) -> bool:
    return bool(re.search(
        r"(?i)\b(first to|to our knowledge|to the best of our knowledge"
        r"|novel(ly)?|pioneering|newly (proposed|introduced|developed)"
        r"|we propose|we introduce|unprecedented)\b",
        str(claim or ""),
    ))


def _is_comparative_claim(claim: str) -> bool:
    return bool(re.search(
        r"(?i)\b(outperform\w*|surpass\w*|better than|superior to|exceed\w*"
        r"|state.?of.?the.?art|sota|best.{0,15}baseline|competitive with"
        r"|compared (to|against|with))\b",
        str(claim or ""),
    ))


def _extract_positioning_context(markdown: str) -> str:
    sec = re.search(
        r"(?ims)^##\s+\*?\*?2\.\s+Technical\s+Positioning\*?\*?\s*$\n(?P<body>.*?)(?=^##\s+|\Z)",
        str(markdown or ""),
    )
    if not sec:
        return ""
    body = sec.group("body")
    table_lines = [ln for ln in body.splitlines() if ln.strip().startswith("|")]
    if not table_lines:
        return body.strip()[:1500]
    return "\n".join(table_lines)[:1500]


# Cap on how much of the ablation block we send to the LLM. Most ablation
# blocks are << 4000 chars; this guard prevents pathological reports from
# blowing the context.
_ABLATION_BLOCK_CHAR_LIMIT = 8000

_TYPE_SPECIFIC_RULES = (
    "\n\nType-specific rules:\n"
    "- [THEORETICAL] claims (theorem, proof, derivation, formal guarantee): "
    "'supported' requires a formal proof/theorem anchored to a numbered equation, lemma, or proposition. "
    "'partially_supported' for informal argument or sketch without full derivation. "
    "'in_conflict' for verbal description only with no formal anchor.\n"
    "- [METHODOLOGICAL] claims (architecture, algorithm, pipeline, design component): "
    "'supported' requires the component explicitly described in a named section, figure, algorithm box, or equation. "
    "'partially_supported' if high-level module is shown but a critical sub-component is absent.\n"
    "- [NOVELTY] claims ('first to', 'novel', 'to our knowledge'): "
    "finding prior contradicting work → 'in_conflict'. "
    "Absence of contradiction alone does not establish novelty — use 'partially_supported' when evidence is insufficient. "
    "'supported' requires positive evidence (e.g. positioning comparison) showing no directly comparable prior work.\n"
    "- [COMPARATIVE] claims ('outperforms', 'surpasses', 'SOTA'): "
    "'supported' requires the specific baseline(s) named AND numeric results for both method and baseline present; "
    "gap must clearly exceed the reported error bar (>= 2 sigma if given)."
)


def _format_execution_matches_block(matches: list[dict[str, Any]]) -> str:
    """Format execution reproduction matches into a readable context block."""
    if not matches:
        return ""
    lines = [
        "Execution reproduction results (treat as authoritative for empirical claims):"
    ]
    for m in matches[:20]:
        key = str(m.get("paper_key") or m.get("observed_key") or "").strip()
        paper_val = m.get("paper_value")
        obs_val = m.get("observed_value")
        if paper_val is None or obs_val is None:
            continue
        try:
            delta_pct = abs(float(paper_val) - float(obs_val)) / max(abs(float(paper_val)), 1e-9) * 100
        except (TypeError, ValueError):
            continue
        lines.append(
            f"  {key}: paper={paper_val}, reproduced={obs_val}, Δ={delta_pct:.1f}%"
        )
    return "\n".join(lines)


def _build_llm_prompt(
    *,
    claims: list[dict[str, Any]],
    ablation_block: str,
    positioning_context: str = "",
    execution_matches: list[dict[str, Any]] | None = None,
) -> str:
    has_novelty = any(_is_novelty_claim(e["claim"]) for e in claims)
    claim_blocks: list[str] = []
    for entry in claims:
        claim_type = entry.get("type", "").lower().strip()
        flags: list[str] = []
        if claim_type == "theoretical":
            flags.append("THEORETICAL")
        elif claim_type == "methodological":
            flags.append("METHODOLOGICAL")
        else:
            # empirical or unknown — use regex to detect novelty/comparative sub-types
            if _is_novelty_claim(entry["claim"]):
                flags.append("NOVELTY")
            if _is_comparative_claim(entry["claim"]):
                flags.append("COMPARATIVE")
        flag_str = f" [{', '.join(flags)}]" if flags else ""

        lines = [
            f"--- claim id={entry['id']}{flag_str} ---",
            f"Claim: {entry['claim']}",
            f"Evidence: {entry['evidence']}",
            f"Location: {entry['location']}",
        ]

        claim_blocks.append("\n".join(lines))

    trimmed_ablation = ablation_block.strip()
    if len(trimmed_ablation) > _ABLATION_BLOCK_CHAR_LIMIT:
        trimmed_ablation = (
            trimmed_ablation[:_ABLATION_BLOCK_CHAR_LIMIT]
            + "\n... [truncated]"
        )
    if not trimmed_ablation:
        trimmed_ablation = "(no ablation section in this report)"

    positioning_block = ""
    if has_novelty and positioning_context:
        positioning_block = (
            "\n\nTechnical positioning (use when auditing [NOVELTY] claims):\n"
            + positioning_context
        )

    exec_block = ""
    if execution_matches:
        formatted = _format_execution_matches_block(execution_matches)
        if formatted:
            exec_block = "\n\n" + formatted

    execution_rules = (
        "\n\nExecution evidence rules (apply when execution comparison data is present below):\n"
        "- First determine the metric's direction (higher-is-better or lower-is-better) from "
        "the metric name and claim context — do not assume higher is always better.\n"
        "- Then assess whether the observed delta is meaningful: a 0.1% gap on a saturated "
        "benchmark (e.g. ImageNet top-1 above 85%) is different from the same absolute gap "
        "on a noisy task. Use your knowledge of the benchmark and metric to judge significance.\n"
        "- If the delta is clearly in the wrong direction (e.g. FID went up when the claim "
        "says generation quality improved), that is strong evidence against 'supported'.\n"
        "- Agent self-notes are provided for context only. Do not treat them as verdicts — "
        "the same model that wrote the report produced them and may be optimistic."
    )

    return (
        "You are an independent reviewer auditing claim rows from a paper review report. "
        "Make a single holistic judgment per claim using all available evidence: "
        "the claim text, supporting evidence, execution reproduction data, and positioning context.\n\n"
        "Decision rules:\n"
        "- 'supported' requires the evidence to directly justify the claim's "
        "verb. For comparative claims, the gap vs. the strongest comparator "
        "must be reasonably larger than the reported error bar (>= 2 sigma "
        "if sigma is given). Infer metric direction from context.\n"
        "- 'partially_supported' applies when the evidence supports the "
        "qualitative direction but the magnitude or scope is weaker than "
        "what the claim asserts, or when only a subset of the claim's "
        "components is evidenced.\n"
        "- 'inconclusive' applies when the gap is within 1 sigma of the "
        "comparator, the comparator is not actually evaluable from the "
        "evidence, or the evidence is anecdotal (case study/appendix) "
        "rather than tabular.\n"
        "- 'in_conflict' applies when the evidence contradicts the claim's "
        "verb (e.g., paper value lower than the strongest comparator on a "
        "higher-is-better metric).\n"
        + _TYPE_SPECIFIC_RULES
        + execution_rules
        + positioning_block
        + exec_block
        + "\n\nFor ablation_missing_components, look across every methodological "
        "claim that enumerates a list of components (\"X with A, B, and C\", "
        "\"system consisting of A, B, C\", etc.) and list any component name "
        "that the ablation block clearly does not exercise. Use the original "
        "wording from the claim. If everything is covered (or no methodological "
        "claim enumerates components), return an empty list.\n\n"
        f"Claims to audit (n={len(claims)}):\n\n"
        + "\n\n".join(claim_blocks)
        + "\n\nAblation section to compare against:\n"
        + trimmed_ablation
        + "\n\n"
        'Respond ONLY with one JSON object: '
        '{"verdicts": [...], "ablation_missing_components": [...]}.'
    )


def _resolve_default_llm_call() -> Callable[[str], dict[str, Any]]:
    """Resolve the project-configured LLM into a single-prompt callable.

    Raises if the LLM client cannot be imported. The returned callable
    raises ``RuntimeError`` when ``llm_json`` reports a transport-level
    error (the caller is expected not to degrade silently).
    """
    from llm.client import llm_json, resolve_llm_config

    cfg = resolve_llm_config()

    def _call(prompt: str) -> dict[str, Any]:
        result = llm_json(
            prompt=prompt,
            system=_LLM_SYSTEM_PROMPT,
            cfg=cfg,
            module="report_generation",
        )
        if isinstance(result, dict) and result.get("status") == "error":
            raise RuntimeError(
                "LLM call failed: "
                f"{result.get('error')} "
                f"(provider={result.get('provider')}, model={result.get('model')})"
            )
        return result

    return _call


def _collect_claim_entries(claims_body: str) -> tuple[
    list[str],
    int,
    int,
    list[dict[str, Any]],
]:
    """Locate the claims table inside Section 3 body and return a list of
    per-row entries with cell handles for in-place editing.

    Returns ``(lines, status_idx, assessment_idx, entries)``. ``entries[i]``
    carries the row line index and parsed cells so callers can mutate them
    and rejoin.
    """
    # Use split("\n") rather than splitlines() so trailing-newline structure
    # round-trips correctly when we rejoin.
    lines = claims_body.split("\n")
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
        return lines, -1, -1, []

    headers = _split_table_row(lines[header_idx])
    claim_idx = _find_column_index(headers, "claim")
    evidence_idx = _find_column_index(headers, "evidence")
    assessment_idx = _find_column_index(headers, "assessment")
    status_idx = _find_column_index(headers, "status")
    location_idx = _find_column_index(headers, "location")
    type_idx = _find_column_index(headers, "type")
    importance_idx = _find_column_index(headers, "importance")
    if claim_idx < 0 or evidence_idx < 0 or status_idx < 0:
        return lines, status_idx, assessment_idx, []

    entries: list[dict[str, Any]] = []
    j = header_idx + 2
    next_id = 0
    while j < len(lines):
        s = lines[j].strip()
        if not (s.startswith("|") and s.endswith("|")):
            break
        cells = _normalize_claim_table_cells(_split_table_row(lines[j]), headers)
        if len(cells) <= max(claim_idx, evidence_idx, status_idx):
            j += 1
            continue
        location_text = (
            _strip_html(cells[location_idx])
            if 0 <= location_idx < len(cells)
            else ""
        )
        entries.append(
            {
                "id": next_id,
                "row_line_idx": j,
                "cells": cells,
                "claim": _strip_html(cells[claim_idx]),
                "evidence": _strip_html(cells[evidence_idx]),
                "location": location_text,
                "type": _strip_html(cells[type_idx]).lower().strip() if 0 <= type_idx < len(cells) else "",
                "importance": _strip_html(cells[importance_idx]).lower().strip() if 0 <= importance_idx < len(cells) else "",
            }
        )
        next_id += 1
        j += 1
    return lines, status_idx, assessment_idx, entries


def audit_review_markdown(
    markdown: str,
    *,
    llm_call: Callable[[str], dict[str, Any]] | None = None,
    execution_alignment: dict[str, Any] | None = None,
) -> tuple[str, ReportAuditOutcome]:
    """Apply all audits and return ``(updated_markdown, outcome)``.

    The LLM call is mandatory: when ``llm_call`` is None we resolve the
    project-default LLM via ``llm.client.resolve_llm_config``. Failures
    propagate to the caller rather than degrading silently.
    """
    text = str(markdown or "")
    outcome = ReportAuditOutcome()

    claims_section = _CLAIMS_HEADER_PATTERN.search(text)
    entries: list[dict[str, Any]] = []
    lines: list[str] = []
    status_idx = -1
    assessment_idx = -1
    claims_body = ""
    if claims_section is not None:
        claims_body = claims_section.group("body")
        lines, status_idx, assessment_idx, entries = _collect_claim_entries(claims_body)

    if not entries:
        # No usable claims table; the structural axis audit still runs so
        # the report's positioning matrix gets reviewed even when claims
        # are missing.
        ratio, axis_bullet = audit_axis_self_selection(text)
        outcome.axis_self_selection_ratio = ratio
        if axis_bullet:
            outcome.extra_weaknesses.append(axis_bullet)
            text = inject_weaknesses(text, [axis_bullet])
        return text, outcome

    if llm_call is None:
        llm_call = _resolve_default_llm_call()

    # Pre-pass: extract agent self-tags and execution statuses from every
    # entry before building the prompt. Self-tags are stripped from the
    # display cells so the LLM audits independently — the agent verdict is
    # applied only as a post-LLM conservative cap, never shown to the LLM.
    agent_notes: dict[int, tuple[str, str]] = {}
    for entry in entries:
        cells = entry["cells"]
        original_status_cell = cells[status_idx] if 0 <= status_idx < len(cells) else ""
        entry["execution_status"] = _normalize_status(original_status_cell)
        if 0 <= assessment_idx < len(cells):
            agent_verdict, agent_reason, cleaned = _extract_self_tag(cells[assessment_idx])
            if cleaned != cells[assessment_idx]:
                cells[assessment_idx] = cleaned
            if agent_verdict:
                agent_notes[entry["id"]] = (agent_verdict, agent_reason)

    ablation_match = _ABLATION_PATTERN.search(text)
    ablation_block = ablation_match.group("body") if ablation_match else ""

    positioning_context = _extract_positioning_context(text)

    execution_matches: list[dict[str, Any]] = []
    if isinstance(execution_alignment, dict):
        raw_matches = execution_alignment.get("matches")
        if isinstance(raw_matches, list):
            execution_matches = [m for m in raw_matches if isinstance(m, dict)]

    prompt = _build_llm_prompt(
        claims=[
            {
                "id": entry["id"],
                "claim": entry["claim"],
                "evidence": entry["evidence"],
                "location": entry["location"],
                "type": entry.get("type", ""),
                "execution_status": entry["execution_status"],
            }
            for entry in entries
        ],
        ablation_block=ablation_block,
        positioning_context=positioning_context,
        execution_matches=execution_matches,
    )
    raw = llm_call(prompt) or {}
    if not isinstance(raw, dict):
        raw = {}
    outcome.llm_raw = {
        k: v
        for k, v in raw.items()
        if k in {"verdicts", "ablation_missing_components"}
    }

    verdicts_raw = raw.get("verdicts")
    if not isinstance(verdicts_raw, list):
        verdicts_raw = []
    missing_raw = raw.get("ablation_missing_components")
    if not isinstance(missing_raw, list):
        missing_raw = []

    verdict_by_id: dict[int, dict[str, Any]] = {}
    for item in verdicts_raw:
        if not isinstance(item, dict):
            continue
        try:
            cid = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        verdict_by_id[cid] = item

    new_lines = list(lines)
    for entry in entries:
        cells = entry["cells"]
        normalized_original = entry["execution_status"]

        result = ClaimAuditResult(
            original_status=normalized_original,
            final_status=normalized_original,
        )

        # Restore agent self-tag metadata for tracking (already stripped from cell).
        agent_note = agent_notes.get(entry["id"])
        if agent_note:
            result.agent_self_verdict = agent_note[0]
            result.agent_self_reason = agent_note[1]
            if agent_note[1]:
                result.notes.append(f"Agent self-note: {agent_note[1]}")

        llm_item = verdict_by_id.get(entry["id"], {})
        llm_verdict = _verdict_to_label(str(llm_item.get("verdict") or ""))
        llm_reason = str(llm_item.get("reason") or "").strip()
        result.llm_verdict = llm_verdict
        result.llm_reason = llm_reason
        if llm_reason:
            result.notes.append(f"LLM: {llm_reason}")

        # The LLM audits independently (no agent verdict in its prompt).
        # After the LLM decides, apply the agent self-verdict as a conservative
        # cap: take whichever of LLM or agent is the more critical verdict.
        # Fall back to execution pre-assessment when the LLM returns nothing,
        # and to "inconclusive" when neither source has a usable verdict.
        if llm_verdict:
            agent_note = agent_notes.get(entry["id"])
            agent_verdict_label = _verdict_to_label(agent_note[0]) if agent_note else ""
            if agent_verdict_label and agent_verdict_label in _STATUS_RANK and llm_verdict in _STATUS_RANK:
                new_label = _cap_status(llm_verdict, agent_verdict_label)
            else:
                new_label = llm_verdict
        elif normalized_original and normalized_original != "pending":
            new_label = normalized_original
        else:
            new_label = "inconclusive"
        result.final_status = new_label

        # Always rejoin the row: even if status didn't change, the assessment
        # cell may have had a self-tag stripped above.
        if new_label and new_label != normalized_original:
            cells[status_idx] = _format_status_html(new_label)
        new_lines[entry["row_line_idx"]] = "| " + " | ".join(_cell_safe(c) for c in cells) + " |"
        outcome.claim_results.append(result)

    new_claims_body = "\n".join(new_lines)
    if new_claims_body != claims_body:
        text = (
            text[: claims_section.start("body")]
            + new_claims_body
            + text[claims_section.end("body") :]
        )

    # Structural audit on the updated markdown.
    ratio, axis_bullet = audit_axis_self_selection(text)
    outcome.axis_self_selection_ratio = ratio

    cleaned_missing: list[str] = []
    for item in missing_raw:
        token = str(item or "").strip()
        if token and token not in cleaned_missing:
            cleaned_missing.append(token)
    outcome.ablation_components_missing = cleaned_missing

    bullets: list[str] = []
    for r in outcome.claim_results:
        # Only emit a downgrade bullet when the rank actually moved toward
        # more conservative. Promotions from Pending (empty original status)
        # to a real verdict are not downgrades and should not appear here.
        orig_rank = _STATUS_RANK.get(r.original_status, -1)
        final_rank = _STATUS_RANK.get(r.final_status, -1)
        if orig_rank >= 0 and final_rank > orig_rank:
            reason = r.llm_reason or r.agent_self_reason or "audit cap"
            bullets.append(
                f"Status downgraded to {r.final_status.title()} on a claim - {reason}"
            )
    if axis_bullet:
        bullets.append(axis_bullet)
    if cleaned_missing:
        if len(cleaned_missing) == 1:
            bullets.append(
                "The methodological claims enumerate a component that the "
                f"ablation tables do not cover: {cleaned_missing[0]}."
            )
        else:
            bullets.append(
                "The methodological claims enumerate components that the "
                "ablation tables do not cover: "
                + ", ".join(cleaned_missing)
                + "."
            )

    outcome.extra_weaknesses = bullets
    if bullets:
        text = inject_weaknesses(text, bullets)

    return text, outcome
