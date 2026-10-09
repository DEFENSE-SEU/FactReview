"""Render final v2 records without extracting claims or changing assessments."""

from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from review.report.evidence_tables import table_passage_lines
from schemas.claim import ClaimLocation, ClaimStatus, Evidence
from schemas.review import FinalReview

STATUS_ORDER = [ClaimStatus.FLAWED, ClaimStatus.QUESTIONED, ClaimStatus.UNVERIFIED, ClaimStatus.SUPPORTED]


def _text(value) -> str:
    # Keep paper/model text from injecting headings or HTML into the report.
    text = html.escape(str(value)).replace("\n", " ").replace("|", "&#124;")
    return re.sub(r"([\\`*_\[\]{}()#+.!~-])", r"\\\1", text)


def validate_publication_language(value) -> None:
    if isinstance(value, dict):
        for item in value.values():
            validate_publication_language(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            validate_publication_language(item)
        return
    if not isinstance(value, str):
        return
    text = " ".join(value.split())
    patterns = (
        r"\b(?:recommend(?:ation|ed)?\s*:?\s*(?:to\s+)?(?:accept(?:ance|ing)?|reject(?:ion|ing)?)|accept(?:ance)?\s+recommendation|reject(?:ion)?\s+recommendation)\b",
        r"\bdecision\s*:\s*(?:(?:weak|strong)\s+)?(?:accept(?:ed)?|reject(?:ed)?)\b",
        r"\b(?:paper|submission|manuscript|work)\s+(?:(?:should|must)\s+be|is)\s+(?:accept|reject)ed\b",
        r"\b(?:vote(?:d|s)?|voting)(?:\s+is)?\s+(?:to\s+(?:accept|reject)|for\s+(?:the\s+)?(?:acceptance|rejection)\s+of)\s+(?:(?:this|the|that|a)\s+)?(?:paper|submission|manuscript|work)\b",
        r"(?:建议|推荐|决定)\s*(?:录用|接收|拒稿)(?:该|这篇)?(?:论文|稿件)?",
    )
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
        raise ValueError("Review records contain publication recommendation language")


def _location(loc: ClaimLocation) -> str:
    parts = []
    if loc.page:
        parts.append(f"page {loc.page}")
    if loc.section:
        parts.append(f"section {_text(loc.section)}")
    if loc.char_start is not None:
        parts.append(f"characters {loc.char_start}–{loc.char_end}")
    return "; ".join(parts)


def _source_index(claims, findings):
    sources, occurrences = {}, []
    groups = [(claim.id, claim.evidence) for claim in claims]
    groups += [(f"finding {index}", finding.evidence) for index, finding in enumerate(findings, 1)]
    for owner, evidence in groups:
        for ordinal, item in enumerate(evidence, 1):
            occurrence = {
                "anchor": f"factreview-evidence-{len(occurrences) + 1:06d}",
                "label": f"E{len(occurrences) + 1:04d}",
                "owner": f"{owner}, evidence {ordinal}",
                "usages": [],
            }
            for index, pointer in enumerate([item.pointer, *item.additional_pointers]):
                role = f"additional {index}" if index else "primary"
                usage = {
                    "anchor": occurrence["anchor"] + (f"-source-{index + 1:02d}" if index else ""),
                    "label": occurrence["label"] + (f" / {role}" if item.additional_pointers else ""),
                    "owner": occurrence["owner"],
                    "source": None,
                }
                key = (pointer.locator, pointer.page, pointer.line, pointer.key, pointer.quote)
                if pointer.quote:
                    if key not in sources:
                        digest = hashlib.sha256(
                            json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                        ).hexdigest()
                        sources[key] = {
                            "anchor": f"factreview-source-{digest}",
                            "label": f"S{len(sources) + 1:04d}",
                            "occurrences": [],
                        }
                    source = sources[key]
                    usage["source"] = source
                    source["occurrences"].append(usage)
                occurrence["usages"].append(usage)
            occurrences.append(occurrence)
    return iter(occurrences)


def _pointer_location(pointer):
    location = [pointer.locator]
    location.extend(
        f"{name} {value}"
        for name, value in (("page", pointer.page), ("line", pointer.line), ("key", pointer.key))
        if value is not None
    )
    return _text("; ".join(location))


def _passage(pointer, usage=None):
    lines = []
    source = usage["source"] if usage else None
    repeated = source is not None and source["occurrences"][0] is not usage
    if repeated:
        lines.append(f"  - Passage: [Source {source['label']}](#{source['anchor']}) (same exact source).")
    elif pointer.quote:
        if source:
            backlinks = ", ".join(
                f"[{_text(row['owner'])} / {row['label']}](#{row['anchor']})" for row in source["occurrences"]
            )
            lines.append(
                f'  - <a id="{source["anchor"]}"></a>Source {source["label"]}; occurrences: {backlinks}.'
            )
        if re.search(r"<(?:table|tr|td|th)\b", pointer.quote, re.I):
            table_lines = table_passage_lines(pointer.quote, _text)
            if table_lines is None:
                lines.append("  - Table layout unavailable; original passage follows.")
                lines.append(f"  - Passage: {_text(pointer.quote)}")
            else:
                lines.extend(table_lines)
        else:
            lines.append(f"  - Passage: {_text(pointer.quote)}")
    return lines


def _evidence(item: Evidence, occurrence=None, *, reference_context=False) -> list[str]:
    source = "paper-internal" if item.source == "paper_internal" else item.source
    label = "reference source; discrepancy unconfirmed" if reference_context else item.direction
    lines = [
        f"- **{source} / {label}**; sufficient: {str(item.sufficient).lower()}; "
        f"covers: {_text(', '.join(item.covered) or 'no claim conditions')}. "
        f"Pointer: {_pointer_location(item.pointer)}."
    ]
    if occurrence:
        lines[0] += f' <a id="{occurrence["anchor"]}"></a>Evidence {occurrence["label"]}.'
    lines.extend(_passage(item.pointer, occurrence["usages"][0] if occurrence else None))
    for index, pointer in enumerate(item.additional_pointers, 1):
        usage = occurrence["usages"][index] if occurrence else None
        line = f"  - Additional pointer {index}: {_pointer_location(pointer)}."
        if usage:
            line += f' <a id="{usage["anchor"]}"></a>Source usage {usage["label"]}.'
        lines.append(line)
        lines.extend(_passage(pointer, usage))
    if item.note:
        lines.append(f"  - Detail: {_text(item.note)}")
    if item.source == "execution":
        lines.append(
            f"  - Aligned: {str(item.aligned).lower()}; provenance: "
            f"{_text(item.provenance.model_dump_json() if item.provenance else 'unavailable')}"
        )
    return lines


def ordered_claims(review: FinalReview):
    order = {status: index for index, status in enumerate(STATUS_ORDER)}
    return sorted(
        review.claims, key=lambda claim: (order[claim.status], claim.importance != "core", claim.id)
    )


def execution_summary(review: FinalReview) -> dict:
    """Count retained records; an attempt is not a successful reproduction."""
    return {
        "plan_records": len(review.ledger),
        "recorded_attempts": sum(
            len(row["attempts"]) for row in review.ledger if isinstance(row.get("attempts"), list)
        ),
        "claims_with_aligned_execution_evidence": sum(
            any(e.source == "execution" and e.aligned is True and e.affects_claim for e in claim.evidence)
            for claim in review.claims
        ),
        "outcomes_incomplete": "execution" in review.incomplete_stages,
    }


def _theory_derivations(claim):
    if not claim.theory_derivations:
        return []
    lines = [
        "",
        "Theory derivation traces:",
        "",
        "Model reasoning is recorded below. Program validation checks structure and source correspondence; mathematical correctness requires review.",
    ]
    from review.report.advice import theory_source_integrity

    try:
        theory_source_integrity(claim)
    except (OSError, ValueError) as exc:
        lines += [
            "",
            "**Current Theory source integrity is unavailable.** " + _text(str(exc)),
            "Trace states below describe the retained verification. Its recorded source artifacts no longer validate.",
        ]
    for record in claim.theory_derivations:
        lines += [
            "",
            f"- Phase: {_text(record.phase)}; state: {_text(record.state)}; "
            f"adopted: {str(record.adopted).lower()}; conditions: {_text(', '.join(record.covered))}.",
        ]
        if record.source_pointer:
            lines += [f"  - Original proof location: {_pointer_location(record.source_pointer)}."]
            lines.extend(_passage(record.source_pointer))
        trace = record.trace
        if trace:
            lines += [
                f"  - Goal: {_text(trace.goal)}.",
                f"  - Outcome: {_text(trace.outcome)}. {_text(trace.completion_reason)}",
            ]
            for kind, entries in (
                ("Assumption", trace.assumptions),
                ("Step", trace.steps),
                ("Gap", trace.gaps),
            ):
                for entry in entries:
                    if kind == "Assumption":
                        detail = f"{entry.id} ({entry.status}): {entry.text}"
                    elif kind == "Step":
                        detail = (
                            f"{entry.id}: {entry.statement}. Reason: {entry.reason}. "
                            f"Assumptions: {', '.join(entry.assumption_ids) or 'none'}. "
                            f"Previous steps: {', '.join(entry.previous_step_ids) or 'none'}"
                        )
                    else:
                        detail = f"{entry.at}: {entry.reason}. Needed: {entry.needed}"
                    lines.append(f"  - {kind}: {_text(detail)}.")
                    for source in entry.sources:
                        lines.append(f"  - Source: {_pointer_location(source.pointer)}.")
                        lines.extend(_passage(source.pointer))
        lines.extend(f"  - Limitation: {_text(issue)}" for issue in record.issues)
        if record.audit_pointer:
            lines.append(f"  - Trace audit: {_text(record.audit_pointer)}.")
    return lines


def _checked_report(review):
    from review.report.advice import checked_review
    from screening.reference_corrections import checked_correction

    result = checked_review(review)
    for finding in result.findings:
        if finding.reference_correction:
            finding.reference_correction = checked_correction(finding.reference_correction)
    return result


def _reference_correction(correction):
    if correction is None:
        return []
    lines = ["", f"Reference metadata suggestion: {_text(correction.state)}.", _text(correction.reason), ""]
    if correction.state == "metadata_candidate":
        # A metadata title may contain Markdown fences; preserve the exact BibTeX
        # inside a longer fence so it remains literal source content.
        runs = re.findall(r"`+", correction.corrected_bibtex)
        fence = "`" * max(3, max((len(value) + 1 for value in runs), default=0))
        lines += [
            fence + "bibtex",
            correction.corrected_bibtex,
            fence,
            "",
            "| Field | Retrieved value | Source record |",
            "|---|---|---|",
        ]
        for field in correction.fields:
            lines.append(
                f"| {_text(field.field)} | {_text(field.value)} | "
                f"{_text(field.backend)} / {_text(field.record_id)}: {_text(field.url)} |"
            )
        lines += ["", f"Identity: {_text(correction.identity_identifier)}."]
    lines.append(f"Original reference check: {_text(correction.raw_result_pointer)}.")
    if correction.records_pointer:
        lines.append(
            f"Metadata provenance: {_text(correction.records_pointer)}; "
            f"SHA256: {_text(correction.records_sha256)}."
        )
    return lines


def verification_limitations(
    *,
    issues=None,
    figure_coverage=None,
    figure_context_coverage=None,
    table_coverage=None,
    table_context_coverage=None,
    writing_coverage=None,
    anonymity_policy=None,
    token_usage=None,
) -> list[str]:
    """Make coverage and cost uncertainty visible even when there are no findings."""
    limitations = list(issues or [])
    if writing_coverage is not None:
        missing = writing_coverage.get("failed", 0) + writing_coverage.get("unavailable", 0)
        if missing:
            limitations.append(
                f"Writing screening is incomplete: {writing_coverage.get('failed', 0)} failed and "
                f"{writing_coverage.get('unavailable', 0)} unavailable out of {writing_coverage.get('total', 0)} sections."
            )
    if anonymity_policy == "unspecified":
        limitations.append(
            "Submission anonymity policy was unspecified; anonymity violations were not assessed."
        )
    if figure_coverage is not None:
        if not figure_coverage.get("total", 0):
            limitations.append("No figure inputs were available for visual checks.")
        missing = figure_coverage.get("failed", 0) + figure_coverage.get("unavailable", 0)
        if missing:
            limitations.append(
                f"Figure screening is incomplete: {figure_coverage.get('failed', 0)} failed and "
                f"{figure_coverage.get('unavailable', 0)} unavailable out of {figure_coverage.get('total', 0)} figures."
            )
    if figure_context_coverage is not None:
        missing = sum(figure_context_coverage.get(key, 0) for key in ("failed", "unavailable", "unrecorded"))
        if missing:
            limitations.append(
                f"Figure page-context confirmation is incomplete: {figure_context_coverage.get('failed', 0)} failed, "
                f"{figure_context_coverage.get('unavailable', 0)} unavailable and "
                f"{figure_context_coverage.get('unrecorded', 0)} unrecorded. Crop checks have separate coverage."
            )
    if token_usage:
        failed = token_usage.get("failed_requests", 0)
        unavailable = token_usage.get("unavailable_usage_requests", 0)
        if failed:
            limitations.append(f"{failed} model call attempt(s) failed; see the recorded check limitations.")
        if unavailable:
            limitations.append(
                f"Provider-reported token usage is unavailable for {unavailable} call attempt(s); "
                "recorded token totals may be incomplete or estimated."
            )
        limitations.extend(token_usage.get("warnings", []))
    if table_coverage is not None:
        missing = table_coverage.get("failed", 0) + table_coverage.get("unavailable", 0)
        if missing:
            limitations.append(
                f"Table visual screening is incomplete: {table_coverage.get('failed', 0)} failed and "
                f"{table_coverage.get('unavailable', 0)} unavailable out of {table_coverage.get('total', 0)} tables."
            )
    if table_context_coverage is not None:
        missing = sum(table_context_coverage.get(key, 0) for key in ("failed", "unavailable", "unrecorded"))
        if missing:
            limitations.append(
                f"Table page-context confirmation is incomplete: {table_context_coverage.get('failed', 0)} failed, "
                f"{table_context_coverage.get('unavailable', 0)} unavailable and "
                f"{table_context_coverage.get('unrecorded', 0)} unrecorded. Crop checks have separate coverage."
            )
    return list(dict.fromkeys(limitations))


def render_markdown(
    review: FinalReview,
    *,
    issues: list[str] | None = None,
    figure_coverage=None,
    figure_context_coverage=None,
    table_coverage=None,
    table_context_coverage=None,
    writing_coverage=None,
    anonymity_policy=None,
    token_usage=None,
) -> str:
    review = _checked_report(review)
    claims = ordered_claims(review)
    occurrences = _source_index(claims, review.findings)
    lines = [
        f"# FactReview — {_text(review.paper_key)}",
        "",
    ]
    if review.run_status == "partial":
        lines += [
            f"**Partial review — incomplete stages: {_text(', '.join(review.incomplete_stages))}.**",
            "Counts cover retained claims only. Missing claims or evidence cannot establish an absence of paper problems.",
            "",
        ]
    lines += [
        "## 1. Overview",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {status.value} | {review.summary_counts[status]} |" for status in STATUS_ORDER)
    execution = execution_summary(review)
    execution_lines = [
        "",
        "### Execution coverage",
        "",
        "| Plan records | Recorded attempts | Claims with aligned execution evidence |",
        "|---:|---:|---:|",
        f"| {execution['plan_records']} | {execution['recorded_attempts']} | "
        f"{execution['claims_with_aligned_execution_evidence']} |",
        "",
        "Pipeline completion describes delivery of the available checks. Recorded attempts can fail or remain unaligned.",
    ]
    if review.execution_requested is not None:
        lines += execution_lines
        lines += [f"Execution requested: {str(review.execution_requested).lower()}.", ""]
        if execution["outcomes_incomplete"]:
            lines += ["Execution records are incomplete; additional attempts or outcomes may be unknown.", ""]
    if token_usage is not None:
        lines += [
            "",
            "### Model call accounting",
            "",
            "| Call attempts | Failed attempts | Usage unavailable | Input images |",
            "|---:|---:|---:|---:|",
            f"| {token_usage.get('requests', 0)} | {token_usage.get('failed_requests', 0)} | "
            f"{token_usage.get('unavailable_usage_requests', 0)} | {token_usage.get('image_count', 0)} |",
        ]
    lines += ["", "### Items requiring attention", ""]
    concerns = [claim for claim in claims if claim.status in {ClaimStatus.FLAWED, ClaimStatus.QUESTIONED}]
    lines += [
        f"- {claim.id} ({claim.status.value}, {claim.importance}): {_text(claim.text)}" for claim in concerns
    ] or ["No claim is assessed as flawed or questioned."]
    lines += ["", "## 2. Claim list", ""]
    for claim in claims:
        advice_targets = {}
        lines += [
            f"### {_text(claim.id)} — {claim.status.value}",
            "",
            _text(claim.text),
            "",
            f"Location: {_location(claim.loc)}. Importance: {claim.importance}.",
            "",
            "Conditions:",
            "",
        ]
        lines += [
            f"- {_text(condition.id)}: {_text(json.dumps(condition.model_dump(exclude={'id'}), ensure_ascii=False))}"
            for condition in claim.conditions
        ]
        if claim.source_refs:
            lines += ["", "Original manuscript sources (claim provenance):", ""]
            lines += [
                f"- {_text(ref.source_block_id)}; {_location(ref.loc)}; "
                f"conditions: {_text(', '.join(ref.covered))}. Quote: {_text(ref.source_quote)}"
                for ref in claim.source_refs
            ]
        lines += ["", f"Evidence needs: {', '.join(claim.needs) or 'none'}.", "", "Evidence:", ""]
        if not claim.evidence:
            lines.append("No evidence is available for assessment.")
        for index, item in enumerate(claim.evidence):
            occurrence = next(occurrences)
            advice_targets[f"/evidence/{index}"] = occurrence["anchor"]
            lines.extend(_evidence(item, occurrence))
        lines.extend(_theory_derivations(claim))
        if claim.verification_limitations:
            lines += ["", "System verification limitations:", ""]
            for limitation in claim.verification_limitations:
                lines.append(
                    f"- {_text(limitation.stage)}: {_text(limitation.reason)}. "
                    f"Conditions: {_text(', '.join(limitation.condition_ids))}. "
                    "The system operator should repair or retry this check. The failure alone does not identify missing author material."
                )
        if claim.advice is not None:
            lines += ["", "Reviewer advice:", ""]
            if claim.advice.state == "unavailable":
                lines.append(f"Advice unavailable: {_text(claim.advice.failure_reason)}")
            else:
                from review.report.advice import advice_input

                data = advice_input(claim, review.ledger, version=claim.advice.input_version)
                for item in claim.advice.items:
                    lines.append(f"- {_text(item.text)} Conditions: {_text(', '.join(item.condition_ids))}.")
                    if item.action == "verification_followup":
                        lines.append(
                            "  - Action for the system operator: repair or retry the recorded verification."
                        )
                    for ref in item.basis_refs:
                        if ref in advice_targets:
                            lines.append(
                                f"  - Basis: [evidence {_text(ref.rsplit('/', 1)[1])}](#{advice_targets[ref]})."
                            )
                        else:
                            # Context and missing coverage remain visible beside
                            # the advice; they do not become verification evidence.
                            lines.append(
                                f"  - Basis {_text(ref)}: {_text(json.dumps(data['basis'][ref]['content'], ensure_ascii=False))}"
                            )
                lines.append(
                    "Advice wording is a model interpretation; references and input consistency were checked."
                )
            if claim.advice.audit_pointer:
                lines.append(
                    f"Advice audit: {_text(claim.advice.audit_pointer)}; input SHA256: {_text(claim.advice.input_sha256)}."
                )
        lines += ["", "Questions for authors:", ""]
        lines += [
            f"- {_text(question.text)} Reason: {_text(question.reason)}" for question in claim.questions
        ] or ["None recorded."]
        if claim.notes:
            lines += ["", "Notes:", "", *[f"- {_text(note)}" for note in claim.notes]]
        lines.append("")
    lines += ["## 3. Other findings", ""]
    if writing_coverage is not None:
        lines += [
            "### Writing screening coverage",
            "",
            "| Total sections | Checked | Failed | Unavailable |",
            "|---:|---:|---:|---:|",
            f"| {writing_coverage.get('total', 0)} | {writing_coverage.get('checked', 0)} | "
            f"{writing_coverage.get('failed', 0)} | {writing_coverage.get('unavailable', 0)} |",
            "",
            f"Submission anonymity policy: {_text(anonymity_policy or 'unspecified')}.",
            "",
        ]
    if figure_coverage is not None:
        lines += [
            "### Figure screening coverage",
            "",
            "| Total figures | Checked | Failed | Unavailable |",
            "|---:|---:|---:|---:|",
            f"| {figure_coverage.get('total', 0)} | {figure_coverage.get('checked', 0)} | "
            f"{figure_coverage.get('failed', 0)} | {figure_coverage.get('unavailable', 0)} |",
            "",
            "Counts refer to parsed figure crops; panels may be separate inputs. "
            "Checked figures can still have uncertain observations; see verification limitations.",
            "",
        ]
    if figure_context_coverage is not None:
        lines += [
            "### Figure page-context coverage",
            "",
            "| Total figures | Checked | Failed | Unavailable | Not requested | Unrecorded |",
            "|---:|---:|---:|---:|---:|---:|",
            f"| {figure_context_coverage.get('total', 0)} | {figure_context_coverage.get('checked', 0)} | "
            f"{figure_context_coverage.get('failed', 0)} | {figure_context_coverage.get('unavailable', 0)} | "
            f"{figure_context_coverage.get('not_requested', 0)} | {figure_context_coverage.get('unrecorded', 0)} |",
            "",
            "Page-context confirmation checks selected crop observations. Checked confirmations can remain uncertain. "
            "Printed-size legibility uses the original 96 dpi crop; labels outside that crop have no legibility result.",
            "",
        ]
    if not review.findings:
        lines.append("No additional findings recorded.")
    if table_coverage is not None:
        lines += [
            "### Table visual screening coverage",
            "",
            "| Total tables | Checked | Failed | Unavailable |",
            "|---:|---:|---:|---:|",
            f"| {table_coverage.get('total', 0)} | {table_coverage.get('checked', 0)} | "
            f"{table_coverage.get('failed', 0)} | {table_coverage.get('unavailable', 0)} |",
            "",
            "Parsed-text table checks are recorded separately. Checked visuals can still contain uncertain observations.",
            "",
        ]
    if table_context_coverage is not None:
        lines += [
            "### Table page-context coverage",
            "",
            "| Total tables | Checked | Failed | Unavailable | Not requested | Unrecorded |",
            "|---:|---:|---:|---:|---:|---:|",
            f"| {table_context_coverage.get('total', 0)} | {table_context_coverage.get('checked', 0)} | "
            f"{table_context_coverage.get('failed', 0)} | {table_context_coverage.get('unavailable', 0)} | "
            f"{table_context_coverage.get('not_requested', 0)} | {table_context_coverage.get('unrecorded', 0)} |",
            "",
            "Page-context confirmation checks selected table observations and caption associations. "
            "Checked confirmations can remain uncertain. Captions recovered outside the original crop "
            "have no new printed-size legibility judgment; the original 96 dpi crop remains that check's input.",
            "",
        ]
    for finding in review.findings:
        lines += [
            f"### {finding.kind} — {_text(finding.level)}",
            "",
            f"Location: {_location(finding.loc)}.",
            "",
            _text(finding.text),
            "",
        ]
        for item in finding.evidence:
            lines.extend(
                _evidence(
                    item,
                    next(occurrences),
                    reference_context=(
                        finding.kind == "reference"
                        and finding.level == "metadata_candidate"
                        and not item.affects_claim
                        and not item.sufficient
                        and not item.covered
                    ),
                )
            )
        lines.extend(_reference_correction(finding.reference_correction))
        lines.append("")
    limitations = verification_limitations(
        issues=issues,
        figure_coverage=figure_coverage,
        figure_context_coverage=figure_context_coverage,
        table_coverage=table_coverage,
        table_context_coverage=table_context_coverage,
        writing_coverage=writing_coverage,
        anonymity_policy=anonymity_policy,
        token_usage=token_usage,
    )
    if limitations:
        lines += ["### Verification limitations", "", *[f"- {_text(issue)}" for issue in limitations], ""]
    lines += ["", "## 4. Execution ledger", ""]
    if not review.ledger:
        if "execution" in review.incomplete_stages:
            lines.append(
                "Execution failed and no complete execution record was recovered. Runs may have started; their outcomes and cleanup state are unknown. No new execution evidence was adopted."
            )
        else:
            lines.append("Execution was not run; no new execution evidence was produced.")
    for index, entry in enumerate(review.ledger, 1):
        lines += [
            f"### Run {index}",
            "",
            "```json",
            json.dumps(entry, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def write_review(
    review: FinalReview,
    output_dir: Path,
    *,
    issues=None,
    render_pdf=True,
    token_usage=None,
    figure_coverage=None,
    figure_context_coverage=None,
    table_coverage=None,
    table_context_coverage=None,
    writing_coverage=None,
    anonymity_policy=None,
) -> dict:
    result = _checked_report(review)
    validate_publication_language(
        [result.model_dump(), issues or [], (token_usage or {}).get("warnings", [])]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.claims = ordered_claims(result)
    result.review_markdown = render_markdown(
        result,
        issues=issues,
        figure_coverage=figure_coverage,
        figure_context_coverage=figure_context_coverage,
        table_coverage=table_coverage,
        table_context_coverage=table_context_coverage,
        writing_coverage=writing_coverage,
        anonymity_policy=anonymity_policy,
        token_usage=token_usage,
    )
    markdown = output_dir / "final_review.md"
    artifact = output_dir / "final_review.json"
    markdown.write_text(result.review_markdown, encoding="utf-8")
    artifact.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    outputs = {"markdown": str(markdown), "json": str(artifact)}
    if render_pdf:
        from review.report.pdf_renderer import build_review_report_pdf

        advice_models = sorted(
            {
                f"{claim.advice.provider}/{claim.advice.model}"
                for claim in result.claims
                if claim.advice is not None and claim.advice.state == "generated"
            }
        )
        try:
            content = build_review_report_pdf(
                workspace_title=f"FactReview {review.paper_key}",
                source_pdf_name=review.paper_key,
                run_id=review.run_id,
                status=(
                    f"{result.run_status}; {execution_summary(result)['recorded_attempts']} recorded execution attempts"
                    + ("; execution records incomplete" if "execution" in result.incomplete_stages else "")
                )
                if result.execution_requested is not None
                else result.run_status,
                decision=None,
                estimated_cost=0,
                actual_cost=None,
                exported_at=datetime.now(UTC),
                meta_review={},
                reviewers=[],
                raw_output=None,
                final_report_markdown=result.review_markdown,
                agent_model=("v2 renderer; claim advice: " + ", ".join(advice_models))
                if advice_models
                else "deterministic v2 report",
                # Evidence paths and identifiers must survive PDF rendering literally.
                implicit_math=False,
                token_usage=token_usage if token_usage is not None else {"unavailable": True},
            )
            pdf = output_dir / "final_review.pdf"
            pdf.write_bytes(content)
            outputs["pdf"] = str(pdf)
        except Exception as exc:
            outputs["pdf_error"] = f"{type(exc).__name__}: {exc}"
    return outputs
