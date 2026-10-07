"""Render final v2 records without extracting claims or changing assessments."""

from __future__ import annotations

import html
import json
import re
from datetime import UTC, datetime
from pathlib import Path

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


def _evidence(item: Evidence) -> list[str]:
    pointer = item.pointer
    source = "paper-internal" if item.source == "paper_internal" else item.source
    location = [pointer.locator]
    location.extend(
        f"{name} {value}"
        for name, value in (("page", pointer.page), ("line", pointer.line), ("key", pointer.key))
        if value is not None
    )
    lines = [
        f"- **{source} / {item.direction}**; sufficient: {str(item.sufficient).lower()}; "
        f"covers: {_text(', '.join(item.covered) or 'no claim conditions')}. "
        f"Pointer: {_text('; '.join(location))}."
    ]
    if pointer.quote:
        lines.append(f"  - Passage: {_text(pointer.quote)}")
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


def render_markdown(review: FinalReview, *, issues: list[str] | None = None) -> str:
    claims = ordered_claims(review)
    lines = [
        f"# FactReview — {_text(review.paper_key)}",
        "",
        "## 1. Overview",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {status.value} | {review.summary_counts[status]} |" for status in STATUS_ORDER)
    lines += ["", "### Items requiring attention", ""]
    concerns = [claim for claim in claims if claim.status in {ClaimStatus.FLAWED, ClaimStatus.QUESTIONED}]
    lines += [
        f"- {claim.id} ({claim.status.value}, {claim.importance}): {_text(claim.text)}" for claim in concerns
    ] or ["No claim is assessed as flawed or questioned."]
    lines += ["", "## 2. Claim list", ""]
    for claim in claims:
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
        lines += ["", f"Evidence needs: {', '.join(claim.needs) or 'none'}.", "", "Evidence:", ""]
        if not claim.evidence:
            lines.append("No evidence is available for assessment.")
        for item in claim.evidence:
            lines.extend(_evidence(item))
        lines += ["", "Questions for authors:", ""]
        lines += [
            f"- {_text(question.text)} Reason: {_text(question.reason)}" for question in claim.questions
        ] or ["None recorded."]
        if claim.notes:
            lines += ["", "Notes:", "", *[f"- {_text(note)}" for note in claim.notes]]
        lines.append("")
    lines += ["## 3. Other findings", ""]
    if not review.findings:
        lines.append("No additional findings recorded.")
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
            lines.extend(_evidence(item))
        lines.append("")
    if issues:
        lines += ["### Verification limitations", "", *[f"- {_text(issue)}" for issue in issues], ""]
    lines += ["", "## 4. Execution ledger", ""]
    if not review.ledger:
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
    review: FinalReview, output_dir: Path, *, issues=None, render_pdf=True, token_usage=None
) -> dict:
    validate_publication_language([review.model_dump(), issues or []])
    output_dir.mkdir(parents=True, exist_ok=True)
    result = review.model_copy(deep=True)
    result.claims = ordered_claims(result)
    result.review_markdown = render_markdown(result, issues=issues)
    markdown = output_dir / "final_review.md"
    artifact = output_dir / "final_review.json"
    markdown.write_text(result.review_markdown, encoding="utf-8")
    artifact.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    outputs = {"markdown": str(markdown), "json": str(artifact)}
    if render_pdf:
        from review.report.pdf_renderer import build_review_report_pdf

        try:
            content = build_review_report_pdf(
                workspace_title=f"FactReview {review.paper_key}",
                source_pdf_name=review.paper_key,
                run_id=review.run_id,
                status="completed",
                decision=None,
                estimated_cost=0,
                actual_cost=None,
                exported_at=datetime.now(UTC),
                meta_review={},
                reviewers=[],
                raw_output=None,
                final_report_markdown=result.review_markdown,
                agent_model="deterministic v2 report",
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
