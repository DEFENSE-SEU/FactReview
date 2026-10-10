"""A compact, deterministic visual summary of v2 claim records."""

import json
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape

from review.report.v2 import STATUS_ORDER, execution_summary, ordered_claims, validate_publication_language
from schemas.review import FinalReview

COLORS = {"flawed": "#b42318", "questioned": "#b54708", "unverified": "#475467", "supported": "#067647"}


def teaser_payload(review: FinalReview) -> dict:
    """The same retained records are available even when visual export fails."""
    claims = ordered_claims(review)
    return {
        "paper_key": review.paper_key,
        "run_status": review.run_status,
        "incomplete_stages": review.incomplete_stages,
        "delivery_checks": [check.model_dump(mode="json") for check in review.delivery_checks],
        "execution": execution_summary(review),
        "counts": {status.value: review.summary_counts[status] for status in STATUS_ORDER},
        "claims": [
            {
                "id": claim.id,
                "text": claim.text,
                "status": claim.status.value,
                "source_types": sorted({item.source for item in claim.evidence}),
            }
            for claim in claims
        ],
    }


def write_teaser(review: FinalReview, output_dir: Path) -> dict[str, str]:
    validate_publication_language(review.model_dump())
    output_dir.mkdir(parents=True, exist_ok=True)
    claims = ordered_claims(review)
    payload = teaser_payload(review)
    data = output_dir / "teaser.json"
    data.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    prompt = output_dir / "teaser_prompt.md"
    prompt.write_text(
        "Create a claim-level review summary from these records. Preserve all four statuses and source types. "
        "Use counts exactly as supplied. Include no publication recommendation.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    elements = [
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="32" y="42" font-size="24" font-weight="bold">FactReview · Evidence summary</text>',
    ]
    y = 74
    for line in textwrap.wrap(review.paper_key, 82):
        elements.append(f'<text x="32" y="{y}" font-size="17">{escape(line)}</text>')
        y += 22
    y += 26
    execution = payload["execution"]
    elements.append(
        f'<text x="32" y="{y}" font-size="14">Recorded execution attempts: {execution["recorded_attempts"]}; claims with aligned execution evidence: {execution["claims_with_aligned_execution_evidence"]}.</text>'
    )
    y += 28
    if review.run_status == "partial":
        elements.append(
            f'<text x="32" y="{y}" font-size="16" fill="#b42318">Partial review: {escape(", ".join(review.incomplete_stages))}. Counts cover retained claims only.</text>'
        )
        y += 30
    for index, status in enumerate(STATUS_ORDER):
        x = 32 + index * 218
        elements += [
            f'<rect x="{x}" y="{y}" width="204" height="62" rx="8" fill="{COLORS[status]}"/>',
            f'<text x="{x + 12}" y="{y + 26}" fill="white" font-size="17">{status.value}</text>',
            f'<text x="{x + 12}" y="{y + 49}" fill="white" font-size="20">{review.summary_counts[status]}</text>',
        ]
    y += 94
    for claim in claims:
        elements.append(
            f'<text x="32" y="{y}" font-size="16" font-weight="bold" fill="{COLORS[claim.status]}">{escape(claim.id)} · {claim.status.value}</text>'
        )
        y += 24
        for line in textwrap.wrap(claim.text, 100):
            elements.append(f'<text x="32" y="{y}" font-size="15">{escape(line)}</text>')
            y += 21
        sources = ", ".join(sorted({item.source for item in claim.evidence})) or "no evidence"
        elements.append(
            f'<text x="32" y="{y}" font-size="13" fill="#475467">Sources: {escape(sources)}</text>'
        )
        y += 40
    elements.append(
        f'<text x="32" y="{y}" font-size="13">See the report for conditions, evidence pointers, questions and execution details.</text>'
    )
    svg = output_dir / "teaser.svg"
    svg.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="920" height="{y + 34}" viewBox="0 0 920 {y + 34}" font-family="Arial,sans-serif">'
        + "\n".join(elements)
        + "</svg>",
        encoding="utf-8",
    )
    return {"json": str(data), "prompt": str(prompt), "image": str(svg)}
