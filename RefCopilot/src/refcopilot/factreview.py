"""FactReview integration: produces the JSON / Markdown shapes that
FactReview's refcheck stage consumes. See :func:`format_factreview_markdown`
for the embedded-summary policy (errors only by default).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from refcopilot.bibtex_suggest import suggest_bibtex
from refcopilot.models import Severity, Verdict
from refcopilot.pipeline import RefCopilotPipeline
from refcopilot.report import to_factreview_dict

logger = logging.getLogger(__name__)


def check_references(
    paper: str,
    *,
    api_key: str | None = None,
    output_file: str | None = None,
    debug: bool = False,
    enable_parallel: bool = True,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Run RefCopilot on *paper* and return a structured result dict.

    The dict's top-level keys are ``ok``, ``total_refs``, ``errors``,
    ``warnings``, ``unverified``, ``error_message``, ``issues``,
    ``error_details``, ``warning_details``, ``unverified_details``, and
    ``report_file`` — the schema written to ``reference_check.json``.
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    try:
        pipeline = RefCopilotPipeline(
            s2_api_key=api_key,
            use_llm_verify=True,
            max_workers=max_workers if enable_parallel else 1,
        )
        report = pipeline.run(paper)
    except Exception as exc:
        logger.exception("RefCopilot run failed")
        return _failure_payload(str(exc), output_file)

    payload = to_factreview_dict(report, report_file=str(output_file or ""))

    if output_file:
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            _write_text_report(report, Path(output_file))
        except Exception as exc:
            logger.warning("could not write %s: %s", output_file, exc)

    return payload


def format_factreview_markdown(
    result: dict[str, Any],
    *,
    max_issues: int = 20,
    include_warnings: bool = False,
    include_unverified: bool = False,
) -> str:
    """Render *result* as Markdown for FactReview's review report.

    By default only error rows (fabricated / hallucinated references) appear
    so the embedded section in the final review stays compact. Set
    ``include_warnings=True`` and/or ``include_unverified=True`` to widen the
    listing. The standalone CLI uses :func:`refcopilot.report.to_markdown`
    instead, which always emits all severities.
    """
    if not isinstance(result, dict):
        return ""

    lines: list[str] = ["## Reference Check (RefCopilot)\n"]
    if not result.get("ok"):
        msg = result.get("error_message") or "Reference check did not complete."
        lines.append("RefCopilot was enabled but the run did not complete successfully.")
        lines.append(f"- Error: {msg}")
        return "\n".join(lines).rstrip() + "\n"

    total_refs = int(result.get("total_refs") or 0)
    errors = int(result.get("errors") or 0)
    warnings = int(result.get("warnings") or 0)
    unverified = int(result.get("unverified") or 0)
    lines.append(
        f"- References processed: `{total_refs}`; errors: `{errors}`; "
        f"warnings: `{warnings}`; unverified: `{unverified}`."
    )
    report_file = str(result.get("report_file") or "")
    if report_file:
        lines.append(f"- Detail file: `{report_file}`")
    lines.append("")

    issues = result.get("issues") or []
    sections: list[tuple[str, list[dict[str, Any]]]] = [
        ("Errors", [i for i in issues if i.get("severity") == "error"]),
    ]
    if include_warnings:
        sections.append(("Warnings", [i for i in issues if i.get("severity") == "warning"]))
    if include_unverified:
        sections.append(("Unverified", [i for i in issues if i.get("severity") == "unverified"]))

    rendered_any = False
    for heading, rows in sections:
        if not rows:
            continue
        rendered_any = True
        lines.append(f"### {heading}")
        for index, issue in enumerate(rows[:max_issues], start=1):
            lines.append(_format_factreview_issue(issue, index))
        if len(rows) > max_issues:
            lines.append(f"- {len(rows) - max_issues} additional item(s) omitted.")
        lines.append("")

    if not rendered_any:
        lines.append("No fabricated references detected.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _format_factreview_issue(issue: dict[str, Any], index: int) -> str:
    title = (issue.get("reference_title") or "(untitled)").strip()
    type_ = issue.get("type") or "unknown"
    details = issue.get("details") or ""
    head = f"{index}. **{title}** (`{type_}`)"
    if details:
        head += f" — {details}"

    suggested = (issue.get("corrected_bibtex") or "").strip()
    if suggested:
        head += "\n\n   Suggested replacement (data sources annotated as comments):\n"
        head += "\n".join(f"   {line}" for line in ("```bibtex", *suggested.splitlines(), "```"))
    return head


def _failure_payload(error_message: str, output_file: str | None) -> dict[str, Any]:
    return {
        "ok": False,
        "total_refs": 0,
        "errors": 0,
        "warnings": 0,
        "unverified": 0,
        "error_message": error_message,
        "issues": [],
        "error_details": [],
        "warning_details": [],
        "unverified_details": [],
        "report_file": str(output_file or ""),
    }


def _write_text_report(report, path: Path) -> None:
    lines = []
    for c in report.checked:
        if c.verdict == Verdict.VALID:
            continue
        lines.append("=" * 70)
        lines.append(f"Reference: {c.reference.title or c.reference.bibkey or '(untitled)'}")
        lines.append(f"Authors  : {', '.join(c.reference.authors)}")
        lines.append(f"Year     : {c.reference.year or ''}")
        lines.append(f"Verdict  : {c.verdict.value}")
        for issue in c.issues:
            lines.append(
                f"  - [{issue.severity.value}/{issue.category.value}/{issue.code}] {issue.message}"
            )
            if issue.suggestion:
                lines.append(f"      Suggestion: {issue.suggestion}")
        if any(i.severity == Severity.WARNING for i in c.issues) and c.merged is not None:
            bibtex = suggest_bibtex(c.reference, c.merged)
            if bibtex:
                lines.append("")
                lines.append("Suggested replacement:")
                lines.extend(bibtex.splitlines())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
