"""Report serializers: Markdown for end users, dict for FactReview's refcheck stage."""

from __future__ import annotations

from typing import Any

from refcopilot.models import (
    CheckedReference,
    Issue,
    IssueCategory,
    Report,
    Severity,
    Verdict,
)

_MAX_TEXT = 4000


def to_markdown(report: Report, *, max_issues: int = 50) -> str:
    s = report.summary
    paper = report.paper or {}
    lines: list[str] = ["## Reference Check (RefCopilot)\n"]
    lines.append(
        f"Input: `{paper.get('input', '?')}` (kind: `{paper.get('kind', '?')}`)\n"
    )
    lines.append(
        f"- References processed: `{s.total_refs}`; "
        f"errors: `{s.errors}`; warnings: `{s.warnings}`; unverified: `{s.unverified}`."
    )
    if s.by_category:
        cats = ", ".join(f"`{k}`: {v}" for k, v in sorted(s.by_category.items()))
        lines.append(f"- By category: {cats}")
    lines.append("")

    grouped = {
        "Errors": [c for c in report.checked if c.verdict == Verdict.ERROR],
        "Warnings": [c for c in report.checked if c.verdict == Verdict.WARNING],
        "Unverified": [c for c in report.checked if c.verdict == Verdict.UNVERIFIED],
    }
    rendered = False
    for heading, rows in grouped.items():
        if not rows:
            continue
        rendered = True
        lines.append(f"### {heading}")
        for index, c in enumerate(rows[:max_issues], start=1):
            lines.append(_format_checked_reference(c, index))
        if len(rows) > max_issues:
            lines.append(f"- {len(rows) - max_issues} additional item(s) omitted.")
        lines.append("")

    if not rendered:
        lines.append("All references verified successfully.")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _format_checked_reference(c: CheckedReference, index: int) -> str:
    title = (c.reference.title or c.reference.bibkey or "(untitled)").strip()
    authors = ", ".join(c.reference.authors[:3])
    if len(c.reference.authors) > 3:
        authors += ", ..."
    head = f"{index}. **{title}**" + (f" — {authors}" if authors else "")
    body = []
    for issue in c.issues:
        body.append(
            f"   - [{issue.severity.value}/{issue.category.value}/{issue.code}] {issue.message}"
            + (f"  _Suggestion: {issue.suggestion}_" if issue.suggestion else "")
        )
    return head + "\n" + "\n".join(body) if body else head


# ---------------------------------------------------------------------------
# FactReview JSON shape (written to ``reference_check.json`` by the refcheck stage)
# ---------------------------------------------------------------------------


_CATEGORY_TO_TYPE = {
    IssueCategory.FAKE: "hallucination",
    IssueCategory.OUTDATED: "outdated",
    IssueCategory.INCOMPLETE: "incomplete",
}


def to_factreview_dict(report: Report, *, report_file: str = "") -> dict[str, Any]:
    """Serialize *report* into the dict shape FactReview's refcheck stage stores."""
    issues: list[dict[str, Any]] = []
    for c in report.checked:
        if c.issues:
            for issue in c.issues:
                issues.append(_issue_to_factreview(c, issue))
        elif c.verdict == Verdict.UNVERIFIED:
            issues.append(_unverified_to_factreview(c))

    error_details = [i for i in issues if i["severity"] == "error"]
    warning_details = [i for i in issues if i["severity"] == "warning"]
    unverified_details = [i for i in issues if i["severity"] == "unverified"]

    return {
        "ok": True,
        "total_refs": report.summary.total_refs,
        "errors": report.summary.errors,
        "warnings": report.summary.warnings,
        "unverified": report.summary.unverified,
        "error_message": "",
        "issues": issues,
        "error_details": error_details,
        "warning_details": warning_details,
        "unverified_details": unverified_details,
        "report_file": report_file,
    }


def _issue_to_factreview(c: CheckedReference, issue: Issue) -> dict[str, Any]:
    severity = issue.severity.value
    if severity == Severity.INFO.value:
        severity = "warning"

    cited_url = c.reference.url or ""
    verified_url = ""
    if c.merged and c.merged.url:
        verified_url = c.merged.url

    type_label = (
        f"{_CATEGORY_TO_TYPE.get(issue.category, issue.category.value)}::{issue.code}"
    )

    return {
        "severity": severity,
        "type": type_label,
        "reference_title": _truncate(c.reference.title, 500),
        "reference_year": str(c.reference.year or ""),
        "cited_url": _truncate(cited_url, 1000),
        "verified_url": _truncate(verified_url, 1000),
        "details": _truncate(issue.message + (f" ({issue.suggestion})" if issue.suggestion else ""), _MAX_TEXT),
        "raw_reference": _truncate(c.reference.raw, _MAX_TEXT),
        "corrected_plaintext": "",
        "corrected_bibtex": "",
        "corrected_bibitem": "",
    }


def _unverified_to_factreview(c: CheckedReference) -> dict[str, Any]:
    return {
        "severity": "unverified",
        "type": "unverified::no_match",
        "reference_title": _truncate(c.reference.title, 500),
        "reference_year": str(c.reference.year or ""),
        "cited_url": _truncate(c.reference.url or "", 1000),
        "verified_url": "",
        "details": "Could not verify reference (no records found on arXiv or Semantic Scholar).",
        "raw_reference": _truncate(c.reference.raw, _MAX_TEXT),
        "corrected_plaintext": "",
        "corrected_bibtex": "",
        "corrected_bibitem": "",
    }


def _truncate(value: Any, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 16)].rstrip() + "\n...(truncated)"
