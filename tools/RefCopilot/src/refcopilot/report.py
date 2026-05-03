"""Report serializers: JSON / Markdown / legacy refchecker-compatible dict."""

from __future__ import annotations

import json
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


def to_json(report: Report) -> str:
    return report.model_dump_json(indent=2)


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
# Legacy schema (compatible with FactReview's existing reference_check.json)
# ---------------------------------------------------------------------------


_CATEGORY_TO_TYPE = {
    IssueCategory.FAKE: "hallucination",
    IssueCategory.OUTDATED: "outdated",
    IssueCategory.INCOMPLETE: "incomplete",
}


def to_legacy_dict(report: Report, *, report_file: str = "") -> dict[str, Any]:
    issues_legacy: list[dict[str, Any]] = []
    for c in report.checked:
        if c.issues:
            for issue in c.issues:
                issues_legacy.append(_issue_to_legacy(c, issue))
        elif c.verdict == Verdict.UNVERIFIED:
            issues_legacy.append(_unverified_to_legacy(c))

    error_details = [i for i in issues_legacy if i["severity"] == "error"]
    warning_details = [i for i in issues_legacy if i["severity"] == "warning"]
    unverified_details = [i for i in issues_legacy if i["severity"] == "unverified"]

    return {
        "ok": True,
        "total_refs": report.summary.total_refs,
        "errors": report.summary.errors,
        "warnings": report.summary.warnings,
        "unverified": report.summary.unverified,
        "error_message": "",
        "issues": issues_legacy,
        "error_details": error_details,
        "warning_details": warning_details,
        "unverified_details": unverified_details,
        "report_file": report_file,
    }


def _issue_to_legacy(c: CheckedReference, issue: Issue) -> dict[str, Any]:
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


def _unverified_to_legacy(c: CheckedReference) -> dict[str, Any]:
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


def write_legacy_json(report: Report, path: str, *, report_file: str = "") -> None:
    payload = to_legacy_dict(report, report_file=report_file)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
