from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from llm.client import LLMConfig, llm_json, resolve_llm_config

from .final_report import analyze_report_language, validate_final_report

_HIGH_SEVERITY_LEVELS = {"critical", "high", "major"}


@dataclass
class FinalReportAuditIssue:
    problem_type: str
    severity: str
    section: str
    review_excerpt: str
    paper_evidence: str
    suggested_fix: str
    should_fix: bool = True


@dataclass
class FinalReportAuditIteration:
    iteration: int
    audit_summary: str
    issues: list[FinalReportAuditIssue] = field(default_factory=list)
    issue_count: int = 0
    high_severity_issue_count: int = 0
    revised: bool = False
    accepted: bool = False
    validation_ok: bool = False
    validation_reason: str | None = None
    validation_message: str | None = None
    compatibility_ok: bool = False
    compatibility_message: str | None = None
    error: str | None = None


@dataclass
class FinalReportAuditResult:
    enabled: bool
    applied: bool
    iterations_run: int
    max_iterations: int
    stop_reason: str
    final_markdown: str
    iterations: list[FinalReportAuditIteration] = field(default_factory=list)
    llm_provider: str | None = None
    llm_model: str | None = None
    source_markdown_chars: int = 0
    source_markdown_chars_sent: int = 0
    report_markdown_chars_sent: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _truncate_for_model(text: str, limit: int) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= limit:
        return normalized
    if limit <= 2000:
        return normalized[:limit]
    head = int(limit * 0.65)
    tail = max(0, limit - head - len("\n\n[... truncated ...]\n\n"))
    return normalized[:head].rstrip() + "\n\n[... truncated ...]\n\n" + normalized[-tail:].lstrip()


def _coerce_issue_list(raw_issues: Any) -> list[FinalReportAuditIssue]:
    issues: list[FinalReportAuditIssue] = []
    if not isinstance(raw_issues, list):
        return issues
    for row in raw_issues:
        if not isinstance(row, dict):
            continue
        problem_type = str(row.get("problem_type") or "").strip() or "unknown"
        severity = str(row.get("severity") or "").strip().lower() or "medium"
        section = str(row.get("section") or "").strip()
        review_excerpt = str(row.get("review_excerpt") or "").strip()
        paper_evidence = str(row.get("paper_evidence") or "").strip()
        suggested_fix = str(row.get("suggested_fix") or "").strip()
        should_fix = bool(row.get("should_fix", True))
        issues.append(
            FinalReportAuditIssue(
                problem_type=problem_type,
                severity=severity,
                section=section,
                review_excerpt=review_excerpt,
                paper_evidence=paper_evidence,
                suggested_fix=suggested_fix,
                should_fix=should_fix,
            )
        )
    return issues


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _extract_heading_signature(markdown: str) -> list[str]:
    signature: list[str] = []
    for line in str(markdown or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            signature.append(_normalize_space(stripped))
    return signature


def _collect_table_shapes(markdown: str) -> list[tuple[str, int]]:
    lines = str(markdown or "").splitlines()
    tables: list[tuple[str, int]] = []
    index = 0
    while index + 1 < len(lines):
        header = lines[index].strip()
        separator = lines[index + 1].strip()
        if not (header.startswith("|") and header.endswith("|") and re.fullmatch(r"\|[ :\-|]+\|", separator)):
            index += 1
            continue
        row_count = 0
        cursor = index + 2
        while cursor < len(lines):
            row = lines[cursor].strip()
            if not (row.startswith("|") and row.endswith("|")):
                break
            row_count += 1
            cursor += 1
        tables.append((_normalize_space(header), row_count))
        index = max(cursor, index + 2)
    return tables


def _extract_subheading_signature(markdown: str) -> list[str]:
    signature: list[str] = []
    for line in str(markdown or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("###"):
            signature.append(_normalize_space(stripped))
    return signature


def _check_format_compatibility(original_markdown: str, candidate_markdown: str) -> tuple[bool, str]:
    original_headings = _extract_heading_signature(original_markdown)
    candidate_headings = _extract_heading_signature(candidate_markdown)
    if original_headings != candidate_headings:
        return False, "heading signature changed"

    original_subheadings = _extract_subheading_signature(original_markdown)
    candidate_subheadings = _extract_subheading_signature(candidate_markdown)
    if original_subheadings != candidate_subheadings:
        return False, "subheading signature changed"

    original_tables = _collect_table_shapes(original_markdown)
    candidate_tables = _collect_table_shapes(candidate_markdown)
    if len(original_tables) != len(candidate_tables):
        return False, "table count changed"
    for index, (original_table, candidate_table) in enumerate(
        zip(original_tables, candidate_tables, strict=False), start=1
    ):
        original_header, original_row_count = original_table
        candidate_header, candidate_row_count = candidate_table
        if original_header != candidate_header:
            return False, f"table {index} header changed"
        if original_row_count != candidate_row_count:
            return False, f"table {index} row count changed"

    return True, "ok"


def _build_audit_system_prompt() -> str:
    return (
        "You are a strict paper-review fact auditor.\n"
        "Your job is to compare a generated final review against the source paper markdown and find only "
        "evidence-grounded mismatches.\n"
        "Return valid JSON only.\n"
        "Rules:\n"
        "- Use the paper markdown as the sole source of truth.\n"
        "- Flag only concrete problems: factual error, overclaim, understatement, missing condition, "
        "causal overreach, wrong number/dataset/setting/result/conclusion.\n"
        "- Do not invent paper evidence.\n"
        "- If evidence is insufficient, do not claim an error.\n"
        "- Output schema:\n"
        "{"
        '"audit_summary": str, '
        '"issues": [{"problem_type": str, "severity": str, "section": str, "review_excerpt": str, '
        '"paper_evidence": str, "suggested_fix": str, "should_fix": bool}]'
        "}\n"
        "- Severity should be one of critical/high/medium/low.\n"
        "- If no material issue exists, return issues as []."
    )


def _build_audit_user_prompt(
    *, iteration: int, max_iterations: int, paper_markdown: str, review_markdown: str
) -> str:
    return (
        f"Audit iteration {iteration} of {max_iterations}.\n\n"
        "Compare the current final review against the source paper and list only evidence-grounded mismatches.\n\n"
        "Source paper markdown:\n"
        "```markdown\n"
        f"{paper_markdown}\n"
        "```\n\n"
        "Current final review markdown:\n"
        "```markdown\n"
        f"{review_markdown}\n"
        "```"
    )


def _build_revision_system_prompt(*, output_language: str) -> str:
    return (
        "You are revising a paper review under a strict fixed markdown template.\n"
        "Return valid JSON only.\n"
        "Rules:\n"
        "- Revise only the content that is directly implicated by the provided audit issues.\n"
        "- Keep the exact same headings, section numbering, section order, subheadings, table headers, and row counts.\n"
        "- Do not add or remove sections, subheadings, table rows, or table columns.\n"
        "- Do not rewrite unrelated text.\n"
        f"- Keep the revised markdown in {output_language}.\n"
        "- Output schema:\n"
        '{"revision_summary": str, "revised_markdown": str}'
    )


def _build_revision_user_prompt(
    *,
    review_markdown: str,
    issues: list[FinalReportAuditIssue],
    paper_markdown: str,
) -> str:
    issue_lines = []
    for index, issue in enumerate(issues, start=1):
        issue_lines.append(
            f"{index}. section={issue.section or 'unknown'} | type={issue.problem_type} | severity={issue.severity}\n"
            f"review_excerpt: {issue.review_excerpt or 'N/A'}\n"
            f"paper_evidence: {issue.paper_evidence or 'N/A'}\n"
            f"suggested_fix: {issue.suggested_fix or 'N/A'}"
        )
    joined_issues = "\n\n".join(issue_lines) if issue_lines else "No issues."
    return (
        "Revise the current final review by fixing only the audited issues below.\n\n"
        "Audit issues:\n"
        f"{joined_issues}\n\n"
        "Source paper markdown:\n"
        "```markdown\n"
        f"{paper_markdown}\n"
        "```\n\n"
        "Current final review markdown:\n"
        "```markdown\n"
        f"{review_markdown}\n"
        "```"
    )


def audit_and_refine_final_report(
    *,
    final_markdown: str,
    source_markdown: str,
    max_iterations: int,
    max_source_chars: int,
    max_review_chars: int,
    model: str,
    min_english_words: int,
    min_chinese_chars: int,
    force_english_output: bool,
) -> FinalReportAuditResult:
    current_markdown = str(final_markdown or "").strip()
    source_text = str(source_markdown or "").strip()
    if not current_markdown or not source_text:
        return FinalReportAuditResult(
            enabled=False,
            applied=False,
            iterations_run=0,
            max_iterations=max(0, int(max_iterations or 0)),
            stop_reason="missing_input",
            final_markdown=current_markdown,
            source_markdown_chars=len(source_text),
            source_markdown_chars_sent=0,
            report_markdown_chars_sent=0,
        )

    sent_source = _truncate_for_model(source_text, max(8000, int(max_source_chars or 0)))
    sent_report = _truncate_for_model(current_markdown, max(4000, int(max_review_chars or 0)))
    language = analyze_report_language(current_markdown).primary_language
    output_language = "Simplified Chinese" if language == "zh-CN" else "English"

    base_cfg = resolve_llm_config(model=model)
    cfg = LLMConfig(
        provider=base_cfg.provider,
        model=base_cfg.model,
        base_url=base_cfg.base_url,
        api_key=base_cfg.api_key,
        temperature=0.0,
        max_tokens=max(2500, min(8000, int(max_review_chars or 0) // 4 + 1500)),
    )

    result = FinalReportAuditResult(
        enabled=True,
        applied=False,
        iterations_run=0,
        max_iterations=max(0, int(max_iterations or 0)),
        stop_reason="max_iterations_reached",
        final_markdown=current_markdown,
        llm_provider=cfg.provider,
        llm_model=cfg.model,
        source_markdown_chars=len(source_text),
        source_markdown_chars_sent=len(sent_source),
        report_markdown_chars_sent=len(sent_report),
    )

    if result.max_iterations <= 0:
        result.stop_reason = "disabled_by_config"
        return result

    for iteration in range(1, result.max_iterations + 1):
        prompt = _build_audit_user_prompt(
            iteration=iteration,
            max_iterations=result.max_iterations,
            paper_markdown=sent_source,
            review_markdown=_truncate_for_model(result.final_markdown, max_review_chars),
        )
        payload = llm_json(prompt=prompt, system=_build_audit_system_prompt(), cfg=cfg)
        audit_iteration = FinalReportAuditIteration(
            iteration=iteration,
            audit_summary=str(payload.get("audit_summary") or "").strip(),
        )
        result.iterations_run = iteration
        result.iterations.append(audit_iteration)

        if str(payload.get("status") or "").strip().lower() == "error":
            audit_iteration.error = str(payload.get("error") or "unknown llm error").strip()
            result.stop_reason = "llm_error"
            break

        issues = _coerce_issue_list(payload.get("issues"))
        audit_iteration.issues = issues
        audit_iteration.issue_count = len(issues)
        audit_iteration.high_severity_issue_count = sum(
            1 for item in issues if item.severity.lower() in _HIGH_SEVERITY_LEVELS and item.should_fix
        )

        if audit_iteration.issue_count <= 0:
            audit_iteration.compatibility_ok = True
            audit_iteration.validation_ok = True
            audit_iteration.validation_message = "No revision needed."
            result.stop_reason = "no_issues_found"
            break

        issues_to_fix = [item for item in issues if item.should_fix]
        if not issues_to_fix:
            audit_iteration.compatibility_ok = True
            audit_iteration.validation_ok = True
            audit_iteration.validation_message = "No actionable issue."
            result.stop_reason = "no_actionable_issues"
            break

        revision_payload = llm_json(
            prompt=_build_revision_user_prompt(
                review_markdown=_truncate_for_model(result.final_markdown, max_review_chars),
                issues=issues_to_fix,
                paper_markdown=sent_source,
            ),
            system=_build_revision_system_prompt(output_language=output_language),
            cfg=cfg,
        )
        if str(revision_payload.get("status") or "").strip().lower() == "error":
            audit_iteration.error = str(revision_payload.get("error") or "unknown llm error").strip()
            result.stop_reason = "llm_error"
            break

        revised_markdown = str(revision_payload.get("revised_markdown") or "").strip()
        if not revised_markdown:
            audit_iteration.compatibility_ok = False
            audit_iteration.compatibility_message = "empty revised_markdown"
            result.stop_reason = "no_further_revision"
            break

        audit_iteration.revised = revised_markdown != result.final_markdown
        compatibility_ok, compatibility_message = _check_format_compatibility(
            result.final_markdown,
            revised_markdown,
        )
        audit_iteration.compatibility_ok = compatibility_ok
        audit_iteration.compatibility_message = compatibility_message
        if not compatibility_ok:
            result.stop_reason = "revision_changed_fixed_format"
            break

        validation = validate_final_report(
            markdown=revised_markdown,
            min_english_words=min_english_words,
            min_chinese_chars=min_chinese_chars,
            force_english_output=force_english_output,
        )
        audit_iteration.validation_ok = bool(validation.ok)
        audit_iteration.validation_reason = validation.reason
        audit_iteration.validation_message = validation.message
        if not validation.ok:
            result.stop_reason = "revision_failed_validation"
            break

        if revised_markdown != result.final_markdown:
            result.applied = True
            result.final_markdown = revised_markdown
            audit_iteration.accepted = True
        else:
            result.stop_reason = "no_further_revision"
            break

    return result
