from __future__ import annotations

import re
from dataclasses import dataclass

_ENGLISH_WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?")
_CHINESE_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_MARKDOWN_CODE_FENCE_PATTERN = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_PATTERN = re.compile(r"`[^`\n]+`")
_MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

_REQUIRED_SECTION_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("1. Metadata", ("1. metadata", "metadata")),
    ("2. Technical Positioning", ("2. technical positioning", "technical positioning")),
    ("3. Claims", ("3. claims", "claims")),
    ("4. Summary", ("4. summary", "summary")),
    ("5. Experiment", ("5. experiment", "experiment", "experiments")),
]


@dataclass
class LanguageStats:
    primary_language: str
    english_words: int
    chinese_chars: int
    english_ratio: float
    chinese_ratio: float


@dataclass
class FinalReportValidation:
    ok: bool
    reason: str | None
    message: str
    language_stats: LanguageStats
    missing_sections: list[str]


def _extract_section_body(markdown_text: str, section_no: int, title: str) -> str:
    pattern = re.compile(
        rf"(?ims)^##\s+(?:\*\*)?{section_no}\.\s+{re.escape(title)}(?:\*\*)?\s*$\n"
        r"(?P<body>.*?)(?=^##\s+|\Z)"
    )
    match = pattern.search(str(markdown_text or ""))
    return str(match.group("body") or "") if match else ""


def _parse_markdown_tables(markdown_text: str) -> list[tuple[list[str], list[list[str]]]]:
    lines = str(markdown_text or "").splitlines()
    tables: list[tuple[list[str], list[list[str]]]] = []
    i = 0
    while i + 1 < len(lines):
        header = lines[i].strip()
        sep = lines[i + 1].strip()
        if not (header.startswith("|") and header.endswith("|") and re.fullmatch(r"\|[ :\-|]+\|", sep)):
            i += 1
            continue
        headers = [_normalize_cell(c) for c in header.strip("|").split("|")]
        rows: list[list[str]] = []
        j = i + 2
        while j < len(lines):
            raw = lines[j].strip()
            if not (raw.startswith("|") and raw.endswith("|")):
                break
            rows.append([_normalize_cell(c) for c in raw.strip("|").split("|")])
            j += 1
        tables.append((headers, rows))
        i = max(j, i + 2)
    return tables


def _normalize_cell(text: str) -> str:
    s = re.sub(r"<[^>]+>", "", str(text or ""))
    s = s.replace("**", "").replace("__", "").replace("`", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _validate_technical_positioning(markdown: str) -> str | None:
    body = _extract_section_body(markdown, 2, "Technical Positioning")
    if not body.strip():
        return "Technical Positioning section is empty."
    tables = _parse_markdown_tables(body)
    if not tables:
        return "Technical Positioning must contain one niche-positioning matrix table."
    headers, rows = tables[0]
    if len(headers) < 3:
        return "Technical Positioning table must include Research domain, Method, and at least one niche dimension."
    if headers[0].lower() != "research domain" or headers[1].lower() != "method":
        return "Technical Positioning table first two columns must be exactly Research domain | Method."
    if len(headers) > 10:
        return "Technical Positioning table must use 3-8 niche-dimension columns after Research domain and Method."
    if not rows:
        return "Technical Positioning table must include method rows."
    for row in rows:
        cells = row + [""] * max(0, len(headers) - len(row))
        for value in cells[2 : len(headers)]:
            if value not in {"√", "×", "✓", "✗"}:
                return "Technical Positioning niche-dimension cells must contain only √ or ×."
    return None


def _validate_claims(markdown: str) -> str | None:
    body = _extract_section_body(markdown, 3, "Claims")
    if not body.strip():
        return "Claims section is empty."
    tables = _parse_markdown_tables(body)
    if not tables:
        return "Claims section must contain a claim/evidence/assessment/location table."
    headers, rows = tables[0]
    lowered = [h.lower() for h in headers]
    for required in ("claim", "evidence", "assessment", "location"):
        if not any(required in h for h in lowered):
            return f"Claims table missing required column: {required}."
    if any("status" in h for h in lowered):
        return (
            "Claims table must not include Status before system assessment; status is appended automatically."
        )
    if len(rows) != 3:
        return "Claims table must contain exactly 3 core claims."
    evidence_idx = next((i for i, h in enumerate(lowered) if "evidence" in h), -1)
    assessment_idx = next((i for i, h in enumerate(lowered) if "assessment" in h), -1)
    location_idx = next((i for i, h in enumerate(lowered) if "location" in h), -1)
    for row in rows:
        cells = row + [""] * max(0, len(headers) - len(row))
        if evidence_idx >= 0 and cells[evidence_idx].lower() in {"", "not found", "unknown", "n/a"}:
            return "Each claim must include manuscript evidence or Not found in manuscript."
        if assessment_idx >= 0 and not cells[assessment_idx].strip():
            return "Each claim must include an initial assessment."
        if location_idx >= 0 and not cells[location_idx].strip():
            return "Each claim must include a manuscript location."
    return None


def _validate_experiment(markdown: str) -> str | None:
    body = _extract_section_body(markdown, 5, "Experiment")
    if not body.strip():
        return "Experiment section is empty."
    if not re.search(r"(?im)^###?\s+(?:\*\*)?Main Result(?:\*\*)?\s*$", body):
        return "Experiment section must include Main Result."
    if not re.search(r"(?im)^###?\s+(?:\*\*)?Ablation Result(?:\*\*)?\s*$", body):
        return "Experiment section must include Ablation Result."
    if re.search(r"(?i)\b(sampled|representative|selected rows|truncated|partial list)\b", body):
        return "Experiment section must not sample or truncate reported experiment results."
    tables = _parse_markdown_tables(body)
    if not tables:
        return "Experiment section must contain tables for reported experimental results."
    ablation_match = re.search(
        r"(?ims)^###?\s+(?:\*\*)?Ablation Result(?:\*\*)?\s*$\n(?P<ablation>.*?)(?=^##\s+|\Z)",
        body,
    )
    if ablation_match:
        ablation_body = str(ablation_match.group("ablation") or "")
        ablation_tables = _parse_markdown_tables(ablation_body)
        for _headers, rows in ablation_tables:
            if rows and _normalize_cell(rows[0][0]).lower() in {"optimal setup", "not found in manuscript"}:
                break
        else:
            if ablation_tables:
                return (
                    "Each ablation table must begin with an 'Optimal setup' anchor row "
                    "(Ablation Dimension = 'Optimal setup', Difference (Δ) = 0)."
                )
    return None


def validate_final_report_logic(markdown: str) -> str | None:
    for validator in (_validate_technical_positioning, _validate_claims, _validate_experiment):
        issue = validator(markdown)
        if issue:
            return issue
    return None


def _extract_markdown_headings(markdown_text: str) -> list[str]:
    headings: list[str] = []
    for line in str(markdown_text or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        text = stripped.lstrip("#").strip().lower()
        if text:
            headings.append(text)
    return headings


def find_missing_required_sections(markdown_text: str) -> list[str]:
    headings = _extract_markdown_headings(markdown_text)
    if not headings:
        return [label for label, _ in _REQUIRED_SECTION_GROUPS]

    missing: list[str] = []
    for label, aliases in _REQUIRED_SECTION_GROUPS:
        if not any(any(alias in heading for heading in headings) for alias in aliases):
            missing.append(label)
    return missing


def _sanitize_markdown_for_length_count(text: str) -> str:
    normalized = str(text or "")
    if not normalized:
        return ""
    normalized = _MARKDOWN_CODE_FENCE_PATTERN.sub(" ", normalized)
    normalized = _INLINE_CODE_PATTERN.sub(" ", normalized)
    normalized = _MARKDOWN_LINK_PATTERN.sub(r"\1", normalized)
    normalized = _URL_PATTERN.sub(" ", normalized)
    normalized = normalized.replace("|", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def analyze_report_language(text: str) -> LanguageStats:
    cleaned = _sanitize_markdown_for_length_count(text)
    english_words = len(_ENGLISH_WORD_PATTERN.findall(cleaned))
    chinese_chars = len(_CHINESE_CHAR_PATTERN.findall(cleaned))

    total_units = english_words + chinese_chars
    if total_units <= 0:
        return LanguageStats(
            primary_language="en",
            english_words=english_words,
            chinese_chars=chinese_chars,
            english_ratio=0.0,
            chinese_ratio=0.0,
        )

    chinese_ratio = chinese_chars / total_units
    english_ratio = english_words / total_units
    primary = "zh-CN" if chinese_ratio > 0.5 else "en"
    return LanguageStats(
        primary_language=primary,
        english_words=english_words,
        chinese_chars=chinese_chars,
        english_ratio=english_ratio,
        chinese_ratio=chinese_ratio,
    )


def validate_final_report(
    *,
    markdown: str,
    min_english_words: int,
    min_chinese_chars: int,
    force_english_output: bool = True,
) -> FinalReportValidation:
    text = str(markdown or "").strip()
    if not text:
        return FinalReportValidation(
            ok=False,
            reason="markdown_required",
            message="Final report markdown is empty.",
            language_stats=analyze_report_language(""),
            missing_sections=[label for label, _ in _REQUIRED_SECTION_GROUPS],
        )

    missing_sections = find_missing_required_sections(text)
    stats = analyze_report_language(text)

    if force_english_output and stats.chinese_chars > 0:
        return FinalReportValidation(
            ok=False,
            reason="english_required",
            message="Final report must be written in English only for this deployment.",
            language_stats=stats,
            missing_sections=[],
        )

    if missing_sections:
        return FinalReportValidation(
            ok=False,
            reason="final_report_sections_not_met",
            message="Final report missing required sections: " + ", ".join(missing_sections),
            language_stats=stats,
            missing_sections=missing_sections,
        )

    if min_english_words > 0 and stats.primary_language == "en" and stats.english_words < min_english_words:
        return FinalReportValidation(
            ok=False,
            reason="final_report_length_not_met",
            message=(
                f"English report is too short: {stats.english_words} words, required >= {min_english_words}."
            ),
            language_stats=stats,
            missing_sections=[],
        )

    if (
        min_chinese_chars > 0
        and stats.primary_language == "zh-CN"
        and stats.chinese_chars < min_chinese_chars
    ):
        return FinalReportValidation(
            ok=False,
            reason="final_report_length_not_met",
            message=(
                f"Chinese report is too short: {stats.chinese_chars} chars, required >= {min_chinese_chars}."
            ),
            language_stats=stats,
            missing_sections=[],
        )

    logic_issue = validate_final_report_logic(text)
    if logic_issue:
        return FinalReportValidation(
            ok=False,
            reason="final_report_logic_not_met",
            message=logic_issue,
            language_stats=stats,
            missing_sections=[],
        )

    return FinalReportValidation(
        ok=True,
        reason=None,
        message="ok",
        language_stats=stats,
        missing_sections=[],
    )
