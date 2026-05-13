from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from common.pipeline_context import resolve_artifact_path
from schemas.paper import Paper, PaperMetadata, Section

_HEADING_RE = re.compile(r"(?m)^(#{1,4})\s+(.+?)\s*$")


def sections_from_markdown(markdown: str) -> list[Section]:
    matches = list(_HEADING_RE.finditer(markdown or ""))
    if not matches:
        body = (markdown or "").strip()
        return [Section(id="sec_01", title="Full text", text=body, char_start=0, char_end=len(markdown or ""))]

    sections: list[Section] = []
    for idx, match in enumerate(matches, start=1):
        start = match.end()
        end = matches[idx].start() if idx < len(matches) else len(markdown)
        text = (markdown[start:end] or "").strip()
        title = match.group(2).strip()
        if not text and sections:
            continue
        sections.append(
            Section(
                id=f"sec_{idx:02d}",
                title=title,
                level=len(match.group(1)),
                text=text,
                char_start=start,
                char_end=end,
            )
        )
    return sections


def title_from_markdown(markdown: str, *, fallback: str) -> str:
    match = _HEADING_RE.search(markdown or "")
    if match:
        title = match.group(2).strip()
        if title:
            return title
    return fallback


def paper_from_parse_payload(
    *,
    repo_root: Path,
    paper_key: str,
    parse_payload: dict[str, Any],
) -> Paper:
    mineru_md = resolve_artifact_path(repo_root, parse_payload.get("mineru_markdown_path"))
    if mineru_md is None or not mineru_md.exists():
        raise FileNotFoundError("claim_extract requires parse-stage mineru_markdown_path")
    markdown = mineru_md.read_text(encoding="utf-8", errors="ignore")

    source_pdf = resolve_artifact_path(repo_root, parse_payload.get("source_pdf"))
    if source_pdf is None:
        source_pdf = Path(str(parse_payload.get("source_pdf") or "paper.pdf"))

    key = str(paper_key or "").strip() or source_pdf.stem or "paper"
    title = title_from_markdown(markdown, fallback=key)
    return Paper(
        metadata=PaperMetadata(paper_key=key, title=title),
        pdf_path=source_pdf,
        markdown_path=mineru_md,
        sections=sections_from_markdown(markdown),
        tables=[],
        figures=[],
        backend=str(parse_payload.get("markdown_provider") or "mineru"),
    )
