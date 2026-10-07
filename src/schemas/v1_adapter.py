"""Explicit, one-way reading of historical artifacts. Never invoked by v2 validation.

Old labels are retained as provenance. Their translation is a display migration;
it does not establish v2 evidence sufficiency or re-assess the old run.
"""

from __future__ import annotations

import html
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from pydantic import Field

from schemas.claim import ClaimStatus, Contract

_LABELS = {
    "supported": ClaimStatus.SUPPORTED,
    "paper_supported": ClaimStatus.SUPPORTED,
    "partially_supported": ClaimStatus.QUESTIONED,
    "in_conflict": ClaimStatus.QUESTIONED,
    "inconclusive": ClaimStatus.UNVERIFIED,
}


def adapt_v1_label(value: str) -> ClaimStatus:
    """Map known historical labels conservatively; reject unknown labels."""
    text = html.unescape(re.sub(r"<[^>]*>", "", value)).lower().strip()
    text = re.sub(r"^[^a-z]+", "", text)
    text = text.strip("*_` ")
    text = re.sub(r"[\s-]+", "_", text)
    try:
        return _LABELS[text]
    except KeyError as exc:
        raise ValueError(f"unknown v1 claim label: {value!r}") from exc


class LegacyClaimRecord(Contract):
    """Historical display data kept separate from validated v2 Claim records."""

    id: str
    text: str = ""
    original_label: str
    status: ClaimStatus
    location: Any = None
    evidence: Any = None
    raw: dict[str, Any] = Field(default_factory=dict)
    reassessed: bool = False


class LegacyArtifact(Contract):
    source_path: str
    raw: Any
    claims: list[LegacyClaimRecord] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class _HTMLTables(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self.table: list[list[str]] | None = None
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self.table = []
        elif tag == "tr":
            self.row = []
        elif tag in {"td", "th"}:
            self.cell = []
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.cell is not None and self.row is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None and self.table is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None


def _markdown_claims(text: str, issues: list[str]) -> list[dict[str, Any]]:
    parser = _HTMLTables()
    parser.feed(text)
    tables = parser.tables
    # Historical reports contain either HTML tables or pipe-delimited Markdown.
    current: list[list[str]] = []
    for line in [*text.splitlines(), ""]:
        if line.strip().startswith("|"):
            row = [cell.strip().replace(r"\|", "|")
                   for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
            if not all(re.fullmatch(r"[:\-\s]+", cell) for cell in row):
                current.append(row)
        elif current:
            tables.append(current)
            current = []
    records = []
    for rows in tables:
        if not rows:
            continue
        headers = [cell.strip().lower() for cell in rows[0]]
        if "claim" not in headers or "status" not in headers:
            continue
        for row_number, cells in enumerate(rows[1:], 2):
            if len(cells) == len(headers):
                records.append(dict(zip(headers, cells, strict=True)))
            else:
                issues.append(f"Malformed historical claim-table row {row_number}; preserved in raw.")
    return records


def read_v1_artifact(path: str | Path) -> LegacyArtifact:
    """Read old JSON/Markdown/HTML without modifying it or inventing evidence."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    raw: Any = json.loads(text) if source.suffix.lower() == ".json" else text
    result = LegacyArtifact(source_path=str(source), raw=raw)
    if isinstance(raw, str):
        records = _markdown_claims(raw, result.issues)
    elif isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict):
        records = raw.get("claims", raw.get("assessments", raw.get("claim_results", [])))
        if not records and any(key in raw for key in ("label", "final_status")):
            records = [raw]
    else:
        records = []
    if not isinstance(records, list):
        result.issues.append("Historical claim collection is not a list; preserved in raw.")
        return result
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            result.issues.append(f"Claim {index} has no structured record; preserved in raw.")
            continue
        label = str(record.get("label", record.get("final_status", record.get("status", ""))))
        if not label:
            result.issues.append(f"Claim {index} has no historical label; preserved in raw.")
            continue
        try:
            status = adapt_v1_label(label)
        except ValueError as exc:
            result.issues.append(str(exc))
            continue
        result.claims.append(LegacyClaimRecord(
            id=str(record.get("id", record.get("claim_id", f"legacy_{index}"))),
            text=str(record.get("text", record.get("claim", ""))),
            original_label=label,
            status=status,
            location=record.get("location"),
            evidence=record.get("evidence"),
            raw=record,
        ))
    return result
