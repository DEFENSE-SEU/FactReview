"""Visual table screening, retaining parsed-text checks as a separate boundary."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from schemas.claim import Contract, Evidence, EvidencePointer, Finding
from schemas.materials import SharedMaterials, TableMaterial
from screening.checks import ask
from screening.figures import _printed_size_verified

CATEGORIES = {"self_containedness", "legibility", "text_table_consistency"}


class TableCheckRecord(Contract):
    table_id: str
    status: Literal["checked", "failed", "unavailable"]
    finding_count: int = Field(default=0, ge=0)
    printed_size_verified: bool = False
    issues: list[str] = Field(default_factory=list)


def check_visual_tables(
    materials: SharedMaterials,
    *,
    call=None,
    recover_errors: bool = False,
    records: list[TableCheckRecord] | None = None,
) -> tuple[list[Finding], list[str]]:
    """Commit findings only after each complete table response is validated."""
    findings, issues = [], []
    if records is None:
        records = []
    for table in materials.tables:
        table_issues = list(table.issues)
        if not table.printed_crop_path or not Path(table.printed_crop_path).is_file() or table.loc is None:
            table_issues.append(f"{table.id}: table visual check unavailable; crop or location missing")
            issues.extend(table_issues)
            records.append(TableCheckRecord(table_id=table.id, status="unavailable", issues=table_issues))
            continue
        try:
            verified = _printed_size_verified(table)
        except (OSError, ValueError, OverflowError) as exc:
            table_issues.append(
                f"{table.id}: table visual check unavailable; invalid printed-size input: {exc}"
            )
            issues.extend(table_issues)
            records.append(TableCheckRecord(table_id=table.id, status="unavailable", issues=table_issues))
            continue
        try:
            table_findings, observed_issues = _check_table(table, call=call, printed_size_verified=verified)
            table_issues.extend(observed_issues)
        except Exception as exc:
            if not recover_errors:
                raise
            table_issues.append(f"table visual check failed for {table.id}: {exc}")
            issues.extend(table_issues)
            records.append(
                TableCheckRecord(
                    table_id=table.id, status="failed", printed_size_verified=verified, issues=table_issues
                )
            )
            continue
        if not verified:
            table_issues.append(
                f"{table.id}: printed-size legibility unavailable because physical bounding box metadata is missing; other checks used the available pixels and context."
            )
            for finding in table_findings:
                if finding.level == "legibility":
                    table_issues.append(
                        f"{table.id}: legibility check uncertain because physical printed size is unverified: {finding.text}"
                    )
            table_findings = [finding for finding in table_findings if finding.level != "legibility"]
        findings.extend(table_findings)
        issues.extend(table_issues)
        records.append(
            TableCheckRecord(
                table_id=table.id,
                status="checked",
                finding_count=len(table_findings),
                printed_size_verified=verified,
                issues=table_issues,
            )
        )
    return findings, issues


def _check_table(table: TableMaterial, *, call=None, printed_size_verified=False):
    result = ask(
        "Inspect this original table crop, its caption, supplied table footnotes, and EVERY body reference. "
        "Use the pixels to check self_containedness (row/column headers, units, meanings of bold, "
        "asterisks or other marks), legibility, and text_table_consistency. Interpret marks using "
        "the visible table, caption, table footnotes and supplied context; report unexplained marks only when their "
        "meaning is ambiguous. Check concrete agreements or contradictions between this table and "
        "its caption/body references. Do not assess scientific claim validity or numerical significance. "
        "When printed_size_verified is true, judge legibility at the attached physical printed size "
        "of 96 dpi. When false, mark legibility uncertain. When caption_ambiguous is true, "
        "caption-dependent self_containedness and text_table_consistency are uncertain. "
        "The parsed assignment cannot establish a manuscript defect. Do not infer defects from "
        "style or aesthetics. Return JSON {findings: [{category, disposition, text}]}; category is "
        "self_containedness, legibility, or text_table_consistency; disposition is issue, clear, "
        "or uncertain. Give concrete visible evidence in text. Use clear for positive observations, "
        "uncertain for insufficient pixels/context, and findings=[] when no defect or uncertainty is found.",
        {
            "table_id": table.id,
            "block_id": table.block_id,
            "caption": table.caption,
            "footnotes": table.footnotes,
            "caption_ambiguous": table.caption_ambiguous,
            "references": [block.model_dump() for block in table.references],
            "printed_dpi": table.printed_dpi,
            "printed_size_verified": printed_size_verified,
        },
        module="screening_tables.visual",
        call=call,
        images=[table.printed_crop_path],
    )
    if not isinstance(result.get("findings"), list):
        raise ValueError("table visual response must contain a findings list")
    findings, issues = [], []
    for row in result["findings"]:
        if not isinstance(row, dict) or row.get("category") not in CATEGORIES:
            raise ValueError("table visual findings require one of the three specified categories")
        if row.get("disposition") not in {"issue", "clear", "uncertain"}:
            raise ValueError("table visual disposition must be issue, clear, or uncertain")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("table visual findings require a nonempty explanation")
        if row["disposition"] == "clear":
            continue
        if row["disposition"] == "uncertain":
            issues.append(f"{table.id}: {row['category']} check uncertain: {row['text']}")
            continue
        if table.caption_ambiguous and row["category"] in {"self_containedness", "text_table_consistency"}:
            issues.append(
                f"{table.id}: {row['category']} unconfirmed because parser caption assignment is ambiguous: {row['text']}"
            )
            continue
        quote = table.caption or next((reference.text for reference in table.references), "")
        if not quote:
            issues.append(f"{table.id}: {row['text']} (caption/body reference unavailable)")
            continue
        findings.append(
            Finding(
                kind="table",
                loc=table.loc,
                level=row["category"],
                text=row["text"],
                evidence=[
                    Evidence(
                        source="paper_internal",
                        pointer=EvidencePointer(
                            locator=table.printed_crop_path, quote=quote, page=table.loc.page, key=table.id
                        ),
                        direction="flaw",
                        sufficient=False,
                        note=row["text"],
                        affects_claim=False,
                    )
                ],
            )
        )
    return findings, issues
