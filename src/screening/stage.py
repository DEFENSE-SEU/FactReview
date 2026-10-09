"""L1 boundary: extract once, retain findings and explicit check failures."""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field

from schemas.claim import Claim, Contract, Finding
from schemas.materials import SharedMaterials
from screening.checks import check_tables, check_writing
from screening.claims import ClaimExtractionError, extract_claims
from screening.figures import FigureCheckRecord, check_figures
from screening.references import check_bibliography
from screening.tables import TableCheckRecord, check_visual_tables
from screening.writing import WritingSectionRecord


class ScreeningResult(Contract):
    claims: list[Claim]
    findings: list[Finding] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    figure_checks: list[FigureCheckRecord] = Field(default_factory=list)
    figure_coverage: dict[str, int] = Field(default_factory=dict)
    figure_context_coverage: dict[str, int] = Field(default_factory=dict)
    table_checks: list[TableCheckRecord] = Field(default_factory=list)
    table_coverage: dict[str, int] = Field(default_factory=dict)
    writing_checks: list[WritingSectionRecord] = Field(default_factory=list)
    writing_coverage: dict[str, int] = Field(default_factory=dict)
    anonymity_policy: Literal["unspecified", "required", "not_required"] = "unspecified"
    claim_extraction_status: Literal["ok", "failed"] = "ok"


class ScreeningFailure(ClaimExtractionError):
    """Independent screening finished, while claim extraction remains failed."""

    def __init__(self, message: str, result: ScreeningResult):
        super().__init__(message)
        self.result = result


def screen_paper(
    materials: SharedMaterials,
    output_dir: Path,
    *,
    call=None,
    reference_checker=None,
    anonymity_policy="unspecified",
):
    # Failed extraction cannot become an apparently successful review with zero claims.
    result = ScreeningResult(claims=[], anonymity_policy=anonymity_policy)
    extraction_error = None
    try:
        result.claims = extract_claims(
            materials, call=call, max_source_repairs=int(os.environ.get("CLAIM_SOURCE_MAX_REPAIRS", "3"))
        )
    except Exception as exc:
        extraction_error = f"{type(exc).__name__}: {exc}"
        result.claim_extraction_status = "failed"
        result.issues.append(f"Claim extraction failed: {extraction_error}")
    for name, check in (
        (
            "writing",
            lambda: check_writing(
                materials,
                call=call,
                issues=result.issues,
                records=result.writing_checks,
                anonymity_policy=result.anonymity_policy,
                recover_errors=True,
            ),
        ),
        ("tables", lambda: check_tables(materials, call=call)),
    ):
        try:
            result.findings.extend(check())
        except Exception as exc:
            result.issues.append(f"{name} check failed: {exc}")
    represented_tables = {table.block_id for table in materials.tables}
    missing_tables = [
        block for block in materials.blocks if block.kind == "table" and block.id not in represented_tables
    ]
    for block in missing_tables:
        issue = (
            f"{block.id}: table visual material unavailable; rebuild shared materials from the original PDF."
        )
        result.table_checks.append(TableCheckRecord(table_id=block.id, status="unavailable", issues=[issue]))
        result.issues.append(issue)
    for name, check in (
        (
            "figures",
            lambda: check_figures(materials, call=call, recover_errors=True, records=result.figure_checks),
        ),
        (
            "table visuals",
            lambda: check_visual_tables(
                materials, call=call, recover_errors=True, records=result.table_checks
            ),
        ),
        (
            "references",
            lambda: check_bibliography(materials, output_dir, checker=reference_checker, call=call),
        ),
    ):
        try:
            findings, issues = check()
            result.findings.extend(findings)
            result.issues.extend(issues)
        except Exception as exc:
            result.issues.append(f"{name} check failed: {exc}")
    result.writing_coverage = {
        "total": len(result.writing_checks),
        **{
            status: sum(record.status == status for record in result.writing_checks)
            for status in ("checked", "failed", "unavailable")
        },
    }
    result.figure_coverage = {
        "total": len(materials.figures),
        **{
            status: sum(record.status == status for record in result.figure_checks)
            for status in ("checked", "failed", "unavailable")
        },
    }
    result.figure_context_coverage = {
        "total": len(materials.figures),
        **{
            status: sum(record.context_status == status for record in result.figure_checks)
            for status in ("not_requested", "checked", "failed", "unavailable")
        },
        "unrecorded": max(0, len(materials.figures) - len(result.figure_checks)),
    }
    result.table_coverage = {
        "total": len(materials.tables) + len(missing_tables),
        **{
            status: sum(record.status == status for record in result.table_checks)
            for status in ("checked", "failed", "unavailable")
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "screening.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    if extraction_error is not None:
        raise ScreeningFailure(extraction_error, result)
    return result
