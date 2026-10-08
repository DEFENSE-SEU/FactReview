"""L1 boundary: extract once, retain findings and explicit check failures."""

import os
from pathlib import Path

from pydantic import Field

from schemas.claim import Claim, Contract, Finding
from schemas.materials import SharedMaterials
from screening.checks import check_tables, check_writing
from screening.claims import extract_claims
from screening.figures import FigureCheckRecord, check_figures
from screening.references import check_bibliography


class ScreeningResult(Contract):
    claims: list[Claim]
    findings: list[Finding] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    figure_checks: list[FigureCheckRecord] = Field(default_factory=list)
    figure_coverage: dict[str, int] = Field(default_factory=dict)


def screen_paper(materials: SharedMaterials, output_dir: Path, *, call=None, reference_checker=None):
    # Failed extraction cannot become an apparently successful review with zero claims.
    result = ScreeningResult(
        claims=extract_claims(
            materials, call=call, max_source_repairs=int(os.environ.get("CLAIM_SOURCE_MAX_REPAIRS", "3"))
        )
    )
    for name, check in (
        ("writing", lambda: check_writing(materials, call=call, issues=result.issues)),
        ("tables", lambda: check_tables(materials, call=call)),
    ):
        try:
            result.findings.extend(check())
        except Exception as exc:
            result.issues.append(f"{name} check failed: {exc}")
    for name, check in (
        (
            "figures",
            lambda: check_figures(materials, call=call, recover_errors=True, records=result.figure_checks),
        ),
        ("references", lambda: check_bibliography(materials, output_dir, checker=reference_checker, call=call)),
    ):
        try:
            findings, issues = check()
            result.findings.extend(findings)
            result.issues.extend(issues)
        except Exception as exc:
            result.issues.append(f"{name} check failed: {exc}")
    result.figure_coverage = {
        "total": len(materials.figures),
        **{
            status: sum(record.status == status for record in result.figure_checks)
            for status in ("checked", "failed", "unavailable")
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "screening.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result
