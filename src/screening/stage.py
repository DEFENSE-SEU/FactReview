"""L1 boundary: extract once, retain findings and explicit check failures."""

from pathlib import Path

from pydantic import Field

from schemas.claim import Claim, Contract, Finding
from schemas.materials import SharedMaterials
from screening.checks import check_tables, check_writing
from screening.claims import extract_claims
from screening.figures import check_figures
from screening.references import check_bibliography


class ScreeningResult(Contract):
    claims: list[Claim]
    findings: list[Finding] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


def screen_paper(materials: SharedMaterials, output_dir: Path, *, call=None, reference_checker=None):
    # Failed extraction cannot become an apparently successful review with zero claims.
    result = ScreeningResult(claims=extract_claims(materials, call=call))
    for name, check in (("writing", check_writing), ("tables", check_tables)):
        try:
            result.findings.extend(check(materials, call=call))
        except Exception as exc:
            result.issues.append(f"{name} check failed: {exc}")
    for name, check in (
        ("figures", lambda: check_figures(materials, call=call)),
        ("references", lambda: check_bibliography(materials, output_dir, checker=reference_checker)),
    ):
        try:
            findings, issues = check()
            result.findings.extend(findings)
            result.issues.extend(issues)
        except Exception as exc:
            result.issues.append(f"{name} check failed: {exc}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "screening.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result
