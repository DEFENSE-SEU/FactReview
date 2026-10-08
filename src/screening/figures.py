"""Inspect each figure at its physical printed dimensions using actual image pixels."""

import math
from pathlib import Path
from typing import Literal

from pydantic import Field

from schemas.claim import Contract, Evidence, EvidencePointer, Finding
from schemas.materials import FigureMaterial, SharedMaterials
from screening.checks import ask

CATEGORIES = {"self_containedness", "legibility", "text_figure_consistency"}


class FigureCheckRecord(Contract):
    figure_id: str
    status: Literal["checked", "failed", "unavailable"]
    finding_count: int = Field(default=0, ge=0)
    printed_size_verified: bool = False
    issues: list[str] = Field(default_factory=list)


def _printed_size_verified(figure: FigureMaterial) -> bool:
    """Check the actual submitted pixels against the parser's physical box."""
    from PIL import Image

    if figure.printed_dpi != 96:
        raise ValueError("printed-size input must use 96 dpi")
    with Image.open(figure.printed_crop_path) as image:
        image.load()
        size = image.size
    if figure.bbox_points is None:
        # Older serialized materials may have the image but no physical box.
        # Preserve their readability while making the missing verification explicit.
        return False
    x1, y1, x2, y2 = figure.bbox_points
    if not all(math.isfinite(value) for value in figure.bbox_points) or x2 <= x1 or y2 <= y1:
        raise ValueError("printed-size input has an invalid physical bounding box")
    expected = (max(1, round((x2 - x1) * 96 / 72)), max(1, round((y2 - y1) * 96 / 72)))
    if size != expected:
        raise ValueError(f"printed-size pixels {size} do not match physical bounding box {expected}")
    return True


def check_figures(
    materials: SharedMaterials,
    *,
    call=None,
    recover_errors: bool = False,
    records: list[FigureCheckRecord] | None = None,
) -> tuple[list[Finding], list[str]]:
    """Inspect all figures; production can retain other figures after one fails.

    Direct callers retain the strict response-validation contract by default.
    Each figure is validated completely before any of its findings are retained.
    """
    findings, issues = [], []
    if records is None:
        records = []
    for figure in materials.figures:
        if not figure.printed_crop_path or not Path(figure.printed_crop_path).is_file() or figure.loc is None:
            issue = f"{figure.id}: figure check unavailable; crop or location missing"
            issues.append(issue)
            records.append(FigureCheckRecord(figure_id=figure.id, status="unavailable", issues=[issue]))
            continue
        try:
            verified = _printed_size_verified(figure)
        except (OSError, ValueError, OverflowError) as exc:
            issue = f"{figure.id}: figure check unavailable; invalid printed-size input: {exc}"
            issues.append(issue)
            records.append(FigureCheckRecord(figure_id=figure.id, status="unavailable", issues=[issue]))
            continue
        try:
            figure_findings, figure_issues = _check_figure(figure, call=call, printed_size_verified=verified)
        except Exception as exc:
            if not recover_errors:
                raise
            issue = f"figures check failed for {figure.id}: {exc}"
            issues.append(issue)
            records.append(
                FigureCheckRecord(
                    figure_id=figure.id, status="failed", printed_size_verified=verified, issues=[issue]
                )
            )
            continue
        if recover_errors and not verified:
            figure_issues.append(
                f"{figure.id}: printed-size legibility unavailable because physical bounding box metadata "
                "is missing; the remaining figure checks used the available pixels and context."
            )
            for finding in figure_findings:
                if finding.level == "legibility":
                    figure_issues.append(
                        f"{figure.id}: legibility check uncertain because physical printed size could not "
                        f"be verified: {finding.text}"
                    )
            figure_findings = [finding for finding in figure_findings if finding.level != "legibility"]
        findings.extend(figure_findings)
        issues.extend(figure_issues)
        records.append(
            FigureCheckRecord(
                figure_id=figure.id,
                status="checked",
                finding_count=len(figure_findings),
                printed_size_verified=verified,
                issues=figure_issues,
            )
        )
    return findings, issues


def _check_figure(
    figure: FigureMaterial, *, call=None, printed_size_verified: bool = False
) -> tuple[list[Finding], list[str]]:
    findings, issues = [], []
    result = ask(
        "Inspect the attached cropped image, its caption, and EVERY supplied body reference. "
        "When printed_size_verified is true, the image is downscaled to its printed size at 96 dpi: "
        "judge legibility from those pixels. When false, the physical printed size is unverified; "
        "classify legibility concerns as uncertain and assess the other categories from available context. "
        "Return JSON {findings: [{category, disposition, text}]}. Each disposition must be "
        "issue, clear, or uncertain. Use issue only for a concrete observed defect; text must "
        "identify the defect and visible evidence. Use clear for normal or positive observations "
        "such as readable labels or agreement with the caption. Use uncertain when the supplied "
        "image or context cannot establish whether a defect exists. Return an empty findings list "
        "when no defects or uncertainties are found. Allowed categories: self_containedness "
        "(legend, axis labels/units, panel labels), legibility, text_figure_consistency. "
        "Evaluate labels and legends together with the supplied caption; report omissions only "
        "when they leave the figure ambiguous. "
        "When caption_ambiguous is true, the parser may have combined captions from neighboring figures. "
        "Treat caption-dependent self_containedness and text_figure_consistency as uncertain; "
        "the parser's assignment cannot establish a manuscript defect. Assess legibility from the pixels. "
        "Exclude colour, style and aesthetics. Describe concrete visible evidence. "
        "Tables are handled from parsed text separately.",
        {
            "figure_id": figure.id,
            "caption": figure.caption,
            "caption_ambiguous": figure.caption_ambiguous,
            "references": [b.model_dump() for b in figure.references],
            "printed_dpi": figure.printed_dpi,
            "printed_size_verified": printed_size_verified,
        },
        module="screening_figures",
        call=call,
        images=[figure.printed_crop_path],
    )
    if not isinstance(result.get("findings"), list):
        raise ValueError("figure response must contain a findings list")
    for row in result["findings"]:
        if not isinstance(row, dict):
            raise ValueError("each figure finding must be an object")
        if row.get("category") not in CATEGORIES:
            raise ValueError("figure findings are restricted to the three specified categories")
        if row.get("disposition") not in ("issue", "clear", "uncertain"):
            raise ValueError("figure finding disposition must be issue, clear, or uncertain")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("figure finding text must be a nonempty explanation")
        if row["disposition"] == "clear":
            continue
        if row["disposition"] == "uncertain":
            issues.append(f"{figure.id}: {row['category']} check uncertain: {row['text']}")
            continue
        if figure.caption_ambiguous and row["category"] in {
            "self_containedness",
            "text_figure_consistency",
        }:
            issues.append(
                f"{figure.id}: {row['category']} unconfirmed because parser caption assignment is ambiguous: {row['text']}"
            )
            continue
        # A visual observation points to the actual crop. The caption is a
        # contextual quote; absence of a caption remains visible in issues.
        quote = figure.caption or next((b.text for b in figure.references), "")
        if not quote:
            issues.append(f"{figure.id}: {row['text']} (caption/body reference unavailable)")
            continue
        findings.append(
            Finding(
                kind="figure",
                loc=figure.loc,
                level=row["category"],
                text=row["text"],
                evidence=[
                    Evidence(
                        source="paper_internal",
                        pointer=EvidencePointer(
                            locator=figure.printed_crop_path,
                            quote=quote,
                            page=figure.loc.page,
                            key=figure.id,
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
