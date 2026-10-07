"""Inspect each figure at its physical printed dimensions using actual image pixels."""

from pathlib import Path

from schemas.claim import Evidence, EvidencePointer, Finding
from schemas.materials import SharedMaterials
from screening.checks import ask

CATEGORIES = {"self_containedness", "legibility", "text_figure_consistency"}


def check_figures(materials: SharedMaterials, *, call=None) -> tuple[list[Finding], list[str]]:
    findings, issues = [], []
    for figure in materials.figures:
        if not figure.printed_crop_path or not Path(figure.printed_crop_path).is_file() or figure.loc is None:
            issues.append(f"{figure.id}: figure check unavailable; crop or location missing")
            continue
        result = ask(
            "Inspect the attached cropped image, its caption, and EVERY supplied body reference. "
            "The image is downscaled to its printed size at 96 dpi: judge legibility from those pixels. "
            "Return JSON {findings: [{category, text}]}. Allowed categories: self_containedness "
            "(legend, axis labels/units, panel labels), legibility, text_figure_consistency. "
            "Exclude colour, style and aesthetics. Describe concrete visible evidence. "
            "Tables are handled from parsed text separately.",
            {
                "figure_id": figure.id,
                "caption": figure.caption,
                "references": [b.model_dump() for b in figure.references],
                "printed_dpi": figure.printed_dpi,
            },
            module="screening_figures",
            call=call,
            images=[figure.printed_crop_path],
        )
        if not isinstance(result.get("findings"), list):
            raise ValueError("figure response must contain a findings list")
        for row in result["findings"]:
            if row.get("category") not in CATEGORIES:
                raise ValueError("figure findings are restricted to the three specified categories")
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
