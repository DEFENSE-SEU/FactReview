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
