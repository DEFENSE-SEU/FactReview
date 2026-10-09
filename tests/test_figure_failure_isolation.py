"""Production L1 retains completed visual checks when another figure fails."""

import json
from pathlib import Path

import pytest
from PIL import Image

from schemas.claim import ClaimLocation
from schemas.materials import FigureMaterial, MaterialBlock, SharedMaterials
from screening.stage import screen_paper


@pytest.fixture
def visual_materials(tmp_path):
    text = "Our method is stable."
    manuscript = tmp_path / "paper.md"
    manuscript.write_text(text, encoding="utf-8")
    figures = []
    for index in range(1, 4):
        image = tmp_path / f"figure_{index}_printed.png"
        Image.new("RGB", (96, 48), "white").save(image)
        figures.append(
            FigureMaterial(
                id=f"figure_{index}",
                caption=f"Figure {index}: Model result.",
                loc=ClaimLocation(page=1),
                printed_crop_path=str(image),
                bbox_points=(0, 0, 72, 36),
            )
        )
    return SharedMaterials(
        paper_key="visual-isolation",
        source_pdf=str(tmp_path / "paper.pdf"),
        markdown=text,
        markdown_path=str(manuscript),
        content_list_path="",
        provider="fixture",
        blocks=[MaterialBlock(id="b1", text=text, loc=ClaimLocation(page=1))],
        figures=figures,
    )


def model_call(visual_response, seen):
    def call(**kwargs):
        if kwargs["module"] == "screening.claims":
            return {
                "status": "ok",
                "claims": [
                    {
                        "text": "Our method is stable.",
                        "source_block_id": "b1",
                        "source_quote": "Our method is stable.",
                        "conditions": [{"id": "stability", "description": "method stability"}],
                        "needs": ["Experiments"],
                        "importance": "core",
                    }
                ],
            }
        if kwargs["module"] == "screening_figures":
            figure = json.loads(kwargs["prompt"])["figure_id"]
            seen.append(figure)
            return visual_response(figure)
        return {"findings": []}

    return call


def visible_issue(figure):
    return {"category": "legibility", "disposition": "issue", "text": f"{figure}: labels overlap."}


@pytest.mark.parametrize("failure", ["provider", "malformed", "exception"])
def test_production_l1_retains_other_figures_and_persists_failed_coverage(
    visual_materials, tmp_path, failure
):
    seen = []

    def response(figure):
        if figure == "figure_2":
            if failure == "provider":
                return {"status": "error", "error": "fixture provider unavailable"}
            if failure == "exception":
                raise TimeoutError("fixture timeout")
            # The first row is valid, but this entire figure must be rejected.
            return {"findings": [visible_issue(figure), {"category": "colour", "disposition": "issue"}]}
        return {"findings": [visible_issue(figure)]}

    output = tmp_path / "screening"
    result = screen_paper(visual_materials, output, call=model_call(response, seen))
    assert seen == ["figure_1", "figure_2", "figure_3"]
    assert [finding.evidence[0].pointer.key for finding in result.findings] == ["figure_1", "figure_3"]
    assert result.figure_coverage == {"total": 3, "checked": 2, "failed": 1, "unavailable": 0}
    assert [record.status for record in result.figure_checks] == ["checked", "failed", "checked"]
    assert [record.finding_count for record in result.figure_checks] == [1, 0, 1]
    assert all(record.printed_size_verified for record in result.figure_checks)
    stored = json.loads((output / "screening.json").read_text(encoding="utf-8"))
    assert stored["figure_coverage"] == result.figure_coverage
    assert len(stored["findings"]) == 2
    assert stored["figure_checks"][1]["issues"]
    assert any("figures check failed for figure_2" in issue for issue in result.issues)


@pytest.mark.parametrize("invalid", ["missing", "corrupt", "pixel_size", "dpi", "bbox"])
def test_unavailable_or_wrong_scale_image_is_reported_without_model_call(visual_materials, tmp_path, invalid):
    figure = visual_materials.figures[1]
    image = Path(figure.printed_crop_path)
    if invalid == "missing":
        image.unlink()
    elif invalid == "corrupt":
        image.write_bytes(b"not an image")
    elif invalid == "pixel_size":
        Image.new("RGB", (192, 96), "white").save(image)
    elif invalid == "dpi":
        figure.printed_dpi = 200
    else:
        figure.bbox_points = (0, 0, float("inf"), 36)
    seen = []
    result = screen_paper(
        visual_materials, tmp_path / "screening", call=model_call(lambda figure: {"findings": []}, seen)
    )
    assert seen == ["figure_1", "figure_3"]
    assert result.figure_coverage == {"total": 3, "checked": 2, "failed": 0, "unavailable": 1}
    assert result.figure_checks[1].status == "unavailable"
    assert not result.figure_checks[1].printed_size_verified
    assert any("figure_2: figure check unavailable" in issue for issue in result.issues)


def test_zero_figures_persist_explicit_zero_coverage(visual_materials, tmp_path):
    visual_materials.figures = []
    seen = []
    result = screen_paper(
        visual_materials, tmp_path / "screening", call=model_call(lambda figure: {"findings": []}, seen)
    )
    assert not seen
    assert result.figure_coverage == {"total": 0, "checked": 0, "failed": 0, "unavailable": 0}
    assert result.figure_checks == []


def test_legacy_image_without_bbox_keeps_explicit_unverified_scale(visual_materials, tmp_path):
    visual_materials.figures[1].bbox_points = None
    seen = []
    result = screen_paper(
        visual_materials, tmp_path / "screening", call=model_call(lambda figure: {"findings": []}, seen)
    )
    assert len(seen) == 3
    assert [record.printed_size_verified for record in result.figure_checks] == [True, False, True]
    assert any("figure_2: printed-size legibility unavailable" in issue for issue in result.issues)
    assert any("printed-size legibility unavailable" in issue for issue in result.figure_checks[1].issues)
    from review.report.v2 import write_review
    from schemas.review import FinalReview

    output = write_review(
        FinalReview(paper_key=visual_materials.paper_key, run_id="fixture", claims=result.claims),
        tmp_path / "report",
        issues=result.issues,
        figure_coverage=result.figure_coverage,
        render_pdf=False,
    )
    report = Path(output["markdown"]).read_text(encoding="utf-8")
    assert "figure\\_2: printed\\-size legibility unavailable" in report


def test_production_cannot_confirm_printed_legibility_without_physical_box(visual_materials, tmp_path):
    visual_materials.figures[1].bbox_points = None
    seen = []
    result = screen_paper(
        visual_materials,
        tmp_path / "screening",
        call=model_call(
            lambda figure: {
                "findings": [
                    visible_issue(figure),
                    {"category": "self_containedness", "disposition": "issue", "text": "Axis has no units."},
                ]
            },
            seen,
        ),
    )
    second = [finding for finding in result.findings if finding.evidence[0].pointer.key == "figure_2"]
    assert second == []
    record = result.figure_checks[1]
    assert record.context_status == "unavailable"
    assert record.crop_response["findings"][1] == {
        "category": "self_containedness",
        "disposition": "issue",
        "text": "Axis has no units.",
    }
    assert any("original-page context unavailable" in issue for issue in record.issues)
    assert any(
        "self_containedness unconfirmed original crop observation: Axis has no units." in issue
        for issue in record.issues
    )
    assert any("figure_2: legibility check uncertain" in issue for issue in result.issues)
