from pathlib import Path

import pytest
from PIL import Image

from schemas.claim import ClaimLocation
from schemas.materials import FigureMaterial, MaterialBlock, SharedMaterials
from screening.figures import check_figures
from screening.references import check_bibliography
from screening.writing import check_tables, check_writing


@pytest.fixture
def materials(tmp_path):
    text = "Our methods is fast. Figure 1 shows the model."
    return SharedMaterials(
        paper_key="tiny",
        source_pdf=str(tmp_path / "paper.pdf"),
        markdown=text,
        markdown_path=str(tmp_path / "paper.md"),
        content_list_path="",
        provider="fixture",
        blocks=[MaterialBlock(id="b1", text=text, loc=ClaimLocation(page=1, section="intro"))],
    )


def test_writing_preserves_original_sentence_and_levels(materials):
    result = check_writing(
        materials,
        call=lambda **kwargs: {
            "findings": [
                {
                    "block_id": "b1",
                    "quote": "Our methods is fast.",
                    "text": "Use 'Our methods are fast'.",
                    "level": "definite_error",
                }
            ]
        },
    )
    assert result[0].evidence[0].pointer.quote == "Our methods is fast."
    assert result[0].loc.page == 1
    assert result[0].level == "definite_error"


@pytest.mark.parametrize("change", [{"quote": "Invented sentence"}, {"level": "aesthetics"}])
def test_writing_rejects_untraceable_or_out_of_scope_findings(materials, change):
    row = {
        "block_id": "b1",
        "quote": "Our methods is fast.",
        "text": "Fix grammar",
        "level": "definite_error",
    }
    with pytest.raises(ValueError):
        check_writing(materials, call=lambda **kwargs: {"findings": [{**row, **change}]})


def test_figures_send_printed_pixels_caption_all_references(materials, tmp_path):
    path = tmp_path / "printed.png"
    Image.new("RGB", (96, 48), "white").save(path)
    materials.figures = [
        FigureMaterial(
            id="figure_1",
            caption="Figure 1: Model.",
            loc=ClaimLocation(page=1),
            printed_crop_path=str(path),
            references=materials.blocks,
        )
    ]
    calls = []

    def call(**kwargs):
        calls.append(kwargs)
        return {
            "findings": [
                {"category": "text_figure_consistency", "text": "The caption refers to absent panel (c)."}
            ]
        }

    result, issues = check_figures(materials, call=call)
    assert not issues
    assert result[0].level == "text_figure_consistency"
    assert calls[0]["images"] == [str(path)]
    assert "Figure 1: Model." in calls[0]["prompt"]
    assert "Our methods is fast." in calls[0]["prompt"]
    assert "printed size" in calls[0]["system"]
    with pytest.raises(ValueError):
        check_figures(
            materials, call=lambda **kwargs: {"findings": [{"category": "colour", "text": "Use blue"}]}
        )


def test_missing_figure_crop_is_explicit(materials):
    materials.figures = [FigureMaterial(id="figure_1")]
    findings, issues = check_figures(materials, call=lambda **kwargs: pytest.fail("no pixels available"))
    assert findings == [] and "crop or location missing" in issues[0]


def test_table_checks_use_parsed_text_without_images(materials):
    materials.blocks.append(
        MaterialBlock(id="table1", text="Method | Time", kind="table", loc=ClaimLocation(page=2))
    )

    def call(**kwargs):
        assert "images" not in kwargs
        return {
            "findings": [{"block_id": "table1", "quote": "Method | Time", "text": "Time units are absent."}]
        }

    assert check_tables(materials, call=call)[0].kind == "table"


def test_reference_check_only_receives_bibliography(materials, tmp_path):
    materials.bibliography = [
        MaterialBlock(id="ref1", text="A. Author. Real Work. 2020.", loc=ClaimLocation(page=5))
    ]

    def checker(*, paper):
        assert Path(paper).read_text(encoding="utf-8") == materials.bibliography[0].text
        return {
            "ok": True,
            "total_refs": 1,
            "issues": [{"reference_title": "Real Work", "severity": "warning", "details": "Year differs"}],
        }

    findings, issues = check_bibliography(materials, tmp_path / "references", checker=checker)
    assert not issues
    assert findings[0].kind == "reference"
    assert "reference_check.json#issues.0" in findings[0].text


@pytest.mark.parametrize("status", ["error", "failed", "unknown"])
def test_model_failure_is_not_an_empty_success(materials, status):
    with pytest.raises(RuntimeError):
        check_writing(materials, call=lambda **kwargs: {"status": status, "findings": []})


def test_nonempty_error_cannot_be_empty_success(materials):
    with pytest.raises(RuntimeError):
        check_writing(materials, call=lambda **kwargs: {"status": "ok", "findings": [], "error": "offline"})


@pytest.mark.parametrize(
    "response",
    [
        {"ok": True, "total_refs": 0, "issues": []},
        {"ok": True, "issues": []},
        {"ok": True, "total_refs": 1},
    ],
)
def test_reference_check_reports_empty_processing_and_malformed_results(materials, tmp_path, response):
    materials.bibliography = materials.blocks
    findings, issues = check_bibliography(materials, tmp_path, checker=lambda **kwargs: response)
    assert not findings
    assert issues


def test_screening_keeps_failed_checks_visible(materials, tmp_path):
    from screening.stage import screen_paper

    def call(**kwargs):
        if kwargs["module"] == "screening.claims":
            return {
                "status": "ok",
                "claims": [
                    {
                        "text": "Our methods is fast.",
                        "source_block_id": "b1",
                        "source_quote": "Our methods is fast.",
                        "conditions": [{"id": "speed", "description": "method speed"}],
                        "needs": ["Experiments"],
                        "importance": "core",
                    }
                ],
            }
        return {"status": "error", "error": "offline"}

    result = screen_paper(materials, tmp_path, call=call)
    assert len(result.claims) == 1
    assert any("writing check failed" in issue for issue in result.issues)
    assert any("no bibliography" in issue for issue in result.issues)
    assert (tmp_path / "screening.json").is_file()
