import json
from pathlib import Path

import fitz
import pytest
from PIL import Image

from common import run_stats
from schemas.claim import ClaimLocation
from schemas.materials import FigureMaterial, MaterialBlock, PageImage, SharedMaterials
from screening.figures import check_figures
from screening.references import check_bibliography
from screening.writing import check_tables, check_writing


@pytest.fixture
def materials(tmp_path):
    text = "Our methods is fast. Figure 1 shows the model."
    (tmp_path / "paper.md").write_text(text, encoding="utf-8")
    return SharedMaterials(
        paper_key="tiny",
        source_pdf=str(tmp_path / "paper.pdf"),
        markdown=text,
        markdown_path=str(tmp_path / "paper.md"),
        content_list_path="",
        provider="fixture",
        blocks=[MaterialBlock(id="b1", text=text, loc=ClaimLocation(page=1, section="intro"))],
    )


@pytest.fixture
def writing_materials(materials, tmp_path):
    image_path = tmp_path / "page_1.png"
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((40, 50), materials.markdown)
        pdf.save(materials.source_pdf)
        page.get_pixmap().save(image_path)
        materials.pages = [
            PageImage(
                page=1, path=str(image_path), width_points=page.rect.width, height_points=page.rect.height
            )
        ]
    return materials


def test_writing_preserves_original_sentence_and_levels(writing_materials):
    materials = writing_materials

    def call(**kwargs):
        if kwargs["module"] == "screening_writing.validation":
            assert kwargs["images"] == [materials.pages[0].path]
            return {
                "results": [
                    {
                        "candidate_id": "writing_1",
                        "classification": "manuscript_error",
                        "explanation": "The original sentence has incorrect plural agreement.",
                    }
                ]
            }
        return {
            "findings": [
                {
                    "block_id": "b1",
                    "quote": "Our methods is fast.",
                    "text": "Use 'Our methods are fast'.",
                    "level": "definite_error",
                }
            ]
        }

    result = check_writing(
        materials,
        call=call,
    )
    assert result[0].evidence[0].pointer.quote == "Our methods is fast."
    assert result[0].loc.page == 1
    assert result[0].level == "definite_error"
    assert "original PDF page 1 confirmed" in result[0].evidence[0].note


def test_writing_batches_original_pdf_review_and_excludes_ocr_style_uncertainty(materials, tmp_path):
    quotes = [
        "Our methods is fast.",
        "A well studied model.",
        "The selector is dim(r).",
        "This improves it.",
        "The notation is x y.",
    ]
    texts = [" ".join(quotes[:3]), " ".join(quotes[3:])]
    materials.blocks = [
        MaterialBlock(id=f"b{i}", text=text, loc=ClaimLocation(page=i)) for i, text in enumerate(texts, 1)
    ]
    materials.markdown = "\n".join(texts)
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
    with fitz.open() as pdf:
        for number, text in enumerate(texts, 1):
            page = pdf.new_page()
            page.insert_text((40, 50), text.replace("dim(r)", "dir(r)"))
            image_path = tmp_path / f"original_{number}.png"
            page.get_pixmap().save(image_path)
            materials.pages.append(
                PageImage(
                    page=number,
                    path=str(image_path),
                    width_points=page.rect.width,
                    height_points=page.rect.height,
                )
            )
        pdf.save(materials.source_pdf)
    classifications = ["manuscript_error", "style", "parser_artifact", "clarity_issue", "uncertain"]
    calls = []

    def call(**kwargs):
        calls.append(kwargs)
        if kwargs["module"] == "screening_writing":
            assert "discretionary style" in kwargs["system"]
            return {
                "findings": [
                    {
                        "block_id": "b1" if i < 3 else "b2",
                        "quote": quote,
                        "text": f"Candidate {i + 1}",
                        "level": "definite_error",
                    }
                    for i, quote in enumerate(quotes)
                ]
            }
        payload = json.loads(kwargs["prompt"])
        assert kwargs["images"] == [materials.pages[payload["page"] - 1].path]
        assert len(payload["candidates"]) == (3 if payload["page"] == 1 else 2)
        return {
            "results": [
                {
                    "candidate_id": row["candidate_id"],
                    "classification": classifications[int(row["candidate_id"].split("_")[1]) - 1],
                    "explanation": "Compared with the printed sentence and context.",
                }
                for row in payload["candidates"]
            ]
        }

    issues = []
    findings = check_writing(materials, call=call, issues=issues)
    assert len(calls) == 3
    assert [f.level for f in findings] == ["definite_error", "clarity_issue"]
    assert [f.evidence[0].pointer.quote for f in findings] == [quotes[0], quotes[3]]
    assert len(issues) == 3 and all(
        any(label in issue for issue in issues) for label in ["style", "parser_artifact", "uncertain"]
    )
    assert "dim(r)" in materials.markdown


@pytest.mark.parametrize("missing", ["pages", "image_file"])
def test_writing_missing_original_image_cannot_confirm_candidates(writing_materials, missing):
    materials = writing_materials
    if missing == "pages":
        materials.pages = []
    else:
        Path(materials.pages[0].path).unlink()
    calls = []

    def call(**kwargs):
        calls.append(kwargs)
        return {
            "findings": [
                {
                    "block_id": "b1",
                    "quote": "Our methods is fast.",
                    "text": "Fix agreement",
                    "level": "definite_error",
                }
            ]
        }

    issues = []
    assert check_writing(materials, call=call, issues=issues) == []
    assert len(calls) == 1 and "original PDF page image unavailable" in issues[0]


def test_writing_checks_every_quote_before_any_visual_validation(writing_materials):
    calls = []

    def call(**kwargs):
        calls.append(kwargs)
        assert kwargs["module"] == "screening_writing"
        return {
            "findings": [
                {"block_id": "b1", "quote": quote, "text": "Fix agreement", "level": "definite_error"}
                for quote in ("Our methods is fast.", "A fabricated second sentence.")
            ]
        }

    with pytest.raises(ValueError, match="located manuscript block"):
        check_writing(writing_materials, call=call)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "response",
    [
        {"results": []},
        {
            "results": [
                {"candidate_id": "other", "classification": "manuscript_error", "explanation": "Confirmed"}
            ]
        },
        {
            "results": [
                {
                    "candidate_id": "writing_1",
                    "classification": "manuscript_error",
                    "explanation": "Confirmed",
                }
            ]
            * 2
        },
        {"results": [{"candidate_id": "writing_1", "classification": "unknown", "explanation": "Confirmed"}]},
        {"status": "error", "error": "offline"},
    ],
)
def test_writing_invalid_or_failed_visual_review_is_explicit_and_nondecisive(writing_materials, response):
    def call(**kwargs):
        if kwargs["module"] == "screening_writing.validation":
            return response
        return {
            "findings": [
                {
                    "block_id": "b1",
                    "quote": "Our methods is fast.",
                    "text": "Fix agreement",
                    "level": "definite_error",
                }
            ]
        }

    issues = []
    assert check_writing(writing_materials, call=call, issues=issues) == []
    assert "original PDF validation failed" in issues[0]


def test_writing_requires_an_existing_grounded_artifact(materials):
    Path(materials.markdown_path).unlink()
    with pytest.raises(ValueError, match="existing artifact"):
        check_writing(
            materials,
            call=lambda **kwargs: {
                "findings": [
                    {
                        "block_id": "b1",
                        "quote": "Our methods is fast.",
                        "text": "Use plural agreement.",
                        "level": "definite_error",
                    }
                ]
            },
        )


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


@pytest.fixture
def figure_materials(materials, tmp_path):
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
    return materials


def test_figures_send_printed_pixels_caption_all_references(figure_materials):
    materials = figure_materials
    path = Path(materials.figures[0].printed_crop_path)
    calls = []

    def call(**kwargs):
        calls.append(kwargs)
        return {
            "findings": [
                {
                    "category": "text_figure_consistency",
                    "disposition": "issue",
                    "text": "The caption refers to absent panel (c).",
                }
            ]
        }

    records = []
    result, issues = check_figures(materials, call=call, records=records)
    assert result == []
    assert records[0].context_status == "unavailable"
    assert records[0].crop_response == {
        "findings": [
            {
                "category": "text_figure_consistency",
                "disposition": "issue",
                "text": "The caption refers to absent panel (c).",
            }
        ]
    }
    assert any("original-page context unavailable" in issue for issue in issues)
    assert any("The caption refers to absent panel (c)." in issue for issue in issues)
    assert calls[0]["images"] == [str(path)]
    assert "Figure 1: Model." in calls[0]["prompt"]
    assert "Our methods is fast." in calls[0]["prompt"]
    assert "printed size" in calls[0]["system"]
    with pytest.raises(ValueError):
        check_figures(
            materials, call=lambda **kwargs: {"findings": [{"category": "colour", "text": "Use blue"}]}
        )


def test_figure_positive_observations_do_not_become_flaws(figure_materials):
    findings, issues = check_figures(
        figure_materials,
        call=lambda **kwargs: {
            "findings": [
                {
                    "category": "legibility",
                    "disposition": "clear",
                    "text": "Most node labels and bottom panel titles are legible at the printed size.",
                },
                {
                    "category": "text_figure_consistency",
                    "disposition": "clear",
                    "text": "图注与图中的中心节点 Christopher Nolan 一致。",
                },
                {
                    "category": "legibility",
                    "disposition": "issue",
                    "text": "The relation label Born-in_inv is too small to read at the printed size.",
                },
            ]
        },
    )
    assert not issues
    assert len(findings) == 1
    assert "Born-in_inv" in findings[0].text
    assert findings[0].evidence[0].direction == "flaw"


def test_ambiguous_parser_captions_block_contextual_flaws_and_preserve_legibility(figure_materials):
    figure_materials.figures[0].caption_ambiguous = True
    figure_materials.figures[0].caption = "Figure 4: Left result.\nFigure 5: Right result."

    def call(**kwargs):
        assert json.loads(kwargs["prompt"])["caption_ambiguous"] is True
        return {
            "findings": [
                {"category": category, "disposition": "issue", "text": "A candidate defect."}
                for category in ("self_containedness", "text_figure_consistency", "legibility")
            ]
        }

    findings, issues = check_figures(figure_materials, call=call)
    assert [finding.level for finding in findings] == ["legibility"]
    assert [issue for issue in issues if "parser caption assignment is ambiguous" in issue] == [
        f"figure_1: {category} unconfirmed because parser caption assignment is ambiguous: A candidate defect."
        for category in ("self_containedness", "text_figure_consistency")
    ]
    assert any("original-page context unavailable" in issue for issue in issues)


def test_uncertain_figure_observation_remains_an_explicit_issue(figure_materials):
    findings, issues = check_figures(
        figure_materials,
        call=lambda **kwargs: {
            "findings": [
                {
                    "category": "text_figure_consistency",
                    "disposition": "uncertain",
                    "text": "The crop cuts off panel (c), so its presence cannot be assessed.",
                }
            ]
        },
    )
    assert findings == []
    assert (
        "figure_1: text_figure_consistency check uncertain: "
        "The crop cuts off panel (c), so its presence cannot be assessed."
    ) in issues
    assert any("original-page context unavailable" in issue for issue in issues)


@pytest.mark.parametrize("disposition", [None, "unknown", True, 1, {}])
def test_figure_missing_or_invalid_disposition_is_rejected(figure_materials, disposition):
    row = {"category": "legibility", "text": "Labels look readable."}
    if disposition is not None:
        row["disposition"] = disposition
    with pytest.raises(ValueError, match="disposition"):
        check_figures(figure_materials, call=lambda **kwargs: {"findings": [row]})


@pytest.mark.parametrize("text", [None, "", " ", 1])
def test_figure_explanation_is_required_even_for_clear_results(figure_materials, text):
    with pytest.raises(ValueError, match="nonempty explanation"):
        check_figures(
            figure_materials,
            call=lambda **kwargs: {
                "findings": [{"category": "legibility", "disposition": "clear", "text": text}]
            },
        )


def test_malformed_figure_observation_is_rejected(figure_materials):
    with pytest.raises(ValueError, match="must be an object"):
        check_figures(figure_materials, call=lambda **kwargs: {"findings": ["readable"]})


def test_screening_records_unknown_figure_disposition_as_failure(figure_materials, tmp_path):
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
        if kwargs["module"] == "screening_figures":
            return {"findings": [{"category": "legibility", "text": "Labels look readable."}]}
        return {"findings": []}

    result = screen_paper(figure_materials, tmp_path, call=call)
    assert not any(finding.kind == "figure" for finding in result.findings)
    assert any("figures check failed" in issue and "disposition" in issue for issue in result.issues)
    assert "disposition" in (tmp_path / "screening.json").read_text(encoding="utf-8")


def test_reference_partial_processing_retains_coverage_issue(materials, tmp_path):
    materials.bibliography = materials.blocks * 2
    findings, issues = check_bibliography(
        materials,
        tmp_path,
        checker=lambda **kwargs: {"ok": True, "total_refs": 1, "issues": []},
    )
    assert findings == []
    assert len(issues) == 1 and "coverage needs review" in issues[0]
    assert "processed 1 entries from 2 parser bibliography blocks" in issues[0]


def test_missing_figure_crop_is_explicit(materials):
    materials.figures = [FigureMaterial(id="figure_1")]
    findings, issues = check_figures(materials, call=lambda **kwargs: pytest.fail("no pixels available"))
    assert findings == [] and "crop or location missing" in issues[0]


def test_table_checks_use_parsed_text_without_images(materials):
    materials.blocks.append(
        MaterialBlock(id="table1", text="Method | Time", kind="table", loc=ClaimLocation(page=2))
    )
    materials.markdown += "\nMethod | Time"
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")

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
    materials.markdown += "\n" + materials.bibliography[0].text
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")

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


@pytest.mark.parametrize(
    ("response", "status"),
    [
        ({"ok": True, "total_refs": 1, "issues": []}, "ok"),
        ({"ok": False, "error_message": "fixture checker unavailable"}, "failed"),
        ({"ok": True, "total_refs": 0, "issues": []}, "failed"),
        ({"ok": True, "total_refs": 1}, "failed"),
    ],
)
def test_reference_check_statistics_follow_actual_outcome(materials, tmp_path, response, status):
    materials.bibliography = materials.blocks

    def checker(**kwargs):
        run_stats.record_llm_call(usage={"input_tokens": 5}, model="reference-fixture")
        return response

    with run_stats.run_scope(tmp_path / "stats.json"), run_stats.module_scope("analysis"):
        check_bibliography(materials, tmp_path / "references", checker=checker)
        stats = run_stats.read()
        assert run_stats.current_module() == "analysis"
    row = stats["modules"]["reference_check"]
    assert row["status"] == status
    assert row["duration_sec"] > 0
    assert row["token_usage"]["input_tokens"] == 5
    assert stats["modules"]["analysis"]["token_usage"]["requests"] == 0


def test_reference_check_statistics_preserve_exception_and_skipped_states(materials, tmp_path):
    def checker(**kwargs):
        raise RuntimeError("fixture reference failure")

    with run_stats.run_scope(tmp_path / "stats.json"):
        check_bibliography(materials, tmp_path / "references", checker=checker)
        assert run_stats.read()["modules"]["reference_check"]["status"] == "skipped"
        materials.bibliography = materials.blocks
        with pytest.raises(RuntimeError, match="fixture reference failure"):
            check_bibliography(materials, tmp_path / "references", checker=checker)
        row = run_stats.read()["modules"]["reference_check"]
        assert row["status"] == "failed"
        assert row["duration_sec"] > 0
        assert row["warnings"] == ["fixture reference failure"]


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


def test_screening_persists_unconfirmed_writing_diagnostics(materials, tmp_path):
    from screening.stage import screen_paper

    def call(**kwargs):
        if kwargs["module"] == "screening.claims":
            return {
                "status": "ok",
                "claims": [
                    {
                        "text": "The method is fast.",
                        "source_block_id": "b1",
                        "source_quote": "Our methods is fast.",
                        "conditions": [{"id": "speed", "description": "method speed"}],
                        "needs": ["Experiments"],
                        "importance": "core",
                    }
                ],
            }
        assert kwargs["module"] == "screening_writing"
        return {
            "findings": [
                {
                    "block_id": "b1",
                    "quote": "Our methods is fast.",
                    "text": "Fix agreement",
                    "level": "definite_error",
                }
            ]
        }

    result = screen_paper(materials, tmp_path, call=call)
    assert not any(finding.kind == "writing" for finding in result.findings)
    assert any("original PDF page image unavailable" in issue for issue in result.issues)
    saved = json.loads((tmp_path / "screening.json").read_text(encoding="utf-8"))
    assert any("original PDF page image unavailable" in issue for issue in saved["issues"])
