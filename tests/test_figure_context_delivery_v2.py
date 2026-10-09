"""Page-context coverage survives screening failures and reaches delivered reports."""

import json
from pathlib import Path

import pytest
from pypdf import PdfReader

from review.report.v2 import write_review
from schemas.review import FinalReview
from screening.figures import FigureCheckRecord
from screening.stage import screen_paper
from tests import test_figure_failure_isolation as visual_fixtures
from tests import test_pipeline_v2 as fixtures

offline_boundaries = fixtures.offline_boundaries
tiny_inputs = fixtures.tiny_inputs
visual_materials = visual_fixtures.visual_materials


@pytest.mark.parametrize("crash", [False, True])
def test_context_failure_counts_are_independent_of_successful_crops(
    visual_materials, tmp_path, monkeypatch, crash
):
    monkeypatch.setattr("screening.stage.extract_claims", lambda *a, **k: [])
    for name in ("check_writing", "check_tables"):
        monkeypatch.setattr(f"screening.stage.{name}", lambda *a, **k: [])
    for name in ("check_visual_tables", "check_bibliography"):
        monkeypatch.setattr(f"screening.stage.{name}", lambda *a, **k: ([], []))

    def figures(materials, *, records, **kwargs):
        records.append(FigureCheckRecord(figure_id="figure_1", status="checked", context_status="failed"))
        if crash:
            raise RuntimeError("Local figure stage failure")
        records.extend(
            [
                FigureCheckRecord(figure_id="figure_2", status="checked", context_status="unavailable"),
                FigureCheckRecord(figure_id="figure_3", status="checked", context_status="not_requested"),
            ]
        )
        return [], []

    monkeypatch.setattr("screening.stage.check_figures", figures)
    result = screen_paper(visual_materials, tmp_path / "screening")
    assert result.figure_coverage["checked"] == (1 if crash else 3)
    assert result.figure_context_coverage == {
        "total": 3,
        "checked": 0,
        "failed": 1,
        "unavailable": 0 if crash else 1,
        "not_requested": 0 if crash else 1,
        "unrecorded": 2 if crash else 0,
    }
    saved = json.loads((tmp_path / "screening/screening.json").read_text("utf-8"))
    assert saved["figure_context_coverage"] == result.figure_context_coverage
    files = write_review(
        FinalReview(paper_key="context", run_id="offline"),
        tmp_path / "report",
        figure_coverage=result.figure_coverage,
        figure_context_coverage=result.figure_context_coverage,
        render_pdf=True,
    )
    md = Path(files["markdown"]).read_text("utf-8")
    assert r"Figure page\-context confirmation is incomplete: 1 failed" in md
    assert "Checked confirmations can remain uncertain" in md
    assert "labels outside that crop have no legibility result" in md
    pdf_text = " ".join(" ".join(page.extract_text().split()) for page in PdfReader(files["pdf"]).pages)
    assert "Figure page-context coverage" in pdf_text
    assert "1 failed" in pdf_text and f"{2 if crash else 0} unrecorded" in pdf_text


def test_context_coverage_reaches_pipeline_summary_and_final_report(tiny_inputs, monkeypatch):
    def figures(materials, *, records, **kwargs):
        for figure in materials.figures:
            records.append(FigureCheckRecord(figure_id=figure.id, status="checked", context_status="failed"))
        return [], ["Selected page-context check failed"]

    monkeypatch.setattr("screening.stage.check_figures", figures)
    summary, _, _, _ = fixtures.run_tiny(tiny_inputs, monkeypatch, render_pdf=False)
    assert summary["figure_coverage"]["checked"] == 1
    assert summary["figure_context_coverage"]["failed"] == 1
    assert any("page-context confirmation is incomplete" in issue for issue in summary["issues"])
    report = Path(summary["outputs"]["report_markdown"]).read_text("utf-8")
    assert "Figure page-context coverage" in report
    assert r"Selected page\-context check failed" in report
