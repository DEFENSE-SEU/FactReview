"""Table context coverage remains distinct from crop results in delivery."""

import json
from pathlib import Path

import pytest
from pypdf import PdfReader

from review.report.v2 import write_review
from schemas.review import FinalReview
from screening.stage import screen_paper
from screening.tables import TableCheckRecord
from tests import test_pipeline_v2 as fixtures
from tests import test_table_vision_v2 as table_fixtures

offline_boundaries = fixtures.offline_boundaries
tiny_inputs = fixtures.tiny_inputs
table_paper = table_fixtures.table_paper
visual_materials = table_fixtures.visual_materials


@pytest.mark.parametrize("crash", [False, True])
def test_table_context_failure_preserves_crop_coverage(visual_materials, tmp_path, monkeypatch, crash):
    monkeypatch.setattr("screening.stage.extract_claims", lambda *a, **k: [])
    for name in ("check_writing", "check_tables"):
        monkeypatch.setattr(f"screening.stage.{name}", lambda *a, **k: [])
    for name in ("check_figures", "check_bibliography"):
        monkeypatch.setattr(f"screening.stage.{name}", lambda *a, **k: ([], []))

    def tables(materials, *, records, **kwargs):
        records.append(
            TableCheckRecord(table_id=materials.tables[0].id, status="checked", context_status="failed")
        )
        if crash:
            raise RuntimeError("Local table stage failure")
        records.append(
            TableCheckRecord(
                table_id=materials.tables[1].id, status="checked", context_status="not_requested"
            )
        )
        return [], []

    monkeypatch.setattr("screening.stage.check_visual_tables", tables)
    result = screen_paper(visual_materials, tmp_path / "screening")
    assert result.table_coverage["checked"] == (1 if crash else 2)
    assert result.table_context_coverage == {
        "total": 2,
        "checked": 0,
        "failed": 1,
        "unavailable": 0,
        "not_requested": 0 if crash else 1,
        "unrecorded": 1 if crash else 0,
    }
    saved = json.loads((tmp_path / "screening/screening.json").read_text("utf-8"))
    assert saved["table_context_coverage"] == result.table_context_coverage
    files = write_review(
        FinalReview(paper_key="tables", run_id="offline"),
        tmp_path / "report",
        table_coverage=result.table_coverage,
        table_context_coverage=result.table_context_coverage,
        render_pdf=True,
    )
    md = Path(files["markdown"]).read_text("utf-8")
    assert r"Table page\-context confirmation is incomplete: 1 failed" in md
    assert "Checked confirmations can remain uncertain" in md
    assert "have no new printed-size legibility judgment" in md
    pdf = " ".join(" ".join(page.extract_text().split()) for page in PdfReader(files["pdf"]).pages)
    assert "Table page-context coverage" in pdf
    assert f"{1 if crash else 0} unrecorded" in pdf


def test_table_context_is_forwarded_by_pipeline(tiny_inputs, monkeypatch):
    def tables(materials, *, records, **kwargs):
        for table in materials.tables:
            records.append(
                TableCheckRecord(table_id=table.id, status="checked", context_status="unavailable")
            )
        return [], ["Original table caption context unavailable"]

    monkeypatch.setattr("screening.stage.check_visual_tables", tables)
    summary, _, _, _ = fixtures.run_tiny(tiny_inputs, monkeypatch, render_pdf=False)
    assert summary["table_coverage"]["total"] > 0
    assert summary["table_context_coverage"]["unavailable"] == summary["table_coverage"]["total"]
    assert any("Table page-context confirmation is incomplete" in issue for issue in summary["issues"])
    report = Path(summary["outputs"]["report_markdown"]).read_text("utf-8")
    assert "Table page-context coverage" in report and "Original table caption context unavailable" in report
