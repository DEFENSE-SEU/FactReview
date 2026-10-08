"""Full pipeline reports expose visual coverage and incomplete usage accounting."""

import json
from pathlib import Path
from types import SimpleNamespace

import fitz
from pypdf import PdfReader

from common import run_stats
from llm.client import LLMConfig
from pipeline_v2 import run_v2_pipeline
from preprocessing.parse.mineru_adapter import MineruParseResult
from review.report.v2 import write_review
from schemas.review import FinalReview
from verification.contracts import BranchResult


def test_real_pipeline_preserves_mixed_visual_coverage_and_cost_limitations(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "screening.claims.resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None)
    )
    monkeypatch.setattr(
        "screening.checks.resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None)
    )
    monkeypatch.setattr(
        "screening.checks.resolve_vlm_config", lambda **kwargs: LLMConfig("mock", "fixture", None, None)
    )
    text = "The method is stable."
    rows = [{"type": "text", "text": text, "page_idx": 0}]
    for index in range(1, 4):
        row = {"type": "image", "image_caption": f"Figure {index}: Result.", "page_idx": 0}
        if index < 3:
            row.update(bbox=[10, 100, 110, 150], bbox_space="pdf_points")
        rows.append(row)
    markdown = "\n\n".join(row.get("text", row.get("image_caption", "")) for row in rows)
    pdf_path = tmp_path / "paper.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page(width=300, height=300)
        page.insert_text((10, 30), text)
        page.draw_rect((10, 100, 110, 150))
        pdf.save(pdf_path)

    class Parser:
        async def parse_pdf(self, **kwargs):
            return MineruParseResult(markdown, rows, None, "mock", {}, "mock-mineru")

    seen = []

    def call(**kwargs):
        module = kwargs["module"]
        if module == "screening_figures":
            figure_id = json.loads(kwargs["prompt"])["figure_id"]
            seen.append(figure_id)
            failed = figure_id == "figure_2"
            run_stats.record_llm_call(
                module=module,
                provider="mock",
                model="fixture",
                usage={},
                failed=failed,
                image_count=1,
                prompt=kwargs["prompt"],
                system=kwargs["system"],
                response_text='{"findings": []}',
            )
            return {"status": "error", "error": "fixture visual timeout"} if failed else {"findings": []}
        run_stats.record_llm_call(
            module=module, provider="mock", model="fixture", usage={"input_tokens": 5, "output_tokens": 2}
        )
        if module == "screening.claims":
            return {
                "status": "ok",
                "claims": [
                    {
                        "text": text,
                        "source_block_id": "block_1",
                        "source_quote": text,
                        "conditions": [{"id": "stability", "description": "method stability"}],
                        "needs": ["Experiments"],
                        "importance": "core",
                    }
                ],
            }
        return {"findings": []}

    summary = run_v2_pipeline(
        SimpleNamespace(
            paper_pdf=str(pdf_path),
            paper_key="visual",
            run_root=str(tmp_path / "runs"),
            submission_deadline="2021-01-31",
            run_execution=False,
        ),
        parser=Parser(),
        call=call,
        branches={"Experiments": lambda *args: BranchResult()},
        global_literature=lambda *args: BranchResult(),
        render_pdf=True,
    )
    assert summary["stages"]["report"] == "ok", summary["stage_errors"]
    assert seen == ["figure_1", "figure_2"]
    assert summary["figure_coverage"] == {"total": 3, "checked": 1, "failed": 1, "unavailable": 1}
    usage = summary["model_usage"]
    assert usage["failed_requests"] == 1
    assert usage["unavailable_usage_requests"] == 2
    assert usage["image_count"] == 2
    assert usage["estimated"] is True
    assert any("image token cost is unknown" in issue for issue in summary["issues"])
    assert any("Figure screening is incomplete" in issue for issue in summary["issues"])
    persisted = json.loads(
        (Path(summary["run_dir"]) / "full_pipeline_summary.json").read_text(encoding="utf-8")
    )
    assert persisted["figure_coverage"] == summary["figure_coverage"]
    assert persisted["model_usage"] == usage
    assert Path(summary["outputs"]["visual_calls"]).is_dir()
    report = Path(summary["outputs"]["report_markdown"]).read_text(encoding="utf-8")
    assert "| Total figures | Checked | Failed | Unavailable |" in report
    assert "| 3 | 1 | 1 | 1 |" in report
    assert "Figure screening is incomplete" in report
    assert "image token cost is unknown" in report
    assert "Provider\\-reported token usage is unavailable" in report
    assert "Model call accounting" in report
    assert sum(line.startswith("## ") for line in report.splitlines()) == 4
    parsed = json.loads(Path(summary["outputs"]["report_json"]).read_text(encoding="utf-8"))
    assert parsed["findings"] == []
    assert "Figure screening coverage" in parsed["review_markdown"]
    pdf_text = "\n".join(page.extract_text() for page in PdfReader(summary["outputs"]["report_pdf"]).pages)
    assert "Figure screening coverage" in pdf_text
    assert "image token cost is unknown" in pdf_text


def test_report_explicitly_exposes_zero_visual_inputs(tmp_path):
    output = write_review(
        FinalReview(paper_key="no-figures", run_id="fixture"),
        tmp_path,
        render_pdf=False,
        figure_coverage={"total": 0, "checked": 0, "failed": 0, "unavailable": 0},
    )
    report = Path(output["markdown"]).read_text(encoding="utf-8")
    assert "| 0 | 0 | 0 | 0 |" in report
    assert "No figure inputs were available for visual checks" in report


def test_failed_only_model_usage_is_unavailable_in_pdf(tmp_path):
    from pipeline_v2 import _model_usage

    with run_stats.run_scope(tmp_path / "stats.json"):
        run_stats.record_llm_call(module="analysis", failed=True, image_count=1)
        usage = _model_usage(run_stats.with_totals(run_stats.read()))
    assert usage["unavailable"] is True
    output = write_review(
        FinalReview(paper_key="failed-vlm", run_id="fixture"), tmp_path / "report", token_usage=usage
    )
    report = Path(output["markdown"]).read_text(encoding="utf-8")
    assert "| 1 | 1 | 1 | 1 |" in report
    assert "Token totals are incomplete" in report
    pdf_text = "\n".join(page.extract_text() for page in PdfReader(output["pdf"]).pages)
    assert "Unavailable" in pdf_text
    assert "Input 0 | Output 0 | Total 0" not in pdf_text
