"""End-to-end pipeline smoke tests.

Marked ``e2e`` and skipped by default; run with ``pytest -m e2e``. The first
test catches the most common refactoring breakage — a stage_runner module
that no longer imports — without setting up the heavy bridge_state machinery
the real pipeline needs. The second test exercises the report → audit →
teaser tail (the path users iterate on most often) end-to-end with stubbed
LLMs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.e2e


def test_all_stage_runners_import_and_expose_run_entry() -> None:
    # If any stage's public ``run_*_stage`` symbol disappears or its imports
    # break, every per-stage CLI in scripts/ stops working — pipeline_full
    # also imports them at module load. Catching this in a single smoke is
    # cheaper than a per-stage check.
    from fact_generation.execution.stage_runner import run_execution_stage
    from fact_generation.positioning.stage_runner import run_positioning_stage
    from fact_generation.refcheck.stage_runner import run_refcheck_stage
    from preprocessing.claim_extract.stage_runner import run_claim_extract_stage
    from preprocessing.parse.stage_runner import run_parse_stage
    from review.report.stage_runner import run_report_stage
    from review.teaser.stage_runner import run_teaser_stage

    for fn in (
        run_parse_stage,
        run_claim_extract_stage,
        run_refcheck_stage,
        run_positioning_stage,
        run_execution_stage,
        run_report_stage,
        run_teaser_stage,
    ):
        assert callable(fn)

    # pipeline_full glues them together; failing to import means the
    # orchestrator can't start.
    from pipeline_full import run_full_pipeline

    assert callable(run_full_pipeline)


def test_audit_then_teaser_tail_runs_end_to_end(tmp_path: Path) -> None:
    """Report-audit → teaser is the chain users iterate on most often.

    This test feeds a minimal review markdown through ``audit_review_markdown``
    (with a stubbed LLM) and then through ``generate_teaser_figure`` (with
    image generation disabled so no Gemini key is needed). Together they
    exercise the markdown round-trip the report stage_runner performs.
    """
    from review.report.claim_audit import audit_review_markdown
    from review.teaser.teaser import generate_teaser_figure

    review_md = (
        "## 1. Summary\nTinyMethod tackles X.\n\n"
        "## 2. Technical Positioning\n"
        "| Research domain | Method | A | B |\n"
        "| --- | --- | --- | --- |\n"
        "| Other | Baseline | × | √ |\n"
        "| This Work | TinyMethod | √ | √ |\n\n"
        "## 3. Claims\n"
        "| Claim | Evidence | Assessment | Status | Location |\n"
        "|---|---|---|---|---|\n"
        "| TinyMethod is leading on FB15k. | Table 1: 0.355 vs 0.30. | "
        'ok | <span style="color: green;">✓ Supported</span> | Table 1 |\n'
        "## 4. Summary\nThe system improves on baselines.\n\n"
        "**Strengths:**\n- Clear ablations.\n\n"
        "**Weaknesses:**\n- Limited to one benchmark.\n\n"
        "## 5. Experiment\n"
        "Main Result:\nLocation: Table 1\n"
        "| Method | MRR |\n|---|---|\n| Baseline | 0.30 |\n| TinyMethod | 0.355 |\n\n"
        "Ablation Result:\nLocation: Table 2\n"
        "| Dim | Cfg | Full | Paper | Δ |\n|---|---|---|---|---|\n"
        "| A | no | 1.0 | 0.5 | -0.5 |\n"
    )

    def llm_stub(prompt: str) -> dict[str, Any]:
        return {
            "verdicts": [{"id": 0, "verdict": "partially_supported", "reason": "small gap"}],
            "ablation_missing_components": [],
        }

    audited_md, outcome = audit_review_markdown(review_md, llm_call=llm_stub)
    assert outcome.claim_results[0].final_status == "partially supported"
    assert "⚠ Partially supported" in audited_md

    audited_path = tmp_path / "final_review.md"
    audited_path.write_text(audited_md, encoding="utf-8")

    teaser = generate_teaser_figure(audited_path, output_dir=tmp_path, generate_image=False)
    assert teaser.status == "prompt_only"
    assert teaser.prompt
    assert Path(teaser.prompt_path).exists()
