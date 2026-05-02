from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("agents")
pytest.importorskip("fitz")
pytest.importorskip("openai")
pytest.importorskip("reportlab")

from agent_runtime import agent_tools, runner  # noqa: E402


def test_plain_positioning_dimensions_are_not_replaced_by_semantic_titles(tmp_path: Path) -> None:
    (tmp_path / "semantic_scholar_candidates.json").write_text(
        json.dumps(
            {
                "success": True,
                "papers": [
                    {"id": "R1", "title": "Moonshine Fast Distillation", "year": 2024},
                    {"id": "R2", "title": "SwishNet Fast", "year": 2023},
                ],
            }
        ),
        encoding="utf-8",
    )
    markdown = """## 2. Technical Positioning
Caption.

| Research domain | Method | Ensemble compression | Temperature soft targets |
|---|---|---|---|
| This Work | Knowledge distillation | √ | √ |
"""

    normalized = runner._compact_technical_positioning_reference_labels(markdown, job_dir=tmp_path)

    assert "Ensemble compression" in normalized
    assert "Temperature soft targets" in normalized
    assert "Moonshine Fast" not in normalized


def test_this_work_method_prefers_model_supplied_domain_when_method_cell_is_self_marker() -> None:
    markdown = """## 1. Metadata
- **Title:** Distilling the Knowledge in a Neural Network

## 2. Technical Positioning
Caption mentions Baseline and ImageNet.

| Research domain | Method | Temperature soft targets |
|---|---|---|
| Neural network distillation | This Work | √ |
"""

    normalized = runner._normalize_technical_positioning_layout(markdown)

    assert "| This Work | Neural network distillation | √ |" in normalized
    assert "| This Work | Baseline |" not in normalized
    assert "| This Work | ImageNet |" not in normalized


def test_experiment_child_headings_are_demoted_before_validation() -> None:
    markdown = """## 5. Experiment
## Main Result
Location: Table 1

| Task | Dataset | Metric | Best Baseline | Paper Result | Difference (Δ) |
|---|---|---|---|---|---|
| Classification | D1 | Accuracy | 0.8 | 0.9 | +0.1 |

## Ablation Result
Location: Table 2

| Ablation Dimension | Configuration | Full Model | Paper Result | Difference (Δ) |
|---|---|---|---|---|
| Module | w/o A | 0.9 | 0.8 | -0.1 |
"""

    normalized = runner._demote_experiment_child_headings(markdown)

    assert "## Main Result" not in normalized
    assert "## Ablation Result" not in normalized
    assert "### Main Result" in normalized
    assert "### Ablation Result" in normalized


def test_ablation_full_model_text_is_preserved_during_table_normalization() -> None:
    block = """### Ablation Result
Location: Table 3

| Ablation Dimension | Configuration | Full Model | Paper Result | Difference (Δ) |
|---|---|---|---|---|
| Architecture | Removed layers 3,4 | Full model: Val Top-5 16.5 | Val Top-5 22.1 | +5.6 |
"""

    normalized, _statuses = runner._normalize_experiment_tables_in_block(block)

    assert "Full model: Val Top-5 16.5" in normalized
    assert "| Architecture | Removed layers 3,4 | 22.1 |" not in normalized


def test_section_builder_normalizes_experiment_subsection_heading_levels() -> None:
    markdown = agent_tools._build_final_report_markdown_from_sections(
        {
            "metadata": "- **Title:** Example",
            "technical_positioning": "Positioning.",
            "claims": "Claims.",
            "summary": "Summary.",
            "experiment": "## Main Result\nLocation: Table 1\n\n## Ablation Result\nLocation: Table 2",
        }
    )

    assert "## 5. Experiment\n### Main Result" in markdown
    assert "\n## Main Result" not in markdown
    assert "\n## Ablation Result" not in markdown
