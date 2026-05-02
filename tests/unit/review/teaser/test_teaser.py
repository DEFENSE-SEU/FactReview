from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

import review.teaser.teaser as teaser_figure_module
from review.teaser.teaser import (
    _extract_inline_image_bytes,
    build_teaser_figure_prompt_from_latest_extraction,
    extract_teaser_figure_payload,
    generate_teaser_figure,
)

SAMPLE_MARKDOWN = """## 1. Metadata
- Title: Demo Paper
- Task: Node classification

## 2. Technical Positioning
Figure 1. Model comparison overview.
![overview](figures/overview.png)

| Model | Setting |
| --- | --- |
| Ours | Full |

## 3. Claims
(Status legend: Supported, Inconclusive, In conflict)

| Claim | Evidence | Status |
| --- | --- | --- |
| Claim A | Evidence A | Supported |
| Claim B | Evidence B | Inconclusive |
| Claim C | Evidence C | In conflict |

## 4. Summary
This paper proposes a practical method.
Strengths:
- Clear empirical gains
Weaknesses:
- Limited ablations

## 5. Experiment
### Main Result
Location: Table 1

| Dataset | Score |
| --- | --- |
| Cora | 85.0 |

### Ablation Result
Location: Table 2

| Variant | Score |
| --- | --- |
| w/o X | 82.1 |
"""


def test_build_teaser_prompt_from_latest_extraction(tmp_path: Path) -> None:
    latest_md = tmp_path / "latest_extraction.md"
    latest_md.write_text(SAMPLE_MARKDOWN, encoding="utf-8")

    prompt = build_teaser_figure_prompt_from_latest_extraction(latest_md)

    assert "Title: Demo Paper" in prompt
    assert "Task: Node classification" in prompt
    assert "**Claim:** **Claim A**" in prompt
    assert "Main result location: Table 1" in prompt
    assert "[Template Geometry]" in prompt


def test_generate_teaser_figure_without_gemini_key_writes_prompt(tmp_path: Path, monkeypatch) -> None:
    latest_md = tmp_path / "latest_extraction.md"
    latest_md.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
    monkeypatch.setattr(teaser_figure_module, "_ensure_env_loaded", lambda: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)

    result = generate_teaser_figure(latest_md, output_dir=tmp_path / "artifacts")

    assert result.status == "prompt_only"
    assert result.used_gemini_api is False
    assert Path(result.prompt_path).exists()
    assert result.prompt
    assert "Gemini web app" in result.message


def test_generate_teaser_figure_ignores_openai_key_without_image_base_url(
    tmp_path: Path, monkeypatch
) -> None:
    latest_md = tmp_path / "latest_extraction.md"
    latest_md.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
    monkeypatch.setattr(teaser_figure_module, "_ensure_env_loaded", lambda: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "regular-openai-key")

    result = generate_teaser_figure(latest_md, output_dir=tmp_path / "artifacts")

    assert result.status == "prompt_only"
    assert result.used_gemini_api is False
    assert Path(result.prompt_path).read_text(encoding="utf-8") == result.prompt


def test_generate_teaser_figure_prompt_only_mode_returns_manual_prompt(tmp_path: Path, monkeypatch) -> None:
    latest_md = tmp_path / "latest_extraction.md"
    latest_md.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
    monkeypatch.setattr(teaser_figure_module, "_ensure_env_loaded", lambda: None)
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")

    result = generate_teaser_figure(
        latest_md,
        output_dir=tmp_path / "artifacts",
        generate_image=False,
    )

    assert result.status == "prompt_only"
    assert result.used_gemini_api is False
    assert result.prompt
    assert "Gemini web app" in result.message


def test_extract_inline_image_bytes_finds_nested_base64() -> None:
    raw = b"fake-png-bytes"
    payload = {
        "predictions": [
            {
                "nested": {
                    "bytesBase64Encoded": base64.b64encode(raw).decode("ascii"),
                }
            }
        ]
    }

    assert _extract_inline_image_bytes(payload) == raw


def test_generate_teaser_figure_retries_with_validation_feedback(tmp_path: Path, monkeypatch) -> None:
    latest_md = tmp_path / "latest_extraction.md"
    latest_md.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
    monkeypatch.setattr(teaser_figure_module, "_ensure_env_loaded", lambda: None)
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")
    monkeypatch.setenv("TEASER_TEMPLATE_STRICT", "true")
    monkeypatch.setenv("TEASER_TEMPLATE_MAX_ATTEMPTS", "3")

    image = Image.new("RGB", (8, 8), color=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    png_bytes = buffer.getvalue()

    prompts: list[str] = []

    def fake_call_gemini_image_api(**kwargs):
        prompts.append(kwargs["prompt"])
        return {"imageBytes": base64.b64encode(png_bytes).decode("ascii")}

    validation_results = iter(
        [
            {
                "passed": False,
                "score": 0.52,
                "threshold": 0.78,
                "color_similarity": 0.6,
                "edge_similarity": 0.4,
                "region_scores": [],
                "prompt_feedback": [
                    "Keep the task label as a light rounded badge anchored at the top-right corner inside the header area."
                ],
                "reason": "template_similarity_below_threshold",
            },
            {
                "passed": True,
                "score": 0.89,
                "threshold": 0.78,
                "color_similarity": 0.9,
                "edge_similarity": 0.88,
                "region_scores": [],
                "prompt_feedback": [],
                "reason": "passed",
            },
        ]
    )

    monkeypatch.setattr(teaser_figure_module, "_call_gemini_image_api", fake_call_gemini_image_api)
    monkeypatch.setattr(
        teaser_figure_module, "_validate_generated_teaser_image", lambda _path: next(validation_results)
    )

    result = generate_teaser_figure(latest_md, output_dir=tmp_path / "artifacts")

    assert result.status == "generated"
    assert len(prompts) == 2
    assert "[Retry Corrections]" in prompts[1]
    assert "top-right corner inside the header area" in prompts[1]
    assert Path(result.image_path).exists()
    assert "0.890" in result.message
    assert (tmp_path / "artifacts" / "teaser_figure_validation.json").exists()


def test_extract_teaser_payload_compresses_main_and_ablation_tables() -> None:
    markdown = """## 1. Metadata
- Title: Demo Paper
- Task: Multi-task

## 2. Technical Positioning
Figure 1. Overview.
![overview](figures/overview.png)

## 3. Claims
| Claim | Evidence | Status |
| --- | --- | --- |
| C1 | E1 | Supported |

## 4. Summary
Short summary.
Strengths:
- Strong
Weaknesses:
- Limited

## 5. Experiment
### Main Result
Location: Table Main

| Task | Dataset | Metric | Best Baseline | Paper Result | Difference (Δ) |
| --- | --- | --- | --- | --- | --- |
| Machine translation | WMT14 EN-DE | BLEU | 26.0 | 27.0 | +1.0 |
| Machine translation | WMT14 EN-DE | chrF | 58.0 | 58.2 | +0.2 |
| Machine translation | WMT14 EN-FR | BLEU | 40.0 | 41.2 | +1.2 |
| Parsing | WSJ | F1 | 91.0 | 92.0 | +1.0 |

### Ablation Result
Location: Table Ablation

| Ablation Dimension | Configuration | Full Model | Paper Result | Difference (Δ) |
| --- | --- | --- | --- | --- |
| Attention heads | h=1 | 26.4 | 24.9 | -1.5 |
| Attention heads | h=4 | 26.4 | 25.5 | -0.9 |
| Dropout | p=0.0 | 26.4 | 24.6 | -1.8 |
| Dropout | p=0.2 | 26.4 | 25.5 | -0.9 |
"""
    payload = extract_teaser_figure_payload(markdown)

    assert payload.experiment_main_table is not None
    assert payload.experiment_ablation_table is not None

    main_rows = payload.experiment_main_table.rows
    # Keep one best row per (Task, Dataset): EN-DE + EN-FR + WSJ => 3 rows.
    assert len(main_rows) == 3
    en_de_rows = [r for r in main_rows if r[1] == "WMT14 EN-DE"]
    assert len(en_de_rows) == 1
    assert en_de_rows[0][2] == "BLEU"

    ablation_rows = payload.experiment_ablation_table.rows
    # Keep one strongest row per Ablation Dimension: Attention heads + Dropout => 2 rows.
    assert len(ablation_rows) == 2
    heads_rows = [r for r in ablation_rows if r[0] == "Attention heads"]
    dropout_rows = [r for r in ablation_rows if r[0] == "Dropout"]
    assert len(heads_rows) == 1 and heads_rows[0][1] == "h=1"
    assert len(dropout_rows) == 1 and dropout_rows[0][1] == "p=0.0"


def test_ablation_selection_uses_reference_full_model_not_rowwise_max() -> None:
    markdown = """## 1. Metadata
- Title: Demo Paper
- Task: Demo

## 2. Technical Positioning
Figure 1. Overview.
![overview](figures/overview.png)

## 3. Claims
| Claim | Evidence | Status |
| --- | --- | --- |
| C1 | E1 | Supported |

## 4. Summary
Summary.
Strengths:
- S
Weaknesses:
- W

## 5. Experiment
### Main Result
Location: Table Main
| Task | Dataset | Metric | Best Baseline | Paper Result | Difference (Δ) |
| --- | --- | --- | --- | --- | --- |
| T | D | BLEU | 26.0 | 27.0 | +1.0 |

### Ablation Result
Location: Table Ablation
| Ablation Dimension | Configuration | Full Model | Paper Result |
| --- | --- | --- | --- |
| Attention heads | base | 26.4 | 26.4 |
| Attention heads | h=1 | 26.4 | 24.9 |
| Attention heads | h=4 | 30.0 | 26.8 |
"""
    payload = extract_teaser_figure_payload(markdown)
    assert payload.experiment_ablation_table is not None
    rows = payload.experiment_ablation_table.rows
    assert len(rows) == 1
    # Must pick h=1: effect vs reference full model 26.4 is 1.5;
    # h=4 should not win just because its row-wise full model is 30.0.
    assert rows[0][1] == "h=1"


def test_prompt_claims_are_bold() -> None:
    payload = extract_teaser_figure_payload(SAMPLE_MARKDOWN)
    prompt = teaser_figure_module.build_teaser_figure_prompt(payload)
    assert "**Claim:** **Claim A**" in prompt
