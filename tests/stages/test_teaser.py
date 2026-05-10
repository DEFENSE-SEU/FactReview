"""Teaser stage tests.

The teaser stage builds a prompt from the report's markdown and (optionally)
calls Gemini for image generation. We exercise the prompt-only path here —
the Gemini call is gated behind ``GEMINI_API_KEY`` and would be the
``requires_llm`` branch.
"""

from __future__ import annotations

from pathlib import Path

from review.teaser.teaser import generate_teaser_figure

_REVIEW_MD_FOR_TEASER = """## 1. Summary
TinyMethod tackles X.

## 2. Technical Positioning
| Research domain | Method | A | B |
| --- | --- | --- | --- |
| Other | Baseline | × | √ |
| This Work | TinyMethod | √ | √ |

## 3. Claims
| Claim | Evidence | Assessment | Status | Location |
|---|---|---|---|---|
| TinyMethod is leading on FB15k. | Table 1: 0.355 vs 0.30. | ok | Supported | Table 1 |

## 4. Summary
The system improves on baselines.

**Strengths:**
- Clear ablations.

**Weaknesses:**
- Limited to one benchmark.

## 5. Experiment

Main Result:
Location: Table 1
| Method | MRR |
|---|---|
| Baseline | 0.30 |
| TinyMethod | 0.355 |

Ablation Result:
Location: Table 2
| Dim | Cfg | Full | Paper | Δ |
|---|---|---|---|---|
| A | no | 1.0 | 0.5 | -0.5 |
"""


def test_generate_teaser_returns_prompt_only_when_image_generation_disabled(
    tmp_path: Path,
) -> None:
    md_path = tmp_path / "review.md"
    md_path.write_text(_REVIEW_MD_FOR_TEASER, encoding="utf-8")

    result = generate_teaser_figure(md_path, output_dir=tmp_path, generate_image=False)

    assert result.status == "prompt_only"
    assert result.image_path == ""
    # The prompt file is the deliverable when Gemini is disabled.
    assert result.prompt_path
    assert Path(result.prompt_path).exists()
    assert result.prompt, "prompt must be non-empty so the user can paste it into Gemini"
    assert result.source_markdown_path == str(md_path.resolve())


def test_generate_teaser_falls_back_to_prompt_only_without_api_key(tmp_path: Path, monkeypatch) -> None:
    md_path = tmp_path / "review.md"
    md_path.write_text(_REVIEW_MD_FOR_TEASER, encoding="utf-8")

    # Ensure no key is visible to the resolver regardless of the host env.
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    result = generate_teaser_figure(md_path, output_dir=tmp_path, generate_image=True, gemini_api_key="")

    # generate_image=True but no API key → still prompt-only, no exception.
    assert result.status == "prompt_only"
    assert result.image_path == ""
    assert "API key" in result.message or "image API" in result.message
