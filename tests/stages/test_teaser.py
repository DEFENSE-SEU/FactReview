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


def test_teaser_claim_selection_prioritizes_novelty_and_weak_statuses(tmp_path: Path) -> None:
    md = _REVIEW_MD_FOR_TEASER.replace(
        "| TinyMethod is leading on FB15k. | Table 1: 0.355 vs 0.30. | ok | Supported | Table 1 |\n",
        (
            "| Routine method claim. | Section 3. | ok | Supported | Section 3 |\n"
            "| TinyMethod is the first framework for X. | Related Work comparison. Missing: external check. | partial | Partially supported | Intro |\n"
            "| Routine artifact claim. | Appendix A. | ok | Supported | Appendix A |\n"
            "| TinyMethod beats all baselines. | Table 1 contradicts strongest baseline. | conflict | In conflict | Table 1 |\n"
        ),
    )
    md_path = tmp_path / "review.md"
    md_path.write_text(md, encoding="utf-8")

    result = generate_teaser_figure(md_path, output_dir=tmp_path, generate_image=False)

    selected_block = result.prompt.split("Audited claims table:", 1)[0]
    assert "TinyMethod is the first framework for X" in selected_block
    assert "TinyMethod beats all baselines" in selected_block


def test_teaser_uses_core_claims_when_provided(tmp_path: Path) -> None:
    md_path = tmp_path / "review.md"
    md_path.write_text(_REVIEW_MD_FOR_TEASER, encoding="utf-8")

    result = generate_teaser_figure(
        md_path,
        output_dir=tmp_path,
        generate_image=False,
        core_claims=[
            {"id": "core_claim_01", "text": "TinyMethod is positioned as a first framework for X."},
            {"id": "core_claim_02", "text": "TinyMethod combines retrieval with skill execution."},
            {"id": "core_claim_03", "text": "TinyMethod improves the headline benchmark result."},
        ],
    )

    selected_block = result.prompt.split("Audited claims table:", 1)[0]
    assert "Core claim" in selected_block
    assert "TinyMethod is positioned as a first framework for X" in selected_block
    assert "TinyMethod is leading on FB15k" not in selected_block
