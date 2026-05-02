from review.report.final_report import validate_final_report


def _valid_report() -> str:
    return """
## 1. Metadata
- **Title:** Example Paper
- **Task:** Example task
- **Code:** Not found in manuscript

## 2. Technical Positioning
Overview of the proposed method.

| Research domain | Method | Title-query retrieval | Uses ablation | Code-backed |
|---|---|---|---|---|
| Example domain | Related method | √ | × | × |
| This work | Proposed method | √ | √ | √ |

## 3. Claims
Paper scope: Example scope
Evaluation scope: Example evaluation

| Claim | Evidence | Assessment | Location |
|---|---|---|---|
| Improves benchmark score | Table 1 reports 0.91 accuracy. | Empirical support is present. | Table 1 |
| Defines objective | Equation 1 gives the loss. | The derivation has a paper anchor. | Equation 1 |
| Includes ablation | Table 2 removes one module. | Ablation evidence is present. | Table 2 |

## 4. Summary
Short summary.

Strengths:
- Clear extraction.

Weaknesses:
- Limited scope.

## 5. Experiment
### Main Result
Location: Table 1

| Task | Dataset | Metric | Best Baseline | Paper Result | Difference (Δ) |
|---|---|---|---|---|---|
| Classification | D1 | Accuracy | 0.88(Base) | 0.91 | +0.03 |

### Ablation Result
Location: Table 2

| Ablation Dimension | Configuration | Full Model | Paper Result | Difference (Δ) |
|---|---|---|---|---|
| Optimal setup | Full Model | 0.91 | 0.91 | 0 |
| Module | w/o M | 0.91 | 0.87 | -0.04 |
"""


def test_validate_final_report_accepts_factreview_contract():
    result = validate_final_report(
        markdown=_valid_report(),
        min_english_words=0,
        min_chinese_chars=0,
        force_english_output=False,
    )

    assert result.ok


def test_validate_final_report_rejects_bad_positioning_marks():
    report = _valid_report().replace(
        "| Related method | √ | × | × |", "| Related method | yes | no | maybe |"
    )

    result = validate_final_report(
        markdown=report,
        min_english_words=0,
        min_chinese_chars=0,
        force_english_output=False,
    )

    assert not result.ok
    assert result.reason == "final_report_logic_not_met"
    assert "niche-dimension cells" in result.message


def test_validate_final_report_rejects_missing_claim_assessment():
    report = _valid_report().replace(
        "| Claim | Evidence | Assessment | Location |", "| Claim | Evidence | Location |"
    )

    result = validate_final_report(
        markdown=report,
        min_english_words=0,
        min_chinese_chars=0,
        force_english_output=False,
    )

    assert not result.ok
    assert result.reason == "final_report_logic_not_met"
    assert "assessment" in result.message.lower()
