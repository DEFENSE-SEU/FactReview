from __future__ import annotations

from review.report.final_report_audit import audit_and_refine_final_report


def _valid_report(summary_text: str) -> str:
    return f"""
## 1. Metadata
- **Title**: Attention Is All You Need

## 2. Technical Positioning
| Research domain | Method | Encoder only |
| --- | --- | --- |
| Machine translation | Transformer | √ |

## 3. Claims
| Claim | Evidence | Assessment | Location |
| --- | --- | --- | --- |
| The model removes recurrence. | The paper states the model uses attention without recurrence. | supported | Abstract |
| The paper evaluates on WMT 2014 En-De. | WMT 2014 En-De appears in the experiments section. | supported | Section 5 |
| The base model uses 6 layers. | The architecture section lists 6 layers. | supported | Section 3 |

## 4. Summary
{summary_text}

## 5. Experiment
### Main Result
| Dataset | Metric | Value |
| --- | --- | --- |
| WMT 2014 En-De | BLEU | 27.3 |

### Ablation Result
| Setting | Metric | Value |
| --- | --- | --- |
| Base | BLEU | 27.3 |
""".strip()


def test_audit_stops_when_no_issues(monkeypatch):
    report = _valid_report("This review is already aligned with the paper.")

    def fake_llm_json(*, prompt, system, cfg):
        assert "Source paper markdown" in prompt
        return {
            "audit_summary": "No material mismatch found.",
            "issues": [],
        }

    monkeypatch.setattr("review.report.final_report_audit.llm_json", fake_llm_json)

    result = audit_and_refine_final_report(
        final_markdown=report,
        source_markdown="Attention is used without recurrence.",
        max_iterations=3,
        max_source_chars=20000,
        max_review_chars=12000,
        model="gpt-5.2",
        min_english_words=0,
        min_chinese_chars=0,
        force_english_output=True,
    )

    assert result.enabled is True
    assert result.applied is False
    assert result.iterations_run == 1
    assert result.stop_reason == "no_issues_found"
    assert result.final_markdown == report


def test_audit_applies_revision_and_runs_next_check(monkeypatch):
    original = _valid_report("The review says the paper reports 28.4 BLEU on WMT 2014 En-De.")
    revised = _valid_report("The review says the paper reports 27.3 BLEU on WMT 2014 En-De.")
    calls = {"count": 0}

    def fake_llm_json(*, prompt, system, cfg):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "audit_summary": "One numeric mismatch found.",
                "issues": [
                    {
                        "problem_type": "wrong_number",
                        "severity": "high",
                        "section": "4. Summary",
                        "review_excerpt": "28.4 BLEU",
                        "paper_evidence": "The paper reports 27.3 BLEU on WMT 2014 En-De.",
                        "suggested_fix": "Change 28.4 to 27.3.",
                        "should_fix": True,
                    }
                ],
            }
        if calls["count"] == 2:
            assert "Audit issues:" in prompt
            return {
                "revision_summary": "Updated the BLEU number in the summary only.",
                "revised_markdown": revised,
            }
        return {"audit_summary": "No remaining mismatch.", "issues": []}

    monkeypatch.setattr("review.report.final_report_audit.llm_json", fake_llm_json)

    result = audit_and_refine_final_report(
        final_markdown=original,
        source_markdown="The paper reports 27.3 BLEU on WMT 2014 En-De.",
        max_iterations=3,
        max_source_chars=20000,
        max_review_chars=12000,
        model="gpt-5.2",
        min_english_words=0,
        min_chinese_chars=0,
        force_english_output=True,
    )

    assert result.applied is True
    assert result.iterations_run == 2
    assert result.stop_reason == "no_issues_found"
    assert result.final_markdown == revised
    assert result.iterations[0].accepted is True
    assert result.iterations[0].high_severity_issue_count == 1
    assert result.iterations[0].compatibility_ok is True


def test_audit_rejects_revision_that_changes_fixed_format(monkeypatch):
    original = _valid_report("The review says the paper reports 28.4 BLEU on WMT 2014 En-De.")
    invalid_revised = """
## 1. Metadata
- **Title**: Attention Is All You Need

## 2. Technical Positioning
| Research domain | Method | Encoder only |
| --- | --- | --- |
| Machine translation | Transformer | √ |

## 3. Claims
| Claim | Evidence | Assessment | Location |
| --- | --- | --- | --- |
| Changed row count. | Evidence | supported | Abstract |

## 4. Summary
The review says the paper reports 27.3 BLEU on WMT 2014 En-De.

## 5. Experiment
### Main Result
| Dataset | Metric | Value |
| --- | --- | --- |
| WMT 2014 En-De | BLEU | 27.3 |

### Ablation Result
| Setting | Metric | Value |
| --- | --- | --- |
| Base | BLEU | 27.3 |
""".strip()
    calls = {"count": 0}

    def fake_llm_json(*, prompt, system, cfg):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "audit_summary": "One numeric mismatch found.",
                "issues": [
                    {
                        "problem_type": "wrong_number",
                        "severity": "high",
                        "section": "4. Summary",
                        "review_excerpt": "28.4 BLEU",
                        "paper_evidence": "The paper reports 27.3 BLEU on WMT 2014 En-De.",
                        "suggested_fix": "Change 28.4 to 27.3.",
                        "should_fix": True,
                    }
                ],
            }
        return {
            "revision_summary": "Fixed the number but also changed the claims table.",
            "revised_markdown": invalid_revised,
        }

    monkeypatch.setattr("review.report.final_report_audit.llm_json", fake_llm_json)

    result = audit_and_refine_final_report(
        final_markdown=original,
        source_markdown="The paper reports 27.3 BLEU on WMT 2014 En-De.",
        max_iterations=3,
        max_source_chars=20000,
        max_review_chars=12000,
        model="gpt-5.2",
        min_english_words=0,
        min_chinese_chars=0,
        force_english_output=True,
    )

    assert result.applied is False
    assert result.stop_reason == "revision_changed_fixed_format"
    assert result.final_markdown == original
    assert result.iterations[0].compatibility_ok is False
    assert result.iterations[0].compatibility_message == "table 2 row count changed"
