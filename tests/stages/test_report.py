"""Report stage tests.

Pins down the post-hoc claim audit semantics that drive the final review's
verdict cells: status capping (always one-way toward more conservative),
Pending-promotion to the LLM verdict, agent self-tag reconciliation, and
the structural axis-self-selection / ablation-coverage weakness bullets.

The audit's LLM call is mandatory in production but injectable for tests
via the ``llm_call`` parameter. Each test wires a deterministic stub.
"""

from __future__ import annotations

from typing import Any

import pytest

from review.report.claim_audit import audit_review_markdown


def _stub_llm(verdicts: list[dict[str, Any]], missing: list[str] | None = None):
    """Build a deterministic LLM stub for ``audit_review_markdown``."""

    def _call(prompt: str) -> dict[str, Any]:
        return {"verdicts": verdicts, "ablation_missing_components": missing or []}

    return _call


def test_audit_caps_supported_to_inconclusive_when_llm_disagrees(
    review_md_with_claims_table,
) -> None:
    new_md, outcome = audit_review_markdown(
        review_md_with_claims_table,
        llm_call=_stub_llm([{"id": 0, "verdict": "inconclusive", "reason": "gap within 1 sigma"}]),
    )

    result = outcome.claim_results[0]
    assert result.original_status == "supported"
    assert result.final_status == "inconclusive"
    # Markdown reflects the cap and an audit weakness bullet got injected.
    assert "⚠ Inconclusive" in new_md
    assert "[audit] Status downgraded to Inconclusive" in new_md


def test_audit_promotes_pending_to_llm_verdict() -> None:
    md = (
        "## 3. Claims\n"
        "| Claim | Evidence | Assessment | Status | Location |\n"
        "|---|---|---|---|---|\n"
        "| Method M describes a new attention block. | Section 3.1 introduces M. | "
        "ok | Pending | Section 3.1 |\n"
        "## 4. Summary\n"
        "**Weaknesses:**\n- w\n\n"
    )
    new_md, outcome = audit_review_markdown(
        md, llm_call=_stub_llm([{"id": 0, "verdict": "supported", "reason": "anchored"}])
    )

    result = outcome.claim_results[0]
    assert result.original_status == ""  # Pending normalises to ""
    assert result.final_status == "supported"
    assert "✓ Supported" in new_md
    # Promotions out of Pending must NOT emit a "downgraded" weakness bullet.
    assert not any("Status downgraded" in b for b in outcome.extra_weaknesses)


def test_audit_takes_more_conservative_of_llm_and_agent_self_tag() -> None:
    md = (
        "## 3. Claims\n"
        "| Claim | Evidence | Assessment | Status | Location |\n"
        "|---|---|---|---|---|\n"
        "| Trivial. | Evidence. | "
        "ok [verdict: in_conflict; reason: paper value below comparator] | "
        '<span style="color: green;">✓ Supported</span> | Loc |\n'
        "## 4. Summary\n"
    )
    new_md, outcome = audit_review_markdown(
        md, llm_call=_stub_llm([{"id": 0, "verdict": "partially_supported", "reason": "weak"}])
    )

    result = outcome.claim_results[0]
    # Capping takes the more conservative of the two verdicts; the agent
    # self-tag wins here even though the LLM was more lenient.
    assert result.agent_self_verdict == "in conflict"
    assert result.llm_verdict == "partially supported"
    assert result.final_status == "in conflict"
    # The bracketed self-tag is stripped from the visible cell.
    assert "[verdict:" not in new_md


def test_audit_injects_missing_component_weakness_from_llm() -> None:
    md = (
        "## 3. Claims\n"
        "| Claim | Evidence | Assessment | Status | Location |\n"
        "|---|---|---|---|---|\n"
        "| Method M with A, B, and C. | Sec 2. | ok | Pending | Sec 2 |\n"
        "## 4. Summary\n"
        "**Weaknesses:**\n- w\n\n"
        "## 5. Experiment\n"
        "### Ablation Result\n"
        "| Dim | Cfg | Full | Paper | Δ |\n"
        "|---|---|---|---|---|\n"
        "| A | no | 1.0 | 0.5 | -0.5 |\n"
    )
    new_md, outcome = audit_review_markdown(
        md,
        llm_call=_stub_llm(
            [{"id": 0, "verdict": "partially_supported", "reason": "B and C not ablated"}],
            missing=["B", "C"],
        ),
    )

    assert outcome.ablation_components_missing == ["B", "C"]
    assert any("B" in b and "C" in b and "ablation" in b.lower() for b in outcome.extra_weaknesses)
    assert "[audit]" in new_md.split("## 5.")[0]


def test_audit_skips_llm_when_no_claims_table_but_runs_axis_audit() -> None:
    md = (
        "## 2. Technical Positioning\n"
        "| Research domain | Method | A | B | C |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| Other | X | × | × | × |\n"
        "| Other | Y | × | × | × |\n"
        "| Other | Z | × | × | × |\n"
        "| This Work | Ours | √ | √ | √ |\n"
        "## 4. Summary\n"
        "**Weaknesses:**\n- w\n\n"
    )

    def must_not_call(prompt: str) -> dict[str, Any]:
        raise AssertionError("LLM should not be called when there are no claims")

    new_md, outcome = audit_review_markdown(md, llm_call=must_not_call)
    # Axis audit is structural and runs regardless of whether claims exist.
    assert outcome.claim_results == []
    assert outcome.axis_self_selection_ratio is not None
    assert any("favor the proposed system" in b for b in outcome.extra_weaknesses)
    assert "[audit]" in new_md


def test_audit_propagates_llm_failure_without_silent_fallback() -> None:
    md = (
        "## 3. Claims\n"
        "| Claim | Evidence | Assessment | Status | Location |\n"
        "|---|---|---|---|---|\n"
        "| C. | E. | ok | Pending | L |\n"
        "## 4. Summary\n"
    )

    def boom(prompt: str) -> dict[str, Any]:
        raise RuntimeError("LLM unreachable")

    # Mandatory-LLM contract: a transport failure must surface, not get
    # swallowed into a default verdict.
    with pytest.raises(RuntimeError, match="LLM unreachable"):
        audit_review_markdown(md, llm_call=boom)
