from __future__ import annotations

from review.report.claim_audit import (
    _extract_score_candidates,
    _extract_value_sigma,
    apply_llm_claim_adjudication,
    apply_significance_cap_to_markdown,
    assess_significance_cap,
    audit_ablation_coverage,
    audit_axis_self_selection,
    audit_review_markdown,
    inject_weaknesses,
)


# ---------------------------------------------------------------------------
# Significance cap


def test_value_sigma_extraction() -> None:
    assert _extract_value_sigma("CCA: 59.0 ± 1.9 Resolve Rate") == (59.0, 1.9)
    assert _extract_value_sigma("44.0 +/- 0.5 vs 42.0") == (44.0, 0.5)
    assert _extract_value_sigma("just a value 27.0") is None


def test_score_candidates_excludes_hyphen_version_digits() -> None:
    """Hyphenated version numbers like ``GPT-5.2`` must be filtered.

    Space-separated names like ``Claude 4.5 Opus`` are not filtered (the
    structural mask is intentionally narrow to avoid eating real scores like
    ``at 59.0``). We document that trade-off here: leaked low version digits
    are dominated by real comparator scores in ``max()`` downstream, so the
    audit still picks the right strongest comparator.
    """
    text = (
        "GPT-5.2 + CCA at 59.0 +/- 1.9, compared with OpenAI GPT-5.2 at 55.6, "
        "GPT-5.4 at 57.7, Claude 4.5 Opus at 52.0, and CCA variants at 45.5–54.3."
    )
    cands = _extract_score_candidates(text, exclude_value=59.0)
    # Real comparator scores must all be retained.
    for expected in (55.6, 57.7, 52.0, 45.5, 54.3):
        assert expected in cands, f"Expected {expected} in {cands}"
    assert 59.0 not in cands  # paper value excluded
    # Hyphen-form version digits ARE filtered (they look like "GPT-5.2").
    assert 5.2 not in cands
    assert 5.4 not in cands
    # Space-form version digit (4.5 from "Claude 4.5 Opus") may leak through;
    # downstream max() picks 57.7 as the strongest comparator regardless.
    strongest = max(cands)
    assert strongest == 57.7


def test_score_candidates_work_across_domains() -> None:
    """Cross-domain regression: numbers in CV / graph / biology evidence
    text should not be eaten by the model-name mask."""
    cv_text = "ResNet-50 reaches 76.5 top-1 vs ResNet-101 at 77.4 and EfficientNet-B7 at 84.3."
    cands = _extract_score_candidates(cv_text)
    assert 76.5 in cands and 77.4 in cands and 84.3 in cands
    # Hyphen-form version digits filtered.
    assert 50 not in cands
    assert 101 not in cands

    graph_text = "GCN scores 81.5 on Cora vs GAT at 83.0 and SAGE at 78.2."
    cands = _extract_score_candidates(graph_text)
    assert 81.5 in cands and 83.0 in cands and 78.2 in cands

    bio_text = "AlphaFold2 reaches 92.4 GDT vs RoseTTAFold at 81.3."
    cands = _extract_score_candidates(bio_text)
    assert 92.4 in cands and 81.3 in cands


def test_significance_cap_inconclusive_when_within_one_sigma() -> None:
    # Δ = 59.0 − 57.7 = 1.3, σ = 1.9, ratio ≈ 0.68 < 1 ⇒ inconclusive.
    audit = assess_significance_cap(
        claim="CCA achieves leading reported performance on SWE-Bench-Pro.",
        evidence=(
            "Table 1 reports GPT-5.2 + CCA at 59.0 ± 1.9 Resolve Rate (Pass@1), "
            "compared with OpenAI GPT-5.2 at 55.6, GPT-5.4 at 57.7, "
            "Claude 4.5 Opus + Anthropic at 52.0, and CCA variants at 45.5–54.3."
        ),
    )
    assert audit.superlative is True
    assert audit.significance_cap == "inconclusive"
    assert audit.delta_over_sigma is not None
    assert audit.delta_over_sigma < 1.0
    assert audit.paper_value == 59.0
    assert audit.paper_sigma == 1.9
    assert audit.comparator_value == 57.7


def test_significance_cap_partially_supported_when_within_two_sigma() -> None:
    # Δ = 60.0 − 57.0 = 3.0, σ = 1.9, ratio ≈ 1.58 ⇒ partially supported.
    audit = assess_significance_cap(
        claim="Method outperforms prior baselines.",
        evidence="Method scores 60.0 ± 1.9 vs. baseline 57.0.",
    )
    assert audit.significance_cap == "partially supported"


def test_significance_cap_supported_when_above_two_sigma() -> None:
    # Δ = 60.0 − 54.0 = 6.0, σ = 1.5, ratio = 4 ⇒ no cap.
    audit = assess_significance_cap(
        claim="Method outperforms prior baselines.",
        evidence="Method scores 60.0 ± 1.5 vs. baseline 54.0.",
    )
    assert audit.significance_cap is None


def test_significance_cap_no_superlative_no_change() -> None:
    audit = assess_significance_cap(
        claim="The method introduces a new module.",
        evidence="Section 2 describes the module.",
    )
    assert audit.significance_cap is None
    assert audit.superlative is False


def test_significance_cap_superlative_no_error_bar() -> None:
    audit = assess_significance_cap(
        claim="Method achieves the best result.",
        evidence="Method scores 60.0 vs baseline 55.0.",
    )
    assert audit.significance_cap == "partially supported"


def test_apply_significance_cap_to_markdown_downgrades_supported_to_inconclusive() -> None:
    md = (
        "## 3. Claims\n"
        "(legend)\n\n"
        "| Claim | Evidence | Assessment | Status | Location |\n"
        "|---|---|---|---|---|\n"
        "| CCA achieves leading reported performance on SWE-Bench-Pro. | "
        "Table 1 reports GPT-5.2 + CCA at 59.0 ± 1.9 Resolve Rate (Pass@1), "
        "compared with OpenAI GPT-5.2 at 55.6, GPT-5.4 at 57.7. | "
        "ok | "
        '<span style="color: green;">✓ Supported</span> | Table 1 |\n'
        "## 4. Summary\n"
    )
    new_md, audits = apply_significance_cap_to_markdown(md)
    assert len(audits) == 1
    assert audits[0].significance_cap == "inconclusive"
    assert "⚠ Inconclusive" in new_md
    assert "✓ Supported" not in new_md.split("## 4. Summary")[0]


# ---------------------------------------------------------------------------
# Axis self-selection


def test_axis_self_selection_flagged_when_majority_exclusive_wins() -> None:
    md = (
        "## 2. Technical Positioning\n"
        "Caption.\n\n"
        "| Research domain | Method | Context comp. | Persistent notes | Modular SDK | Repo issues |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| SWE agents | SWE-Agent | × | × | × | √ |\n"
        "| SWE agents | OpenHands | × | × | × | √ |\n"
        "| Test-time | Live-SWE-Agent | × | × | × | √ |\n"
        "| This Work | CCA | √ | √ | √ | √ |\n"
        "## 3. Claims\n"
    )
    ratio, bullet = audit_axis_self_selection(md)
    # 3 of 4 niche cols are exclusive wins ⇒ ratio = 0.75.
    assert ratio is not None and ratio >= 0.6
    assert bullet is not None
    assert "favor the proposed system" in bullet


def test_axis_self_selection_quiet_when_ratio_low() -> None:
    md = (
        "## 2. Technical Positioning\n"
        "| Research domain | Method | A | B |\n"
        "| --- | --- | --- | --- |\n"
        "| Domain | Other | √ | √ |\n"
        "| This Work | Ours | √ | × |\n"
        "## 3. Claims\n"
    )
    _, bullet = audit_axis_self_selection(md)
    assert bullet is None


# ---------------------------------------------------------------------------
# Ablation coverage


def test_extract_components_works_across_domains() -> None:
    """The component extractor must rely on English enumeration patterns,
    not on a curated vocabulary, so that graph/CV/biology papers also work."""
    from review.report.claim_audit import _extract_components_from_claim

    # Generic LLM-scaffold claim.
    cca = _extract_components_from_claim(
        "CCA provides a scalable agent scaffold with context management, "
        "note-taking, and extensions."
    )
    assert "context management" in cca and "note-taking" in cca and "extensions" in cca

    # Graph paper.
    graph = _extract_components_from_claim(
        "GraphNet is a model comprising graph convolution, attention pooling, "
        "and residual layers."
    )
    assert "graph convolution" in graph and "attention pooling" in graph

    # CV paper with version-suffixed component name.
    cv = _extract_components_from_claim(
        "Our model consists of a ResNet-50 backbone, a feature pyramid network, "
        "and a detection head."
    )
    assert "feature pyramid network" in cv
    assert "detection head" in cv
    # ``ResNet-50 backbone`` is a real component; digit content is allowed.
    assert any("backbone" in c for c in cv)

    # Biology paper.
    bio = _extract_components_from_claim(
        "AlphaFold2 includes an Evoformer block, a structure module, and "
        "a recycling iteration."
    )
    assert "evoformer block" in bio
    assert "structure module" in bio

    # Negative cases — no list, must return empty.
    assert _extract_components_from_claim("The method provides a transformer.") == []
    assert _extract_components_from_claim("The model is fast and efficient.") == []


def test_ablation_coverage_flags_missing_components() -> None:
    md = (
        "## 3. Claims\n"
        "| Claim | Evidence | Assessment | Status | Location |\n"
        "|---|---|---|---|---|\n"
        "| CCA provides a scalable agent scaffold with context management, "
        "note-taking, and extensions. | The paper introduces ... | ok | "
        '<span style="color: green;">✓ Supported</span> | Section 2 |\n'
        "## 5. Experiment\n"
        "### Main Result\n"
        "(table)\n\n"
        "### Ablation Result\n"
        "Location: Section 3.2 Table 2; ablation on context management.\n\n"
        "| Ablation Dimension | Configuration | Full Model | Paper Result | Difference (Δ) |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| Optimal setup | full | 48.6 | 48.6 | 0 |\n"
        "| Context management | no | 48.6 | 42.0 | -6.6 |\n"
    )
    missing, bullet = audit_ablation_coverage(md)
    assert "note-taking" in missing or "note taking" in missing
    assert "extensions" in missing
    assert bullet is not None
    assert "not ablated" in bullet


def test_ablation_coverage_quiet_when_all_components_present() -> None:
    md = (
        "## 3. Claims\n"
        "| Claim | Evidence | Assessment | Status | Location |\n"
        "|---|---|---|---|---|\n"
        "| Method provides scaffold with context management. | Section 2 ... | "
        "ok | ok | Section 2 |\n"
        "## 5. Experiment\n"
        "### Ablation Result\n"
        "| Ablation Dimension | Configuration | Full Model | Paper Result | Δ |\n"
        "|---|---|---|---|---|\n"
        "| Context management | no | 1.0 | 0.5 | -0.5 |\n"
    )
    missing, bullet = audit_ablation_coverage(md)
    assert missing == []
    assert bullet is None


# ---------------------------------------------------------------------------
# Weakness injection


def test_inject_weaknesses_appends_audit_bullets() -> None:
    md = (
        "## 4. Summary\n"
        "Some summary text.\n\n"
        "**Strengths:**\n"
        "- A strength\n\n"
        "**Weaknesses:**\n"
        "- Existing weakness\n\n"
        "## 5. Experiment\n"
    )
    out = inject_weaknesses(md, ["First audit weakness", "Second audit weakness"])
    assert "- Existing weakness" in out
    assert "- [audit] First audit weakness" in out
    assert "- [audit] Second audit weakness" in out
    # Audit bullets must come after the existing weakness, not before.
    pos_existing = out.find("Existing weakness")
    pos_audit = out.find("[audit] First")
    assert pos_existing < pos_audit
    # No duplicate Weaknesses heading inserted.
    assert out.count("**Weaknesses:**") == 1


def test_inject_weaknesses_handles_inline_weakness_label() -> None:
    """The agent often writes ``**Weaknesses:** - First...`` on a single line.
    The injector must merge into the existing list rather than create a
    duplicate ``**Weaknesses:**`` heading."""
    md = (
        "## 4. Summary\n"
        "Summary.\n\n"
        "**Strengths:** - A strength.\n"
        "- Another strength.\n\n"
        "**Weaknesses:** - First weakness.\n"
        "- Second weakness.\n\n"
        "## 5. Experiment\n"
    )
    out = inject_weaknesses(md, ["audit one"])
    assert out.count("**Weaknesses:**") == 1
    assert "- [audit] audit one" in out
    # Audit bullet appears after the original weaknesses, not before.
    assert out.index("Second weakness") < out.index("[audit] audit one")
    # Section 5 heading should still be on its own line.
    assert "## 5. Experiment" in out
    assert "[audit] audit one\n\n## 5. Experiment" in out or "[audit] audit one\n## 5. Experiment" in out


# ---------------------------------------------------------------------------
# End-to-end deterministic audit


def test_audit_review_markdown_end_to_end_for_cca_like_paper() -> None:
    md = (
        "## 2. Technical Positioning\n"
        "| Research domain | Method | Context comp. | Persistent notes | Modular SDK |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| SWE agents | SWE-Agent | × | × | × |\n"
        "| SWE agents | OpenHands | × | × | × |\n"
        "| This Work | CCA | √ | √ | √ |\n\n"
        "## 3. Claims\n"
        "(legend)\n\n"
        "| Claim | Evidence | Assessment | Status | Location |\n"
        "|---|---|---|---|---|\n"
        "| CCA provides a scalable agent scaffold with context management, "
        "note-taking, and extensions. | "
        "The paper introduces Confucius SDK with hierarchical working memory and "
        "persistent notes. | ok | "
        '<span style="color: green;">✓ Supported</span> | Section 2 |\n'
        "| CCA achieves leading reported performance on SWE-Bench-Pro. | "
        "Table 1 reports GPT-5.2 + CCA at 59.0 ± 1.9 Resolve Rate, vs. "
        "GPT-5.4 at 57.7, Claude 4.5 Opus at 52.0. | ok | "
        '<span style="color: green;">✓ Supported</span> | Table 1 |\n\n'
        "## 4. Summary\n"
        "Summary.\n\n"
        "**Strengths:**\n"
        "- A strength\n\n"
        "**Weaknesses:**\n"
        "- A weakness\n\n"
        "## 5. Experiment\n"
        "### Main Result\n"
        "(t)\n\n"
        "### Ablation Result\n"
        "Location: Section 3.2 Table 2.\n\n"
        "| Ablation Dimension | Configuration | Full Model | Paper Result | Δ |\n"
        "|---|---|---|---|---|\n"
        "| Optimal setup | full | 48.6 | 48.6 | 0 |\n"
        "| Context management | no | 48.6 | 42.0 | -6.6 |\n"
    )
    new_md, outcome = audit_review_markdown(md)

    # Significance cap fires on Claim 2.
    assert any(c.significance_cap == "inconclusive" for c in outcome.claim_results)
    assert "⚠ Inconclusive" in new_md

    # Axis self-selection fires (3/3 niche dims are exclusive wins).
    assert outcome.axis_self_selection_ratio is not None
    assert outcome.axis_self_selection_ratio >= 0.6

    # Ablation coverage flags note-taking + extensions as missing.
    assert any("note" in c for c in outcome.ablation_components_missing)
    assert "extensions" in outcome.ablation_components_missing

    # Weaknesses injected.
    weakness_section = new_md.split("**Weaknesses:**")[1].split("## 5.")[0]
    assert "[audit]" in weakness_section


# ---------------------------------------------------------------------------
# LLM adjudication wiring


def test_apply_llm_claim_adjudication_caps_to_more_conservative() -> None:
    md = (
        "## 3. Claims\n"
        "| Claim | Evidence | Assessment | Status | Location |\n"
        "|---|---|---|---|---|\n"
        "| Claim A. | Evidence A. | ok | "
        '<span style="color: green;">✓ Supported</span> | Loc A |\n'
        "| Claim B. | Evidence B. | ok | "
        '<span style="color: #E6B800;">⚠ Inconclusive</span> | Loc B |\n'
        "## 4. Summary\n"
    )
    # First call wants to downgrade; second call wants to upgrade.
    responses = [
        {"verdict": "partially_supported", "reason": "evidence partial"},
        {"verdict": "supported", "reason": "looks fine"},
    ]
    calls: list[str] = []

    def fake_call(prompt: str) -> dict[str, str]:
        calls.append(prompt)
        return responses.pop(0)

    new_md, audits = apply_llm_claim_adjudication(md, llm_call=fake_call)
    assert len(audits) == 2
    # First claim downgraded.
    assert "⚠ Partially supported" in new_md
    # Second claim NOT upgraded — capping never moves toward more positive.
    assert "⚠ Inconclusive" in new_md
    assert "✓ Supported" not in new_md.split("## 4. Summary")[0]


def test_apply_llm_claim_adjudication_handles_call_failure_gracefully() -> None:
    md = (
        "## 3. Claims\n"
        "| Claim | Evidence | Assessment | Status | Location |\n"
        "|---|---|---|---|---|\n"
        "| Claim. | Evidence. | ok | "
        '<span style="color: green;">✓ Supported</span> | Loc |\n'
        "## 4. Summary\n"
    )

    def boom(prompt: str) -> dict[str, str]:
        raise RuntimeError("llm down")

    new_md, audits = apply_llm_claim_adjudication(md, llm_call=boom)
    # Markdown unchanged, audit captures the error.
    assert "✓ Supported" in new_md
    assert audits and audits[0]["llm_status"] == "error"
