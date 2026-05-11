from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("agents")
pytest.importorskip("fitz")
pytest.importorskip("openai")
pytest.importorskip("reportlab")

from agent_runtime import runner  # noqa: E402


# ---------------------------------------------------------------------------
# _llm_verify_claim_evidence — LLM status parsing
# ---------------------------------------------------------------------------

class TestLlmVerifyFallbackDefault:
    def _call_with_mock_status(self, mock_status: str) -> str:
        with patch("agent_runtime.runner.resolve_llm_config"), \
             patch("agent_runtime.runner.llm_json") as mock_llm:
            mock_llm.return_value = {"status": mock_status, "assessment": "test assessment"}
            _, status = runner._llm_verify_claim_evidence(
                claim="Model achieves SOTA on all benchmarks",
                evidence="Supporting: Table 1. Missing: none.",
                location="Table 1",
            )
        return status

    def test_unrecognized_status_is_in_conflict(self) -> None:
        # "unclear" matches none of conflict/partial/support → else branch → In conflict
        assert self._call_with_mock_status("unclear") == "In conflict"

    def test_uncertain_status_is_in_conflict(self) -> None:
        # "uncertain" matches none of the substrings
        assert self._call_with_mock_status("uncertain") == "In conflict"

    def test_empty_status_is_in_conflict(self) -> None:
        assert self._call_with_mock_status("") == "In conflict"

    def test_supported_status_recognized(self) -> None:
        assert self._call_with_mock_status("Supported") == "Supported"

    def test_partially_supported_status_recognized(self) -> None:
        assert self._call_with_mock_status("Partially supported") == "Partially supported"

    def test_in_conflict_status_recognized(self) -> None:
        assert self._call_with_mock_status("In conflict") == "In conflict"


# ---------------------------------------------------------------------------
# Dead code removal verification
# ---------------------------------------------------------------------------

def test_dead_code_functions_removed() -> None:
    assert not hasattr(runner, "_build_theoretical_claim_assessment")
    assert not hasattr(runner, "_build_methodological_claim_assessment")
    assert not hasattr(runner, "_score_paper_evidence")
    assert not hasattr(runner, "_paper_status_from_score")


# ---------------------------------------------------------------------------
# _classify_llm_error — actionable error messages
# ---------------------------------------------------------------------------

class TestClassifyLlmError:
    def test_authentication_error(self) -> None:
        msg = runner._classify_llm_error("AuthenticationError: invalid api key")
        assert "API key" in msg

    def test_rate_limit_error(self) -> None:
        msg = runner._classify_llm_error("RateLimitError: 429 too many requests")
        assert "rate limit" in msg.lower()

    def test_connection_error(self) -> None:
        msg = runner._classify_llm_error("ConnectionError: timed out")
        assert "network" in msg.lower() or "connection" in msg.lower()

    def test_model_not_found(self) -> None:
        msg = runner._classify_llm_error("NotFoundError: model not found")
        assert "model" in msg.lower()

    def test_unknown_error_falls_back(self) -> None:
        msg = runner._classify_llm_error("SomeRandomError: unexpected")
        assert "API key" in msg or "error" in msg.lower()


# ---------------------------------------------------------------------------
# _llm_verify_claim_evidence — LLM unavailable raises RuntimeError
# ---------------------------------------------------------------------------

class TestLlmVerifyConservativeFallback:
    def test_llm_exception_raises(self) -> None:
        with patch("agent_runtime.runner.resolve_llm_config"), \
             patch("agent_runtime.runner.llm_json", side_effect=ConnectionError("timed out")):
            with pytest.raises(RuntimeError, match="Network error"):
                runner._llm_verify_claim_evidence(
                    claim="Our method is novel",
                    evidence="Supporting: Table 1. Missing: None.",
                    location="Table 1",
                )

    def test_auth_error_gives_actionable_message(self) -> None:
        with patch("agent_runtime.runner.resolve_llm_config"), \
             patch("agent_runtime.runner.llm_json", return_value={
                 "status": "error", "error": "AuthenticationError: invalid api key"
             }):
            with pytest.raises(RuntimeError, match="API key"):
                runner._llm_verify_claim_evidence(
                    claim="Our method is novel",
                    evidence="Supporting: Table 1. Missing: None.",
                    location="Table 1",
                )


# ---------------------------------------------------------------------------
# _split_evidence_parts — pure function
# ---------------------------------------------------------------------------

class TestSplitEvidenceParts:
    def test_well_formed_both_sections(self) -> None:
        sup, mis = runner._split_evidence_parts(
            "Supporting: Table 1, Figure 2. Missing: ablation study."
        )
        assert "Table 1" in sup and "Figure 2" in sup
        assert "ablation study" in mis

    def test_missing_none_is_parsed(self) -> None:
        _, mis = runner._split_evidence_parts(
            "Supporting: Theorem 3 proof in Appendix A. Missing: None."
        )
        assert mis.lower().startswith("none")

    def test_no_structure_returns_empty_strings(self) -> None:
        sup, mis = runner._split_evidence_parts("Some free-form evidence text.")
        assert sup == "" and mis == ""

    def test_empty_string_returns_empty(self) -> None:
        assert runner._split_evidence_parts("") == ("", "")

    def test_multiline_supporting(self) -> None:
        sup, mis = runner._split_evidence_parts(
            "Supporting: Table 1 shows accuracy 92.3%.\nAlso see Figure 3.\n\nMissing: no CI reported."
        )
        assert "Table 1" in sup
        assert "no CI" in mis


# ---------------------------------------------------------------------------
# _is_comparative_claim — pure function
# ---------------------------------------------------------------------------

class TestIsComparativeClaim:
    def test_outperforms_is_comparative(self) -> None:
        assert runner._is_comparative_claim("Our method outperforms all baselines")

    def test_sota_is_comparative(self) -> None:
        assert runner._is_comparative_claim("Achieves state-of-the-art on GLUE benchmark")

    def test_surpass_is_comparative(self) -> None:
        assert runner._is_comparative_claim("Our model surpasses BERT by 3%")

    def test_compared_against_is_comparative(self) -> None:
        assert runner._is_comparative_claim("Compared against BERT, our model is faster")

    def test_plain_performance_is_not_comparative(self) -> None:
        assert not runner._is_comparative_claim("The model achieves 92% accuracy on ImageNet")

    def test_theoretical_claim_is_not_comparative(self) -> None:
        assert not runner._is_comparative_claim("Theorem 1 proves convergence under convexity")


# ---------------------------------------------------------------------------
# _is_novelty_claim — pure function
# ---------------------------------------------------------------------------

class TestIsNoveltyClaim:
    def test_first_to_is_novelty(self) -> None:
        assert runner._is_novelty_claim("We are the first to apply X to Y")

    def test_to_our_knowledge_is_novelty(self) -> None:
        assert runner._is_novelty_claim("To our knowledge, no prior work addresses this")

    def test_to_the_best_of_our_knowledge_is_novelty(self) -> None:
        assert runner._is_novelty_claim("To the best of our knowledge, this is novel")

    def test_we_propose_is_novelty(self) -> None:
        assert runner._is_novelty_claim("We propose a novel attention mechanism")

    def test_plain_experimental_claim_is_not_novelty(self) -> None:
        assert not runner._is_novelty_claim("Our method achieves 94.2 F1 on CoNLL-2003")

    def test_comparative_claim_is_not_novelty(self) -> None:
        assert not runner._is_novelty_claim("Our model outperforms BERT by 3% on GLUE")


# ---------------------------------------------------------------------------
# _llm_verify_claim_evidence — authored_assessment integration
# ---------------------------------------------------------------------------

class TestLlmVerifyWithAuthoredAssessment:
    def _capture_prompt(self, authored_assessment: str) -> str:
        captured: list[str] = []

        def fake_llm(prompt: str, system: str, cfg: object) -> dict:
            captured.append(prompt)
            return {"status": "Supported", "assessment": "ok"}

        with patch("agent_runtime.runner.resolve_llm_config"), \
             patch("agent_runtime.runner.llm_json", side_effect=fake_llm):
            runner._llm_verify_claim_evidence(
                claim="Model achieves SOTA",
                evidence="Supporting: Table 1. Missing: None.",
                location="Table 1",
                authored_assessment=authored_assessment,
            )
        return captured[0] if captured else ""

    def test_meaningful_assessment_appears_in_prompt(self) -> None:
        prompt = self._capture_prompt("The paper supports this with Table 1 numbers.")
        assert "Reviewer agent's prior assessment" in prompt
        assert "Table 1 numbers" in prompt

    def test_empty_assessment_does_not_appear(self) -> None:
        prompt = self._capture_prompt("")
        assert "Reviewer agent's prior assessment" not in prompt

    def test_na_assessment_is_filtered(self) -> None:
        prompt = self._capture_prompt("n/a")
        assert "Reviewer agent's prior assessment" not in prompt

    def test_none_assessment_is_filtered(self) -> None:
        prompt = self._capture_prompt("none")
        assert "Reviewer agent's prior assessment" not in prompt


# ---------------------------------------------------------------------------
# _llm_verify_claim_evidence — type-differentiated system prompt
# ---------------------------------------------------------------------------

class TestLlmVerifyClaimType:
    def _capture_system(self, claim_type: str, claim: str = "Theorem 1 holds for convex losses") -> str:
        captured: list[str] = []

        def fake_llm(prompt: str, system: str, cfg: object) -> dict:
            captured.append(system)
            return {"status": "Supported", "assessment": "ok"}

        with patch("agent_runtime.runner.resolve_llm_config"), \
             patch("agent_runtime.runner.llm_json", side_effect=fake_llm):
            runner._llm_verify_claim_evidence(
                claim=claim,
                evidence="Supporting: Theorem 2 proof. Missing: None.",
                location="Appendix A",
                claim_type=claim_type,
            )
        return captured[0] if captured else ""

    def test_theoretical_system_prompt_includes_suffix(self) -> None:
        system = self._capture_system("Theoretical")
        assert "THEORETICAL" in system
        assert "formal proof" in system.lower() or "theorem" in system.lower()

    def test_methodological_system_prompt_includes_suffix(self) -> None:
        system = self._capture_system("Methodological")
        assert "METHODOLOGICAL" in system
        assert "algorithm" in system.lower() or "architecture" in system.lower()

    def test_experimental_uses_base_only(self) -> None:
        system = self._capture_system(
            "Experimental",
            claim="Model achieves 92% accuracy",
        )
        assert "THEORETICAL" not in system
        assert "METHODOLOGICAL" not in system

    def test_empty_claim_type_uses_base_only(self) -> None:
        system = self._capture_system("", claim="Model achieves 92% accuracy")
        assert "THEORETICAL" not in system
        assert "METHODOLOGICAL" not in system


# ---------------------------------------------------------------------------
# _llm_verify_claim_evidence — comparative and novelty additions
# ---------------------------------------------------------------------------

class TestLlmVerifyComparativeAndNovelty:
    def _capture_system(self, claim: str) -> str:
        captured: list[str] = []

        def fake_llm(prompt: str, system: str, cfg: object) -> dict:
            captured.append(system)
            return {"status": "In conflict", "assessment": "test"}

        with patch("agent_runtime.runner.resolve_llm_config"), \
             patch("agent_runtime.runner.llm_json", side_effect=fake_llm):
            runner._llm_verify_claim_evidence(
                claim=claim,
                evidence="Supporting: Table 1. Missing: baseline results.",
                location="Table 1",
            )
        return captured[0] if captured else ""

    def test_comparative_claim_adds_baseline_requirement(self) -> None:
        system = self._capture_system("Our method outperforms BERT on all benchmarks")
        assert "COMPARATIVE" in system
        assert "baseline" in system.lower()

    def test_novelty_claim_adds_asymmetric_note(self) -> None:
        system = self._capture_system("To our knowledge, this is the first method to do X")
        assert "NOVELTY" in system
        assert "asymmetric" in system.lower() or "prior work" in system.lower()

    def test_plain_claim_has_no_additions(self) -> None:
        system = self._capture_system("The model achieves 92% accuracy on CIFAR-10")
        assert "COMPARATIVE" not in system
        assert "NOVELTY" not in system

    def test_comparative_and_novelty_both_added(self) -> None:
        system = self._capture_system(
            "We are the first to outperform all prior baselines on this benchmark"
        )
        assert "COMPARATIVE" in system
        assert "NOVELTY" in system


