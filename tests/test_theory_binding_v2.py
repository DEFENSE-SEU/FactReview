"""Exact proof bytes must also belong to the target claim's theorem."""

from pathlib import Path

import pytest

from assessment import assess_claim
from llm.client import LLMConfig
from schemas.claim import Claim, ClaimLocation, ClaimSourceRef, Condition
from schemas.materials import MaterialBlock, SharedMaterials
from screening import checks
from verification.theory import verify_theory


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(checks, "resolve_llm_config", lambda: LLMConfig("mock", "mock", None, None))
    monkeypatch.setattr(checks, "llm_json", lambda **kw: pytest.fail("Unmocked model"))
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **kw: pytest.fail("Network"))
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: pytest.fail("Process"))


def manuscript(tmp_path, *, proof="Proof of Theorem 1. n = 2k; hence n is divisible by two."):
    texts = {
        "a": "Theorem 1: every even integer is divisible by two.",
        "b": "Theorem 2: every algorithm converges.",
        "proof": proof,
    }
    markdown = "\n\n".join(texts.values())
    path = tmp_path / "paper.md"
    path.write_text(markdown, encoding="utf-8")
    blocks = [
        MaterialBlock(
            id=key,
            text=text,
            loc=ClaimLocation(
                page=2 if key == "proof" else 1,
                section="Appendix A" if key == "proof" else "Theory",
                char_start=markdown.index(text),
                char_end=markdown.index(text) + len(text),
            ),
        )
        for key, text in texts.items()
    ]
    return SharedMaterials(
        paper_key="theorem-binding",
        source_pdf=str(tmp_path / "paper.pdf"),
        markdown=markdown,
        markdown_path=str(path),
        content_list_path="",
        provider="fixture",
        blocks=blocks,
    )


def target(materials, block_id="a"):
    block = next(b for b in materials.blocks if b.id == block_id)
    return Claim(
        id="target",
        text=block.text,
        source_block_id=block_id,
        source_quote=block.text,
        loc=block.loc,
        conditions=[Condition(id="c1", description=block.text)],
        needs=["Theory"],
    )


def run(claim, materials, *, main="a", covered=None):
    proof = next(b for b in materials.blocks if b.id == "proof")

    def model(**kwargs):
        if kwargs["module"] == "verification.theory":
            return {"items": [], "appendix_block_ids": ["proof"]}
        return {
            "items": [
                {
                    "block_id": "proof",
                    "quote": proof.text,
                    "step_quote": "n = 2k",
                    "main_block_id": main,
                    "kind": "derivation",
                    "direction": "support",
                    "covered": covered or ["c1"],
                    "fully_supported_conditions": covered or ["c1"],
                    "detail": "Mock declares complete support; source identity must still be checked.",
                }
            ]
        }

    return verify_theory(claim, materials, call=model)


def assert_unconfirmed(claim, result):
    assert result.evidence and not any(e.sufficient for e in result.evidence)
    assert result.issues and "binding" in " ".join(result.issues).lower()
    claim.evidence = result.evidence
    assert assess_claim(claim).status == "unverified"


def test_same_numbered_theorem_and_exact_claim_source_can_support(tmp_path):
    materials = manuscript(tmp_path)
    result = run(target(materials), materials)
    assert result.evidence[0].sufficient and not result.issues


@pytest.mark.parametrize("main", ["a", "b"])
def test_other_theorem_proof_cannot_support_claim_even_with_relabelled_main_id(tmp_path, main):
    materials = manuscript(tmp_path)
    claim = target(materials, "b")
    assert_unconfirmed(claim, run(claim, materials, main=main))


def test_proof_of_other_theorem_merely_citing_target_does_not_change_identity(tmp_path):
    materials = manuscript(
        tmp_path, proof="Proof of Theorem 1. Using Theorem 2, n = 2k; hence divisibility follows."
    )
    claim = target(materials, "b")
    assert_unconfirmed(claim, run(claim, materials, main="b"))


def test_discussing_another_proof_does_not_replace_the_declared_proof_target(tmp_path):
    materials = manuscript(
        tmp_path,
        proof="Proof of Theorem 1. We reuse the proof of Theorem 2: n = 2k; hence divisibility follows.",
    )
    claim = target(materials, "b")
    assert_unconfirmed(claim, run(claim, materials, main="b"))


@pytest.mark.parametrize("change", ["missing", "wrong_quote", "wrong_page"])
def test_unverifiable_or_legacy_source_anchor_retains_insufficient_observation(tmp_path, change):
    materials = manuscript(tmp_path)
    claim = target(materials)
    if change == "missing":
        claim.source_block_id = claim.source_quote = None
    elif change == "wrong_quote":
        claim.source_quote = "Theorem 1: invented text."
    else:
        claim.loc = ClaimLocation(page=99)
    assert_unconfirmed(claim, run(claim, materials))


def test_source_reference_coverage_cannot_expand_primary_anchor_to_other_condition(tmp_path):
    materials = manuscript(tmp_path)
    claim = target(materials)
    claim.conditions.append(Condition(id="c2", description=materials.blocks[1].text))
    claim.source_refs = [
        ClaimSourceRef(
            source_block_id=b.id,
            source_quote=b.text,
            loc=b.loc,
            covered=[condition],
        )
        for b, condition in zip(materials.blocks[:2], ["c1", "c2"], strict=True)
    ]
    assert run(claim, materials, covered=["c1"]).evidence[0].sufficient
    assert_unconfirmed(claim, run(claim, materials, covered=["c1", "c2"]))


@pytest.mark.parametrize("proof_target", ["Theorem 11", "Lemma 1"])
def test_theorem_kind_and_complete_number_must_match(tmp_path, proof_target):
    materials = manuscript(tmp_path, proof=f"Proof of {proof_target}. n = 2k; hence the result follows.")
    claim = target(materials)
    assert_unconfirmed(claim, run(claim, materials))


def test_generic_theorem_reference_requires_unique_main_theorem(tmp_path):
    materials = manuscript(tmp_path, proof="Proof. n = 2k; hence the theorem holds.")
    claim = target(materials)
    assert_unconfirmed(claim, run(claim, materials))


def test_a_source_assertion_without_proof_identity_cannot_support_itself(tmp_path):
    materials = manuscript(tmp_path, proof="n = 2k; therefore it follows.")
    claim = target(materials)
    assert_unconfirmed(claim, run(claim, materials))


def test_original_mrr_claim_cannot_be_supported_by_the_convergence_proof(tmp_path):
    materials = manuscript(tmp_path)
    materials.blocks[1].text = "A test MRR is 0.4."
    materials.markdown = "\n\n".join(b.text for b in materials.blocks)
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
    materials.blocks[1].loc = ClaimLocation(page=1, section="Results")
    claim = target(materials, "b")
    assert_unconfirmed(claim, run(claim, materials, main="b"))


def test_duplicate_numbered_theorem_declarations_are_ambiguous(tmp_path):
    materials = manuscript(tmp_path)
    materials.blocks[1].text = "Theorem 1: every algorithm converges."
    claim = target(materials)
    assert_unconfirmed(claim, run(claim, materials))


def main_derivation(claim, materials, quote, step):
    return verify_theory(
        claim,
        materials,
        call=lambda **kwargs: {
            "items": [
                {
                    "block_id": "a",
                    "quote": quote,
                    "step_quote": step,
                    "kind": "derivation",
                    "direction": "support",
                    "covered": ["c1"],
                    "fully_supported_conditions": ["c1"],
                    "detail": "The derivation establishes this claim under its stated assumptions.",
                }
            ]
        },
    )


def replace_main(materials, text):
    materials.blocks[0].text = text
    materials.blocks[0].loc = ClaimLocation(page=1, section="Theory")
    materials.markdown = "\n\n".join(b.text for b in materials.blocks)
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")


def test_unlabelled_main_text_derivation_retains_exact_step_and_model_semantics(tmp_path):
    materials = manuscript(tmp_path)
    text = "The bound follows from the variational objective. By Jensen's inequality, x = y; hence the bound."
    replace_main(materials, text)
    claim = target(materials)
    result = main_derivation(claim, materials, text, "x = y")
    assert result.evidence[0].sufficient and not result.issues


def test_same_block_numbered_statement_followed_by_local_proof_can_support(tmp_path):
    materials = manuscript(tmp_path)
    statement = "Theorem 1: every even integer is divisible by two."
    proof = "Proof. n = 2k; hence n is divisible by two."
    replace_main(materials, statement + "\n" + proof)
    claim = target(materials)
    claim.source_quote = statement
    result = main_derivation(claim, materials, proof, "n = 2k")
    assert result.evidence[0].sufficient and not result.issues


def test_exact_mrr_source_cannot_inherit_theorem_identity_from_earlier_same_block(tmp_path):
    materials = manuscript(tmp_path)
    source = "A test MRR is 0.4."
    replace_main(materials, materials.blocks[0].text + " " + source + " We use Adam.")
    claim = target(materials)
    claim.text = claim.source_quote = source
    assert_unconfirmed(claim, run(claim, materials))
