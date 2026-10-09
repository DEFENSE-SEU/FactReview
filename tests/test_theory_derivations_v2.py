"""Structured model derivations retain source and proof gates; no external calls."""

import copy
import json
from pathlib import Path

import pytest

from assessment import assess_claim
from llm.client import LLMConfig
from schemas.claim import Claim, ClaimLocation, Condition
from schemas.materials import MaterialBlock, SharedMaterials
from screening import checks
from verification.theory import verify_theory


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(
        checks, "resolve_llm_config", lambda: LLMConfig("mock", "derivation-fixture", None, None)
    )
    monkeypatch.setattr(checks, "llm_json", lambda **kw: pytest.fail("Unmocked model"))
    monkeypatch.setattr("httpx.Client.send", lambda *a, **kw: pytest.fail("Network"))
    monkeypatch.setattr("httpx.AsyncClient.send", lambda *a, **kw: pytest.fail("Network"))
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **kw: pytest.fail("Network"))
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: pytest.fail("Process"))


def paper(tmp_path):
    text = "For real x and y, (x+y)^2 = x*x + 2*x*y + y*y by distributivity."
    path = tmp_path / "paper.md"
    path.write_text(text, encoding="utf-8")
    loc = ClaimLocation(page=1, section="Theory", char_start=0, char_end=len(text))
    materials = SharedMaterials(
        paper_key="derivation-control",
        source_pdf=str(tmp_path / "paper.pdf"),
        markdown=text,
        markdown_path=str(path),
        content_list_path="",
        provider="fixture",
        blocks=[MaterialBlock(id="b1", text=text, loc=loc)],
    )
    claim = Claim(
        id="claim1",
        text=text,
        loc=loc,
        source_block_id="b1",
        source_quote=text,
        conditions=[Condition(id="c1", description="For all real x and y")],
        needs=["Theory"],
    )
    item = {
        "block_id": "b1",
        "quote": text,
        "step_quote": "(x+y)^2 = x*x + 2*x*y + y*y",
        "covered": ["c1"],
        "fully_supported_conditions": ["c1"],
        "kind": "derivation",
        "direction": "support",
        "detail": "Distributivity establishes the stated equality.",
    }
    trace = {
        "goal": "Establish the stated square expansion for real x and y.",
        "assumptions": [
            {
                "id": "a1",
                "text": "x and y are real",
                "status": "paper_explicit",
                "sources": [{"block_id": "b1", "quote": "For real x and y"}],
            }
        ],
        "steps": [
            {
                "id": "s1",
                "statement": "(x+y)^2 = (x+y)(x+y)",
                "reason": "Definition of the square.",
                "assumption_ids": ["a1"],
                "previous_step_ids": [],
                "sources": [],
            },
            {
                "id": "s2",
                "statement": "(x+y)^2 = x*x + 2*x*y + y*y",
                "reason": "Distribute multiplication and combine equal cross terms.",
                "assumption_ids": ["a1"],
                "previous_step_ids": ["s1"],
                "sources": [{"block_id": "b1", "quote": item["step_quote"]}],
            },
        ],
        "gaps": [],
        "outcome": "completed",
        "completion_reason": "The final equality is the target expansion.",
    }
    response = {
        "schema_version": "theory-derivation-v1",
        "items": [item],
        "appendix_block_ids": [],
        "derivations": [{"item_index": 0, "trace": trace}],
    }
    return claim, materials, response


def invoke(claim, materials, response):
    return verify_theory(claim, materials, call=lambda **kwargs: copy.deepcopy(response))


def test_versioned_derivation_is_source_grounded_and_keeps_all_steps(tmp_path):
    claim, materials, response = paper(tmp_path)
    result = invoke(claim, materials, response)
    assert result.evidence[0].sufficient
    record = result.theory_derivations[0]
    assert record.state == "validated" and record.adopted
    assert record.trace.outcome == "completed"
    assert len(record.trace.steps) == 2
    source = record.trace.assumptions[0].sources[0]
    assert source.pointer.quote == "For real x and y"
    assert source.pointer.locator == str(Path(materials.markdown_path).resolve())
    assert source.pointer.key == "chars:0-16"
    assert claim.evidence == [] and claim.source_quote == materials.markdown


def test_completed_trace_does_not_upgrade_partial_flags(tmp_path):
    claim, materials, response = paper(tmp_path)
    response["items"][0]["fully_supported_conditions"] = []
    result = invoke(claim, materials, response)
    assert result.theory_derivations[0].state == "validated"
    assert not result.evidence[0].sufficient


@pytest.mark.parametrize(
    "change",
    ["missing_trace", "future_step", "unknown_assumption", "normalized_quote", "unstated", "partial"],
)
def test_new_incomplete_or_invalid_trace_never_preserves_full_positive_support(tmp_path, change):
    claim, materials, response = paper(tmp_path)
    trace = response["derivations"][0]["trace"]
    if change == "missing_trace":
        response["derivations"] = []
    elif change == "future_step":
        trace["steps"][0]["previous_step_ids"] = ["s2"]
    elif change == "unknown_assumption":
        trace["steps"][0]["assumption_ids"] = ["foreign"]
    elif change == "normalized_quote":
        trace["assumptions"][0]["sources"][0]["quote"] = "For real  x and y"
    elif change == "unstated":
        trace["assumptions"][0]["status"] = "required_unstated"
    elif change == "partial":
        trace["outcome"] = "partial"
        trace["gaps"] = [
            {
                "at": "goal",
                "reason": "Final equality is unresolved.",
                "needed": "Complete the argument.",
                "sources": [],
            }
        ]
    result = invoke(claim, materials, response)
    assert not any(e.sufficient for e in result.evidence)
    assert result.theory_derivations and result.issues
    claim.evidence = result.evidence
    assert assess_claim(claim).status == "unverified"


def test_legacy_response_retains_old_evidence_and_marks_trace_unavailable(tmp_path):
    claim, materials, response = paper(tmp_path)
    legacy = {"items": response["items"]}
    result = invoke(claim, materials, legacy)
    assert result.evidence[0].sufficient and not result.issues
    assert result.theory_derivations[0].state == "legacy_unavailable"
    assert result.theory_derivations[0].trace is None


def test_default_prompt_requests_explicit_version_and_preserves_item_schema(tmp_path):
    claim, materials, response = paper(tmp_path)

    def model(**kwargs):
        payload = json.loads(kwargs["prompt"])
        assert payload["output_schema"]["properties"]["schema_version"]["const"] == "theory-derivation-v1"
        assert "TheoryItem" in payload["output_schema"]["$defs"]
        return response

    assert verify_theory(claim, materials, call=model).evidence[0].sufficient


@pytest.mark.parametrize("identifier", [True, "0", 0.0, -1, 7])
def test_noncanonical_or_unknown_item_index_cannot_preserve_full_support(tmp_path, identifier):
    claim, materials, response = paper(tmp_path)
    response["derivations"][0]["item_index"] = identifier
    result = invoke(claim, materials, response)
    assert not result.evidence[0].sufficient
    assert result.theory_derivations[0].state == "invalid"


@pytest.mark.parametrize("malformation", ["missing", "wrong_quote", "duplicate_then_valid"])
def test_invalid_trace_keeps_independent_healthy_item(tmp_path, malformation):
    claim, materials, response = paper(tmp_path)
    claim.conditions.append(Condition(id="c2", description="The same equality for all real inputs"))
    response["items"].append(
        {**response["items"][0], "covered": ["c2"], "fully_supported_conditions": ["c2"]}
    )
    healthy = {"item_index": 1, "trace": copy.deepcopy(response["derivations"][0]["trace"])}
    if malformation == "missing":
        response["derivations"] = [healthy]
    elif malformation == "wrong_quote":
        response["derivations"][0]["trace"]["assumptions"][0]["sources"][0]["quote"] = "not original"
        response["derivations"].append(healthy)
    else:
        response["derivations"] *= 3
        response["derivations"].append(healthy)
    result = invoke(claim, materials, response)
    assert [e.sufficient for e in result.evidence] == [False, True]
    assert [r.state for r in result.theory_derivations] == ["invalid", "validated"]


@pytest.mark.parametrize("change", ["block_replaced", "block_loc", "markdown", "file_bytes", "path", "claim"])
def test_source_or_target_change_during_request_is_rejected(tmp_path, change):
    claim, materials, response = paper(tmp_path)

    def model(**kwargs):
        if change == "block_replaced":
            materials.blocks[0] = materials.blocks[0].model_copy(update={"text": "Changed manuscript"})
        elif change == "block_loc":
            materials.blocks[0].loc.page = 2
        elif change == "markdown":
            materials.markdown += " changed"
        elif change == "file_bytes":
            Path(materials.markdown_path).write_text("changed", encoding="utf-8")
        elif change == "path":
            materials.markdown_path = str(tmp_path / "other.md")
        else:
            claim.conditions[0].description = "A changed target"
        return response

    with pytest.raises(ValueError, match="changed"):
        verify_theory(claim, materials, call=model)


@pytest.mark.parametrize(
    "change", ["version", "null_version", "mixed_legacy", "no_derivations", "empty_items_unknown_trace"]
)
def test_invalid_envelope_is_visible_and_cannot_be_empty_success(tmp_path, change):
    claim, materials, response = paper(tmp_path)
    response["items"] = []
    response["derivations"] = []
    if change == "version":
        response["schema_version"] = "future-version"
    elif change == "null_version":
        response["schema_version"] = None
    elif change == "mixed_legacy":
        del response["schema_version"]
    elif change == "no_derivations":
        del response["derivations"]
    else:
        response["derivations"] = [{"item_index": 0, "trace": {}}]
    with pytest.raises(ValueError, match=r"version|Version|derivations|item_index"):
        invoke(claim, materials, response)


def test_generated_proof_does_not_replace_missing_author_proof(tmp_path):
    claim, materials, response = paper(tmp_path)
    response["items"][0]["kind"] = "no_proof"
    result = invoke(claim, materials, response)
    assert not result.evidence and not result.questions
    assert result.verification_limitations[0].kind == "evidence_validation_failed"
    assert result.verification_limitations[0].responsibility == "system"
    assert result.theory_derivations[0].state == "invalid"
    assert "no_proof" in result.theory_derivations[0].issues[0]


def test_explicit_unable_trace_keeps_missing_steps_and_reason(tmp_path):
    claim, materials, response = paper(tmp_path)
    response["items"][0]["kind"] = "no_proof"
    trace = response["derivations"][0]["trace"]
    trace.update(
        steps=[],
        outcome="unable",
        completion_reason="No author derivation is available.",
        gaps=[{"at": "goal", "reason": "Proof absent", "needed": "An author derivation", "sources": []}],
    )
    result = invoke(claim, materials, response)
    assert not result.evidence and result.questions
    record = result.theory_derivations[0]
    assert record.state == "validated" and record.trace.outcome == "unable"
    assert record.trace.gaps[0].needed == "An author derivation"


def appendix_case(tmp_path, *, wrong_target=False):
    from tests.test_theory_binding_v2 import manuscript, target

    materials = manuscript(tmp_path)
    claim = target(materials, "b" if wrong_target else "a")
    main = next(b for b in materials.blocks if b.id == claim.source_block_id)
    proof = next(b for b in materials.blocks if b.id == "proof")
    first = {
        "schema_version": "theory-derivation-v1",
        "appendix_block_ids": ["proof"],
        "items": [
            {
                "block_id": main.id,
                "quote": main.text,
                "covered": ["c1"],
                "fully_supported_conditions": [],
                "kind": "no_proof",
                "direction": "support",
                "detail": "The appendix must be inspected.",
            }
        ],
        "derivations": [
            {
                "item_index": 0,
                "trace": {
                    "goal": claim.text,
                    "assumptions": [],
                    "steps": [],
                    "gaps": [
                        {
                            "at": "goal",
                            "reason": "Appendix not inspected",
                            "needed": "The selected proof",
                            "sources": [],
                        }
                    ],
                    "outcome": "unable",
                    "completion_reason": "First pass lacks proof context.",
                },
            }
        ],
    }
    second = {
        "schema_version": "theory-derivation-v1",
        "appendix_block_ids": [],
        "items": [
            {
                "block_id": proof.id,
                "quote": proof.text,
                "step_quote": "n = 2k",
                "covered": ["c1"],
                "fully_supported_conditions": ["c1"],
                "kind": "derivation",
                "direction": "support",
                "main_block_id": main.id,
                "detail": "The model declares this proof complete.",
            }
        ],
        "derivations": [
            {
                "item_index": 0,
                "trace": {
                    "goal": claim.text,
                    "assumptions": [],
                    "steps": [
                        {
                            "id": "s1",
                            "statement": "n is a multiple of two.",
                            "reason": "The displayed equality gives divisibility.",
                            "assumption_ids": [],
                            "previous_step_ids": [],
                            "sources": [{"block_id": proof.id, "quote": "n = 2k"}],
                        }
                    ],
                    "gaps": [],
                    "outcome": "completed",
                    "completion_reason": "The displayed proof reaches the conclusion.",
                },
            }
        ],
    }
    return claim, materials, first, second


@pytest.mark.parametrize("wrong_target", [False, True])
def test_appendix_keeps_both_passes_and_original_proof_identity_gate(tmp_path, wrong_target):
    claim, materials, first, second = appendix_case(tmp_path, wrong_target=wrong_target)
    calls = []

    def model(**kwargs):
        calls.append(kwargs["module"])
        payload = json.loads(kwargs["prompt"])
        assert payload["claim"] == claim.model_dump(mode="json")
        assert payload["allowed_theory_source_block_ids"] == (
            ["a", "b"] if len(calls) == 1 else ["a", "b", "proof"]
        )
        assert "source_refs describe extraction provenance" in kwargs["system"]
        return copy.deepcopy(first if len(calls) == 1 else second)

    result = verify_theory(claim, materials, call=model, output_dir=tmp_path / "audit")
    assert calls == ["verification.theory", "verification.theory.appendix"]
    assert [r.phase for r in result.theory_derivations] == ["main", "appendix"]
    assert [r.adopted for r in result.theory_derivations] == [False, True]
    assert all(r.state == "validated" for r in result.theory_derivations)
    assert result.evidence[0].sufficient is not wrong_target
    assert all(Path(r.audit_pointer).is_file() for r in result.theory_derivations)
    assert len(list((tmp_path / "audit").glob("*.json"))) == 2


def test_appendix_trace_cannot_be_consumed_before_requested_context(tmp_path):
    claim, materials, first, second = appendix_case(tmp_path)
    first["appendix_block_ids"] = []
    first["derivations"] = second["derivations"]
    result = invoke(claim, materials, first)
    assert not result.evidence
    assert "outside this pass" in result.theory_derivations[0].issues[0]


def test_audit_and_generated_explanations_redact_credentials_without_changing_source(tmp_path, monkeypatch):
    claim, materials, response = paper(tmp_path)
    secret = "mock-secret-derivation-only"
    base = "https://user:mock-password@example.invalid:8443/v1?api_key=mock-token"
    cfg = LLMConfig("mock", "fixture", base, secret)
    monkeypatch.setattr(checks, "resolve_llm_config", lambda: cfg)
    trace = response["derivations"][0]["trace"]
    trace["goal"] += f" {secret} {base}"
    trace["steps"][0]["reason"] += f" {secret}"
    response["items"][0]["detail"] += f" {secret}"
    result = verify_theory(claim, materials, call=lambda **kw: response, output_dir=tmp_path / "audit")
    assert result.evidence[0].sufficient
    assert result.evidence[0].pointer.quote == materials.markdown
    serialized = result.model_dump_json() + "".join(
        p.read_text(encoding="utf-8") for p in (tmp_path / "audit").glob("*.json")
    )
    for value in (secret, "mock-password", "mock-token", base):
        assert value not in serialized
    assert "https://example.invalid:8443/v1" in serialized
    assert response["derivations"][0]["trace"]["goal"].endswith(base)


def test_failed_request_is_saved_and_does_not_fabricate_trace(tmp_path):
    claim, materials, _ = paper(tmp_path)

    def failure(**kwargs):
        raise TimeoutError("deliberate offline timeout")

    with pytest.raises(RuntimeError, match="timeout"):
        verify_theory(claim, materials, call=failure, output_dir=tmp_path / "audit")
    audit = json.loads(next((tmp_path / "audit").glob("*.json")).read_text(encoding="utf-8"))
    assert audit["status"] == "failed" and audit["transport"] == "injected"
    assert audit["request"]["claim"]["id"] == claim.id
    assert audit["source_hashes"] and "response" not in audit


async def test_dispatch_assessment_and_serialization_preserve_derivation_records(tmp_path):
    from verification.dispatch import VerificationResult, verify_claims

    claim, materials, response = paper(tmp_path)
    result = await verify_claims(
        [claim],
        materials,
        tmp_path / "verify",
        branches={
            "Theory": lambda c, m: invoke(c, m, response),
        },
    )
    saved = VerificationResult.model_validate_json(
        (tmp_path / "verify" / "verification.json").read_text(encoding="utf-8")
    )
    assert saved == result
    assessed = assess_claim(saved.claims[0])
    assert (
        assessed.status == "supported" and assessed.theory_derivations == result.claims[0].theory_derivations
    )
    old = claim.model_dump(exclude={"theory_derivations", "verification_limitations"})
    assert Claim.model_validate(old).theory_derivations == []


def test_required_nonzero_assumption_is_retained_without_rewriting_claim(tmp_path):
    claim, materials, response = paper(tmp_path)
    text = "For every real x, x/x = 1."
    loc = ClaimLocation(page=1, section="Theory", char_start=0, char_end=len(text))
    materials.markdown = materials.blocks[0].text = text
    materials.blocks[0].loc = loc
    Path(materials.markdown_path).write_text(text, encoding="utf-8")
    claim.text = claim.source_quote = text
    claim.loc = loc
    claim.conditions[0].description = "For every real x, including zero"
    item = response["items"][0]
    item.update(quote=text, step_quote="x/x = 1")
    response["derivations"][0]["trace"] = {
        "goal": "Check the assertion for every real x.",
        "assumptions": [{"id": "a1", "text": "x is nonzero", "status": "required_unstated", "sources": []}],
        "steps": [],
        "gaps": [
            {
                "at": "a1",
                "reason": "The quotient is undefined at zero.",
                "needed": "Restrict the domain to nonzero x.",
                "sources": [{"block_id": "b1", "quote": text}],
            }
        ],
        "outcome": "partial",
        "completion_reason": "The nonzero assumption is absent from the author's statement.",
    }
    original = claim.model_dump(mode="json")
    result = invoke(claim, materials, response)
    assert not result.evidence[0].sufficient and result.evidence[0].direction == "support"
    assert result.theory_derivations[0].trace.assumptions[0].status == "required_unstated"
    assert result.theory_derivations[0].trace.gaps[0].sources[0].pointer.quote == text
    assert claim.model_dump(mode="json") == original


def test_notation_model_cannot_change_frozen_manuscript_after_trace_validation(tmp_path):
    from schemas.materials import PageImage

    claim, materials, response = paper(tmp_path)
    image = tmp_path / "mock_page.png"
    image.write_bytes(b"mock pixels; never sent externally")
    materials.pages = [PageImage(page=1, path=str(image), width_points=100, height_points=100)]
    response["items"][0].update(kind="notation", direction="flaw")

    def model(**kwargs):
        if kwargs["module"] == "verification.theory":
            return response
        assert kwargs["module"] == "verification.theory.notation"
        Path(materials.markdown_path).write_text("altered source", encoding="utf-8")
        return {"classification": "manuscript_issue", "explanation": "Injected confirmation."}

    with pytest.raises(ValueError, match="artifact changed"):
        verify_theory(claim, materials, call=model)
