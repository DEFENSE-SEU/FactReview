"""Concern applicability is separately grounded and scoped to original conditions."""

import copy
import json
from pathlib import Path

import pytest

from assessment import assess_claim
from llm.client import LLMConfig
from schemas.claim import Claim, ClaimLocation, ClaimSourceRef, Condition, TheoryDerivationRecord
from schemas.materials import MaterialBlock, PageImage, SharedMaterials
from tests.test_theory_derivations_v2 import paper
from verification.theory import verify_theory


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Theory concerns must use fixed mocked model boundaries")

    cfg = LLMConfig("mock", "concern-fixture", None, None)
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: cfg)
    monkeypatch.setattr("screening.checks.resolve_vlm_config", lambda **kwargs: cfg)
    monkeypatch.setattr("screening.checks.llm_json", blocked)
    monkeypatch.setattr("requests.sessions.Session.request", blocked)
    monkeypatch.setattr("httpx.Client.send", blocked)
    monkeypatch.setattr("httpx.AsyncClient.send", blocked)
    monkeypatch.setattr("subprocess.run", blocked)


def outside_scope_response(tmp_path):
    claim, materials, response = paper(tmp_path)
    item = copy.deepcopy(response["items"][0])
    item.update(
        kind="edge_case",
        direction="flaw",
        fully_supported_conditions=[],
        detail=(
            "The discussion omits complex numbers. The target explicitly quantifies only real x and y, "
            "so this omission does not affect the stated identity."
        ),
    )
    response["items"].append(item)
    trace = copy.deepcopy(response["derivations"][0])
    trace["item_index"] = 1
    response["derivations"].append(trace)
    return claim, materials, response


def scope_decision(index=1, condition="c1", disposition="outside_scope"):
    return {
        "item_index": index,
        "condition_id": condition,
        "disposition": disposition,
        "target_sources": [{"block_id": "b1", "quote": "For real x and y"}],
        "trace_step_ids": ["s2"],
        "trace_gap_indices": [],
        "scope_reason": "The original target and the established expansion concern real values only.",
        "resolution": "The complex-domain omission leaves this real-domain target unchanged.",
    }


def assessed(claim, result):
    return assess_claim(
        claim.model_copy(
            deep=True,
            update={
                "evidence": result.evidence,
                "notes": result.issues,
                "theory_derivations": result.theory_derivations,
                "verification_limitations": result.verification_limitations,
            },
        )
    )


def test_unrelated_concern_cannot_override_healthy_support(tmp_path):
    claim, materials, response = outside_scope_response(tmp_path)
    original = copy.deepcopy(response)
    calls = []

    def model(**kwargs):
        calls.append(kwargs["module"])
        if kwargs["module"] == "verification.theory.concern_scope":
            payload = json.loads(kwargs["prompt"])
            assert payload["claim"] == claim.model_dump(mode="json")
            return {"schema_version": "theory-concern-v1", "items": [scope_decision()]}
        return copy.deepcopy(response)

    result = verify_theory(claim, materials, call=model, output_dir=tmp_path / "audits")
    assert assessed(claim, result).status == "supported"
    assert calls == ["verification.theory", "verification.theory.concern_scope"]
    flaws = [e for e in result.evidence if e.direction == "flaw"]
    assert flaws and all(not e.affects_claim and not e.concern and not e.sufficient for e in flaws)
    record = result.theory_derivations[1].concern_reviews[0]
    assert record.state == "validated" and record.decision.disposition == "outside_scope"
    assert record.target_sources[0].quote == "For real x and y" and record.audit_pointer
    assert original == response


def test_invalid_concern_trace_is_a_local_system_limitation(tmp_path):
    claim, materials, response = outside_scope_response(tmp_path)
    response["derivations"][1]["trace"]["steps"][0]["assumption_ids"] = ["unknown_assumption"]
    calls = []

    def model(**kwargs):
        calls.append(kwargs["module"])
        return copy.deepcopy(response)

    result = verify_theory(claim, materials, call=model, output_dir=tmp_path / "audits")
    assert assessed(claim, result).status == "supported"
    assert calls == ["verification.theory"]
    assert result.theory_derivations[1].state == "invalid"
    assert result.evidence[0].sufficient and result.evidence[0].direction == "support"
    assert result.verification_limitations[0].kind == "evidence_validation_failed"
    assert result.verification_limitations[0].condition_ids == ["c1"]
    assert not result.questions


def division_case(tmp_path, *, completed=False):
    text = "For all real x, x/x = 1."
    path = tmp_path / "paper.md"
    path.write_text(text, encoding="utf-8")
    loc = ClaimLocation(page=1, section="Theory", char_start=0, char_end=len(text))
    claim = Claim(
        id="division",
        text=text,
        loc=loc,
        source_block_id="b1",
        source_quote=text,
        conditions=[Condition(id="c1", description="All real x, including zero")],
        needs=["Theory"],
    )
    materials = SharedMaterials(
        paper_key="division",
        source_pdf=str(tmp_path / "paper.pdf"),
        markdown=text,
        markdown_path=str(path),
        content_list_path="",
        provider="fixture",
        blocks=[MaterialBlock(id="b1", text=text, loc=loc)],
    )
    assumption = {
        "id": "a1",
        "text": "x is real",
        "status": "paper_explicit",
        "sources": [{"block_id": "b1", "quote": "For all real x"}],
    }
    trace = {
        "goal": "Check the universal quotient claim at zero.",
        "assumptions": [assumption],
        "steps": [
            {
                "id": "s1",
                "statement": "At x=0, x/x is undefined over the reals.",
                "reason": "Real division has no zero denominator.",
                "assumption_ids": ["a1"],
                "previous_step_ids": [],
                "sources": [],
            }
        ],
        "gaps": []
        if completed
        else [
            {
                "at": "goal",
                "reason": "The intended domain is unclear.",
                "needed": "Clarify whether x=0 is excluded.",
                "sources": [{"block_id": "b1", "quote": text}],
            }
        ],
        "outcome": "completed" if completed else "partial",
        "completion_reason": "The original universal assertion includes an undefined quotient."
        if completed
        else "A nonzero restriction could resolve the statement.",
    }
    response = {
        "schema_version": "theory-derivation-v1",
        "appendix_block_ids": [],
        "items": [
            {
                "block_id": "b1",
                "quote": text,
                "covered": ["c1"],
                "fully_supported_conditions": [],
                "kind": "edge_case" if completed else "missing_assumption",
                "direction": "flaw",
                "detail": "Zero belongs to the declared real domain but division by zero is undefined.",
            }
        ],
        "derivations": [{"item_index": 0, "trace": trace}],
    }
    decision = {
        **scope_decision(0, disposition="closed_disproof" if completed else "answerable_concern"),
        "target_sources": [{"block_id": "b1", "quote": text}],
        "trace_step_ids": ["s1"],
        "trace_gap_indices": [] if completed else [0],
        "scope_reason": "The stated domain includes zero, where this quotient is undefined.",
        "resolution": "The original universal claim is false at zero; a restricted claim would change it."
        if completed
        else "Ask whether the intended statement restricts x to nonzero values.",
    }
    return claim, materials, response, decision


def run_case(tmp_path, claim, materials, response, scope, *, hook=None):
    calls = []

    def model(**kwargs):
        calls.append(kwargs["module"])
        if kwargs["module"] == "verification.theory.concern_scope":
            if hook:
                hook(kwargs)
            if isinstance(scope, Exception):
                raise scope
            return copy.deepcopy(scope)
        return copy.deepcopy(response)

    result = verify_theory(claim, materials, call=model, output_dir=tmp_path / "audit")
    return result, calls


@pytest.mark.parametrize("completed,status", [(False, "questioned"), (True, "flawed")])
def test_relevant_gap_and_closed_counterexample_have_distinct_flags(tmp_path, completed, status):
    claim, materials, response, decision = division_case(tmp_path, completed=completed)
    result, calls = run_case(
        tmp_path, claim, materials, response, {"schema_version": "theory-concern-v1", "items": [decision]}
    )
    assert assessed(claim, result).status == status
    assert result.evidence[0].concern and result.evidence[0].affects_claim
    assert result.evidence[0].sufficient is completed
    assert result.evidence[0].overturnable is (not completed)
    assert calls == ["verification.theory", "verification.theory.concern_scope"]
    record = result.theory_derivations[0].concern_reviews[0]
    assert record.validation_scope == "structure_source_and_model_scope_judgment"
    audit_path, pointer = record.audit_pointer.split("#")
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    assert pointer == "/reviews/0" and audit["reviews"][0] == record.model_dump(mode="json")
    assert audit["response"]["items"][0] == decision


@pytest.mark.parametrize(
    "change",
    [
        "empty",
        "version",
        "missing_version",
        "foreign_condition",
        "foreign_step",
        "foreign_gap",
        "wrong_quote",
        "wrong_block",
        "extra_field",
        "error",
        "exception",
        "padded",
        "duplicate",
        "invalid_then_valid",
        "third_resurrection",
    ],
)
def test_bad_scope_never_removes_healthy_support_or_promotes_concern(tmp_path, change):
    claim, materials, response = outside_scope_response(tmp_path)
    decision = scope_decision(disposition="answerable_concern")
    scope = {"schema_version": "theory-concern-v1", "items": [decision]}
    if change == "empty":
        scope["items"] = []
    elif change == "version":
        scope["schema_version"] = "legacy"
    elif change == "missing_version":
        scope.pop("schema_version")
    elif change == "foreign_condition":
        decision["condition_id"] = "c2"
    elif change == "foreign_step":
        decision["trace_step_ids"] = ["foreign"]
    elif change == "foreign_gap":
        decision["trace_gap_indices"] = [42]
    elif change == "wrong_quote":
        decision["target_sources"][0]["quote"] = "For complex x and y"
    elif change == "wrong_block":
        decision["target_sources"][0]["block_id"] = "other"
    elif change == "extra_field":
        decision["fully_supported"] = True
    elif change == "error":
        scope = {"status": "error", "error": "service unavailable"}
    elif change == "exception":
        scope = RuntimeError("service unavailable")
    elif change == "padded":
        decision["condition_id"] = " c1 "
    elif change == "duplicate":
        scope["items"].append(copy.deepcopy(decision))
    elif change in {"invalid_then_valid", "third_resurrection"}:
        invalid = copy.deepcopy(decision)
        invalid["condition_id"] = " c1 "
        invalid["trace_step_ids"] = ["foreign"]
        scope["items"] = [invalid, decision] + (
            [copy.deepcopy(decision)] if change == "third_resurrection" else []
        )
    result, calls = run_case(tmp_path, claim, materials, response, scope)
    assert assessed(claim, result).status == "supported"
    assert len(calls) == 2
    assert result.evidence[0].sufficient and all(not e.affects_claim for e in result.evidence[1:])
    assert result.verification_limitations and not result.questions
    assert result.theory_derivations[1].concern_reviews[0].state in {"invalid", "unavailable"}


@pytest.mark.parametrize("change", ["partial", "missing_closing_step"])
def test_closed_disproof_requires_complete_trace_and_closing_reference(tmp_path, change):
    claim, materials, response, decision = division_case(tmp_path, completed=(change != "partial"))
    decision["disposition"] = "closed_disproof"
    if change == "missing_closing_step":
        trace = response["derivations"][0]["trace"]
        step = copy.deepcopy(trace["steps"][0])
        step.update(id="s2", previous_step_ids=["s1"])
        trace["steps"].append(step)
    result, _ = run_case(
        tmp_path, claim, materials, response, {"schema_version": "theory-concern-v1", "items": [decision]}
    )
    assert assessed(claim, result).status == "unverified"
    assert not result.evidence[0].affects_claim
    assert result.verification_limitations


def test_conflicting_full_support_retains_original_assessment_priority(tmp_path):
    claim, materials, response, decision = division_case(tmp_path, completed=True)
    support = copy.deepcopy(response["items"][0])
    support.update(
        kind="derivation", direction="support", step_quote="x/x = 1", fully_supported_conditions=["c1"]
    )
    response["items"].append(support)
    response["derivations"].append(
        {"item_index": 1, "trace": copy.deepcopy(response["derivations"][0]["trace"])}
    )
    result, _ = run_case(
        tmp_path, claim, materials, response, {"schema_version": "theory-concern-v1", "items": [decision]}
    )
    assert any(e.sufficient and e.direction == "support" for e in result.evidence)
    assert any(e.sufficient and e.direction == "flaw" for e in result.evidence)
    assert assessed(claim, result).status == "questioned"


@pytest.mark.parametrize("mutation", ["claim", "markdown", "file", "block"])
def test_scope_callback_cannot_change_frozen_inputs(tmp_path, mutation):
    claim, materials, response, decision = division_case(tmp_path)

    def change(_):
        if mutation == "claim":
            claim.conditions[0].description = "nonzero x"
        elif mutation == "markdown":
            materials.markdown += " changed"
        elif mutation == "file":
            Path(materials.markdown_path).write_text("changed", encoding="utf-8")
        else:
            materials.blocks[0].text += " changed"

    with pytest.raises(ValueError, match="changed"):
        run_case(
            tmp_path,
            claim,
            materials,
            response,
            {"schema_version": "theory-concern-v1", "items": [decision]},
            hook=change,
        )
    audit = next((tmp_path / "audit").glob("concern-*.json"))
    assert json.loads(audit.read_text(encoding="utf-8"))["response"]["items"][0] == decision


def test_versionless_concern_cannot_bypass_scope_requirement(tmp_path):
    claim, materials, response, _ = division_case(tmp_path)
    response.pop("schema_version")
    response.pop("derivations")
    result, calls = run_case(tmp_path, claim, materials, response, {})
    assert assessed(claim, result).status == "unverified" and len(calls) == 1
    assert result.theory_derivations[0].state == "legacy_unavailable"
    assert result.verification_limitations[0].kind == "evidence_validation_failed"


@pytest.mark.parametrize("completed", [False, True])
def test_printed_notation_is_scoped_after_visual_confirmation(tmp_path, completed):
    claim, materials, response, decision = division_case(tmp_path, completed=completed)
    image = tmp_path / "page.png"
    image.write_bytes(b"fixed-original-page")
    materials.pages = [PageImage(page=1, path=str(image), dpi=96, width_points=100, height_points=100)]
    response["items"][0]["kind"] = "notation"
    calls = []

    def model(**kw):
        calls.append(kw["module"])
        if kw["module"] == "verification.theory.notation":
            return {
                "classification": "manuscript_issue",
                "explanation": "The printed domain says all real x.",
            }
        if kw["module"] == "verification.theory.concern_scope":
            payload = json.loads(kw["prompt"])
            assert payload["candidates"][0]["notation_confirmation"]["classification"] == "manuscript_issue"
            return {"schema_version": "theory-concern-v1", "items": [decision]}
        return copy.deepcopy(response)

    result = verify_theory(claim, materials, call=model, output_dir=tmp_path / "audit")
    assert calls == [
        "verification.theory",
        "verification.theory.notation",
        "verification.theory.concern_scope",
    ]
    assert assessed(claim, result).status == ("flawed" if completed else "questioned")
    review = result.theory_derivations[0].concern_reviews[0]
    assert str(image) in review.source_hashes and result.evidence[0].sufficient is completed


def test_healthy_support_does_not_add_scope_call(tmp_path):
    claim, materials, response = paper(tmp_path)
    result, calls = run_case(tmp_path, claim, materials, response, {})
    assert calls == ["verification.theory"] and assessed(claim, result).status == "supported"
    assert result.theory_derivations[0].concern_reviews == []


def test_condition_source_ranges_and_invalid_neighbor_are_isolated(tmp_path):
    claim, materials, response, decision = division_case(tmp_path)
    second = "For nonzero real x, x/x = 1."
    start = len(materials.markdown) + 2
    loc = ClaimLocation(page=1, section="Theory", char_start=start, char_end=start + len(second))
    materials.markdown += "\n\n" + second
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
    materials.blocks.append(MaterialBlock(id="b2", text=second, loc=loc))
    claim.conditions.append(Condition(id="c2", description="Nonzero real x"))
    claim.source_refs = [
        ClaimSourceRef(source_block_id="b1", source_quote=claim.source_quote, loc=claim.loc, covered=["c1"]),
        ClaimSourceRef(source_block_id="b2", source_quote=second, loc=loc, covered=["c2"]),
    ]
    response["items"][0]["covered"] = ["c1", "c2"]
    foreign = copy.deepcopy(decision)
    foreign["condition_id"] = "c2"
    result, _ = run_case(
        tmp_path,
        claim,
        materials,
        response,
        {"schema_version": "theory-concern-v1", "items": [decision, foreign]},
    )
    evidence = {e.covered[0]: e for e in result.evidence}
    assert evidence["c1"].concern and evidence["c1"].affects_claim
    assert not evidence["c2"].concern and not evidence["c2"].affects_claim
    assert result.verification_limitations[0].condition_ids == ["c2"]
    assert [r.state for r in result.theory_derivations[0].concern_reviews] == ["validated", "invalid"]


def test_duplicate_item_pair_cannot_revoke_separate_healthy_concern(tmp_path):
    claim, materials, response, decision = division_case(tmp_path)
    response["items"].append(copy.deepcopy(response["items"][0]))
    response["derivations"].append(
        {"item_index": 1, "trace": copy.deepcopy(response["derivations"][0]["trace"])}
    )
    healthy = copy.deepcopy(decision)
    healthy["item_index"] = 1
    result, calls = run_case(
        tmp_path,
        claim,
        materials,
        response,
        {"schema_version": "theory-concern-v1", "items": [decision, copy.deepcopy(decision), healthy]},
    )
    assert len(calls) == 2
    assert not result.evidence[0].affects_claim and result.evidence[1].concern
    assert assessed(claim, result).status == "questioned"


def test_scope_page_mutation_is_checked_after_callback(tmp_path):
    claim, materials, response, decision = division_case(tmp_path)
    image = tmp_path / "page.png"
    image.write_bytes(b"original")
    materials.pages = [PageImage(page=1, path=str(image), width_points=10, height_points=10)]
    response["items"][0]["kind"] = "notation"

    def model(**kwargs):
        if kwargs["module"] == "verification.theory.notation":
            return {"classification": "manuscript_issue", "explanation": "All real values are printed."}
        if kwargs["module"] == "verification.theory.concern_scope":
            image.write_bytes(b"changed")
            return {"schema_version": "theory-concern-v1", "items": [decision]}
        return response

    with pytest.raises(ValueError, match="notation page"):
        verify_theory(claim, materials, call=model, output_dir=tmp_path / "audit")
    audit = json.loads(next((tmp_path / "audit").glob("concern-*.json")).read_text(encoding="utf-8"))
    assert (
        audit["response"]["items"][0] == decision and audit["validation_status"] == "rejected_source_change"
    )


def test_scope_explanations_are_sanitized_after_exact_binding(tmp_path, monkeypatch):
    secret = "fixture-secret-123456789"
    cfg = LLMConfig("openai-compatible", "fixture", "https://example.invalid/v1", secret)
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: cfg)
    claim, materials, response, decision = division_case(tmp_path)
    decision["scope_reason"] += " " + secret
    result, _ = run_case(
        tmp_path, claim, materials, response, {"schema_version": "theory-concern-v1", "items": [decision]}
    )
    assert result.evidence[0].concern
    assert secret not in result.model_dump_json()
    assert all(secret not in path.read_text(encoding="utf-8") for path in (tmp_path / "audit").glob("*.json"))
    assert decision["scope_reason"].endswith(secret)


def test_audit_directory_failure_preserves_support_without_model_scope(tmp_path, monkeypatch):
    claim, materials, response = outside_scope_response(tmp_path)
    original = Path.write_text

    def write(path, *args, **kwargs):
        if path.name.startswith("concern-"):
            raise PermissionError("scope audit directory denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write)
    result, calls = run_case(tmp_path, claim, materials, response, {})
    assert calls == ["verification.theory"]
    assert assessed(claim, result).status == "supported"
    record = result.theory_derivations[1].concern_reviews[0]
    assert record.state == "unavailable" and record.audit_pointer is None
    assert any("denied" in issue for issue in record.issues)


def test_old_derivation_record_remains_readable_without_concern_reviews(tmp_path):
    claim, materials, response = paper(tmp_path)
    result, _ = run_case(tmp_path, claim, materials, response, {})
    raw = result.theory_derivations[0].model_dump(mode="json")
    raw.pop("concern_reviews")
    assert TheoryDerivationRecord.model_validate(raw).concern_reviews == []
