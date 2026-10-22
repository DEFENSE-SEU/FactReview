"""The condensed advice request keeps every basis reference and the existing acceptance gates."""

import copy
import json
from pathlib import Path

import pytest

from llm.client import LLMConfig
from review.report import advice
from review.report.advice_request import build_request, project_input, requirements
from schemas.claim import AdviceItem, Claim, Condition, Evidence, EvidencePointer
from schemas.limitations import VerificationLimitation
from schemas.review import FinalReview


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Advice request controls must mock every external boundary")

    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr(advice, "llm_json", forbidden)
    monkeypatch.setattr(advice, "resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None))


def claim(status="unverified"):
    return Claim(
        id="c",
        text="The effect holds.",
        loc={"page": 1},
        conditions=[Condition(id="a", description="Original condition")],
        needs=["Experiments"],
        status=status,
    )


def evidence(direction="support", **kwargs):
    return Evidence(
        source="paper_internal",
        pointer=EvidencePointer(locator="original.pdf", page=1, quote="Original"),
        covered=["a"],
        direction=direction,
        **kwargs,
    )


def test_projection_keeps_every_basis_ref_and_condenses_long_records():
    c = claim()
    c.evidence = [
        evidence(
            note="N" * 5000,
            additional_pointers=[EvidencePointer(locator="appendix.pdf", page=2, quote="Extra original")],
        )
    ]
    c.notes = ["Repeated exact note", "Repeated exact note"]
    data = advice.advice_input(c, [{"plan": {"claim_id": "c", "condition_ids": ["a"]}, "attempts": [{"error": "x"}]}])
    frozen = copy.deepcopy(data)
    projected = project_input(data)
    assert set(projected["basis"]) == set(data["basis"])
    row = projected["basis"]["/evidence/0"]["content"]
    assert row["direction"] == "support" and row["quote"] == {"quote": "Original", "page": 1}
    assert len(row["note"]) < 700 and "additional_pointers" not in row and "locator" not in json.dumps(row)
    assert projected["claim"]["status"] == "unverified" and "loc" not in projected["claim"]
    assert "source_files" not in projected and data == frozen


def test_projection_rejects_unknown_top_level_input():
    data = advice.advice_input(claim(), [])
    data["future_scientific_record"] = {"important": True}
    with pytest.raises(ValueError):
        project_input(data)


@pytest.mark.parametrize(
    "status,items,expected",
    [
        ("supported", [evidence(sufficient=True), evidence(sufficient=False)], [["/evidence/0"]]),
        (
            "flawed",
            [evidence("flaw", sufficient=True, overturnable=False), evidence("flaw", sufficient=True)],
            [["/evidence/0"]],
        ),
        (
            "questioned",
            [evidence(sufficient=True), evidence("flaw", sufficient=True)],
            [["/evidence/0"], ["/evidence/1"]],
        ),
        (
            "questioned",
            [evidence(concern=True), evidence("flaw", sufficient=True)],
            [["/evidence/0", "/evidence/1"]],
        ),
    ],
)
def test_requirement_alternatives_match_real_status_gates(status, items, expected):
    c = claim(status)
    c.evidence = items
    data = advice.advice_input(c, [])
    catalog = requirements(c, data)
    assert catalog["conditions"][0]["one_or_more_from_each_group"] == expected
    assert catalog["required_condition_ids"] == (["a"] if status == "supported" else [])
    refs = [group[0] for group in expected]
    valid = AdviceItem(text="Original-condition advice.", condition_ids=["a"], basis_refs=refs)
    advice.validate_items(c, data, [valid])
    # Omit one genuinely required evidence direction/group; the old gate still rejects.
    for index in range(len(refs)):
        bad_refs = refs[:index] + refs[index + 1 :]
        bad = valid.model_copy(update={"basis_refs": bad_refs})
        with pytest.raises(ValueError):
            advice.validate_items(c, data, [bad])


def test_unverified_catalog_requires_every_condition_local_system_failure():
    c = claim()
    c.conditions.append(Condition(id="b", description="Already supported"))
    c.evidence = [evidence(sufficient=True).model_copy(update={"covered": ["b"]})]
    c.verification_limitations = [
        VerificationLimitation(
            claim_id="c", condition_ids=[cid], stage="Code", kind="branch_failed", reason=reason
        )
        for cid, reason in [
            ("a", "Source service failed"),
            ("a", "Index service failed"),
            ("b", "Separate failure"),
        ]
    ]
    data = advice.advice_input(c, [])
    catalog = requirements(c, data)
    assert catalog["required_condition_ids"] == ["a"]
    assert len(catalog["conditions"]) == 1
    row = catalog["conditions"][0]
    assert row["required_basis_refs"] == [
        "/coverage_gaps/a",
        "/verification_limitations/0",
        "/verification_limitations/1",
    ]
    assert row["required_action"] == "verification_followup"
    item = AdviceItem(
        text="Repair the recorded source checks.",
        condition_ids=["a"],
        basis_refs=row["required_basis_refs"],
        action="verification_followup",
    )
    advice.validate_items(c, data, [item])
    for ref in row["required_basis_refs"]:
        with pytest.raises(ValueError):
            advice.validate_items(
                c, data, [item.model_copy(update={"basis_refs": [r for r in item.basis_refs if r != ref]})]
            )
    with pytest.raises(ValueError):
        advice.validate_items(c, data, [item.model_copy(update={"action": "reviewer_advice"})])


def test_nonusable_evidence_cannot_become_an_eligible_condition():
    c = claim("questioned")
    c.evidence = [evidence(concern=True, affects_claim=False)]
    c.evidence.append(
        Evidence(
            source="execution",
            pointer=EvidencePointer(locator="run.json", key="result"),
            covered=["a"],
            direction="flaw",
            concern=True,
            aligned=False,
        )
    )
    assert requirements(c, advice.advice_input(c, []))["conditions"] == []


def test_public_generation_audits_full_input_and_exact_projected_request(tmp_path):
    c = claim()
    c.notes = ["Long original note " * 100]
    review = FinalReview(paper_key="fixture", run_id="mock", claims=[c])
    frozen = review.model_dump(mode="json")
    seen = []

    def call(**kwargs):
        request = json.loads(kwargs["prompt"].split("\nADVICE_DATA_JSON:\n", 1)[1])
        seen.append(request)
        assert "notes" not in request["claim"]
        # Long records are clipped on the wire; the audit keeps the full input.
        assert request["basis"]["/notes/0"]["content"].startswith(c.notes[0][:600])
        assert len(request["basis"]["/notes/0"]["content"]) < len(c.notes[0])
        assert request["item_requirements"]["required_condition_ids"] == ["a"]
        return {
            "status": "ok",
            "claim_id": "c",
            "items": [
                {
                    "text": "Please provide evidence for this condition.",
                    "condition_ids": ["a"],
                    "basis_refs": ["/coverage_gaps/a"],
                }
            ],
        }

    result = advice.generate_advice(review, tmp_path, call=call)
    assert result.counts == {"generated": 1, "unavailable": 0} and len(seen) == 1
    assert review.model_dump(mode="json") == frozen
    audit = json.loads(Path(result.review.claims[0].advice.audit_pointer).read_text("utf-8"))
    assert audit["request"] == seen[0] == build_request(c, audit["input"])
    assert audit["request_sha256"] == advice._digest(seen[0])
    assert audit["input_sha256"] == advice._digest(advice.advice_input(c, []))
    assert audit["input"]["claim"]["notes"] == c.notes
    assert set(seen[0]["basis"]) == set(audit["input"]["basis"])


def test_extra_unreferenced_item_still_rejects_entire_response(tmp_path):
    c = claim()
    c.notes = ["Supplemental context"]
    review = FinalReview(paper_key="fixture", run_id="mock", claims=[c])
    calls = []

    def call(**kwargs):
        calls.append(kwargs)
        return {
            "status": "ok",
            "claim_id": "c",
            "items": [
                {
                    "text": "Please provide evidence.",
                    "condition_ids": ["a"],
                    "basis_refs": ["/coverage_gaps/a"],
                },
                {"text": "Extra context.", "condition_ids": ["a"], "basis_refs": ["/notes/0"]},
            ],
        }

    result = advice.generate_advice(review, tmp_path, call=call)
    assert result.counts == {"generated": 0, "unavailable": 1} and len(calls) == 1
    assert result.review.claims[0].advice.items == []
    audit = json.loads(Path(result.review.claims[0].advice.audit_pointer).read_text("utf-8"))
    assert len(audit["response"]["items"]) == 2
