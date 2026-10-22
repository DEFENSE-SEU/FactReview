"""V2 contracts reject ambiguity at the stage boundary and preserve provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from schemas.claim import (
    AuthorQuestion,
    Claim,
    ClaimLocation,
    ClaimStatus,
    Condition,
    Evidence,
    EvidenceNeed,
    ExecutionPlan,
    ExecutionProvenance,
    ExecutionTask,
    Finding,
)
from schemas.review import FinalReview
from schemas.v1_adapter import adapt_v1_label, read_v1_artifact


def condition(condition_id: str = "fb237") -> Condition:
    return Condition(id=condition_id, dataset=condition_id, metric="MRR", settings={"split": "test"})


def claim(**changes) -> Claim:
    return Claim.model_validate({
        "id": "c1", "text": "Improves MRR on both datasets.",
        "loc": {"page": 5, "section": "results", "char_start": 10, "char_end": 45},
        "conditions": [condition().model_dump(), condition("wn18rr").model_dump()],
        "needs": ["Experiments", "Code"], "importance": "core", **changes,
    })


def evidence(**changes) -> Evidence:
    return Evidence.model_validate({
        "source": "paper_internal",
        "pointer": {"locator": "paper.pdf", "page": 5, "quote": "MRR 0.355"},
        "covered": ["fb237"], "direction": "support", "sufficient": True, **changes,
    })


def plan(**changes) -> ExecutionPlan:
    return ExecutionPlan.model_validate({
        "id": "p1", "claim_id": "c1", "condition_ids": ["fb237", "wn18rr"],
        "target_conditions": [condition().model_dump(), condition("wn18rr").model_dump()],
        "task": {"entry_script": "evaluate.py", "config": "config.json"},
        "run_mode": "evaluation", "y_paper": {"fb237": 0.355, "wn18rr": 0.479},
        "feasibility": "ready", "priority": "high", **changes,
    })


def test_claim_roundtrip_has_explicit_coverage_and_four_statuses() -> None:
    record = claim(evidence=[evidence().model_dump()], questions=[{
        "text": "Which seed produced Table 2?", "claim_id": "c1", "reason": "Seed absent",
    }], notes=["Only one setting covered."])
    assert Claim.model_validate_json(record.model_dump_json()) == record
    assert record.needs == [EvidenceNeed.EXPERIMENTS, EvidenceNeed.CODE]
    assert record.status is ClaimStatus.UNVERIFIED
    assert set(ClaimStatus) == {"supported", "flawed", "questioned", "unverified"}
    assert record.conditions[0].id == record.evidence[0].covered[0]
    # The LLM integration can request the actual nested JSON schema.
    assert "conditions" in Claim.model_json_schema()["required"]


@pytest.mark.parametrize("changes", [
    {"conditions": []},
    {"conditions": [condition().model_dump(), condition().model_dump()]},
    {"needs": ["Experiments", "Experiments"]},
    {"needs": ["empirical"]},
    {"status": "in_conflict"},
    {"type": "empirical"},
    {"loc": {}},
    {"loc": {"page": 0}},
    {"loc": {"char_start": 4}},
    {"loc": {"char_start": 4, "char_end": 4}},
    {"questions": [{"text": "Why?", "claim_id": "other"}]},
    {"evidence": [evidence(covered=["unclaimed"]).model_dump()]},
])
def test_claim_rejects_invalid_coverage_location_and_routing(changes: dict) -> None:
    with pytest.raises(ValidationError):
        claim(**changes)


@pytest.mark.parametrize(("source", "pointer"), [
    ("paper_internal", {"locator": "paper.pdf", "page": 3, "quote": "A specific result."}),
    ("theory", {"locator": "paper.md", "key": "sec_3", "quote": "x = y"}),
    ("literature", {"locator": "arxiv:1911.03082", "quote": "The retrieved passage."}),
    ("literature", {"locator": "doi:10.1234/abc", "quote": "The retrieved passage."}),
    ("literature", {"locator": "https://example.org/paper", "quote": "The retrieved passage."}),
    ("code", {"locator": "src/model.py", "line": 42}),
    ("execution", {"locator": "runs/eval/metrics.json", "key": "MRR"}),
])
def test_source_specific_pointer_roundtrip(source: str, pointer: dict) -> None:
    item = evidence(source=source, pointer=pointer, aligned=True if source == "execution" else None)
    assert Evidence.model_validate_json(item.model_dump_json()) == item


@pytest.mark.parametrize("changes", [
    {"pointer": {"locator": " "}},
    {"pointer": {"locator": "paper.pdf", "page": 1}},
    {"pointer": {"locator": "paper.pdf", "quote": "Text"}},
    {"source": "literature", "pointer": {"locator": "https://", "quote": "Text"}},
    {"source": "literature", "pointer": {"locator": "arxiv:", "quote": "Text"}},
    {"source": "literature", "pointer": {"locator": "doi:10.1234/paper"}},
    {"source": "code", "pointer": {"locator": "model.py"}},
    {"source": "execution", "pointer": {"locator": "metrics.json"}},
    {"source": "execution", "pointer": {"locator": "metrics.json", "key": "MRR"}, "aligned": False},
    {"source": "execution", "pointer": {"locator": "metrics.json", "key": "MRR"}},
    {"covered": []},
    {"covered": ["fb237", "fb237"]},
])
def test_evidence_rejects_uncheckable_or_unaligned_sufficiency(changes: dict) -> None:
    with pytest.raises(ValidationError):
        evidence(**changes)


def test_concern_and_execution_provenance_survive_roundtrip() -> None:
    item = evidence(
        source="execution", pointer={"locator": "metrics.json", "key": "MRR"},
        direction="flaw", concern=True, overturnable=False, affects_claim=True, aligned=True,
        provenance=ExecutionProvenance(
            released_artifact=True, artifact_kind="logs", environment_explanation_possible=False,
            run_id="r1", command=["python", "analyse.py"], runtime_conditions=[condition()],
        ).model_dump(),
    )
    assert Evidence.model_validate_json(item.model_dump_json()) == item
    assert item.provenance.released_artifact
    assert item.concern and not item.overturnable


def test_execution_plan_roundtrip_preserves_two_values_for_one_metric() -> None:
    value = plan()
    assert ExecutionPlan.model_validate_json(value.model_dump_json()) == value
    assert value.y_paper == {"fb237": 0.355, "wn18rr": 0.479}
    assert value.task == ExecutionTask(entry_script="evaluate.py", config="config.json")
    blocked = plan(feasibility="blocked", blocker="Released weights are missing", task={})
    assert blocked.blocker == "Released weights are missing"


@pytest.mark.parametrize("changes", [
    {"feasibility": "blocked"},
    {"blocker": "Missing weights"},
    {"task": {}},
    {"condition_ids": ["fb237"]},
    {"condition_ids": ["fb237", "fb237"]},
    {"priority": "urgent"},
    {"run_mode": "test"},
    {"y_paper": {"MRR": 0.355}},
    {"y_paper": {"fb237": float("nan"), "wn18rr": 0.479}},
    {"y_paper": {"fb237": float("inf"), "wn18rr": 0.479}},
    {"target_conditions": [{"id": "fb237", "dataset": "fb237"}, condition("wn18rr").model_dump()]},
])
def test_execution_plan_rejects_ambiguous_or_incomplete_targets(changes: dict) -> None:
    with pytest.raises(ValidationError):
        plan(**changes)


def test_finding_question_location_and_review_roundtrip() -> None:
    issue = Finding(kind="writing", loc=ClaimLocation(section="intro"),
                    evidence=[evidence()], level="definite_error", text="Misspelled term.")
    question = AuthorQuestion(text="What assumption is needed?", reason="Unstated assumption")
    assert AuthorQuestion.model_validate_json(question.model_dump_json()) == question
    review = FinalReview(paper_key="tiny", run_id="r1", claims=[claim()], findings=[issue], ledger=[{"run_id": "r1"}])
    assert FinalReview.model_validate_json(review.model_dump_json()) == review
    assert review.summary_counts == {"supported": 0, "flawed": 0, "questioned": 0, "unverified": 1}


@pytest.mark.parametrize(("old", "new"), [
    ("supported", "supported"), ("partially_supported", "questioned"),
    ("in_conflict", "questioned"), ("inconclusive", "unverified"),
    ('<span>✗ In conflict</span>', "questioned"), ("Paper-supported", "supported"),
    ("**In conflict**", "questioned"), ("**Supported**", "supported"),
])
def test_v1_label_adapter_never_promotes_conflict_to_flawed(old: str, new: str) -> None:
    assert adapt_v1_label(old) == new
    with pytest.raises(ValueError):
        adapt_v1_label("flawed")


def test_v1_json_adapter_preserves_raw_and_never_invents_pointers(tmp_path: Path) -> None:
    raw = {"assessments": [{"claim_id": "c1", "label": "in_conflict", "rationale": "Old judgement",
                            "evidence": [{"kind": "execution", "locator": "t1"}]}]}
    path = tmp_path / "review.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    old_bytes = path.read_bytes()
    result = read_v1_artifact(path)
    assert result.raw == raw
    assert result.claims[0].status is ClaimStatus.QUESTIONED
    assert result.claims[0].original_label == "in_conflict"
    assert result.claims[0].evidence == raw["assessments"][0]["evidence"]
    assert not result.claims[0].reassessed
    assert result.claims[0].location is None
    assert path.read_bytes() == old_bytes


def test_v1_markdown_and_html_tables_are_readable(tmp_path: Path) -> None:
    markdown = "| Claim | Status | Evidence | Location |\n|---|---|---|---|\n| Result | In conflict | Table 1 | §3 |"
    html_table = "<table><tr><th>Claim</th><th>Status</th></tr><tr><td>Result</td><td>In conflict</td></tr></table>"
    for index, text in enumerate([markdown, html_table]):
        path = tmp_path / f"report{index}.md"
        path.write_text(text, encoding="utf-8")
        result = read_v1_artifact(path)
        assert result.claims[0].text == "Result"
        assert result.claims[0].status is ClaimStatus.QUESTIONED
        assert result.raw == text


def test_v1_unknown_labels_are_preserved_and_reported(tmp_path: Path) -> None:
    raw = {"claim_results": [{"final_status": "unrecognized", "notes": ["Do not lose me"]}]}
    path = tmp_path / "claim_audit.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    result = read_v1_artifact(path)
    assert result.claims == []
    assert "unrecognized" in result.issues[0]
    assert result.raw == raw


def test_v1_markdown_keeps_escaped_pipes_and_reports_malformed_rows(tmp_path: Path) -> None:
    text = (
        "| Claim | Status | Evidence |\n|---|---|---|\n"
        "| Uses A \\| B | **In conflict** | Section 3 |\n"
        "| Malformed | Supported | Extra | Fourth cell |\n"
    )
    path = tmp_path / "report.md"
    path.write_text(text, encoding="utf-8")
    result = read_v1_artifact(path)
    assert result.claims[0].text == "Uses A | B"
    assert result.claims[0].original_label == "**In conflict**"
    assert result.claims[0].status is ClaimStatus.QUESTIONED
    assert "Malformed historical claim-table row" in result.issues[0]
    assert result.raw == text
