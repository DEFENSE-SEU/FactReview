"""Explicit mocked traces and visual classifications; no model-quality claim."""

import copy
import json
from pathlib import Path

import pytest

from assessment import assess_claim
from llm.client import LLMConfig
from schemas.claim import Claim, ClaimLocation, Condition
from schemas.materials import MaterialBlock, SharedMaterials
from tests.test_theory_notation_source_v2 import context
from verification.theory import verify_theory


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Only explicit mocked Theory boundaries are allowed")

    cfg = LLMConfig("mock", "integrated-trace", None, None)
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: cfg)
    monkeypatch.setattr("screening.checks.resolve_vlm_config", lambda **kwargs: cfg)
    monkeypatch.setattr("screening.checks.llm_json", blocked)
    monkeypatch.setattr("httpx.Client.send", blocked)
    monkeypatch.setattr("httpx.AsyncClient.send", blocked)
    monkeypatch.setattr("requests.sessions.Session.request", blocked)
    monkeypatch.setattr("subprocess.run", blocked)


def integrated_case(tmp_path):
    texts = [
        "For every real x, a(x) = x*x.",
        "For every real x, b(x) = 2*x.",
        "We define F(x) = a(x) + b(x).",
        "For every real x, F(x) = x*x + 2*x.",
    ]
    markdown = "\n\n".join(texts)
    path = tmp_path / "paper.md"
    path.write_text(markdown, encoding="utf-8")
    blocks = []
    start = 0
    for index, text in enumerate(texts):
        blocks.append(
            MaterialBlock(
                id=f"b{index + 1}",
                text=text,
                loc=ClaimLocation(page=1, section="Method", char_start=start, char_end=start + len(text)),
            )
        )
        start += len(text) + 2
    materials = SharedMaterials(
        paper_key="integrated-definition",
        source_pdf=str(tmp_path / "paper.pdf"),
        markdown=markdown,
        markdown_path=str(path),
        content_list_path="",
        provider="fixture",
        blocks=blocks,
    )
    claim = Claim(
        id="integrated",
        text=texts[3],
        source_block_id="b4",
        source_quote=texts[3],
        loc=blocks[3].loc,
        conditions=[Condition(id="c1", description="For every real x")],
        needs=["Theory"],
    )
    trace = {
        "goal": texts[3],
        "assumptions": [
            {
                "id": f"a{index + 1}",
                "text": text,
                "status": "paper_explicit",
                "sources": [{"block_id": f"b{index + 1}", "quote": text}],
            }
            for index, text in enumerate(texts[:3])
        ],
        "steps": [
            {
                "id": "s1",
                "statement": "For every real x, F(x) = x*x + b(x).",
                "reason": "Substitute the first explicitly defined function into F.",
                "assumption_ids": ["a1", "a3"],
                "previous_step_ids": [],
                "sources": [],
            },
            {
                "id": "s2",
                "statement": texts[3],
                "reason": "Substitute the second explicit definition, retaining the real domain.",
                "assumption_ids": ["a2"],
                "previous_step_ids": ["s1"],
                "sources": [],
            },
        ],
        "gaps": [],
        "outcome": "completed",
        "completion_reason": "Three separately quoted definitions establish the full target.",
    }
    response = {
        "schema_version": "theory-derivation-v1",
        "appendix_block_ids": [],
        "items": [
            {
                "block_id": "b3",
                "quote": texts[2],
                "step_quote": "F(x) = a(x) + b(x)",
                "covered": ["c1"],
                "fully_supported_conditions": ["c1"],
                "kind": "derivation",
                "direction": "support",
                "detail": "The primary step and the complete source-bound trace establish the target.",
            }
        ],
        "derivations": [{"item_index": 0, "trace": trace}],
    }
    return claim, materials, response


def assess(claim, result):
    target = claim.model_copy(deep=True)
    target.evidence.extend(result.evidence)
    target.theory_derivations.extend(result.theory_derivations)
    target.verification_limitations.extend(result.verification_limitations)
    return assess_claim(target)


@pytest.mark.parametrize("change", [None, "foreign_source", "future_step", "gap", "partial_flag"])
def test_complete_trace_can_join_loaded_sources_without_relaxing_guards(tmp_path, change):
    claim, materials, response = integrated_case(tmp_path)
    trace = response["derivations"][0]["trace"]
    if change == "foreign_source":
        trace["assumptions"][1]["sources"][0]["block_id"] = "foreign"
    elif change == "future_step":
        trace["steps"][0]["previous_step_ids"] = ["s2"]
    elif change == "gap":
        trace["gaps"] = [
            {
                "at": "goal",
                "reason": "An explicit unresolved qualifier.",
                "needed": "A source.",
                "sources": [],
            }
        ]
    elif change == "partial_flag":
        response["items"][0]["fully_supported_conditions"] = []
    original = copy.deepcopy(response)
    calls = []

    def model(**kwargs):
        calls.append(kwargs["module"])
        payload = json.loads(kwargs["prompt"])
        assert payload["claim"] == claim.model_dump(mode="json")
        assert payload["allowed_theory_source_block_ids"] == ["b1", "b2", "b3", "b4"]
        return copy.deepcopy(response)

    result = verify_theory(claim, materials, call=model, output_dir=tmp_path / "audit")
    assert calls == ["verification.theory"]
    assert assess(claim, result).status == ("supported" if change is None else "unverified")
    assert response == original
    if change is None:
        record = result.theory_derivations[0]
        assert {s.block_id for a in record.trace.assumptions for s in a.sources} == {"b1", "b2", "b3"}
        assert result.evidence[0].pointer.quote == materials.blocks[2].text
        for assumption in record.trace.assumptions:
            source = assumption.sources[0]
            start, end = map(int, source.pointer.key[6:].split("-"))
            assert Path(source.pointer.locator).read_text("utf-8")[start:end] == source.pointer.quote


def run_notation(
    tmp_path,
    *,
    healthy=None,
    healthy_first=False,
    classification="parser_artifact",
    second_condition=False,
    duplicates=False,
):
    claim, materials, response, _, _ = context(tmp_path)
    response["items"][0]["fully_supported_conditions"] = []
    if second_condition:
        claim.conditions.append(Condition(id="c2", description="For nonnegative real x and y"))
        response["items"][0]["covered"] = ["c1", "c2"]
    if duplicates:
        response["items"].append(copy.deepcopy(response["items"][0]))
    if healthy is not None:
        support = copy.deepcopy(response["items"][0])
        support.update(
            kind="derivation",
            direction="support",
            covered=["c1"],
            fully_supported_conditions=["c1"] if healthy else [],
        )
        response["items"].insert(0 if healthy_first else len(response["items"]), support)
    base_trace = response["derivations"][0]["trace"]
    response["derivations"] = [
        {"item_index": i, "trace": copy.deepcopy(base_trace)} for i in range(len(response["items"]))
    ]
    original = copy.deepcopy(response)
    calls = []

    def model(**kwargs):
        calls.append(kwargs["module"])
        if kwargs["module"] == "verification.theory.notation":
            return {
                "classification": classification,
                "explanation": "Explicit mock: the rendered symbol is ordinary multiplication; the parsed superscript is not printed.",
            }
        assert kwargs["module"] == "verification.theory"
        return copy.deepcopy(response)

    result = verify_theory(claim, materials, call=model, output_dir=tmp_path / "audit")
    assert response == original
    assert "verification.theory.concern_scope" not in calls
    assert not result.questions
    return claim, result, calls


def test_suppressed_parser_artifact_keeps_condition_level_system_limitation(tmp_path):
    claim, result, calls = run_notation(tmp_path)
    assert calls == ["verification.theory", "verification.theory.notation"]
    assert result.evidence == [] and assess(claim, result).status == "unverified"
    (limitation,) = result.verification_limitations
    assert limitation.kind == "source_context_unavailable" and limitation.responsibility == "system"
    assert limitation.condition_ids == ["c1"] and "parser_artifact" in limitation.reason


@pytest.mark.parametrize("healthy_first", [True, False])
def test_healthy_support_is_preserved_in_either_item_order(tmp_path, healthy_first):
    claim, result, _ = run_notation(tmp_path, healthy=True, healthy_first=healthy_first)
    assert assess(claim, result).status == "supported"
    assert result.evidence[0].sufficient and not result.verification_limitations


def test_only_unresolved_condition_gets_parser_limitation(tmp_path):
    claim, result, _ = run_notation(tmp_path, healthy=True, second_condition=True)
    (limitation,) = result.verification_limitations
    assert limitation.condition_ids == ["c2"]
    assert result.evidence[0].sufficient and result.evidence[0].covered == ["c1"]
    assert assess(claim, result).status == "unverified"


def test_partial_support_does_not_erase_parser_limitation(tmp_path):
    claim, result, _ = run_notation(tmp_path, healthy=False)
    assert not result.evidence[0].sufficient
    assert result.verification_limitations[0].condition_ids == ["c1"]
    assert assess(claim, result).status == "unverified"


def test_repeated_parser_observations_keep_one_condition_limitation(tmp_path):
    _, result, calls = run_notation(tmp_path, duplicates=True)
    assert calls.count("verification.theory.notation") == 2
    assert len(result.verification_limitations) == 1


def test_uncertain_pixels_do_not_claim_confirmed_parser_artifact(tmp_path):
    _, result, _ = run_notation(tmp_path, classification="uncertain")
    assert not result.verification_limitations and not result.evidence
    assert any("uncertain" in issue for issue in result.issues)
