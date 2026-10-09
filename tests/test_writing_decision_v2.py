"""Mock protocol controls; these tests do not measure model semantic accuracy."""

import copy
import json
from pathlib import Path

import pytest

from common import run_stats
from llm.client import LLMConfig
from screening.checks import ask, check_writing
from tests.test_writing_sections_v2 import candidate, make_materials


def decision(candidate_id="writing_1", **change):
    return {
        "candidate_id": candidate_id,
        "decision": "accept",
        "confirmed_kind": "grammar",
        "reason": "visible_defect",
        "explanation": "The visible plural subject conflicts with its singular verb.",
        **change,
    }


def page_review(*rows, version="writing-decision-v1"):
    return {"version": version, "results": list(rows)}


def run_candidate(tmp_path, response, *, level="definite_error"):
    materials = make_materials(tmp_path, [("intro", "These results is fixed.")])
    calls, records = [], []

    def model(**kwargs):
        calls.append(kwargs)
        if kwargs["module"] == "screening_writing.validation":
            return response
        return {"findings": [candidate(materials.blocks[0].model_dump(), level=level)]}

    findings = check_writing(materials, call=model, records=records, recover_errors=True)
    return findings, records, calls


@pytest.mark.parametrize(
    "level,kind,expected",
    [
        ("clarity_issue", "grammar", "definite_error"),
        ("definite_error", "consequential_ambiguity", "clarity_issue"),
    ],
)
def test_confirmed_kind_sets_level_without_candidate_level_in_visual_payload(tmp_path, level, kind, expected):
    findings, records, calls = run_candidate(
        tmp_path, page_review(decision(confirmed_kind=kind)), level=level
    )
    assert [f.level for f in findings] == [expected]
    assert records[0].status == "checked" and records[0].confirmed_count == 1
    assert len(calls) == 2
    payload = json.loads(calls[1]["prompt"])
    assert "level" not in payload["candidates"][0]
    assert payload["candidates"][0]["text"] == "Correct the identified agreement error."
    assert payload["output_schema"]["properties"]["version"]["const"] == "writing-decision-v1"


@pytest.mark.parametrize("reason", ["parser_artifact", "style", "candidate_not_supported", "not_applicable"])
def test_explicit_reject_is_audited_and_never_becomes_finding(tmp_path, reason):
    response = page_review(decision(decision="reject", confirmed_kind=None, reason=reason))
    with run_stats.run_scope(tmp_path / "run_stats.json"):
        findings, records, calls = run_candidate(tmp_path, response)
    assert findings == [] and records[0].status == "checked"
    assert any(reason in issue for issue in records[0].issues)
    assert len(calls) == 2
    audit = json.loads(next((tmp_path / "visual_calls").glob("*.json")).read_text(encoding="utf-8"))
    assert audit["response"] == response


def test_uncertain_is_unavailable_and_empty_or_failed_is_not_rejection(tmp_path):
    response = page_review(decision(decision="uncertain", confirmed_kind=None, reason="insufficient_context"))
    findings, records, _ = run_candidate(tmp_path, response)
    assert findings == [] and records[0].status == "unavailable"
    assert any("insufficient_context" in issue for issue in records[0].issues)


@pytest.mark.parametrize(
    "response",
    [
        page_review(decision(decision="reject", reason="style")),
        page_review(decision(decision="uncertain", reason="insufficient_context")),
        page_review(decision(confirmed_kind=None)),
        page_review(decision(confirmed_kind="cross_reference")),
        page_review(decision(reason="style")),
        page_review(decision(decision="reject", confirmed_kind=None, reason="visible_defect")),
        page_review(decision(decision="uncertain", confirmed_kind=None, reason="parser_artifact")),
        page_review(decision(decision=True)),
        page_review(decision(confirmed_kind="manuscript_error")),
        page_review(decision(explanation="  ")),
        page_review(decision("other")),
        page_review(decision(), decision()),
        page_review(),
        page_review(decision(), version="writing-decision-v2"),
        {"results": [decision()]},
        {
            "results": [
                {
                    "candidate_id": "writing_1",
                    "classification": "manuscript_error",
                    "explanation": "Visible error.",
                }
            ]
        },
        page_review(
            {"candidate_id": "writing_1", "classification": "clarity_issue", "explanation": "Legacy result."}
        ),
        {"status": "error", "error": "offline"},
    ],
)
def test_invalid_decisions_preserve_page_failure_boundary(tmp_path, response):
    findings, records, calls = run_candidate(tmp_path, response)
    assert findings == [] and records[0].status == "failed"
    assert len(calls) == 2 and any("original PDF validation failed" in issue for issue in records[0].issues)


def test_bad_decision_rolls_back_section_and_preserves_healthy_neighbor(tmp_path):
    materials = make_materials(
        tmp_path, [("bad", "These results is fixed."), ("good", "Our tests is small.")]
    )
    records, calls = [], []

    def model(**kwargs):
        calls.append(kwargs["module"])
        payload = json.loads(kwargs["prompt"])
        if kwargs["module"] == "screening_writing":
            return {"findings": [candidate(b) for b in payload["blocks"]]}
        item = decision(payload["candidates"][0]["candidate_id"])
        if payload["page"] == 1:
            item["reason"] = "style"
        return page_review(item)

    findings = check_writing(materials, call=model, records=records, recover_errors=True)
    assert [f.loc.section for f in findings] == ["good"]
    assert [r.status for r in records] == ["failed", "checked"]
    assert calls == ["screening_writing", "screening_writing.validation"] * 2


@pytest.mark.parametrize("conflict", [False, True])
def test_same_reference_allegation_has_explicit_mock_accept_or_reject(tmp_path, conflict):
    target = "Table 10. Latency is 4 seconds." if conflict else "Table 10. Accuracy is 0.9."
    materials = make_materials(tmp_path, [("intro", "Table 10 reports accuracy 0.9."), ("appendix", target)])
    materials.blocks[1].kind = "table"
    validations = []

    def model(**kwargs):
        payload = json.loads(kwargs["prompt"])
        if kwargs["module"] == "screening_writing":
            return {
                "findings": [
                    candidate(
                        payload["blocks"][0],
                        category="cross_reference",
                        target_kind="table",
                        target_label="10",
                        reference_problem="inconsistent_target",
                        text="The cited table reports a different quantity.",
                    )
                ]
            }
        validations.append(payload)
        assert kwargs["images"] == [page.path for page in materials.pages]
        assert payload["candidates"][0]["targets"][0]["quote"] == target
        return page_review(
            decision(
                decision="accept" if conflict else "reject",
                confirmed_kind="cross_reference" if conflict else None,
                reason="visible_defect" if conflict else "candidate_not_supported",
                explanation="The displayed table measures latency."
                if conflict
                else "The reference and table both report accuracy 0.9.",
            )
        )

    records = []
    findings = check_writing(materials, call=model, records=records)
    assert bool(findings) is conflict and len(validations) == 1
    assert records[0].status == "checked"
    if findings:
        assert findings[0].text == "The cited table reports a different quantity."
        assert findings[0].evidence[0].pointer.quote == materials.blocks[0].text


def test_explanation_prose_is_retained_without_a_negation_word_decoder(tmp_path):
    explanation = "The displayed agreement error requires correction; no mathematical symbol is missing."
    findings, _, _ = run_candidate(tmp_path, page_review(decision(explanation=explanation)))
    assert len(findings) == 1 and explanation in findings[0].evidence[0].note


def test_first_response_is_complete_before_later_candidate_binding_rejection(tmp_path):
    materials = make_materials(tmp_path, [("intro", "These results is fixed.")])
    response = {
        "findings": [
            candidate(materials.blocks[0].model_dump()),
            candidate({"id": "foreign", "text": "Unbound text."}),
        ]
    }
    original = copy.deepcopy(response)
    calls = []

    def model(**kwargs):
        calls.append(kwargs["module"])
        return response

    with run_stats.run_scope(tmp_path / "run_stats.json"), pytest.raises(ValueError, match="own section"):
        check_writing(materials, call=model)
    files = list((tmp_path / "writing_calls").glob("*.json"))
    assert len(files) == 1 and calls == ["screening_writing"]
    audit = json.loads(files[0].read_text(encoding="utf-8"))
    assert audit["response"] == response == original
    assert audit["status"] == "returned" and audit["binding_status"] == "not_evaluated"
    assert audit["payload"]["blocks"][0]["text"] == materials.blocks[0].text
    assert audit["transport"] == "injected" and audit["module"] == "screening_writing"
    assert not (tmp_path / "visual_calls").exists()


@pytest.mark.parametrize("mode", ["error_envelope", "exception", "non_object"])
def test_raw_writing_audit_retains_failed_boundary_without_credentials(tmp_path, monkeypatch, mode):
    cfg = LLMConfig(
        "openai",
        "text-model",
        "https://fixture-user:fixture-password@host.invalid/v1?token=fixture-token",
        "fixture-key",
    )
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: cfg)
    raw = {
        "status": "error",
        "error": f"Failed at {cfg.base_url} using {cfg.api_key}",
        "detail": {"retry": False},
    }
    original = copy.deepcopy(raw)

    def model(**kwargs):
        if mode == "exception":
            raise RuntimeError(raw["error"])
        return raw if mode == "error_envelope" else ["unstructured", "complete response"]

    with run_stats.run_scope(tmp_path / "run_stats.json"), pytest.raises(RuntimeError):
        ask("Read text", {"section": "intro"}, module="screening_writing", call=model)
    path = next((tmp_path / "writing_calls").glob("*.json"))
    text = path.read_text(encoding="utf-8")
    audit = json.loads(text)
    assert audit["status"] == "failed" and audit["error"]
    assert audit["endpoint"] == "https://host.invalid/v1"
    for credential in ("fixture-user", "fixture-password", "fixture-token", "fixture-key"):
        assert credential not in text
    if mode == "error_envelope":
        assert audit["response"]["detail"] == {"retry": False}
    elif mode == "non_object":
        assert audit["response"] == ["unstructured", "complete response"]
    else:
        assert "response" not in audit
    assert raw == original


def test_audit_uses_active_run_and_does_not_change_other_text_checks(tmp_path):
    with run_stats.run_scope(tmp_path / "one" / "run_stats.json"):
        ask("Read text", {"section": "one"}, module="screening_writing", call=lambda **_: {"findings": []})
        with run_stats.run_scope(tmp_path / "two" / "run_stats.json"):
            ask(
                "Read text", {"section": "two"}, module="screening_writing", call=lambda **_: {"findings": []}
            )
        ask("Read tables", {}, module="screening_tables", call=lambda **_: {"findings": []})
    for name in ("one", "two"):
        files = list((tmp_path / name / "writing_calls").glob("*.json"))
        assert len(files) == 1
        assert json.loads(files[0].read_text(encoding="utf-8"))["payload"]["section"] == name


def test_direct_call_without_active_run_keeps_existing_behavior(tmp_path, monkeypatch):
    monkeypatch.delenv("FACTREVIEW_RUN_STATS_PATH", raising=False)
    result = {"findings": []}
    assert ask("Read text", {}, module="screening_writing", call=lambda **_: result) is result
    assert not list(Path(tmp_path).rglob("writing_calls"))
