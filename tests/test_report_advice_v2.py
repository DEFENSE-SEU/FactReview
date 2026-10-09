"""Report advice uses immutable assessed records; every external boundary is mocked."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from assessment import assess_claims
from llm.client import LLMConfig
from schemas.claim import AuthorQuestion, Claim, ClaimStatus, Condition, Evidence, EvidencePointer
from schemas.review import FinalReview


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Advice tests must not call a provider, retrieval or Docker")

    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("llm.client.llm_json", forbidden)
    from review.report import advice

    monkeypatch.setattr(advice, "llm_json", forbidden)
    monkeypatch.setattr(advice, "resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None))


def review(tmp_path):
    source = tmp_path / "source.md"
    source.write_text("A has accuracy 90 on D. B has accuracy 80 on D.", encoding="utf-8")
    ptr = EvidencePointer(locator=str(source), page=1, quote=source.read_text("utf-8"))
    support = Evidence(
        source="paper_internal", pointer=ptr, covered=["a"], direction="support", sufficient=True
    )
    flaw = Evidence(
        source="paper_internal",
        pointer=ptr,
        covered=["a"],
        direction="flaw",
        sufficient=True,
        overturnable=False,
    )
    seeds = [
        Claim(
            id=name,
            text="A reports accuracy 90 on D.",
            loc={"page": 1},
            conditions=[Condition(id="a", dataset="D", metric="accuracy")],
            needs=["Experiments"],
            evidence=items,
            questions=[AuthorQuestion(text="Which protocol was used?", claim_id=name)],
            notes=["Protocol detail has not been recorded."],
        )
        for name, items in (("s", [support]), ("f", [flaw]), ("q", [support, flaw]), ("u", []))
    ]
    return FinalReview(paper_key="fixture", run_id="offline", claims=assess_claims(seeds))


def payload(kwargs):
    assert kwargs["module"] == "report_generation"
    return json.loads(kwargs["prompt"].split("\nADVICE_DATA_JSON:\n", 1)[1])


def valid_response(data):
    claim = data["claim"]
    ids = [c["id"] for c in claim["conditions"]]
    if claim["status"] == "unverified":
        refs = [key for key in data["basis"] if key.startswith("/coverage_gaps/")]
        text = "Please provide checkable evidence for the uncovered conditions."
    else:
        refs = [key for key in data["basis"] if key.startswith("/evidence/")]
        text = {
            "supported": "The available paper evidence supports this stated condition.",
            "flawed": "The recorded contradiction is a problem under this stated condition.",
            "questioned": "How do the authors reconcile the supporting and opposing evidence?",
        }[claim["status"]]
    return {
        "status": "ok",
        "claim_id": claim["id"],
        "items": [{"text": text, "condition_ids": ids, "basis_refs": refs}],
    }


def generate(original, tmp_path, call=None):
    from review.report.advice import generate_advice

    return generate_advice(
        original, tmp_path / "advice", call=call or (lambda **kw: valid_response(payload(kw)))
    )


def test_four_statuses_generate_after_assessment_with_traceable_local_basis(tmp_path):
    original = review(tmp_path)
    frozen = original.model_dump(mode="json")
    seen = []

    def call(**kwargs):
        data = payload(kwargs)
        seen.append((data["claim"]["id"], data["claim"]["status"]))
        return valid_response(data)

    result = generate(original, tmp_path, call)
    assert seen == [("s", "supported"), ("f", "flawed"), ("q", "questioned"), ("u", "unverified")]
    assert result.counts == {"generated": 4, "unavailable": 0}
    assert not result.issues and original.model_dump(mode="json") == frozen
    for old, new in zip(original.claims, result.review.claims, strict=True):
        assert new.model_dump(exclude={"advice"}) == old.model_dump(exclude={"advice"})
        assert new.advice.state == "generated"
        audit = json.loads(Path(new.advice.audit_pointer).read_text("utf-8"))
        assert audit["state"] == "generated" and audit["module"] == "report_generation"
        assert audit["response"]["claim_id"] == new.id
        assert audit["input_sha256"] == new.advice.input_sha256
        assert audit["provider"] == "mock" and audit["model"] == "fixture"
        assert all(ref in audit["input"]["basis"] for item in new.advice.items for ref in item.basis_refs)


@pytest.mark.parametrize(
    "failure", ["raise", "empty", "unknown", "foreign", "duplicate", "status", "recommendation"]
)
def test_bad_claim_advice_is_unavailable_and_healthy_neighbors_survive(tmp_path, failure):
    original = review(tmp_path)

    def call(**kwargs):
        data = payload(kwargs)
        response = valid_response(data)
        if data["claim"]["id"] != "f":
            return response
        if failure == "raise":
            raise RuntimeError("provider failed")
        if failure == "empty":
            response["items"] = []
        elif failure == "unknown":
            response["items"][0]["basis_refs"] = ["/evidence/99"]
        elif failure == "foreign":
            response["claim_id"] = "s"
        elif failure == "duplicate":
            response["items"][0]["condition_ids"] = ["a", "a"]
        elif failure == "status":
            response["new_status"] = "supported"
        elif failure == "recommendation":
            response["items"][0]["text"] = "I recommend accepting this paper."
        return response

    result = generate(original, tmp_path, call)
    assert result.counts == {"generated": 3, "unavailable": 1}
    assert result.review.claims[1].advice.state == "unavailable"
    assert result.review.claims[1].advice.items == []
    assert result.review.claims[1].status == "flawed"
    audit = json.loads(Path(result.review.claims[1].advice.audit_pointer).read_text("utf-8"))
    assert audit["state"] == "unavailable" and audit["failure_reason"]
    assert len(result.issues) == 1


def test_unverified_advice_cannot_borrow_already_supported_condition(tmp_path):
    original = review(tmp_path)
    c = original.claims[0]
    c.conditions.append(Condition(id="b", dataset="Other", metric="accuracy"))
    original.claims = assess_claims([c])

    def call(**kwargs):
        data = payload(kwargs)
        assert "/coverage_gaps/a" not in data["basis"]
        assert "/coverage_gaps/b" in data["basis"]
        return {
            "status": "ok",
            "claim_id": "s",
            "items": [
                {"text": "Provide evidence.", "condition_ids": ["a"], "basis_refs": ["/coverage_gaps/b"]}
            ],
        }

    assert generate(original, tmp_path, call).counts["unavailable"] == 1


def test_conflicting_evidence_requires_both_sides_in_question(tmp_path):
    original = review(tmp_path)
    original.claims = [original.claims[2]]

    def call(**kwargs):
        response = valid_response(payload(kwargs))
        response["items"][0]["basis_refs"] = ["/evidence/0"]
        return response

    assert generate(original, tmp_path, call).counts["unavailable"] == 1


@pytest.mark.parametrize("change", ["claim", "evidence", "conditions", "source"])
def test_callback_mutation_is_detected_without_mutating_returned_assessment(tmp_path, change):
    original = review(tmp_path)
    expected = original.model_dump(mode="json", exclude={"review_markdown"})

    def call(**kwargs):
        data = payload(kwargs)
        if data["claim"]["id"] == "s":
            c = original.claims[0]
            if change == "claim":
                c.status = ClaimStatus.FLAWED
            elif change == "evidence":
                c.evidence[0].covered = []
            elif change == "conditions":
                c.conditions[0].dataset = "altered"
            else:
                Path(c.evidence[0].pointer.locator).write_text("source changed", encoding="utf-8")
        return valid_response(data)

    result = generate(original, tmp_path, call)
    assert result.review.claims[0].advice.state == "unavailable"
    if change == "source":
        assert result.counts == {"generated": 1, "unavailable": 3}
    for actual, frozen in zip(result.review.claims, expected["claims"], strict=True):
        assert actual.model_dump(exclude={"advice"}, mode="json") == {
            k: v for k, v in frozen.items() if k != "advice"
        }


@pytest.mark.parametrize("change", ["status", "basis", "source", "source_deleted", "conditions", "ledger"])
def test_saved_advice_revalidated_on_static_render(tmp_path, change):
    from review.report.v2 import write_review

    original = review(tmp_path)
    original.ledger = [{"plan": {"claim_id": "s", "condition_ids": ["a"]}, "reason": "No weights"}]
    result = generate(original, tmp_path)
    restored = FinalReview.model_validate_json(result.review.model_dump_json())
    c = restored.claims[0]
    if change == "status":
        c.status = ClaimStatus.UNVERIFIED
    elif change == "basis":
        c.advice.items[0].basis_refs = ["/evidence/99"]
    elif change == "conditions":
        c.conditions[0].dataset = "altered"
    elif change == "ledger":
        restored.ledger[0]["reason"] = "Changed reason"
    elif change == "source":
        Path(c.evidence[0].pointer.locator).write_text("source changed", encoding="utf-8")
    else:
        Path(c.evidence[0].pointer.locator).unlink()
    before = restored.model_dump(mode="json")
    output = write_review(restored, tmp_path / "render", render_pdf=False)
    record = FinalReview.model_validate_json(Path(output["json"]).read_text("utf-8"))
    assert next(row for row in record.claims if row.id == "s").advice.state == "unavailable"
    assert "Advice unavailable" in Path(output["markdown"]).read_text("utf-8")
    assert restored.model_dump(mode="json") == before


def test_rendered_advice_links_and_old_questions_preserved(tmp_path):
    from markdown_it import MarkdownIt
    from pypdf import PdfReader

    from review.report.v2 import write_review

    result = generate(review(tmp_path), tmp_path)
    output = write_review(result.review, tmp_path / "render")
    assert "pdf_error" not in output
    text = Path(output["markdown"]).read_text("utf-8")
    assert text.count("Reviewer advice:") == 4 and text.count("Which protocol was used?") == 4
    assert sum(line.startswith("## ") for line in text.splitlines()) == 4
    links = [
        child.attrGet("href")
        for token in MarkdownIt().parse(text)
        for child in token.children or []
        if child.type == "link_open"
    ]
    assert any(link.startswith("#factreview-evidence-") for link in links)
    assert "Reviewer advice" in " ".join(page.extract_text() or "" for page in PdfReader(output["pdf"]).pages)


def test_old_json_reads_with_no_generated_advice(tmp_path):
    original = review(tmp_path).model_dump(mode="json")
    for c in original["claims"]:
        c.pop("advice", None)
    restored = FinalReview.model_validate(original)
    assert all(c.advice is None for c in restored.claims)
    assert all(c.questions for c in restored.claims)


def test_response_extra_actions_cannot_change_claim_contract(tmp_path):
    original = review(tmp_path)

    def call(**kwargs):
        response = valid_response(payload(kwargs))
        response["evidence"] = []
        return response

    assert generate(original, tmp_path, call).counts == {"generated": 0, "unavailable": 4}


def test_invalid_advice_schema_rejects_empty_generated_record():
    from schemas.claim import ClaimAdvice

    with pytest.raises(ValidationError):
        ClaimAdvice(state="generated", items=[], input_sha256="0" * 64)


def test_unaligned_execution_cannot_supply_supported_advice(tmp_path):
    original = review(tmp_path)
    original.claims = [original.claims[0]]
    original.claims[0].evidence.append(
        Evidence(
            source="execution",
            pointer=EvidencePointer(locator="metrics.json", key="score"),
            covered=["a"],
            direction="support",
            aligned=False,
            sufficient=False,
        )
    )

    def call(**kwargs):
        response = valid_response(payload(kwargs))
        response["items"][0]["basis_refs"] = ["/evidence/1"]
        return response

    assert generate(original, tmp_path, call).counts["unavailable"] == 1


def test_ledger_is_claim_local_and_execution_failure_stays_a_limit(tmp_path):
    original = review(tmp_path)
    original.claims = [original.claims[-1]]
    original.ledger = [
        {"plan": {"claim_id": "other", "condition_ids": ["a"]}, "reason": "Other claim secret"},
        {"plan": {"claim_id": "u", "condition_ids": ["a"]}, "reason": "Weights were unavailable"},
    ]

    def call(**kwargs):
        data = payload(kwargs)
        assert len(data["ledger"]) == 1
        assert "Other claim secret" not in kwargs["prompt"]
        response = valid_response(data)
        response["items"][0]["basis_refs"].append("/ledger/0")
        response["items"][0]["text"] = (
            "Please provide the unavailable weights needed to evaluate this condition."
        )
        return response

    result = generate(original, tmp_path, call)
    assert result.counts["generated"] == 1
    assert result.review.claims[0].status == "unverified" and result.review.claims[0].evidence == []


def test_credentials_and_unsafe_error_wording_do_not_escape_into_report(tmp_path, monkeypatch):
    from review.report import advice
    from review.report.v2 import write_review

    monkeypatch.setattr(
        advice,
        "resolve_llm_config",
        lambda: LLMConfig(
            "mock",
            "fixture",
            "https://user:password@example.invalid/v1?api_key=query-secret",
            "fake-private-key",
        ),
    )

    def call(**kwargs):
        raise RuntimeError(
            "Decision: reject; fake-private-key https://user:password@example.invalid/v1?api_key=query-secret"
        )

    result = generate(review(tmp_path), tmp_path, call)
    assert result.counts["unavailable"] == 4
    output = write_review(result.review, tmp_path / "render", render_pdf=False)
    all_text = Path(output["json"]).read_text("utf-8") + "".join(
        Path(c.advice.audit_pointer).read_text("utf-8") for c in result.review.claims
    )
    assert (
        "fake-private-key" not in all_text
        and "query-secret" not in all_text
        and "user:password" not in all_text
    )
    assert "Advice unavailable" in Path(output["markdown"]).read_text("utf-8")


def test_callback_replacing_later_claim_does_not_corrupt_snapshot(tmp_path):
    original = review(tmp_path)
    frozen = original.model_dump(mode="json")

    def call(**kwargs):
        data = payload(kwargs)
        if data["claim"]["id"] == "s":
            original.claims[1] = original.claims[0].model_copy(deep=True)
        return valid_response(data)

    result = generate(original, tmp_path, call)
    assert result.counts == {"generated": 3, "unavailable": 1}
    for actual, old in zip(result.review.claims, frozen["claims"], strict=True):
        assert actual.model_dump(mode="json", exclude={"advice"}) == {
            k: v for k, v in old.items() if k != "advice"
        }


def test_advice_directory_failure_still_delivers_pipeline_report(tmp_path, monkeypatch):
    from tests.test_pipeline_v2 import offline_boundaries, run_tiny, tiny_inputs

    offline_boundaries.__wrapped__(monkeypatch)
    inputs = tiny_inputs.__wrapped__(tmp_path)
    original_mkdir = Path.mkdir

    def fail_advice_directory(path, *args, **kwargs):
        if path.name == "advice":
            raise PermissionError("Synthetic advice directory is unwritable")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_advice_directory)
    summary, model, _, _ = run_tiny(inputs, monkeypatch, render_pdf=False)
    assert summary["stages"]["report"] == "ok"
    assert summary["advice"] == {"generated": 0, "unavailable": 4}
    assert "advice" not in summary["outputs"]
    assert "report_generation" not in model.calls
    assert summary["run_stats"]["modules"]["report_generation"]["token_usage"]["requests"] == 0
    result = FinalReview.model_validate_json(Path(summary["outputs"]["report_json"]).read_text("utf-8"))
    assessed = FinalReview.model_validate_json(
        Path(summary["outputs"]["assessment_snapshot"]).read_text("utf-8")
    )
    old = {claim.id: claim for claim in assessed.claims}
    for claim in result.claims:
        assert claim.model_dump(exclude={"advice"}) == old[claim.id].model_dump(exclude={"advice"})
        assert claim.advice.state == "unavailable" and claim.advice.audit_pointer is None
        assert "Synthetic advice directory is unwritable" in claim.advice.failure_reason
    assert any("advice" in issue and "unwritable" in issue for issue in summary["issues"])
    assert "Advice unavailable" in Path(summary["outputs"]["report_markdown"]).read_text("utf-8")


def test_later_callback_source_change_rechecks_all_advice_and_counts(tmp_path):
    from review.report.v2 import write_review

    original = review(tmp_path)
    before = original.model_dump(mode="json")
    seen = []

    def call(**kwargs):
        data = payload(kwargs)
        seen.append(data["claim"]["id"])
        if data["claim"]["id"] == "u":
            (tmp_path / "source.md").write_text("Changed after the first three calls.", encoding="utf-8")
        return valid_response(data)

    result = generate(original, tmp_path, call)
    assert seen == ["s", "f", "q", "u"]
    assert result.counts == {"generated": 1, "unavailable": 3}
    assert original.model_dump(mode="json") == before
    assert len(result.issues) == 3
    for claim in result.review.claims[:3]:
        assert claim.advice.state == "unavailable" and not claim.advice.items
        assert "source artifact changed" in claim.advice.failure_reason
        assert any(issue.startswith(claim.id + ":") for issue in result.issues)
        audit = json.loads(Path(claim.advice.audit_pointer).read_text("utf-8"))
        assert audit["state"] == "generated"  # Preserve the original successful call history.
    assert result.review.claims[-1].advice.state == "generated"
    outputs = write_review(result.review, tmp_path / "render", render_pdf=False)
    delivered = FinalReview.model_validate_json(Path(outputs["json"]).read_text("utf-8"))
    assert result.counts == {
        state: sum(claim.advice.state == state for claim in delivered.claims)
        for state in ("generated", "unavailable")
    }
    originals = {claim.id: claim for claim in original.claims}
    assert set(originals) == {claim.id for claim in delivered.claims}
    for new in delivered.claims:
        old = originals[new.id]
        assert old.model_dump(exclude={"advice"}) == new.model_dump(exclude={"advice"})
