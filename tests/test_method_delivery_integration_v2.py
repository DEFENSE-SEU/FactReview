"""New method delivery contracts, with every external service mocked."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
from pypdf import PdfReader

from review.report.advice import advice_input, checked_review, generate_advice
from review.report.v2 import render_markdown, write_review
from schemas.claim import AdviceItem, Claim, ClaimAdvice, Condition
from schemas.limitations import VerificationLimitation
from schemas.review import FinalReview
from screening.writing import WritingSectionRecord
from tests import test_pipeline_v2 as pipeline_fixtures
from tests.test_report_advice_v2 import payload, valid_response

offline_boundaries = pipeline_fixtures.offline_boundaries
tiny_inputs = pipeline_fixtures.tiny_inputs


def minimal():
    return FinalReview(
        paper_key="test",
        run_id="offline",
        claims=[
            Claim(
                id="c",
                text="The method is stable.",
                loc={"page": 1},
                conditions=[Condition(id="stability", description="stability")],
                needs=[],
            )
        ],
    )


def limitation():
    return VerificationLimitation(
        claim_id="c",
        condition_ids=["stability"],
        stage="Theory",
        kind="branch_failed",
        reason="Main-text source binding failed.",
    )


def test_legacy_advice_keeps_original_hash_and_new_limitations_invalidate_it():
    # Exact pre-extension serialized input: new fields must not alter this contract.
    original = {
        "claim": {
            "id": "c",
            "text": "The method is stable.",
            "loc": {"page": 1, "section": None, "char_start": None, "char_end": None},
            "source_block_id": None,
            "source_quote": None,
            "source_refs": [],
            "conditions": [
                {
                    "id": "stability",
                    "dataset": None,
                    "metric": None,
                    "settings": {},
                    "description": "stability",
                }
            ],
            "needs": [],
            "importance": "secondary",
            "questions": [],
            "evidence": [],
            "status": "unverified",
            "notes": [],
        },
        "ledger": [],
        "source_files": {},
        "basis": {
            "/coverage_gaps/stability": {
                "condition_ids": ["stability"],
                "content": "The available usable sufficient support does not cover this condition. This alone does not identify a missing artifact or a paper defect.",
            }
        },
    }
    review = minimal()
    claim = review.claims[0]
    assert advice_input(claim, [], version="advice-v1") == original
    digest = hashlib.sha256(
        json.dumps(
            original, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    claim.advice = ClaimAdvice(
        state="generated",
        input_sha256=digest,
        items=[
            AdviceItem(
                text="Additional checkable support is needed.",
                condition_ids=["stability"],
                basis_refs=["/coverage_gaps/stability"],
            )
        ],
    )
    assert checked_review(review).claims[0].advice.state == "generated"
    claim.verification_limitations.append(limitation())
    checked = checked_review(review).claims[0]
    assert checked.advice.state == "unavailable"
    assert "New verification records" in checked.advice.failure_reason
    assert claim.advice.state == "generated"  # Rendering must not mutate the saved input.


@pytest.mark.parametrize(
    "action,include_basis,expected",
    [
        ("author_question", True, "unavailable"),
        ("verification_followup", False, "unavailable"),
        ("verification_followup", True, "generated"),
    ],
)
def test_system_failed_check_requires_operator_followup(tmp_path, action, include_basis, expected):
    review = minimal()
    review.claims[0].verification_limitations.append(limitation())
    before = review.model_dump()

    def call(**kwargs):
        response = valid_response(payload(kwargs))
        item = response["items"][0]
        item["text"] = "Repair the source binding and retry the theory check."
        item["action"] = action
        if include_basis:
            item["basis_refs"].append("/verification_limitations/0")
        return response

    result = generate_advice(review, tmp_path, call=call)
    assert result.review.claims[0].advice.state == expected
    assert review.model_dump() == before
    assert result.review.claims[0].status == "unverified"
    text = render_markdown(result.review)
    assert "System verification limitations" in text
    assert "does not identify missing author material" in text
    if expected == "generated":
        assert "Action for the system operator" in text
        assert result.review.claims[0].advice.input_version == "advice-v2"


def test_pipeline_retains_section_coverage_policy_and_execution_counts(tiny_inputs, monkeypatch):
    tiny_inputs[0].anonymity_policy = "required"
    seen = []

    def writing(materials, **kwargs):
        seen.append((kwargs["anonymity_policy"], kwargs["recover_errors"]))
        for number, status in enumerate(("checked", "failed", "unavailable")):
            kwargs["records"].append(
                WritingSectionRecord(
                    section_id=f"s{number}",
                    section=f"Section {number}",
                    block_ids=[materials.blocks[0].id],
                    status=status,
                    anonymity_policy=kwargs["anonymity_policy"],
                )
            )
        kwargs["issues"].append("Section 1 provider unavailable.")
        return []

    monkeypatch.setattr("screening.stage.check_writing", writing)
    summary, _, _, runner = pipeline_fixtures.run_tiny(tiny_inputs, monkeypatch)
    assert seen == [("required", True)]
    assert summary["writing_coverage"] == {"total": 3, "checked": 1, "failed": 1, "unavailable": 1}
    assert summary["anonymity_policy"] == "required"
    screening = json.loads(Path(summary["outputs"]["screening"]).read_text("utf-8"))
    assert len(screening["writing_checks"]) == 3
    assert screening["writing_coverage"] == summary["writing_coverage"]
    report = json.loads(Path(summary["outputs"]["report_json"]).read_text("utf-8"))
    assert report["execution_requested"] is True and runner.call_count == 1
    assert Path(summary["outputs"]["theory_derivations"]).is_dir()
    assert "Writing screening is incomplete" in report["review_markdown"]
    assert "Submission anonymity policy: required" in report["review_markdown"]
    teaser = json.loads(Path(summary["outputs"]["teaser_json"]).read_text("utf-8"))
    assert teaser["execution"] == {
        "plan_records": 1,
        "recorded_attempts": 1,
        "claims_with_aligned_execution_evidence": 1,
        "outcomes_incomplete": False,
    }
    pdf = "\n".join(p.extract_text() for p in PdfReader(summary["outputs"]["report_pdf"]).pages)
    assert "Writing screening coverage" in pdf
    assert "1 Recorded Execution Attempts" in " ".join(pdf.split())


def test_completed_report_exposes_no_execution_and_partial_records_remain_unknown(tmp_path):
    review = minimal()
    review.execution_requested = True
    result = write_review(review, tmp_path / "completed")
    pdf = "\n".join(p.extract_text() for p in PdfReader(result["pdf"]).pages)
    assert "0 Recorded Execution Attempts" in " ".join(pdf.split())
    assert "| 0 | 0 | 0 |" in Path(result["markdown"]).read_text("utf-8")
    review.run_status, review.incomplete_stages = "partial", ["execution"]
    assert "additional attempts or outcomes may be unknown" in render_markdown(review)


@pytest.mark.parametrize("policy", ["required", "not_required", "unspecified"])
def test_cli_exposes_explicit_anonymity_policy(monkeypatch, policy):
    import pipeline_full

    monkeypatch.setattr(sys, "argv", ["factreview", "paper.pdf", "--anonymity-policy", policy])
    assert pipeline_full.parse_args().anonymity_policy == policy


def test_theory_trace_survives_advice_and_pdf_with_unchanged_assessment(tmp_path):
    from assessment import assess_claim
    from tests.test_theory_derivations_v2 import invoke, paper

    claim, materials, response = paper(tmp_path)
    verified = invoke(claim, materials, response)
    claim.evidence = verified.evidence
    claim.theory_derivations = verified.theory_derivations
    review = FinalReview(paper_key="trace", run_id="offline", claims=[assess_claim(claim)])
    result = generate_advice(review, tmp_path / "advice", call=lambda **kw: valid_response(payload(kw)))
    assert result.counts == {"generated": 1, "unavailable": 0}
    basis = advice_input(result.review.claims[0], [])
    assert basis["basis"]["/theory_derivations/0"]["content"]["trace"] == claim.theory_derivations[
        0
    ].trace.model_dump(mode="json")
    outputs = write_review(result.review, tmp_path / "report")
    persisted = FinalReview.model_validate_json(Path(outputs["json"]).read_text("utf-8"))
    after = persisted.claims[0]
    assert after.status == "supported" and after.evidence == claim.evidence
    assert after.theory_derivations == claim.theory_derivations
    pdf = " ".join(p.extract_text() for p in PdfReader(outputs["pdf"]).pages)
    assert "Theory derivation traces" in pdf and "Previous steps: s1" in pdf
    assert "mathematical correctness requires review" in pdf
    Path(materials.markdown_path).write_text("changed proof", encoding="utf-8")
    assert checked_review(result.review).claims[0].advice.state == "unavailable"


def test_reference_bibtex_survives_report_and_tampering_is_unavailable(tmp_path):
    from schemas.claim import Evidence, EvidencePointer, Finding
    from tests.test_reference_corrections_v2 import correction, files
    from tests.test_reference_records_v2 import report

    data = files(tmp_path, report())
    candidate = correction(data)
    assert candidate.state == "metadata_candidate"
    review = FinalReview(
        paper_key="bibliography",
        run_id="offline",
        findings=[
            Finding(
                kind="reference",
                loc={"page": 1},
                level="metadata_candidate",
                text="Identity-bound metadata for comparison; manuscript error remains unconfirmed.",
                evidence=[
                    Evidence(
                        source="paper_internal",
                        direction="flaw",
                        affects_claim=False,
                        pointer=EvidencePointer(
                            locator=str(tmp_path / "bibliography.txt"),
                            quote=candidate.raw_reference,
                            key="entry:0",
                        ),
                    )
                ],
                reference_correction=candidate,
            )
        ],
    )
    outputs = write_review(review, tmp_path / "report")
    saved = FinalReview.model_validate_json(Path(outputs["json"]).read_text("utf-8"))
    assert saved.findings[0].reference_correction == candidate
    assert saved.findings[0].evidence[0].direction == "flaw"
    assert candidate.corrected_bibtex in Path(outputs["markdown"]).read_text("utf-8")
    pdf = " ".join(p.extract_text() for p in PdfReader(outputs["pdf"]).pages)
    assert "10.1000/one" in pdf and "Field" in pdf and "record-1" in pdf
    assert "paper-internal / flaw" not in pdf
    assert "reference source; discrepancy unconfirmed" in pdf
    review.findings[0].reference_correction.corrected_bibtex += "\n@article{forged,title={Wrong work}}"
    invalid = write_review(review, tmp_path / "invalid", render_pdf=False)
    persisted = FinalReview.model_validate_json(Path(invalid["json"]).read_text("utf-8"))
    assert persisted.findings[0].reference_correction.state == "unavailable"
    assert "@article{forged" not in Path(invalid["markdown"]).read_text("utf-8")
    assert "@article{forged" in review.findings[0].reference_correction.corrected_bibtex


@pytest.mark.parametrize("change", ["before", "after", "deleted", "trace_hash"])
def test_theory_original_hash_is_checked_before_and_after_advice(tmp_path, change):
    from tests.test_theory_derivations_v2 import invoke, paper

    claim, materials, response = paper(tmp_path)
    verified = invoke(claim, materials, response)
    claim.theory_derivations = verified.theory_derivations
    review = FinalReview(paper_key="trace mutation", run_id="offline", claims=[claim])
    original_hashes = dict(claim.theory_derivations[0].source_hashes)
    source = Path(materials.markdown_path)
    seen = []

    def call(**kw):
        seen.append(kw["module"])
        return valid_response(payload(kw))

    if change == "before":
        source.write_text("changed before advice", encoding="utf-8")
    elif change == "deleted":
        source.unlink()
    elif change == "trace_hash":
        claim.theory_derivations[0].trace.assumptions[0].sources[0].artifact_sha256 = "0" * 64
    result = generate_advice(review, tmp_path / "advice", call=call)
    if change == "after":
        assert result.review.claims[0].advice.state == "generated" and seen == ["report_generation"]
        source.write_text("changed after advice", encoding="utf-8")
    else:
        assert not seen
    validated = checked_review(result.review)
    assert validated.claims[0].advice.state == "unavailable"
    assert "Theory source artifact changed" in validated.claims[0].advice.failure_reason
    assert "Current Theory source integrity is unavailable" in render_markdown(validated)
    assert claim.theory_derivations[0].source_hashes == original_hashes
