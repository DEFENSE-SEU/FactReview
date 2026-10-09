"""Coverage follow-up integration; all service and execution boundaries are mocked."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import pipeline_full
import pipeline_v2
from assessment import assess_claims
from schemas.claim import Claim, ClaimLocation, Condition, Evidence, EvidenceNeed, EvidencePointer
from schemas.materials import MaterialBlock, SharedMaterials
from screening import stage
from screening.claim_coverage import ClaimCoverageResult
from tests.test_pipeline_v2 import offline_boundaries as offline_boundaries
from tests.test_pipeline_v2 import tiny_inputs as tiny_inputs
from verification.contracts import BranchResult
from verification.dispatch import verify_claims


def first_claims(materials):
    block = next(b for b in materials.blocks if "A test MRR is 0.4." in b.text)
    return [
        Claim(
            id=f"claim_{index:03d}",
            text=text,
            source_block_id=block.id,
            source_quote=text,
            loc=ClaimLocation(page=1),
            conditions=[Condition(id="c1", description=text)],
            needs=[need],
        )
        for index, (text, need) in enumerate(
            [("A test MRR is 0.4.", EvidenceNeed.EXPERIMENTS), ("We use Adam.", EvidenceNeed.CODE)], 1
        )
    ]


@pytest.fixture
def isolated_stage(monkeypatch, tmp_path):
    text = "A test MRR is 0.4. We use Adam."
    markdown = tmp_path / "paper.md"
    markdown.write_text(text, encoding="utf-8")
    materials = SharedMaterials(
        paper_key="coverage",
        source_pdf=str(tmp_path / "paper.pdf"),
        markdown=text,
        markdown_path=str(markdown),
        content_list_path="",
        provider="fixture",
        blocks=[MaterialBlock(id="b1", text=text, loc=ClaimLocation(page=1))],
    )
    for name in ("check_writing", "check_tables"):
        monkeypatch.setattr(stage, name, lambda *args, **kwargs: [])
    for name in ("check_figures", "check_visual_tables", "check_bibliography"):
        monkeypatch.setattr(stage, name, lambda *args, **kwargs: ([], []))
    return materials


def coverage_result(claims, directory, *, blocked=()):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    audit = directory / "coverage.json"
    summary = {
        "status": "partial" if blocked else "complete",
        "initial_claims": len(claims),
        "final_claims": len(claims),
        "windows_total": 2,
        "windows_reviewed": 2,
        "windows_unreviewed": 0,
        "unresolved_observations": len(blocked),
        "blocked_claim_ids": list(blocked),
        "audit_path": str(audit),
    }
    audit.write_text(json.dumps(summary), encoding="utf-8")
    return ClaimCoverageResult(claims, summary, [], list(blocked))


def healthy_branch(claim, materials):
    code = claim.needs == [EvidenceNeed.CODE]
    pointer = EvidencePointer(locator=materials.source_pdf, page=1, quote=claim.source_quote)
    if code:
        source = Path(materials.markdown_path).with_name("coverage_model.py")
        source.write_text("optimizer = 'Adam'\n", encoding="utf-8")
        pointer = EvidencePointer(locator=str(source), line=1, quote="optimizer = 'Adam'")
    return BranchResult(
        evidence=[
            Evidence(
                source="code" if code else "paper_internal",
                pointer=pointer,
                covered=["c1"],
                direction="support",
                sufficient=True,
            )
        ]
    )


async def test_blocked_claim_keeps_needs_and_cannot_dispatch_or_produce_plans(isolated_stage, tmp_path):
    originals = first_claims(isolated_stage)
    before = [c.model_dump(mode="json") for c in originals]
    code = Mock(side_effect=healthy_branch)
    experiments = Mock(side_effect=AssertionError("Blocked claim reached Experiments"))
    result = await verify_claims(
        originals,
        isolated_stage,
        tmp_path / "verification",
        branches={EvidenceNeed.CODE: code, EvidenceNeed.EXPERIMENTS: experiments},
        blocked_claim_ids=["claim_001"],
    )
    assessed = assess_claims(result.claims)
    assert [c.status.value for c in assessed] == ["unverified", "supported"]
    assert result.dispatched == {"claim_001": [], "claim_002": [EvidenceNeed.CODE]}
    assert result.plans == [] and assessed[0].questions == [] and assessed[0].evidence == []
    assert assessed[0].needs == originals[0].needs
    limitation = assessed[0].verification_limitations[0]
    assert limitation.kind == "claim_extraction_incomplete" and limitation.responsibility == "system"
    assert limitation.condition_ids == ["c1"]
    experiments.assert_not_called()
    assert code.call_count == 1
    assert [c.model_dump(mode="json") for c in originals] == before


def test_stage_adopts_followup_copy_and_preserves_first_pass_audit(isolated_stage, monkeypatch, tmp_path):
    originals = first_claims(isolated_stage)
    before = [c.model_dump(mode="json") for c in originals]
    monkeypatch.setattr(stage, "extract_claims", lambda *args, **kwargs: originals)

    def followup(materials, claims, **kwargs):
        assert materials is not isolated_stage and claims[0] is not originals[0]
        assert (kwargs["window_chars"], kwargs["max_review_calls"], kwargs["max_followup_calls"]) == (
            7000,
            2,
            1,
        )
        claims[0].conditions[0].settings["split"] = "test"
        result = coverage_result(claims, kwargs["output_dir"])
        Path(result.coverage["audit_path"]).write_text(
            json.dumps({"first_pass": before, "summary": result.coverage}), encoding="utf-8"
        )
        result.coverage["large_private_detail"] = "audit only"
        return result

    monkeypatch.setattr(stage, "review_claim_coverage", followup)
    result = stage.screen_paper(
        isolated_stage,
        tmp_path / "screening",
        claim_coverage_window_chars=7000,
        claim_coverage_review_calls=2,
        claim_coverage_followup_calls=1,
    )
    assert result.claims[0].conditions[0].settings == {"split": "test"}
    assert [c.model_dump(mode="json") for c in originals] == before
    assert result.blocked_claim_ids == [] and result.claim_coverage["status"] == "complete"
    assert "large_private_detail" not in result.claim_coverage
    saved = json.loads((tmp_path / "screening/screening.json").read_text("utf-8"))
    assert saved["claim_coverage"] == result.claim_coverage
    assert json.loads(Path(result.claim_coverage["audit_path"]).read_text("utf-8"))["first_pass"] == before


def test_stage_coverage_failure_discards_callback_mutations(isolated_stage, monkeypatch, tmp_path):
    originals = first_claims(isolated_stage)
    before = [c.model_dump(mode="json") for c in originals]
    material_before = isolated_stage.model_dump(mode="json")
    monkeypatch.setattr(stage, "extract_claims", lambda *args, **kwargs: originals)

    def fail(materials, claims, **kwargs):
        claims[0].text = "uncommitted callback mutation"
        materials.markdown = "uncommitted material mutation"
        raise TimeoutError("coverage service unavailable")

    monkeypatch.setattr(stage, "review_claim_coverage", fail)
    result = stage.screen_paper(isolated_stage, tmp_path / "screening")
    assert result.claim_extraction_status == "ok" and result.claim_coverage["status"] == "failed"
    assert result.blocked_claim_ids == []
    assert [c.model_dump(mode="json") for c in result.claims] == before
    assert isolated_stage.model_dump(mode="json") == material_before
    assert any("coverage service unavailable" in issue for issue in result.issues)
    assert all(not c.questions and not c.verification_limitations for c in result.claims)


def run_coverage_pipeline(tiny_inputs, isolated_stage, monkeypatch, followup):
    args, parser = tiny_inputs
    monkeypatch.setattr(stage, "extract_claims", lambda materials, **kwargs: first_claims(materials))
    monkeypatch.setattr(stage, "review_claim_coverage", followup)
    called = []

    def branch(claim, materials):
        called.append(claim.id)
        return healthy_branch(claim, materials)

    async def dispatch(claims, materials, output_dir, **kwargs):
        return await verify_claims(
            claims,
            materials,
            output_dir,
            branches={EvidenceNeed.CODE: branch, EvidenceNeed.EXPERIMENTS: branch},
            blocked_claim_ids=kwargs["blocked_claim_ids"],
        )

    monkeypatch.setattr(pipeline_v2, "verify_claims", dispatch)
    monkeypatch.setattr(
        pipeline_v2,
        "generate_advice",
        lambda review, *args, **kwargs: SimpleNamespace(review=review, issues=[], counts={}),
    )
    forbidden = Mock(side_effect=AssertionError("Unexpected external service or author execution"))
    summary = pipeline_v2.run_v2_pipeline(
        args,
        parser=parser,
        call=forbidden,
        reference_checker=forbidden,
        runner=forbidden,
        render_pdf=False,
    )
    forbidden.assert_not_called()
    review = json.loads(Path(summary["outputs"]["report_json"]).read_text("utf-8"))
    return summary, review, called


def test_cli_budgets_reach_stage_and_blocked_claim_survives_final_report(
    tiny_inputs, isolated_stage, monkeypatch
):
    args, _ = tiny_inputs
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "factreview",
            args.paper_pdf,
            "--claim-coverage-window-chars",
            "7000",
            "--claim-coverage-review-calls",
            "2",
            "--claim-coverage-followup-calls",
            "1",
            "--claim-coverage-validation-calls",
            "3",
        ],
    )
    parsed = pipeline_full.parse_args()
    for key in (
        "claim_coverage_window_chars",
        "claim_coverage_review_calls",
        "claim_coverage_followup_calls",
        "claim_coverage_validation_calls",
    ):
        setattr(args, key, getattr(parsed, key))
    args.report_presentation = "full"

    def followup(materials, claims, **kwargs):
        assert (kwargs["window_chars"], kwargs["max_review_calls"], kwargs["max_followup_calls"]) == (
            7000,
            2,
            1,
        )
        assert kwargs["max_validation_calls"] == 3
        return coverage_result(claims, kwargs["output_dir"], blocked=["claim_001"])

    summary, review, called = run_coverage_pipeline(tiny_inputs, isolated_stage, monkeypatch, followup)
    assert called == ["claim_002"]
    final = {c["id"]: c for c in review["claims"]}
    assert {key: c["status"] for key, c in final.items()} == {
        "claim_001": "unverified",
        "claim_002": "supported",
    }
    assert final["claim_001"]["needs"] == ["Experiments"] and review["ledger"] == []
    verification = json.loads(Path(summary["outputs"]["verification"]).read_text("utf-8"))
    assert verification["plans"] == [] and verification["dispatched"]["claim_001"] == []
    assert summary["claim_coverage"]["blocked_claim_ids"] == ["claim_001"]
    assert summary["outputs"]["claim_coverage"] == summary["claim_coverage"]["audit_path"]
    markdown = Path(summary["outputs"]["report_markdown"]).read_text("utf-8")
    assert "Claim extraction coverage" in markdown and "**partial**" in markdown
    assert "blocked claim IDs: claim\\_001" in markdown and "2 / 2" in markdown
    assert not final["claim_001"]["questions"]


def test_pipeline_coverage_service_failure_is_visible_without_blocking_healthy_claims(
    tiny_inputs, isolated_stage, monkeypatch
):
    def fail(*args, **kwargs):
        raise TimeoutError("coverage service unavailable")

    summary, review, called = run_coverage_pipeline(tiny_inputs, isolated_stage, monkeypatch, fail)
    assert sorted(called) == ["claim_001", "claim_002"]
    assert [c["status"] for c in review["claims"]] == ["supported", "supported"]
    assert summary["claim_coverage"] == {"status": "failed", "initial_claims": 2, "final_claims": 2}
    markdown = Path(summary["outputs"]["report_markdown"]).read_text("utf-8")
    assert "Claim extraction coverage" in markdown and "**failed**" in markdown
    assert "coverage service unavailable" in markdown
    assert all(not c["questions"] and not c["verification_limitations"] for c in review["claims"])
