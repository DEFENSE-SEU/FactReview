import asyncio
import itertools

import pytest

from schemas.claim import Claim, ClaimLocation, Condition, Evidence, EvidenceNeed, EvidencePointer
from schemas.materials import SharedMaterials
from verification.contracts import BranchResult
from verification.dispatch import verify_claims


@pytest.mark.parametrize(
    "branch,coverage", [(EvidenceNeed.CODE, "d1"), (EvidenceNeed.EXPERIMENTS, "foreign")]
)
async def test_rejected_plan_cannot_bypass_branch_or_coverage_checks(branch, coverage, tmp_path):
    from verification.contracts import RejectedPlan

    evidence = Evidence(
        source="paper_internal",
        pointer=EvidencePointer(locator="paper.pdf", page=1, quote="Claim"),
        covered=[coverage],
        direction="support",
        sufficient=True,
    )

    def reject(c, m):
        raise RejectedPlan("Invalid proposed target", BranchResult(evidence=[evidence]))

    result = await verify_claims([claim([branch])], materials(tmp_path), tmp_path, branches={branch: reject})
    assert result.plans == [] and result.claims[0].evidence == []
    assert result.issues and result.claims[0].questions


def claim(needs):
    return Claim(
        id="c1",
        text="Claim",
        loc=ClaimLocation(page=1),
        conditions=[Condition(id="d1", dataset="data", metric="accuracy")],
        needs=needs,
    )


def materials(tmp_path):
    return SharedMaterials(
        paper_key="test",
        source_pdf="paper.pdf",
        markdown="Claim",
        provider="fixture",
        markdown_path=str(tmp_path / "paper.md"),
        content_list_path="",
    )


NEEDS = list(EvidenceNeed)


@pytest.mark.parametrize(
    "needs", [list(itertools.compress(NEEDS, mask)) for mask in itertools.product([False, True], repeat=4)]
)
async def test_dispatch_matrix_calls_exactly_needs(needs, tmp_path):
    calls = []

    def make_branch(name):
        async def branch(c, m):
            calls.append((c.id, name))
            return BranchResult()

        return branch

    result = await verify_claims(
        [claim(needs)], materials(tmp_path), tmp_path, branches={name: make_branch(name) for name in NEEDS}
    )
    assert set(calls) == {("c1", name) for name in needs}
    assert len(calls) == len(needs)
    assert result.dispatched.get("c1", []) == needs


async def test_peer_branches_start_in_parallel(tmp_path):
    started = set()
    ready = asyncio.Event()

    def make_branch(name):
        async def branch(c, m):
            started.add(name)
            if len(started) == 4:
                ready.set()
            await asyncio.wait_for(ready.wait(), timeout=2)
            return BranchResult()

        return branch

    result = await verify_claims(
        [claim(NEEDS)], materials(tmp_path), tmp_path, branches={name: make_branch(name) for name in NEEDS}
    )
    assert not result.issues
    assert len(started) == 4


async def test_branch_failure_records_reason_without_fabricated_evidence(tmp_path):
    def branch(c, m):
        raise ConnectionError("mock retrieval unavailable")

    result = await verify_claims(
        [claim([EvidenceNeed.LITERATURE])],
        materials(tmp_path),
        tmp_path,
        branches={EvidenceNeed.LITERATURE: branch},
    )
    assert not result.claims[0].evidence
    assert result.claims[0].questions[0].claim_id == "c1"
    assert "mock retrieval unavailable" in result.issues[0]


async def test_foreign_condition_coverage_is_rejected(tmp_path):
    evidence = Evidence(
        source="code",
        pointer=EvidencePointer(locator="model.py", line=1),
        covered=["other_claim_condition"],
        direction="support",
        sufficient=True,
    )
    result = await verify_claims(
        [claim([EvidenceNeed.CODE])],
        materials(tmp_path),
        tmp_path,
        branches={EvidenceNeed.CODE: lambda c, m: BranchResult(evidence=[evidence])},
    )
    assert not result.claims[0].evidence
    assert "coverage" in result.issues[0]


async def test_global_search_runs_even_without_any_literature_claim(tmp_path):
    called = []

    def global_search(c, m):
        assert c is None
        called.append(True)
        return BranchResult(issues=["Global search scope recorded"])

    result = await verify_claims(
        [claim([])], materials(tmp_path), tmp_path, branches={}, global_literature=global_search
    )
    assert called == [True]
    assert result.dispatched == {}
    assert result.issues == ["Global search scope recorded"]


@pytest.mark.parametrize("branch_name", [EvidenceNeed.THEORY, EvidenceNeed.CODE, EvidenceNeed.LITERATURE])
async def test_only_experiments_can_emit_plans(branch_name, tmp_path):
    from schemas.claim import ExecutionPlan, ExecutionTask

    record = claim([branch_name])
    plan = ExecutionPlan(
        id="p1",
        claim_id=record.id,
        condition_ids=["d1"],
        target_conditions=record.conditions,
        task=ExecutionTask(entry_script="eval.py"),
        run_mode="evaluation",
        y_paper={"d1": 0.9},
        feasibility="ready",
        priority="high",
    )
    result = await verify_claims(
        [record],
        materials(tmp_path),
        tmp_path,
        branches={branch_name: lambda c, m: BranchResult(plans=[plan])},
    )
    assert not result.plans
    assert "Only Experiments" in result.issues[0]
