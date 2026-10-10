"""Resource selection identities only; every external boundary stays forbidden."""

import copy
import hashlib
from pathlib import Path

import pytest

from preprocessing.materials import index_repository
from schemas.claim import ClaimStatus, EvidenceNeed, ExecutionPlan
from tests import test_execution_v2 as original_fixture_module
from verification.experiments import PlanCandidate, _plan

original_inputs = original_fixture_module.inputs


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Resource selection controls must not call external boundaries")

    for name in (
        "llm.client.llm_json",
        "screening.checks.llm_json",
        "requests.sessions.Session.request",
        "httpx.Client.send",
        "httpx.AsyncClient.send",
        "subprocess.run",
        "fact_generation.execution.v2.docker_runner",
    ):
        monkeypatch.setattr(name, forbidden)


@pytest.fixture
def selected(original_inputs):
    _, claim, materials = original_inputs
    root = Path(materials.repository.root)
    for name, data in {
        "data.json": b'{"examples": [1, 2], "split": "test"}',
        "weights.bin": b"released checkpoint",
        "config.json": b'{"split": "test", "data": "data.json"}',
        "other.json": b'{"examples": [3]}',
    }.items():
        (root / name).write_bytes(data)
    materials.repository = index_repository(root)
    candidate = PlanCandidate(
        targets=[{
            "condition_id": "c1",
            "reported": {
                "block_id": claim.source_block_id,
                "quote": claim.source_quote,
                "token": "0.9",
                "value_context": claim.source_quote,
            },
        }],
        entry_script="eval.py", config="config.json", run_mode="evaluation",
        feasibility="ready", priority="high", data_paths=["data.json"],
        weight_paths=["weights.bin"],
    )
    return claim, materials, candidate


def test_producer_preserves_plan_wide_choices_and_original_blockers(selected):
    from fact_generation.execution.resource_contract import validate_resource_contract

    claim, materials, candidate = selected
    before = claim.model_dump(mode="json")
    plan = _plan(claim, materials, candidate)
    assert plan.task.data_paths == candidate.data_paths
    assert plan.task.weight_paths == candidate.weight_paths
    assert plan.task.resource_contract.selection_scope == "plan_wide_candidates"
    contract = validate_resource_contract(plan, claim, materials)
    assert contract == plan.task.resource_contract
    assert [(row.role, row.path) for row in contract.resources] == [
        ("entry", "eval.py"), ("config", "config.json"),
        ("data", "data.json"), ("weights", "weights.bin"),
    ]
    indexed = {row.path: row.sha256 for row in materials.repository.files}
    assert all(row.sha256 == indexed[row.path] for row in contract.resources)
    assert all("condition_ids" not in row.model_dump() for row in contract.resources)
    assert not {"aligned", "sufficient", "consumed"} & set(contract.model_dump())
    assert claim.model_dump(mode="json") == before and not claim.evidence
    assert ExecutionPlan.model_validate_json(plan.model_dump_json()) == plan
    candidate.data_paths.append("other.json")
    assert plan.task.data_paths == ["data.json"]
    for missing, reason in (("weight_paths", "Released weights"), ("data_paths", "Released data")):
        blocked = _plan(claim, materials, candidate.model_copy(update={missing: []}, deep=True))
        assert blocked.feasibility == "blocked" and reason in blocked.blocker
        assert validate_resource_contract(blocked, claim, materials) is not None


def test_validation_rebuilds_independent_task_choices_and_allows_shared_roles(selected):
    from fact_generation.execution.resource_contract import validate_resource_contract

    claim, materials, candidate = selected
    plan = _plan(claim, materials, candidate)
    same_file = candidate.model_copy(update={"data_paths": ["config.json"]}, deep=True)
    shared = _plan(claim, materials, same_file)
    assert validate_resource_contract(shared, claim, materials) is not None
    assert [row.role for row in shared.task.resource_contract.resources if row.path == "config.json"] == [
        "config", "data",
    ]
    assert not claim.evidence
    for change in ("self_hash", "extra", "duplicate", "role", "replacement", "version"):
        bad = plan.model_copy(deep=True)
        contract = bad.task.resource_contract
        if change == "self_hash":
            contract.resources[-1].sha256 = "0" * 64
        elif change == "extra":
            contract.resources.append(contract.resources[2].model_copy(update={"path": "other.json"}))
        elif change == "duplicate":
            contract.resources.append(contract.resources[2].model_copy())
        elif change == "role":
            contract.resources[2].role = "weights"
        elif change == "replacement":
            other = _plan(claim, materials, candidate.model_copy(update={"data_paths": ["other.json"]}, deep=True))
            bad.task.resource_contract = other.task.resource_contract
        else:
            contract.version = "unknown"
        with pytest.raises(ValueError):
            validate_resource_contract(bad, claim, materials)
    for field in ("data_paths", "weight_paths"):
        repeated = candidate.model_copy(update={field: getattr(candidate, field) * 2}, deep=True)
        with pytest.raises(ValueError):
            _plan(claim, materials, repeated)


def test_scientific_identity_covers_whole_claim_and_history_is_unbound(selected):
    from fact_generation.execution.resource_contract import validate_resource_contract
    from schemas.claim import Condition

    claim, materials, candidate = selected
    claim.conditions.append(Condition(id="c2", dataset="other", metric="accuracy", settings={"split": "test"}))
    plan = _plan(claim, materials, candidate)
    assert set(plan.task.resource_contract.condition_sha256) == {"c1", "c2"}
    assert plan.task.resource_contract.condition_ids == ["c1"]
    enriched = claim.model_copy(deep=True)
    enriched.status, enriched.notes = ClaimStatus.SUPPORTED, ["later audit note"]
    assert validate_resource_contract(plan, enriched, materials) == plan.task.resource_contract
    for change in ("other_condition", "need", "importance", "source", "target", "claim_id"):
        altered_claim, altered_plan = claim.model_copy(deep=True), plan.model_copy(deep=True)
        if change == "other_condition":
            altered_claim.conditions[1].settings["split"] = "validation"
        elif change == "need":
            altered_claim.needs.append(EvidenceNeed.CODE)
        elif change == "importance":
            altered_claim.importance = "core"
        elif change == "source":
            altered_claim.source_quote += " altered"
        elif change == "target":
            altered_plan.target_conditions[0].settings["split"] = "validation"
        else:
            altered_plan.claim_id = "other_claim"
        with pytest.raises(ValueError):
            validate_resource_contract(altered_plan, altered_claim, materials)
    legacy = plan.model_dump(mode="json")
    for key in ("resource_contract", "data_paths", "weight_paths"):
        legacy["task"].pop(key)
    readable = ExecutionPlan.model_validate(legacy)
    assert readable.task.resource_contract is None
    assert validate_resource_contract(readable, claim, materials) is None
    assert not claim.evidence


def test_actual_bytes_unsafe_paths_and_ambiguous_index_fail_closed(selected, monkeypatch):
    from fact_generation.execution.resource_contract import (
        build_resource_contract,
        validate_resource_contract,
    )

    claim, materials, candidate = selected
    plan = _plan(claim, materials, candidate)
    root = Path(materials.repository.root)
    original = (root / "data.json").read_bytes()
    (root / "data.json").write_bytes(b'{"examples": [99]}')
    resealed = plan.model_copy(deep=True)
    resealed.task.resource_contract.resources[2].sha256 = hashlib.sha256((root / "data.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        validate_resource_contract(resealed, claim, materials)
    (root / "data.json").write_bytes(original)
    (root / "weights.bin").unlink()
    with pytest.raises(ValueError):
        validate_resource_contract(plan, claim, materials)
    (root / "weights.bin").write_bytes(b"released checkpoint")
    ambiguous = materials.model_copy(deep=True)
    ambiguous.repository.files.append(copy.deepcopy(ambiguous.repository.files[0]))
    with pytest.raises(ValueError):
        validate_resource_contract(plan, claim, ambiguous)
    unavailable = materials.model_copy(deep=True)
    unavailable.repository.root = str(root / "missing_root")
    with pytest.raises(ValueError):
        validate_resource_contract(plan, claim, unavailable)
    for unsafe in ("../data.json", "/data.json", "C:/data.json", "a/../data.json", "a\\data.json"):
        forged = materials.model_copy(deep=True)
        row = next(row for row in forged.repository.files if row.path == "data.json")
        row.path = unsafe
        with pytest.raises(ValueError):
            build_resource_contract(
                claim, forged, condition_ids=["c1"], entry_script="eval.py", config="config.json",
                data_paths=[unsafe], weight_paths=["weights.bin"],
            )
    original_is_symlink = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == root / "data.json" or original_is_symlink(path))
    with pytest.raises(ValueError):
        validate_resource_contract(plan, claim, materials)
