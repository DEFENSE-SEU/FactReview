import copy
import hashlib
import json
from pathlib import Path

import fitz
import pytest

from schemas.claim import Claim, PredictionProjection, ProjectionScopeDecision
from schemas.materials import SharedMaterials
from verification.execution_projection import (
    ProjectionError,
    bind_projection_target,
    entry_recipe,
    field_inventory,
    projection_context,
)
from verification.experiment_catalog import build_catalog


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    from llm.client import LLMConfig

    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: LLMConfig("mock", "mock", None, None))
    monkeypatch.setattr("screening.checks.llm_json", lambda **kw: pytest.fail("Unmocked LLM"))
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **kw: pytest.fail("Network"))
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: pytest.fail("Process"))


def original_inputs(tmp_path, claim_index=0):
    source = json.loads((Path(__file__).parent / "fixtures/execution_projection_008.json").read_text("utf-8"))
    claim = Claim.model_validate(source["claims"][claim_index])
    materials = SharedMaterials.model_validate(source["materials"])
    repo = tmp_path / "repository"
    for name, text in source["repository_bytes"].items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, "utf-8")
    materials.repository.root = str(repo)
    for row in materials.repository.files:
        row.sha256 = hashlib.sha256((repo / row.path).read_bytes()).hexdigest()
    md = tmp_path / "paper.md"
    md.write_text(materials.markdown, "utf-8")
    materials.markdown_path = str(md)
    pdf = tmp_path / "paper.pdf"
    with fitz.open() as doc:
        doc.new_page()
        doc.new_page()
        doc.save(pdf)
    materials.source_pdf = str(pdf)
    return claim, materials


def proposal_input(claim, materials, tmp_path):
    catalog = build_catalog(claim, materials)
    source_ids = {
        block: next(
            sid
            for sid, row in catalog["sources"].items()
            if row["kind"] == "paper_block" and row["block_id"] == block
        )
        for block in ("block_10", "block_11")
    }
    inventory = field_inventory(claim.conditions[0])
    roles = []
    for path in inventory:
        role = {
            "/dataset": "dataset_identity",
            "/metric": "metric",
            "/description": "conclusion_boundary",
            "/settings/model": "runtime_setting",
            "/settings/accuracy": "reported_value",
            "/settings/examples": "sample_scope",
            "/settings/accuracy_definition": "measurement_definition",
        }.get(path, "conclusion_boundary")
        roles.append(
            {
                "path": path,
                "role": role,
                "source_ids": [
                    source_ids["block_11"] if path.endswith("qualifiers/2") else source_ids["block_10"]
                ],
            }
        )
    proposal = PredictionProjection(
        version="released-predictions-v1",
        recipe="exact_match_accuracy",
        data_path="data/test_predictions.json",
        field_roles=roles,
    )
    review = ProjectionScopeDecision(
        plan_index=0,
        condition_id="c1",
        classification="absolute_fixed_predictions",
        dataset_identity_confirmed=True,
        measurement_definition_confirmed=True,
        all_original_qualifiers_preserved=True,
        confirmed_field_paths=list(inventory),
        source_ids=list(source_ids.values()),
        unresolved=[],
        rationale="The original fixed-sample statement retains all three non-generalization boundaries.",
    )
    block = next(b for b in materials.blocks if b.id == "block_10")
    reported = {
        "block_id": block.id,
        "quote": block.text,
        "token": "0.75",
        "value_context": block.text.split(". Accuracy")[0] + ".",
    }
    selector = {
        "number_id": next(
            key
            for key, value in catalog["numbers"].items()
            if value["block_id"] == block.id and value["token"] == "0.75"
        )
    }
    plan = {
        "targets": [
            {
                "condition_id": "c1",
                "reported": reported,
                "selector": selector,
                "projection": proposal.model_dump(mode="json"),
            }
        ],
        "entry_script": "scripts/evaluate.py",
        "config": "configs/evaluation.json",
        "run_mode": "evaluation",
        "feasibility": "ready",
        "priority": "high",
        "data_paths": [proposal.data_path],
        "weight_paths": [],
    }
    audit = tmp_path / "scope.json"
    audit.write_text(
        json.dumps(
            {
                "input": {
                    "claim": claim.model_dump(mode="json"),
                    "candidate_plans": [plan],
                    "paper_blocks": [b.model_dump(mode="json") for b in materials.blocks],
                    "execution_projection_context": projection_context(claim, materials),
                },
                "response": {
                    "schema_version": "catalog-v2",
                    "plan_projection_reviews": [review.model_dump(mode="json")],
                },
            }
        ),
        "utf-8",
    )
    return proposal, review, reported, selector, plan, audit


def bind(claim, materials, data):
    proposal, review, reported, selector, plan, audit = data
    return bind_projection_target(
        claim,
        claim.conditions[0],
        reported,
        materials,
        selector=selector,
        proposal=proposal,
        entry_script=plan["entry_script"],
        config_path=plan["config"],
        review=review,
        audit_path=audit,
    )


def refresh(materials, name):
    row = next(f for f in materials.repository.files if f.path == name)
    row.sha256 = hashlib.sha256((Path(materials.repository.root) / name).read_bytes()).hexdigest()


def test_original_008_projection_retains_original_claim_and_every_condition_leaf(tmp_path):
    claim, materials = original_inputs(tmp_path)
    before = claim.model_dump(mode="json")
    data = proposal_input(claim, materials, tmp_path)
    binding = bind(claim, materials, data)
    assert binding.version == 2 and binding.value == 0.75 and binding.unit == "fraction"
    assert binding.projection.runtime_target.dataset == "MiniSet"
    assert binding.projection.runtime_target.settings == {"split": "test", "model": "ExactMatch"}
    assert binding.projection.sample_count == 4
    assert {r.path for r in binding.projection.proposal.field_roles} == set(
        field_inventory(claim.conditions[0])
    )
    assert claim.model_dump(mode="json") == before


@pytest.mark.parametrize(
    "change", ["hardcoded", "slice", "filter", "overwritten", "other_data", "mean_score"]
)
def test_entry_recipe_rejects_same_number_without_original_data_dependency(tmp_path, change):
    _, materials = original_inputs(tmp_path)
    path = Path(materials.repository.root) / "scripts/evaluate.py"
    text = path.read_text("utf-8")
    if change == "hardcoded":
        text = text.replace('"value": accuracy', '"value": 0.75')
    elif change == "slice":
        text = text.replace("for row in rows)", "for row in rows[:3])")
    elif change == "filter":
        text = text.replace("for row in rows)", "for row in rows if row['label'])")
    elif change == "overwritten":
        text = text.replace("print(", "accuracy = 0.75\nprint(")
    elif change == "other_data":
        text = text.replace("data/test_predictions.json", "data/other.json")
    else:
        text = text.replace('row["label"] == row["prediction"]', 'row["score"]')
    path.write_text(text, "utf-8")
    refresh(materials, "scripts/evaluate.py")
    with pytest.raises(ProjectionError):
        entry_recipe(
            materials, "scripts/evaluate.py", "configs/evaluation.json", "data/test_predictions.json"
        )


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "runtime_as_prose",
        "unknown_qualifier",
        "foreign_model",
        "wrong_split",
        "extra_setting",
        "comparison",
        "unknown_version",
    ],
)
def test_projection_does_not_discard_original_obligations(tmp_path, change):
    claim, materials = original_inputs(tmp_path)
    data = proposal_input(claim, materials, tmp_path)
    proposal, _review, _reported, _selector, _plan, _audit = data
    if change == "missing":
        proposal.field_roles.pop()
    if change == "duplicate":
        proposal.field_roles.append(proposal.field_roles[-1].model_copy())
    if change == "runtime_as_prose":
        next(
            row for row in proposal.field_roles if row.path == "/settings/model"
        ).role = "conclusion_boundary"
    if change == "unknown_qualifier":
        claim.conditions[0].settings["qualifiers"][0] = "No data augmentation used"
    if change in {"foreign_model", "wrong_split", "extra_setting"}:
        p = Path(materials.repository.root) / "configs/evaluation.json"
        config = json.loads(p.read_text("utf-8"))
        config["settings"].update(
            {
                "foreign_model": {"model": "Another"},
                "wrong_split": {"split": "validation"},
                "extra_setting": {"filter": "easy"},
            }[change]
        )
        p.write_text(json.dumps(config), "utf-8")
        refresh(materials, "configs/evaluation.json")
    if change == "comparison":
        claim.text = "ExactMatch outperforms Another by 10 percentage points on MiniSet test."
    if change == "unknown_version":
        proposal.version = "future"
    with pytest.raises((ProjectionError, ValueError)):
        bind(claim, materials, data)


def test_original_009_composite_count_claim_remains_outside_scalar_recipe(tmp_path):
    claim, materials = original_inputs(tmp_path, 1)
    data = proposal_input(claim, materials, tmp_path)
    with pytest.raises(ProjectionError, match="compound counts"):
        bind(claim, materials, data)


def public_verify(tmp_path, *, change=None, with_observation=False):
    from common.run_stats import run_scope
    from verification.experiments import verify_experiments

    claim, materials = original_inputs(tmp_path)
    data = proposal_input(claim, materials, tmp_path)
    _proposal, review, reported, _selector, plan, _audit = data
    items = (
        [
            {
                "aspect": "correspondence",
                "kind": "paper_support",
                "block_id": reported["block_id"],
                "quote": reported["quote"],
                "covered": ["c1"],
                "fully_supported_conditions": [],
                "detail": "Fixed predictions are reported; this partial observation alone does not prove every qualifier.",
            }
        ]
        if with_observation
        else []
    )
    first = {
        "checked_aspects": ["correspondence", "fairness", "isolation", "stability", "consistency"],
        "items": items,
        "plans": [plan],
    }
    scope = {
        "schema_version": "catalog-v2",
        "conditions": [
            {
                "condition_id": "c1",
                "assertion": "descriptive",
                "matched_controls_required": False,
                "uncertainty_sensitive": False,
                "relation": "none",
                "rationale": "Fixed test predictions only.",
            }
        ],
        "items": [
            {
                "item_index": 0,
                "condition_id": "c1",
                "applicability": "applicable",
                "ground_source_ids": [review.source_ids[0]],
                "rationale": "Original partial observation retained.",
                "full_support": False,
                "qualifiers_complete": False,
                "comparison_objects": "not_comparative",
            }
        ]
        if items
        else [],
        "plan_projection_reviews": [review.model_dump(mode="json")],
    }
    calls = []

    def model(**kwargs):
        calls.append(kwargs)
        if kwargs["module"] == "verification.experiments":
            if change:
                change("first", first, scope, claim, materials)
            return copy.deepcopy(first)
        assert kwargs["module"] == "verification.experiments.scope"
        if change:
            change("scope", first, scope, claim, materials)
        return copy.deepcopy(scope)

    with run_scope(tmp_path / "stats.json"):
        result = verify_experiments(claim, materials, call=model)
    return claim, materials, result, calls


def test_plan_only_uses_existing_second_scope_and_preserves_full_original_claim(tmp_path):
    claim, materials, result, calls = public_verify(tmp_path)
    assert len(calls) == 2
    assert result.plans[0].feasibility == "ready", result.plans[0].blocker
    plan = result.plans[0]
    assert plan.target_conditions == claim.conditions and plan.y_paper == {"c1": 0.75}
    assert plan.task.command == ["python", "-I", "-S", "scripts/evaluate.py"]
    assert not result.evidence
    from schemas.claim import ExecutionPlan
    from verification.experiment_targets import validate_plan_targets

    assert ExecutionPlan.model_validate_json(plan.model_dump_json()) == plan
    assert validate_plan_targets(plan, claim, materials) == plan.target_bindings


@pytest.mark.parametrize(
    "damage",
    [
        "unknown_version",
        "duplicate",
        "omitted",
        "bool_index",
        "string_index",
        "unknown_source",
        "unresolved",
        "incomplete",
        "paper_changed",
        "repo_changed",
    ],
)
def test_bad_projection_is_plan_local_and_retains_partial_paper_observation(tmp_path, damage):
    def change(phase, first, scope, claim, materials):
        if phase == "first" and damage == "unknown_version":
            first["plans"][0]["targets"][0]["projection"]["version"] = "unknown"
        if phase != "scope":
            return
        row = scope["plan_projection_reviews"][0]
        if damage == "duplicate":
            scope["plan_projection_reviews"].append(copy.deepcopy(row))
        elif damage == "omitted":
            scope["plan_projection_reviews"] = []
        elif damage == "bool_index":
            row["plan_index"] = False
        elif damage == "string_index":
            row["plan_index"] = "0"
        elif damage == "unknown_source":
            row["source_ids"] = ["source:missing"]
        elif damage == "unresolved":
            row["classification"] = "unresolved"
        elif damage == "incomplete":
            row["all_original_qualifiers_preserved"] = False
        elif damage == "paper_changed":
            Path(materials.markdown_path).write_text(materials.markdown + "\nchanged", "utf-8")
        elif damage == "repo_changed":
            (Path(materials.repository.root) / "configs/evaluation.json").write_text("{}", "utf-8")

    claim, _materials, result, calls = public_verify(tmp_path, change=change, with_observation=True)
    assert len(calls) == 2
    assert len(result.evidence) == 1 and not result.evidence[0].sufficient
    assert result.evidence[0].pointer.quote == claim.source_quote
    assert not result.plans or result.plans[0].feasibility == "blocked"
    assert result.verification_limitations and not result.questions


def execute_public(tmp_path, *, actual_value=0.75, actual_settings=None, mutate=None):
    from fact_generation.execution.v2 import ExecutionConfig, Observation, RunOutcome, execute_plans

    claim, materials, verified, _ = public_verify(tmp_path)
    assert verified.plans[0].feasibility == "ready", verified.plans[0].blocker
    raw = Observation(
        dataset="MiniSet",
        metric="accuracy",
        settings=actual_settings or {"split": "test", "model": "ExactMatch"},
        value=actual_value,
    )
    calls = []

    def runner(request):
        calls.append(request.model_dump(mode="json"))
        if mutate:
            mutate(request, materials)
        return RunOutcome(
            returncode=0,
            observations=[raw],
            stdout=json.dumps({"observations": [raw.model_dump(mode="json")]}),
        )

    result = execute_plans(
        verified.plans,
        [claim],
        materials,
        tmp_path / "execute",
        config=ExecutionConfig(refine_with_llm=False, training_budget=0),
        runner=runner,
    )
    return claim, materials, result, calls


def test_original_008_reaches_l3_with_independent_fraction_and_sample_count(tmp_path):
    from assessment.rules import assess_claim

    claim, _materials, result, calls = execute_public(tmp_path)
    assert len(calls) == 1
    ledger = result.ledger[0]
    assert not ledger["reason"] and not ledger["repairs"]
    evidence = result.claims[0].evidence[0]
    assert evidence.sufficient and evidence.aligned and evidence.direction == "support"
    assert "no model inference or training performed" in evidence.note
    measurement = json.loads(Path(evidence.pointer.locator).read_text("utf-8"))
    assert measurement["measurement"]["value"] == 0.75 and measurement["measurement"]["unit"] == "fraction"
    assert measurement["sample_count"] == 4 and measurement["numerator"] == 3
    assert measurement["author_observation"]["unit"] is None
    raw = json.loads(Path(measurement["author_observation_artifact"]).read_text("utf-8"))
    assert raw[0] == measurement["author_observation"]
    assert result.claims[0].text == claim.text and result.claims[0].conditions == claim.conditions
    assert assess_claim(result.claims[0]).status.value == "supported"


@pytest.mark.parametrize(
    "damage", ["wrong_value", "wrong_split", "wrong_model", "workspace_data", "source_data", "scope_audit"]
)
def test_execution_never_uses_plan_fields_to_repair_missing_actual_measurement(tmp_path, damage):
    def mutate(request, materials):
        if damage == "workspace_data":
            (Path(request.workspace) / "data/test_predictions.json").write_text("[]", "utf-8")
        elif damage == "source_data":
            (Path(materials.repository.root) / "data/test_predictions.json").write_text("[]", "utf-8")
        elif damage == "scope_audit":
            Path(request.plan.target_bindings["c1"].projection.scope_audit).write_text("{}", "utf-8")

    settings = {
        "split": "validation" if damage == "wrong_split" else "test",
        "model": "Another" if damage == "wrong_model" else "ExactMatch",
    }
    _, _, result, calls = execute_public(
        tmp_path,
        actual_value=0.5 if damage == "wrong_value" else 0.75,
        actual_settings=settings,
        mutate=mutate,
    )
    assert len(calls) == 1
    assert not any(e.sufficient for e in result.claims[0].evidence)
    assert result.ledger[0]["reason"]
    assert result.claims[0].verification_limitations and not result.claims[0].questions


@pytest.mark.parametrize(
    "damage",
    [
        "true_index",
        "float_index",
        "duplicate_third",
        "wrong_condition",
        "malformed",
        "repo_id_as_paper",
        "omitted_field",
        "scope_failure",
    ],
)
def test_independent_projection_tombstones_and_source_domain(tmp_path, damage):
    def change(phase, first, scope, claim, materials):
        if phase != "scope":
            return
        row = scope["plan_projection_reviews"][0]
        if damage == "true_index":
            row["plan_index"] = True
        elif damage == "float_index":
            row["plan_index"] = 0.0
        elif damage == "duplicate_third":
            bad = copy.deepcopy(row)
            bad.pop("classification")
            scope["plan_projection_reviews"] += [bad, copy.deepcopy(row)]
        elif damage == "wrong_condition":
            row["condition_id"] = "c2"
        elif damage == "malformed":
            scope["plan_projection_reviews"] = {"0": row}
        elif damage == "repo_id_as_paper":
            row["source_ids"] = ["configs/evaluation.json"]
        elif damage == "omitted_field":
            row["confirmed_field_paths"].pop()
        else:
            raise RuntimeError("Injected scope transport failure")

    _, _, result, calls = public_verify(tmp_path, change=change, with_observation=True)
    assert len(calls) == 2 and not result.plans
    assert len(result.evidence) == 1 and not result.evidence[0].sufficient


@pytest.mark.parametrize(
    "damage", ["count_equal_ratio", "empty", "missing_key", "mixed_bool", "float_label", "stdlib_shadow"]
)
def test_measurement_recipe_uses_complete_typed_rows_not_scalar_agreement(tmp_path, damage):
    claim, materials = original_inputs(tmp_path)
    p = Path(materials.repository.root) / "data/test_predictions.json"
    rows = json.loads(p.read_text("utf-8"))
    if damage == "count_equal_ratio":
        rows *= 2
    elif damage == "empty":
        rows = []
    elif damage == "missing_key":
        rows[0].pop("label")
    elif damage == "mixed_bool":
        rows[0]["label"], rows[0]["prediction"] = True, 1
    elif damage == "float_label":
        rows[0]["label"], rows[0]["prediction"] = 0.5, 0.5
    else:
        from schemas.materials import RepositoryFile

        shadow = Path(materials.repository.root) / "scripts/json.py"
        shadow.write_text("def loads(value): return []", "utf-8")
        materials.repository.files.append(
            RepositoryFile(
                path="scripts/json.py", kind="source", sha256=hashlib.sha256(shadow.read_bytes()).hexdigest()
            )
        )
    p.write_text(json.dumps(rows), "utf-8")
    refresh(materials, "data/test_predictions.json")
    data = proposal_input(claim, materials, tmp_path)
    if damage == "count_equal_ratio":
        # L2 records paper scope; L3 independently discovers the different count.
        binding = bind(claim, materials, data)
        assert binding.projection.sample_count == 4
        from fact_generation.execution.prediction_measurement import measure_predictions
        from fact_generation.execution.v2 import ExecutionConfig, Observation, RunRequest
        from verification.experiments import PlanCandidate, _plan

        plan = _plan(
            claim,
            materials,
            PlanCandidate.model_validate(data[4]),
            projection_reviews={(0, "c1"): data[1]},
            scope_audit=data[5],
        )
        request = RunRequest(
            plan=plan,
            workspace=materials.repository.root,
            run_dir=str(tmp_path / "exec"),
            command=plan.task.command,
            workdir=".",
            metric_output=None,
            repair_round=0,
            config=ExecutionConfig(),
        )
        with pytest.raises(ProjectionError, match="sample count"):
            measure_predictions(
                request,
                binding,
                materials,
                Observation(
                    dataset="MiniSet",
                    metric="accuracy",
                    settings={"split": "test", "model": "ExactMatch"},
                    value=0.75,
                ),
                tmp_path / "measurement.json",
            )
    else:
        with pytest.raises(ProjectionError):
            bind(claim, materials, data)


def test_two_indexed_dataset_split_decompositions_require_resolution(tmp_path):
    from schemas.materials import RepositoryFile

    claim, materials = original_inputs(tmp_path)
    path = Path(materials.repository.root) / "configs/alternative.json"
    path.write_text(
        json.dumps({"dataset": "MiniSet test", "metric": "accuracy", "settings": {"model": "ExactMatch"}}),
        "utf-8",
    )
    materials.repository.files.append(
        RepositoryFile(
            path="configs/alternative.json",
            kind="config",
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    )
    materials.repository.configs.append("configs/alternative.json")
    with pytest.raises(ProjectionError, match="decompositions"):
        bind(claim, materials, proposal_input(claim, materials, tmp_path))


def test_separate_original_dataset_and_split_have_the_same_bound_identity(tmp_path):
    claim, materials = original_inputs(tmp_path)
    claim.conditions[0].dataset = "MiniSet"
    claim.conditions[0].settings["split"] = "test"
    data = proposal_input(claim, materials, tmp_path)
    proposal, _review, _reported, _selector, plan, audit = data
    next(row for row in proposal.field_roles if row.path == "/settings/split").role = "runtime_setting"
    plan["targets"][0]["projection"] = proposal.model_dump(mode="json")
    record = json.loads(audit.read_text("utf-8"))
    record["input"]["candidate_plans"] = [plan]
    audit.write_text(json.dumps(record), "utf-8")
    binding = bind(claim, materials, data)
    assert binding.projection.runtime_target.dataset == "MiniSet"
    assert binding.projection.runtime_target.settings["split"] == "test"


def test_recipe_is_independent_of_author_variable_names(tmp_path):
    _, materials = original_inputs(tmp_path)
    path = Path(materials.repository.root) / "scripts/evaluate.py"
    import re

    text = path.read_text("utf-8")
    for old, new in {
        "root": "base",
        "config": "configuration",
        "rows": "predictions",
        "correct": "matches",
        "accuracy": "fraction",
        "row": "sample",
    }.items():
        text = re.sub(r"\b" + old + r"\b", new, text)
    path.write_text(text, "utf-8")
    refresh(materials, "scripts/evaluate.py")
    measurement = entry_recipe(
        materials, "scripts/evaluate.py", "configs/evaluation.json", "data/test_predictions.json"
    )
    assert measurement["numerator"] == 3 and measurement["denominator"] == 4


@pytest.mark.parametrize("damage", ["duplicate_target", "bad_projection_type", "no_inference_discharge"])
def test_first_projection_proposal_errors_preserve_healthy_paper_observations(tmp_path, damage):
    def change(phase, first, scope, claim, materials):
        if phase == "first":
            target = first["plans"][0]["targets"][0]
            if damage == "duplicate_target":
                first["plans"][0]["targets"].append(copy.deepcopy(target))
            elif damage == "bad_projection_type":
                target["projection"] = ["unsupported"]
        if phase == "scope" and damage == "no_inference_discharge":
            scope["plan_projection_reviews"][0]["unresolved"] = [
                "Current model inference and weights are required by the original claim."
            ]

    _, _, result, calls = public_verify(tmp_path, change=change, with_observation=True)
    assert len(calls) == 2 and not result.plans and len(result.evidence) == 1
    assert not result.evidence[0].sufficient and result.verification_limitations


@pytest.mark.parametrize("unit,value", [("percent", 75), ("percent", 0.75), (None, 0.76)])
def test_explicit_wrong_scale_and_close_author_scores_are_not_laundered_by_tolerance(tmp_path, unit, value):
    from fact_generation.execution.v2 import Observation, RunOutcome, execute_plans

    claim, materials, result, _ = public_verify(tmp_path)
    observation = Observation(
        dataset="MiniSet",
        metric="accuracy",
        settings={"split": "test", "model": "ExactMatch"},
        value=value,
        unit=unit,
    )
    executed = execute_plans(
        result.plans,
        [claim],
        materials,
        tmp_path / "exec",
        runner=lambda request: RunOutcome(returncode=0, observations=[observation]),
    )
    assert not executed.claims[0].evidence and executed.ledger[0]["reason"]
    assert executed.claims[0].verification_limitations


def test_schema_exposes_projection_definition_with_resolvable_local_refs():
    from verification.experiments import CatalogScopeReviewV2, ExperimentsOutput

    for model in (ExperimentsOutput, CatalogScopeReviewV2):
        schema = model.model_json_schema()

        def walk(value):
            if isinstance(value, dict):
                if "$ref" in value:
                    target = schema
                    for part in value["$ref"].split("/")[1:]:
                        target = target[part]
                    assert target
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(schema)
    assert "PredictionProjection" in ExperimentsOutput.model_json_schema()["$defs"]


@pytest.mark.parametrize("damage", ["pdf", "condition", "audit", "proposal", "command", "legacy_unbound"])
def test_projection_revalidation_prevents_ready_plan_tampering(tmp_path, damage):
    from fact_generation.execution.v2 import execute_plans

    claim, materials, result, _ = public_verify(tmp_path)
    plan = result.plans[0]
    if damage == "pdf":
        Path(materials.source_pdf).write_bytes(b"changed PDF bytes")
    elif damage == "condition":
        plan.target_conditions[0].settings.pop("examples")
    elif damage == "audit":
        Path(plan.target_bindings["c1"].projection.scope_audit).write_text("{}", "utf-8")
    elif damage == "proposal":
        plan.target_bindings["c1"].projection.proposal.field_roles.pop()
    elif damage == "command":
        plan.task.command = ["python", "scripts/evaluate.py", "--filter=easy"]
    else:
        plan.target_bindings = {}
    out = execute_plans(
        [plan],
        [claim],
        materials,
        tmp_path / "execution",
        runner=lambda request: pytest.fail("Invalid plan reached runner"),
    )
    assert out.ledger[0]["reason"] and not out.ledger[0]["approved"]
    assert not any(e.sufficient for e in out.claims[0].evidence)


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "slim",
        "entrypoint",
        "missing",
        "bad_id",
        "extra_deps",
        "custom_image",
        "wrong_version",
        "changed_after",
        "flags",
    ],
)
def test_stdlib_docker_path_uses_inspected_id_and_does_not_build_author_dependencies(
    tmp_path, monkeypatch, damage
):
    from fact_generation.execution import v2
    from util.subprocess_runner import CommandResult

    for name in ("EXECUTION_DOCKER_PAPER_PYTHON_IMAGE", "EXECUTION_DOCKER_EXTRA_PIP_PACKAGES"):
        monkeypatch.delenv(name, raising=False)
    _claim, materials, result, _ = public_verify(tmp_path)
    plan = result.plans[0]
    run_dir = tmp_path / "docker-run"
    run_dir.mkdir()
    request = v2.RunRequest(
        plan=plan,
        workspace=materials.repository.root,
        run_dir=str(run_dir),
        command=plan.task.command,
        workdir=".",
        metric_output=None,
        repair_round=0,
        config=v2.ExecutionConfig(python_version="3.11"),
    )
    if damage == "extra_deps":
        request.dependencies = ["untrusted-package"]
    if damage == "custom_image":
        request.config.docker_options["docker_paper_python_image"] = "custom:latest"
    if damage == "wrong_version":
        request.config.docker_options["docker_paper_python_image"] = "python:3.12-slim"
    if damage == "slim":
        request.config.docker_options["docker_paper_python_image"] = "python:3.11-slim"
    if damage == "flags":
        request.command = ["python", "scripts/evaluate.py"]
    image_id = "sha256:" + "a" * 64
    commands = []
    inspection = [{"Id": image_id, "RepoTags": ["python:3.11"], "Config": {"Entrypoint": None}}]
    if damage == "slim":
        inspection[0]["RepoTags"] = ["python:3.11-slim"]
    if damage == "entrypoint":
        inspection[0]["Config"]["Entrypoint"] = ["custom-interpreter"]
    if damage == "bad_id":
        inspection[0]["Id"] = "mutable:tag"

    def command(argv, **kwargs):
        commands.append(argv)
        if argv[1:3] == ["image", "inspect"]:
            data = copy.deepcopy(inspection)
            if damage == "changed_after" and argv[-1] == image_id:
                data[0]["Id"] = "sha256:" + "b" * 64
            return CommandResult(
                argv, kwargs["cwd"], 1 if damage == "missing" else 0, json.dumps(data), "", 0.001
            )
        assert argv[1] == "run" and image_id in argv
        assert argv[-4:] == ["python", "-I", "-S", "scripts/evaluate.py"]
        payload = {
            "observations": [
                {
                    "dataset": "MiniSet",
                    "metric": "accuracy",
                    "settings": {"split": "test", "model": "ExactMatch"},
                    "value": 0.75,
                }
            ]
        }
        return CommandResult(argv, kwargs["cwd"], 0, json.dumps(payload), "", 0.002)

    monkeypatch.setattr(v2, "run_command", command)
    monkeypatch.setattr(
        v2,
        "docker_ensure_paper_image",
        lambda *a, **kw: pytest.fail("Recipe must not install author dependencies"),
    )
    outcome = v2.docker_runner(request)
    if damage in (None, "slim"):
        assert outcome.returncode == 0 and len(outcome.observations) == 1
        assert [c[1] for c in commands] == ["image", "run", "image"]
        assert commands[-1][-1] == image_id
        assert commands[0][-1] == ("python:3.11-slim" if damage == "slim" else "python:3.11")
        assert outcome.environment["prediction_recipe_environment"]["image_id"] == image_id
    else:
        assert outcome.returncode != 0 and not outcome.observations
        assert not any(c[1] in {"build", "pull"} for c in commands)


def test_transitive_shadow_file_never_removes_the_isolated_launch_requirement(tmp_path):
    from schemas.materials import RepositoryFile
    from verification.experiment_targets import TargetBindingError, validate_plan_targets
    from verification.experiments import PlanCandidate, _plan

    claim, materials = original_inputs(tmp_path)
    path = Path(materials.repository.root) / "scripts/_json.py"
    # Never execute this adversarial input. It models a transitive import hook.
    path.write_text("import builtins\nbuiltins.sum = lambda values: 3\n", "utf-8")
    materials.repository.files.append(
        RepositoryFile(
            path="scripts/_json.py", kind="source", sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
    )
    data = proposal_input(claim, materials, tmp_path)
    plan = _plan(
        claim,
        materials,
        PlanCandidate.model_validate(data[4]),
        projection_reviews={(0, "c1"): data[1]},
        scope_audit=data[5],
    )
    assert plan.feasibility == "ready" and plan.task.command[1:3] == ["-I", "-S"]
    validate_plan_targets(plan, claim, materials)
    plan.task.command = ["python", "scripts/evaluate.py"]
    with pytest.raises(TargetBindingError):
        validate_plan_targets(plan, claim, materials)


def test_untrusted_scratch_metadata_cannot_change_host_fraction_or_count(tmp_path):
    def mutate(request, materials):
        scratch = Path(request.run_dir) / "runtime_scratch"
        scratch.mkdir(exist_ok=True)
        (scratch / "measurement.json").write_text(
            json.dumps({"value": 0.75, "unit": "percent", "sample_count": 99}), "utf-8"
        )

    _, _, result, _ = execute_public(tmp_path, mutate=mutate)
    evidence = result.claims[0].evidence[0]
    audit = json.loads(Path(evidence.pointer.locator).read_text("utf-8"))
    assert audit["sample_count"] == 4 and audit["measurement"]["unit"] == "fraction"
    assert audit["measurement"]["value"] == 0.75


def test_full_data_recomputation_retains_a_real_discrepancy_from_the_reported_target(tmp_path):
    from assessment.rules import assess_claim
    from fact_generation.execution.v2 import RunOutcome, execute_plans
    from verification.experiments import PlanCandidate, _plan

    claim, materials = original_inputs(tmp_path)
    path = Path(materials.repository.root) / "data/test_predictions.json"
    rows = json.loads(path.read_text("utf-8"))
    match = next(row for row in rows if row["label"] == row["prediction"])
    match["prediction"] = str(match["label"]) + "_different"
    # Preserve the categorical type for this separately declared synthetic data mutation.
    if type(match["label"]) is int:
        match["prediction"] = match["label"] + 1
    path.write_text(json.dumps(rows), "utf-8")
    refresh(materials, "data/test_predictions.json")
    data = proposal_input(claim, materials, tmp_path)
    plan = _plan(
        claim,
        materials,
        PlanCandidate.model_validate(data[4]),
        projection_reviews={(0, "c1"): data[1]},
        scope_audit=data[5],
    )
    result = execute_plans(
        [plan],
        [claim],
        materials,
        tmp_path / "exec",
        runner=lambda request: RunOutcome(
            returncode=0,
            observations=[
                {
                    "dataset": "MiniSet",
                    "metric": "accuracy",
                    "settings": {"split": "test", "model": "ExactMatch"},
                    "value": 0.5,
                }
            ],
        ),
    )
    evidence = result.claims[0].evidence[0]
    assert evidence.direction == "flaw" and evidence.sufficient and not evidence.overturnable
    audit = json.loads(Path(evidence.pointer.locator).read_text("utf-8"))
    assert audit["measurement"]["value"] == 0.5 and audit["numerator"] == 2 and audit["sample_count"] == 4
    assert assess_claim(result.claims[0]).status.value == "flawed"
