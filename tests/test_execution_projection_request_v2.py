"""The projection request menu exposes finite choices without granting acceptance."""

import copy
import json
from pathlib import Path

import pytest

from schemas.claim import ClaimLocation
from schemas.materials import MaterialBlock, RepositoryFile
from tests import test_execution_projection_v2 as fixtures
from verification.execution_projection import (
    ProjectionError,
    expected_field_role,
    field_inventory,
    projection_context,
    repository_context_files,
    source_hits,
    source_requirements,
)
from verification.execution_projection_catalog import request_choices as pure_choices
from verification.experiment_catalog import build_catalog

offline = fixtures.offline


def request_choices(claim, materials):
    return pure_choices(claim, materials, repository_files=repository_context_files(materials))


def config_choices(claim, materials):
    menu = request_choices(claim, materials)
    return menu, menu["conditions"]["c1"]["by_config"]["configs/evaluation.json"]


def test_original_008_menu_exposes_all_five_interface_constraints_without_editing_inputs(tmp_path):
    claim, materials = fixtures.original_inputs(tmp_path)
    before = copy.deepcopy((claim.model_dump(mode="json"), materials.model_dump(mode="json")))
    menu, cfg = config_choices(claim, materials)
    assert menu["schema_version"] == "released-prediction-choices-v1"
    assert menu["status"] == "structural_choices_only"
    assert menu["entries"] == ["scripts/evaluate.py"] and menu["configs"] == ["configs/evaluation.json"]
    assert "data/test_predictions.json" in menu["data_candidates"]
    assert "configs/evaluation.json" not in menu["data_candidates"]
    assert "Exactly [projection.data_path]" in menu["resource_contract"]["data_paths"]
    assert "number_id only" in menu["target_contract"]["selector"]
    rows = {row["path"]: row for row in cfg["fields"]}
    assert set(rows) == set(field_inventory(claim.conditions[0]))
    assert rows["/description"]["allowed_roles"] == ["conclusion_boundary"]
    catalog = build_catalog(claim, materials)
    complete = rows["/settings/examples"]["complete_source_ids"]
    assert any(catalog["sources"][sid]["block_id"] == "block_10" for sid in complete)
    assert not any(catalog["sources"][sid]["block_id"] == "block_11" for sid in complete)
    assert cfg["prose_scalar_candidates"]
    for row in cfg["prose_scalar_candidates"]:
        assert row["number_id"] in catalog["numbers"] and row["block_id"] == "block_10"
        assert row["token"] == "0.75" and "cell_id" not in row
    assert before == (claim.model_dump(mode="json"), materials.model_dump(mode="json"))
    assert not {"numerator", "denominator", "observed_accuracy", "expected_success"} & set(menu)


@pytest.mark.parametrize(
    "path,value,config,expected",
    [
        ("/dataset", "D", {"dataset": 5}, "dataset_identity"),
        ("/metric", "accuracy", {"metric": "x"}, "metric"),
        ("/description", "scope", {"description": "x"}, "conclusion_boundary"),
        ("/settings/accuracy", 0.75, {"accuracy": 0.75}, "runtime_setting"),
        ("/settings/examples", 4, {"examples": 4}, "runtime_setting"),
        ("/settings/accuracy_definition", "fraction", {}, "measurement_definition"),
        ("/settings/model", "A", {"model": "A"}, "runtime_setting"),
        ("/settings/accuracy", 0.75, {}, "reported_value"),
        ("/settings/examples", 4, {}, "sample_scope"),
        ("/settings/model", [], {"model": []}, None),
        ("/settings/new", {}, {}, None),
        ("/settings/qualifiers/0", "No ranking against other models", {}, "conclusion_boundary"),
    ],
)
def test_role_helper_preserves_finite_config_precedence(path, value, config, expected):
    assert expected_field_role(path, value, {"settings": config}) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Four fixed test predictions are available.", True),
        ("4 examples", True),
        ("40 examples", False),
        ("<td>Examples</td><td>4</td>", False),
        ("four arbitrary predictions", False),
    ],
)
def test_shared_sample_predicate_exact_equivalence(text, expected):
    req = source_requirements("/settings/examples", 4, "sample_scope")
    assert all(source_hits(req, text).values()) is expected


def add_block(materials, identifier, text):
    start = len(materials.markdown) + 2
    materials.markdown += "\n\n" + text
    materials.blocks.append(
        MaterialBlock(
            id=identifier, text=text, loc=ClaimLocation(page=1, char_start=start, char_end=start + len(text))
        )
    )
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")


def test_combined_source_definition_hits_remain_explicit_without_synthesized_quote(tmp_path):
    claim, materials = fixtures.original_inputs(tmp_path)
    add_block(materials, "definition_a", "Accuracy is a fraction of predictions.")
    add_block(materials, "definition_b", "Prediction and label equality defines a correct result.")
    _, cfg = config_choices(claim, materials)
    row = next(r for r in cfg["fields"] if r["path"] == "/settings/accuracy_definition")
    catalog = build_catalog(claim, materials)
    ids = [
        sid for sid, s in catalog["sources"].items() if s.get("block_id") in {"definition_a", "definition_b"}
    ]
    assert all(sid in row["partial_source_hits"] and sid not in row["complete_source_ids"] for sid in ids)
    hits = set().union(*(row["partial_source_hits"][sid] for sid in ids))
    assert hits == set(row["source_requirements"])
    assert "newline-joined" in row["source_selection"]


def test_unknown_empty_and_escaped_fields_are_not_dropped(tmp_path):
    claim, materials = fixtures.original_inputs(tmp_path)
    claim.conditions[0].settings.update(
        {"new/key~": [], "empty": {}, "qualifiers": ["No data augmentation used"]}
    )
    menu, cfg = config_choices(claim, materials)
    fields = {r["path"]: r for r in cfg["fields"]}
    assert menu["conditions"]["c1"]["original_fields"] == field_inventory(claim.conditions[0])
    for path in ("/settings/new~1key~0", "/settings/empty", "/settings/qualifiers/0"):
        assert fields[path]["allowed_roles"] == [] and fields[path]["unavailable_reason"]


def test_catalog_does_not_read_prediction_data_or_execute_recipe(tmp_path, monkeypatch):
    claim, materials = fixtures.original_inputs(tmp_path)
    visible = repository_context_files(materials)

    def read(path, *a, **kw):
        pytest.fail("The catalog must reuse existing context without reading any new resource")

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(
        "verification.execution_projection.entry_recipe",
        lambda *a, **kw: pytest.fail("Catalog must not evaluate recipe"),
    )
    assert pure_choices(claim, materials, repository_files=visible)["conditions"]


def test_bad_config_is_local_and_healthy_choices_remain(tmp_path):
    claim, materials = fixtures.original_inputs(tmp_path)
    bad = Path(materials.repository.root) / "configs/bad.json"
    bad.write_text("{}", encoding="utf-8")
    import hashlib

    materials.repository.files.append(
        RepositoryFile(
            path="configs/bad.json", kind="config", sha256=hashlib.sha256(bad.read_bytes()).hexdigest()
        )
    )
    materials.repository.configs.append("configs/bad.json")
    menu, cfg = config_choices(claim, materials)
    assert cfg["prose_scalar_candidates"] and any("bad.json unavailable" in s for s in menu["issues"])


def test_public_both_requests_have_identical_catalog_and_whole_claim(tmp_path):
    claim, materials, result, calls = fixtures.public_verify(tmp_path)
    assert len(calls) == 2 and result.plans[0].feasibility == "ready"
    first, scope = [json.loads(c["prompt"]) for c in calls]
    assert first["claim"] == scope["claim"] == claim.model_dump(mode="json")
    assert (
        first["execution_projection_context"]
        == scope["execution_projection_context"]
        == projection_context(claim, materials)
    )
    assert "structural menu" in calls[1]["system"] and "data_paths to exactly" in calls[0]["system"]


@pytest.mark.parametrize("damage", ["absolute", "config_in_data", "cell", "description_role", "sample_table"])
def test_five_existing_input_failures_stay_rejected_with_new_choices(tmp_path, damage):
    def mutate(phase, first, scope, claim, materials):
        if phase != "first":
            return
        plan = first["plans"][0]
        target = plan["targets"][0]
        if damage == "absolute":
            plan["data_paths"] = [str((Path(materials.repository.root) / plan["data_paths"][0]).resolve())]
        elif damage == "config_in_data":
            plan["data_paths"].append(plan["config"])
        elif damage == "cell":
            catalog = build_catalog(claim, materials)
            cell = next(k for k, v in catalog["cells"].items() if v["token"] == "0.75")
            target["selector"] = {"cell_id": cell}
        elif damage == "description_role":
            next(r for r in target["projection"]["field_roles"] if r["path"] == "/description")["role"] = (
                "sample_scope"
            )
        else:
            catalog = build_catalog(claim, materials)
            table = next(k for k, v in catalog["sources"].items() if v.get("block_id") == "block_11")
            next(r for r in target["projection"]["field_roles"] if r["path"] == "/settings/examples")[
                "source_ids"
            ] = [table]

    _, _, result, calls = fixtures.public_verify(tmp_path, change=mutate, with_observation=True)
    assert len(calls) == 2 and (not result.plans or result.plans[0].feasibility == "blocked")
    assert len(result.evidence) == 1 and not result.evidence[0].sufficient
    assert result.verification_limitations and not result.questions


def test_stored_unknown_catalog_cannot_authorize_projection(tmp_path):
    claim, materials = fixtures.original_inputs(tmp_path)
    data = fixtures.proposal_input(claim, materials, tmp_path)
    audit = data[-1]
    raw = json.loads(audit.read_text())
    raw["input"]["execution_projection_context"]["request_choices"]["schema_version"] = "unknown"
    audit.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ProjectionError, match="independently reviewed context"):
        fixtures.bind(claim, materials, data)


def test_data_identifier_works_at_an_arbitrary_indexed_json_path(tmp_path):
    claim, materials = fixtures.original_inputs(tmp_path)
    repo = Path(materials.repository.root)
    old, new = "data/test_predictions.json", "resources/released_rows_v2.json"
    destination = repo / new
    destination.parent.mkdir()
    destination.write_bytes((repo / old).read_bytes())
    row = next(r for r in materials.repository.files if r.path == old)
    row.path = new
    materials.repository.configs = [new if p == old else p for p in materials.repository.configs]
    entry = repo / "scripts/evaluate.py"
    entry.write_text(entry.read_text().replace(old, new))
    fixtures.refresh(materials, "scripts/evaluate.py")
    menu, cfg = config_choices(claim, materials)
    assert new in menu["data_candidates"] and new not in menu["configs"]
    assert old not in menu["data_candidates"] and cfg["prose_scalar_candidates"]


@pytest.mark.parametrize(
    "field,value,role,text,expected",
    [
        (
            "/settings/accuracy_definition",
            "exact equality fraction",
            "measurement_definition",
            "A fraction of predictions equal to labels.",
            True,
        ),
        (
            "/settings/accuracy_definition",
            "exact equality fraction",
            "measurement_definition",
            "A fraction of prediction scores.",
            False,
        ),
        (
            "/settings/qualifiers/0",
            "No repeated-run uncertainty claimed",
            "conclusion_boundary",
            "No repeated-run uncertainty is claimed.",
            True,
        ),
        (
            "/settings/qualifiers/0",
            "No population-performance conclusion claimed",
            "conclusion_boundary",
            "No population-performance conclusion is claimed.",
            True,
        ),
        (
            "/settings/qualifiers/0",
            "No ranking against other models",
            "conclusion_boundary",
            "No ranking against other models.",
            True,
        ),
        (
            "/settings/qualifiers/0",
            "No ranking against other models",
            "conclusion_boundary",
            "We report a ranking against other models.",
            False,
        ),
        ("/settings/qualifiers/0", "No augmentation", "conclusion_boundary", "No augmentation.", False),
    ],
)
def test_shared_source_predicate_retains_old_finite_lexical_requirements(field, value, role, text, expected):
    assert all(source_hits(source_requirements(field, value, role), text).values()) is expected


def test_catalog_source_failure_does_not_drop_original_field_inventory(tmp_path, monkeypatch):
    claim, materials = fixtures.original_inputs(tmp_path)

    def unavailable(*a, **kw):
        raise ValueError("Fixed unavailable source catalog")

    monkeypatch.setattr("verification.execution_projection_catalog.build_catalog", unavailable)
    menu, cfg = config_choices(claim, materials)
    assert set(r["path"] for r in cfg["fields"]) == set(field_inventory(claim.conditions[0]))
    assert not cfg["prose_scalar_candidates"]
    assert any("Fixed unavailable" in issue for issue in menu["issues"])


def test_catalog_does_not_turn_percent_or_other_subject_into_fraction_target(tmp_path):
    claim, materials = fixtures.original_inputs(tmp_path)
    for identifier, text in (
        ("other_subject", "On MiniSet test, model Other has accuracy 0.75."),
        ("percent", "On MiniSet test, model ExactMatch has accuracy 75%."),
        ("gap", "On MiniSet test, model ExactMatch improves accuracy by 0.75."),
    ):
        add_block(materials, identifier, text)
    _, cfg = config_choices(claim, materials)
    assert {r["block_id"] for r in cfg["prose_scalar_candidates"]} == {"block_10"}


def test_historical_v2_record_readability_never_bypasses_changed_recipe_hash(tmp_path):
    from schemas.claim import ExecutionPlan
    from verification.experiment_targets import TargetBindingError, validate_plan_targets

    claim, materials, result, _ = fixtures.public_verify(tmp_path)
    plan = result.plans[0]
    assert validate_plan_targets(plan, claim, materials) == plan.target_bindings
    plan.target_bindings["c1"].projection.recipe_sha256 = "0" * 64
    restored = ExecutionPlan.model_validate_json(plan.model_dump_json())
    with pytest.raises(
        TargetBindingError, match="Paper target binding changed or conflicts with y_paper: c1"
    ):
        validate_plan_targets(restored, claim, materials)
