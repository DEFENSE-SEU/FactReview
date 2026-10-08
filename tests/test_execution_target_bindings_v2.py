"""Original scalar target identity, quantity and persisted consumer contracts."""

import json
from pathlib import Path

import pytest

from schemas.claim import Claim, ClaimLocation, Condition, ExecutionPlan, ExecutionTask
from schemas.materials import MaterialBlock, SharedMaterials
from verification.experiment_catalog import build_catalog
from verification.experiment_targets import (
    TargetBindingError,
    bind_execution_target,
    runtime_target_issue,
    validate_plan_targets,
)
from verification.experiments import PlanCandidate, _plan


def inputs(
    tmp_path, statement="On D test, A has accuracy 90.", *, source=None, settings=None, metric="accuracy"
):
    condition = Condition(
        id="c1", dataset="D", metric=metric, settings=settings or {"method": "A", "split": "test"}
    )
    texts = [statement] if source is None else [statement, source]
    markdown = "\n\n".join(texts)
    path = tmp_path / "paper.md"
    path.write_text(markdown, encoding="utf-8")
    blocks = []
    offset = 0
    for index, text in enumerate(texts):
        blocks.append(
            MaterialBlock(
                id=f"b{index}",
                kind="table" if "<table" in text else "text",
                text=text,
                loc=ClaimLocation(page=index + 1, char_start=offset, char_end=offset + len(text)),
            )
        )
        offset += len(text) + 2
    claim = Claim(
        id="claim",
        text=statement,
        loc=blocks[0].loc,
        source_block_id="b0",
        source_quote=statement,
        conditions=[condition],
        needs=["Experiments"],
    )
    materials = SharedMaterials(
        paper_key="test",
        markdown=markdown,
        markdown_path=str(path),
        source_pdf=str(tmp_path / "paper.pdf"),
        content_list_path="content.json",
        provider="mock",
        blocks=blocks,
    )
    return claim, condition, materials


def report(materials, token="90", index=-1, **changes):
    block = materials.blocks[index]
    return {"block_id": block.id, "quote": block.text, "token": token, **changes}


def make_plan(claim, condition, binding):
    return ExecutionPlan(
        id="plan",
        claim_id=claim.id,
        condition_ids=[condition.id],
        target_conditions=[condition],
        task=ExecutionTask(entry_script="eval.py"),
        run_mode="analysis",
        y_paper={condition.id: binding.value},
        target_bindings={condition.id: binding},
        feasibility="ready",
        priority="high",
    )


@pytest.fixture(autouse=True)
def no_external(monkeypatch):
    def unavailable(*args, **kwargs):
        pytest.fail("Target binding attempted an external request")

    monkeypatch.setattr("socket.socket.connect", unavailable)
    monkeypatch.setattr("socket.create_connection", unavailable)


def test_original_scalar_can_roundtrip_and_revalidate(tmp_path):
    claim, condition, materials = inputs(tmp_path)
    binding = bind_execution_target(claim, condition, report(materials), materials)
    plan = make_plan(claim, condition, binding)
    loaded = ExecutionPlan.model_validate_json(plan.model_dump_json())
    assert validate_plan_targets(loaded, claim, materials) == {"c1": binding}
    assert binding.value == 90 and binding.subject == "A" and binding.unit is None
    assert Path(binding.pointer.locator).read_text(encoding="utf-8") == binding.pointer.quote


@pytest.mark.parametrize(
    "source,token",
    [
        ("On D test, unlike A, B has accuracy 80.", "80"),
        ("On D test, A has accuracy 80 and B has accuracy 90.", "90"),
        ("On D test, A has an accuracy improvement of 90.", "90"),
        ("On D validation, A has accuracy 90.", "90"),
        ("On E test, A has accuracy 90.", "90"),
        ("On D test, A has F1 90.", "90"),
    ],
)
def test_wrong_actor_quantity_or_scope_cannot_supply_target(tmp_path, source, token):
    claim, condition, materials = inputs(tmp_path, source=source)
    with pytest.raises(TargetBindingError):
        bind_execution_target(claim, condition, report(materials, token), materials)


@pytest.mark.parametrize("value", ["4.6", "90", "0.4"])
def test_absolute_measurement_values_are_not_keyword_or_number_blocklisted(tmp_path, value):
    claim, condition, materials = inputs(tmp_path, f"On D test, A has accuracy {value}.")
    binding = bind_execution_target(claim, condition, report(materials, value), materials)
    assert binding.value == float(value)


@pytest.mark.parametrize(
    "text",
    [
        "On D test, A has 90 accuracy.",
        "On the test split of dataset D, method A has 90 accuracy.",
        "For D test, model A records accuracy 90.",
    ],
)
def test_explicit_scalar_predicate_forms_retain_original_value_identity(tmp_path, text):
    claim, condition, materials = inputs(tmp_path, text)
    assert bind_execution_target(claim, condition, report(materials), materials).value == 90


@pytest.mark.parametrize(
    "metric", ["accuracy improvement", "accuracy ratio", "relative accuracy change", "signal-to-noise ratio"]
)
def test_metric_name_alone_cannot_define_absolute_quantity_semantics(tmp_path, metric):
    claim, condition, materials = inputs(tmp_path, f"On D test, A has {metric} 90.", metric=metric)
    with pytest.raises(TargetBindingError, match=r"semantics|non-scalar"):
        bind_execution_target(claim, condition, report(materials), materials)


@pytest.mark.parametrize("changes", [{"seed": [1, 2]}, {"comparison": "B"}, {"protocol": {"train": "X"}}])
def test_single_number_does_not_establish_composite_settings(tmp_path, changes):
    claim, condition, materials = inputs(tmp_path, settings={"method": "A", "split": "test", **changes})
    with pytest.raises(TargetBindingError):
        bind_execution_target(claim, condition, report(materials), materials)


@pytest.mark.parametrize(
    "header,actor",
    [
        ("D test accuracy improvement", "A"),
        ("D test accuracy ratio", "A"),
        ("D test relative accuracy change", "A"),
        ("D test accuracy", "B (unlike A)"),
        ("D train accuracy", "A"),
        ("E test accuracy", "A"),
    ],
)
def test_table_target_requires_exact_absolute_axis_and_actor(tmp_path, header, actor):
    table = f"<table><tr><th>Method</th><th>{header}</th></tr><tr><td>{actor}</td><td>90</td></tr></table>"
    claim, condition, materials = inputs(tmp_path, source=table)
    with pytest.raises(TargetBindingError):
        bind_execution_target(claim, condition, report(materials), materials)


@pytest.mark.parametrize("tags", ["td", "th"])
def test_native_absolute_table_has_a_valid_positive_path(tmp_path, tags):
    table = f"<table><tr><{tags}>Method</{tags}><{tags}>D test accuracy</{tags}></tr><tr><td>A</td><td>90</td></tr><tr><td>B</td><td>90</td></tr></table>"
    claim, condition, materials = inputs(tmp_path, source=table)
    binding = bind_execution_target(claim, condition, report(materials), materials)
    assert binding.value == 90 and binding.selector.cell_id and binding.subject == "A"


def test_table_never_borrows_seed_or_method_from_preceding_data_row(tmp_path):
    table = "<table><tr><th>Method</th><th>Seed</th><th>D test accuracy</th></tr><tr><td>A</td><td>42</td><td>42</td></tr><tr><td>A</td><td>43</td><td>90</td></tr></table>"
    claim, condition, materials = inputs(
        tmp_path,
        "On D test seed 42, A has accuracy 90.",
        source=table,
        settings={"method": "A", "split": "test", "seed": 42},
    )
    with pytest.raises(TargetBindingError, match="uniquely bind"):
        bind_execution_target(claim, condition, report(materials), materials)


@pytest.mark.parametrize(
    "label,value,accepted",
    [
        ("Split", "train", False),
        ("Model", "B", False),
        ("Split", "test", True),
        ("Model", "A", True),
        ("Units", "ms", False),
    ],
)
def test_explicit_right_hand_table_dimensions_have_equal_authority(tmp_path, label, value, accepted):
    table = f"<table><tr><th>Method</th><th>D test accuracy</th><th>{label}</th></tr><tr><td>A</td><td>90</td><td>{value}</td></tr></table>"
    claim, condition, materials = inputs(tmp_path, source=table)
    if accepted:
        assert bind_execution_target(claim, condition, report(materials), materials).value == 90
    else:
        with pytest.raises(TargetBindingError):
            bind_execution_target(claim, condition, report(materials), materials)


@pytest.mark.parametrize(
    "label,value", [("Architecture", "B"), ("Architecture", "42"), ("Configuration", "baseline")]
)
def test_anonymous_claim_cannot_borrow_an_undeclared_named_table_row(tmp_path, label, value):
    table = f"<table><tr><th>{label}</th><th>D test accuracy</th></tr><tr><td>{value}</td><td>90</td></tr></table>"
    claim, condition, materials = inputs(
        tmp_path, "D test accuracy is 90.", source=table, settings={"split": "test"}
    )
    with pytest.raises(TargetBindingError, match="uniquely bind"):
        bind_execution_target(claim, condition, report(materials), materials)


def test_anonymous_dataset_row_retains_a_positive_path(tmp_path):
    table = "<table><tr><th>Dataset</th><th>test accuracy</th></tr><tr><td>D</td><td>90</td></tr></table>"
    claim, condition, materials = inputs(
        tmp_path, "D test accuracy is 90.", source=table, settings={"split": "test"}
    )
    binding = bind_execution_target(claim, condition, report(materials), materials)
    assert binding.subject is None and binding.value == 90


@pytest.mark.parametrize("selected,expected", [("90", True), ("80", False)])
def test_table_multilevel_headers_preserve_metric_identity(tmp_path, selected, expected):
    table = '<table><tr><th rowspan="2">Method</th><th colspan="2">D test</th></tr><tr><th>accuracy</th><th>F1</th></tr><tr><td>A</td><td>90</td><td>80</td></tr></table>'
    claim, condition, materials = inputs(tmp_path, f"On D test, A has accuracy {selected}.", source=table)
    if expected:
        assert bind_execution_target(claim, condition, report(materials, selected), materials).value == 90
    else:
        with pytest.raises(TargetBindingError):
            bind_execution_target(claim, condition, report(materials, selected), materials)


def test_selected_other_table_cannot_borrow_reported_quote(tmp_path):
    table = "<table><tr><th>Method</th><th>D test accuracy</th></tr><tr><td>A</td><td>90</td></tr></table>"
    claim, condition, materials = inputs(tmp_path, source=table + "\n" + table)
    cells = build_catalog(claim, materials)["cells"]
    cell = next(k for k, v in cells.items() if v["table"] == 1 and v["cell_type"] == "number")
    with pytest.raises(TargetBindingError):
        bind_execution_target(
            claim, condition, report(materials, quote=table), materials, selector={"cell_id": cell}
        )


@pytest.mark.parametrize("change", ["value", "claim", "condition", "artifact", "binding"])
def test_persisted_binding_cannot_authorize_tampered_targets(tmp_path, change):
    claim, condition, materials = inputs(tmp_path)
    binding = bind_execution_target(claim, condition, report(materials), materials)
    plan = make_plan(claim, condition, binding)
    if change == "value":
        plan.y_paper["c1"] = 80
    elif change == "claim":
        claim.text = "On D test, A has accuracy 80."
    elif change == "condition":
        condition.settings["split"] = "validation"
    elif change == "artifact":
        Path(materials.markdown_path).write_text(materials.markdown + "\nchanged", encoding="utf-8")
    else:
        plan.target_bindings["c1"].quantity_kind = "difference"
    with pytest.raises(TargetBindingError):
        validate_plan_targets(plan, claim, materials)


def test_legacy_plan_stays_readable_without_acquiring_target_trust(tmp_path):
    claim, condition, materials = inputs(tmp_path)
    binding = bind_execution_target(claim, condition, report(materials), materials)
    raw = make_plan(claim, condition, binding).model_dump()
    del raw["target_bindings"]
    legacy = ExecutionPlan.model_validate(raw)
    assert legacy.y_paper == {"c1": 90} and legacy.target_bindings == {}
    with pytest.raises(TargetBindingError, match="missing"):
        validate_plan_targets(legacy, claim, materials)


@pytest.mark.parametrize("kind", ["empty", "duplicate", "foreign_claim", "missing_ypaper"])
def test_consumer_revalidates_mutated_plan_identity(tmp_path, kind):
    claim, condition, materials = inputs(tmp_path)
    plan = make_plan(claim, condition, bind_execution_target(claim, condition, report(materials), materials))
    if kind == "empty":
        plan.target_conditions = []
        plan.condition_ids = []
        plan.y_paper = {}
        plan.target_bindings = {}
    elif kind == "duplicate":
        plan.target_conditions.append(condition)
    elif kind == "foreign_claim":
        plan.claim_id = "another"
    else:
        plan.y_paper = {}
    with pytest.raises(TargetBindingError):
        validate_plan_targets(plan, claim, materials)


def test_explicit_percent_scale_must_be_supplied_by_runtime(tmp_path):
    claim, condition, materials = inputs(tmp_path, "On D test, A has accuracy 90%.")
    binding = bind_execution_target(claim, condition, report(materials, "90%"), materials)
    assert binding.unit == "%" and binding.value == 90
    assert runtime_target_issue(binding, {}, observation_unit="percent") == ""
    assert runtime_target_issue(binding, {"units": "%"}) == ""
    assert runtime_target_issue(binding, {})
    assert runtime_target_issue(binding, {"units": "ms"})
    assert runtime_target_issue(binding, {"units": "%"}, observation_unit="ms")


def test_wrong_b_plan_is_blocked_even_without_paper_items(tmp_path):
    claim, _, materials = inputs(tmp_path, source="On D test, unlike A, B has accuracy 80.")
    plan = _plan(
        claim,
        materials,
        PlanCandidate(
            targets=[{"condition_id": "c1", "reported": report(materials, "80")}],
            run_mode="analysis",
            feasibility="ready",
            priority="high",
        ),
    )
    assert plan.feasibility == "blocked" and plan.target_bindings == {}
    assert "Unresolved paper target c1" in plan.blocker


def test_real_bert_improvement_is_not_an_absolute_accuracy_target(tmp_path):
    saved = json.loads(
        (Path(__file__).parent / "fixtures/execution_target_bert034_v2.json").read_text(encoding="utf-8")
    )
    claim = Claim.model_validate(saved["claim"])
    block = MaterialBlock.model_validate(saved["primary_block"])
    path = tmp_path / "paper.md"
    path.write_text(block.text, encoding="utf-8")
    materials = SharedMaterials(
        paper_key="bert-original",
        markdown=block.text,
        markdown_path=str(path),
        source_pdf="unused.pdf",
        content_list_path="unused.json",
        provider="original-saved",
        blocks=[block],
    )
    before = claim.model_dump(mode="json")
    with pytest.raises(TargetBindingError, match=r"non-scalar|absolute scalar"):
        bind_execution_target(
            claim,
            claim.conditions[0],
            {"block_id": block.id, "quote": claim.source_quote, "token": "4.6%"},
            materials,
        )
    assert claim.model_dump(mode="json") == before
    assert saved["plan"]["y_paper"] == {"c1": 4.6}
