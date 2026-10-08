"""Production ID operands retain exact roles, source scope and numerical guards."""

import copy

import pytest

from llm.client import LLMConfig
from tests.test_experiments_bindings_v2 import append_block, cell, evidence_for, response, run
from tests.test_prose_numbers_v2 import COMPARE, DECREASE, choose, inputs
from verification.experiment_catalog import build_catalog


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Occurrence production tests must mock all external boundaries")

    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: LLMConfig("mock", "mock", None, None))
    monkeypatch.setattr("screening.checks.llm_json", forbidden)
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)


def production_case(
    tmp_path, text=COMPARE, *, transition=False, difference=False, unit="percent", tokens=None
):
    claim, materials, _ = inputs(tmp_path, text, transition=transition)
    claim.conditions[0].settings["unit"] = unit
    catalog = build_catalog(claim, materials)
    relation = "difference" if difference else ("lt" if transition else "gt")
    left, right = tokens or (("80%", "90%") if transition else ("90%", "80%"))
    comparison = dict(
        case_id=catalog["conditions"]["c1"]["cases"][0]["id"],
        left=dict(kind="prose", number_id=choose(catalog["numbers"], left)),
        right=dict(kind="prose", number_id=choose(catalog["numbers"], right)),
        relation=relation,
    )
    if difference:
        comparison.update(
            difference_number_id=choose(catalog["numbers"], "10"), difference_mode="percentage_points"
        )
    paper = claim, materials, catalog
    candidate, review = response(paper, "p", [dict(comparison=comparison)])
    review["schema_version"] = "catalog-v2"
    review["conditions"][0].update(assertion="descriptive", matched_controls_required=False)
    if transition:
        review["conditions"][0]["setting_scopes"] = {"from_setting": "comparator", "to_setting": "subject"}
    if transition and difference:
        review["conditions"][0].update(
            difference_direction="decrease", difference_direction_quote="a decrease of 10 percentage points"
        )
    return paper, candidate, review


@pytest.mark.parametrize("kind", ["comparison", "difference", "transition", "transition_difference"])
def test_original_shared_prefix_and_named_transition_work_through_production(tmp_path, kind):
    transition = kind.startswith("transition")
    difference = "difference" in kind
    text = DECREASE if transition else COMPARE
    if difference and not transition:
        text += " The recorded difference is 10 percentage points."
    paper, candidate, review = production_case(tmp_path, text, transition=transition, difference=difference)
    original = copy.deepcopy((paper[0].model_dump(), candidate, review))
    result = run(paper, candidate, review)
    assert evidence_for(result, "c1").sufficient, result.issues
    assert not result.issues
    assert original == (paper[0].model_dump(), candidate, review)


@pytest.mark.parametrize(
    "text,transition",
    [
        (
            "On the test split of dataset D we discuss the protocol; on dataset E, method A has 90% accuracy and method B has 80% accuracy.",
            False,
        ),
        (
            "On the test split of dataset D we discuss the protocol; on the dev split of dataset D, method A has 90% accuracy and method B has 80% accuracy.",
            False,
        ),
        (
            "On the test split of dataset D, unlike method A, method B has 90% accuracy, whereas unlike method B, method A has 80% accuracy.",
            False,
        ),
        (
            "On the test split of dataset E, not dataset D, method A has 90% accuracy and method B has 80% accuracy.",
            False,
        ),
        (
            "On the test split of dataset D or E, method A has 90% accuracy and method B has 80% accuracy.",
            False,
        ),
        ("Method A is the reference; method B changes accuracy from 90% in S1 to 80% in S2.", True),
        ("On the test split of dataset D. Method A has 90% accuracy and method B has 80% accuracy.", False),
        (
            "On the test split of dataset D, method A may have 90% accuracy and method B may have 80% accuracy.",
            False,
        ),
    ],
)
def test_reviewer_scope_and_subject_false_accepts_are_rejected_in_production(tmp_path, text, transition):
    paper, candidate, review = production_case(tmp_path, text, transition=transition)
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c1").sufficient
    assert result.issues


@pytest.mark.parametrize("transition", [False, True])
def test_occurrence_swap_cannot_exchange_roles_or_transition_endpoints(tmp_path, transition):
    paper, candidate, review = production_case(
        tmp_path, DECREASE if transition else COMPARE, transition=transition
    )
    row = review["items"][0]["comparisons"][0]
    row["left"], row["right"] = row["right"], row["left"]
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c1").sufficient
    assert result.issues


@pytest.mark.parametrize("version", ["catalog-v3", "legacy", "", None, 1, {}, []])
def test_unknown_version_never_silently_uses_legacy_scope(tmp_path, version):
    from tests.test_experiments_bindings_v2 import ASPECTS, prose_case
    from verification.experiments import ExperimentsOutput, _decode_scope

    paper, candidate, catalog_review = prose_case(tmp_path)
    claim, materials, catalog = paper
    output = ExperimentsOutput(checked_aspects=ASPECTS, items=[candidate], plans=[])
    conditions, decisions, errors, _ = _decode_scope(catalog_review, claim, materials, output, catalog)
    assert not errors
    review = dict(
        conditions=[row.model_dump() for row in conditions.values()],
        items=[row.model_dump() for row in decisions.values()],
    )
    assert evidence_for(run(paper, candidate, review), "c1").sufficient
    review["schema_version"] = version
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c1").sufficient
    assert any("schema_version" in issue for issue in result.issues)


def test_partial_first_pass_is_not_promoted_by_valid_occurrence_scope(tmp_path):
    paper, candidate, review = production_case(tmp_path)
    candidate["fully_supported_conditions"] = []
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c1").sufficient


def test_difference_id_from_another_claim_cannot_supply_own_asserted_gap(tmp_path):
    paper, candidate, review = production_case(
        tmp_path, COMPARE + " The recorded difference is 10 percentage points.", difference=True
    )
    append_block(paper[1], "other", "Another condition has a difference of 10 percentage points.")
    catalog = build_catalog(paper[0], paper[1])
    row = review["items"][0]["comparisons"][0]
    row["difference_number_id"] = next(
        key for key, value in catalog["numbers"].items() if value["block_id"] == "other"
    )
    result = run((paper[0], paper[1], catalog), candidate, review)
    assert not evidence_for(result, "c1").sufficient
    assert any("own source" in issue for issue in result.issues)


@pytest.mark.parametrize("field", ["left_token", "left_source_id", "left_value_context", "difference_token"])
def test_v2_cannot_override_program_resolved_values_with_legacy_fields(tmp_path, field):
    paper, candidate, review = production_case(tmp_path)
    review["items"][0]["comparisons"][0][field] = "90%"
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c1").sufficient
    assert any(field in issue and "Extra inputs" in issue for issue in result.issues)


def test_html_cell_axes_and_original_prose_gap_use_different_id_contracts(tmp_path):
    text = (
        "On the test split of dataset D, A's recorded accuracy is 10 percentage points above B's (Table 1)."
    )
    claim, materials, _ = inputs(tmp_path, text)
    table = (
        "Table 1: Accuracy on the test split of dataset D.\n"
        "<table><tr><th>Method</th><th>Dataset</th><th>Split</th><th>Accuracy (%)</th></tr>"
        "<tr><td>A</td><td>D</td><td>test</td><td>90</td></tr>"
        "<tr><td>B</td><td>D</td><td>test</td><td>80</td></tr></table>"
    )
    append_block(materials, "table", table)
    catalog = build_catalog(claim, materials)
    paper = claim, materials, catalog
    comparison = dict(
        case_id=catalog["conditions"]["c1"]["cases"][0]["id"],
        left=dict(
            kind="cell", cell_id=cell(catalog, "table", 1, 3), label_cell_id=cell(catalog, "table", 1, 0)
        ),
        right=dict(
            kind="cell", cell_id=cell(catalog, "table", 2, 3), label_cell_id=cell(catalog, "table", 2, 0)
        ),
        relation="difference",
        difference_number_id=choose(catalog["numbers"], "10"),
        difference_mode="percentage_points",
    )
    candidate, review = response(paper, "table", [dict(comparison=comparison)])
    review["schema_version"] = "catalog-v2"
    result = run(paper, candidate, review)
    assert evidence_for(result, "c1").sufficient, result.issues
    assert not result.issues
    comparison["left"] = dict(kind="prose", number_id=comparison["left"]["cell_id"])
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c1").sufficient
    assert any("number occurrence" in issue for issue in result.issues)


def test_invalid_occurrence_preserves_other_condition_observations(tmp_path):
    paper, _, _ = production_case(tmp_path)
    claim, materials, _ = paper
    claim.conditions.append(claim.conditions[0].model_copy(update={"id": "c2"}))
    catalog = build_catalog(claim, materials)
    specs = []
    for condition in claim.conditions:
        specs.append(
            dict(
                comparison=dict(
                    case_id=catalog["conditions"][condition.id]["cases"][0]["id"],
                    left=dict(
                        kind="prose",
                        number_id=choose(catalog["numbers"], "90%")
                        if condition.id == "c2"
                        else "unknown-number",
                    ),
                    right=dict(kind="prose", number_id=choose(catalog["numbers"], "80%")),
                    relation="gt",
                )
            )
        )
    candidate, review = response((claim, materials, catalog), "p", specs)
    review["schema_version"] = "catalog-v2"
    result = run((claim, materials, catalog), candidate, review)
    assert not evidence_for(result, "c1").sufficient
    assert evidence_for(result, "c2").sufficient, result.issues
    assert any("c1" in issue for issue in result.issues)


@pytest.mark.parametrize(
    "kind",
    [
        "no_comparator",
        "wrong_unit",
        "conflicting_units",
        "unknown_setting",
        "wrong_scope",
        "unknown_unit",
        "unknown_units",
        "fraction_unit",
    ],
)
def test_occurrences_cannot_skip_original_role_units_or_settings(tmp_path, kind):
    paper, candidate, review = production_case(tmp_path)
    claim, materials, _ = paper
    if kind == "no_comparator":
        claim.conditions[0].settings.pop("comparator")
    elif kind == "wrong_unit":
        claim.conditions[0].settings["unit"] = "ms"
    elif kind == "conflicting_units":
        claim.conditions[0].settings["units"] = "seconds"
    elif kind == "unknown_setting":
        claim.conditions[0].settings["training_data"] = "extra"
    elif kind == "unknown_unit":
        claim.conditions[0].settings["unit"] = "mystery-scale"
    elif kind == "unknown_units":
        claim.conditions[0].settings["units"] = "mystery-scale"
    elif kind == "fraction_unit":
        claim.conditions[0].settings["unit"] = "fraction"
    else:
        review["conditions"][0]["setting_scopes"] = {"split": "subject"}
    catalog = build_catalog(claim, materials)
    review["items"][0]["comparisons"][0]["case_id"] = catalog["conditions"]["c1"]["cases"][0]["id"]
    result = run((claim, materials, catalog), candidate, review)
    assert not evidence_for(result, "c1").sufficient
    assert result.issues


@pytest.mark.parametrize(
    "left_unit,right_unit,expected,valid",
    [("seconds", "s", "seconds", True), ("ms", "s", "seconds", False), ("", "", "percent", False)],
)
def test_occurrence_unit_contract_retains_per_operand_scale(tmp_path, left_unit, right_unit, expected, valid):
    text = COMPARE.replace("90%", f"90 {left_unit}".strip()).replace("80%", f"80 {right_unit}".strip())
    paper, candidate, review = production_case(tmp_path, text, unit=expected, tokens=("90", "80"))
    result = run(paper, candidate, review)
    assert evidence_for(result, "c1").sufficient is valid, result.issues
    assert bool(result.issues) is (not valid)


@pytest.mark.parametrize(
    "text",
    [
        DECREASE.replace("decreases from", "increases from"),
        DECREASE.replace("a decrease of", "an increase of"),
        DECREASE.replace("10 percentage points", "5 percentage points"),
        DECREASE.replace("10 percentage points", "20 percentage points"),
        DECREASE.replace("10 percentage points", "10 percent"),
    ],
)
def test_original_transition_direction_and_gap_cannot_be_ignored_by_lt_scope(tmp_path, text):
    paper, candidate, review = production_case(tmp_path, text, transition=True)
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c1").sufficient
    assert result.issues


@pytest.mark.parametrize(
    "before,after,direction,magnitude,unit",
    [
        ("90%", "80%", "decrease", "10", "percentage points"),
        ("80%", "90%", "increase", "10", "percentage points"),
        ("80%", "90%", "increase", "12.5", "percent"),
        ("80%", "60%", "decrease", "25", "percent"),
    ],
)
@pytest.mark.parametrize("difference", [False, True])
def test_correct_explicit_transition_assertions_remain_supported(
    tmp_path, before, after, direction, magnitude, unit, difference
):
    text = (
        f"On the test split of dataset D, method A has {before} accuracy in setting S1 and {after} accuracy in setting S2. "
        f"The recorded accuracy {direction}s from {before} in S1 to {after} in S2, "
        f"{'an' if direction == 'increase' else 'a'} {direction} of {magnitude} {unit}."
    )
    paper, candidate, review = production_case(tmp_path, text, transition=True, tokens=(after, before))
    expected_relation = "difference" if difference else ("lt" if direction == "decrease" else "gt")
    review["conditions"][0]["relation"] = expected_relation
    review["items"][0]["comparisons"][0]["relation"] = expected_relation
    if difference:
        review["conditions"][0].update(
            difference_direction=direction, difference_direction_quote=f"{direction} of {magnitude} {unit}"
        )
        review["items"][0]["comparisons"][0].update(
            difference_number_id=choose(paper[2]["numbers"], magnitude),
            difference_mode="relative_percent" if unit == "percent" else "percentage_points",
        )
    result = run(paper, candidate, review)
    assert evidence_for(result, "c1").sufficient, result.issues
    assert not result.issues


@pytest.mark.parametrize("relation", ["none", "le", "ge", "gt", "eq"])
def test_scope_cannot_skip_or_weaken_the_original_strict_transition(tmp_path, relation):
    paper, candidate, review = production_case(tmp_path, DECREASE, transition=True)
    review["conditions"][0]["relation"] = relation
    if relation == "none":
        review["items"][0].update(comparisons=[], comparison_objects="not_comparative")
    else:
        review["items"][0]["comparisons"][0]["relation"] = relation
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c1").sufficient
    assert result.issues


def test_difference_scope_must_retain_original_transition_direction(tmp_path):
    paper, candidate, review = production_case(tmp_path, DECREASE, transition=True, difference=True)
    review["conditions"][0]["difference_direction"] = "increase"
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c1").sufficient
    assert any("direction" in issue for issue in result.issues)
