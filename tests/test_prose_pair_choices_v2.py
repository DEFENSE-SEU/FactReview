"""Model inputs expose strict structural choices without deciding sufficient support."""

import copy
import json

import pytest

from llm.client import LLMConfig
from schemas.claim import ClaimSourceRef
from tests.test_experiments_bindings_v2 import ASPECTS, append_block, evidence_for
from tests.test_experiments_prose_occurrences_v2 import production_case
from tests.test_prose_numbers_v2 import COMPARE, DECREASE
from verification.experiment_catalog import build_catalog
from verification.experiments import _bound_prose_pair_choices, verify_experiments


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Pair-choice tests must mock external boundaries")

    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: LLMConfig("mock", "mock", None, None))
    monkeypatch.setattr("screening.checks.llm_json", forbidden)
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)


def capture(paper, candidate, review):
    requests = []

    def model(**kwargs):
        requests.append((kwargs["module"], json.loads(kwargs["prompt"])))
        return (
            dict(checked_aspects=ASPECTS, items=[candidate], plans=[])
            if kwargs["module"] == "verification.experiments"
            else copy.deepcopy(review)
        )

    result = verify_experiments(paper[0], paper[1], call=model)
    assert [module for module, _ in requests] == [
        "verification.experiments",
        "verification.experiments.scope",
    ]
    return result, requests[1][1]


@pytest.mark.parametrize("transition", [False, True])
def test_input_lists_the_direct_result_pair_and_canonical_bare_setting_keys(tmp_path, transition):
    paper, candidate, review = production_case(
        tmp_path, DECREASE if transition else COMPARE, transition=transition
    )
    original = copy.deepcopy((paper[0].model_dump(), candidate, review))
    result, payload = capture(paper, candidate, review)
    choices = payload["bound_prose_pair_choices"]["c1"]
    assert len(choices) == 1
    choice = choices[0]
    assert choice["case_id"] == paper[2]["conditions"]["c1"]["cases"][0]["id"]
    assert choice["left"] == review["items"][0]["comparisons"][0]["left"]
    assert choice["right"] == review["items"][0]["comparisons"][0]["right"]
    assert payload["catalog"]["sentences"][choice["result_sentence_id"]].startswith("On the test split")
    assert choice["structural_only"] is True
    assert "full_support" not in choice and "sufficient" not in choice
    if transition:
        assert choice["canonical_setting_scopes"] == {
            "split": "shared",
            "unit": "shared",
            "method": "shared",
            "from_setting": "comparator",
            "to_setting": "subject",
        }
    else:
        assert choice["canonical_setting_scopes"]["method"] == "subject"
        assert choice["canonical_setting_scopes"]["comparator"] == "comparator"
    assert evidence_for(result, "c1").sufficient, result.issues
    assert payload["output_schema"]["properties"]["schema_version"]["const"] == "catalog-v2"
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
        (DECREASE.replace("decreases from", "increases from"), True),
        (DECREASE.replace("10 percentage points", "5 percentage points"), True),
        (DECREASE.replace("10 percentage points", "10 percent"), True),
    ],
)
def test_invalid_prose_scopes_and_original_assertions_never_enter_choices(tmp_path, text, transition):
    paper, _, _ = production_case(tmp_path, text, transition=transition)
    assert _bound_prose_pair_choices(*paper[:2], paper[2]) == {"c1": []}


@pytest.mark.parametrize(
    "change",
    [
        "unknown_unit",
        "fraction_unit",
        "conflicting_unit",
        "missing_unit",
        "mismatched_units",
        "missing_comparator",
        "reversed_transition",
    ],
)
def test_choices_use_the_same_unit_and_original_role_guards(tmp_path, change):
    transition = change == "reversed_transition"
    text = DECREASE if transition else COMPARE
    tokens = None
    if change == "missing_unit":
        text, tokens = COMPARE.replace("%", ""), ("90", "80")
    elif change == "mismatched_units":
        text, tokens = COMPARE.replace("90%", "90 ms").replace("80%", "80 s"), ("90", "80")
    paper, _, _ = production_case(tmp_path, text, transition=transition, tokens=tokens)
    condition = paper[0].conditions[0]
    if change == "unknown_unit":
        condition.settings["unit"] = "mystery-scale"
    elif change == "fraction_unit":
        condition.settings["unit"] = "fraction"
    elif change == "conflicting_unit":
        condition.settings["units"] = "seconds"
    elif change == "missing_comparator":
        condition.settings.pop("comparator")
    elif change == "reversed_transition":
        condition.settings.update(from_setting="S2", to_setting="S1")
    catalog = build_catalog(paper[0], paper[1])
    assert _bound_prose_pair_choices(paper[0], paper[1], catalog) == {"c1": []}


def test_repeated_endpoints_stay_catalogued_but_only_direct_results_form_the_choice(tmp_path):
    paper, _, _ = production_case(tmp_path, DECREASE, transition=True)
    before = copy.deepcopy(paper[2])
    choices = _bound_prose_pair_choices(*paper[:2], paper[2])["c1"]
    assert len(paper[2]["numbers"]) == 5 and len(choices) == 1
    for side in ("left", "right"):
        number = paper[2]["numbers"][choices[0][side]["number_id"]]
        assert number["sentence"].startswith("On the test split")
    assert before == paper[2]


@pytest.mark.parametrize("second_full", [False, True])
def test_input_choices_never_upgrade_partial_or_rewrite_wrong_scope_dictionary(tmp_path, second_full):
    paper, candidate, review = production_case(tmp_path, DECREASE, transition=True)
    if second_full:
        review["conditions"][0]["setting_scopes"] = {
            "dataset": "shared",
            "metric": "shared",
            "settings.method": "shared",
            "settings.from_setting": "comparator",
            "settings.to_setting": "subject",
        }
    else:
        review["items"][0]["full_support"] = False
    original = copy.deepcopy(review)
    result, payload = capture(paper, candidate, review)
    assert len(payload["bound_prose_pair_choices"]["c1"]) == 1
    assert not evidence_for(result, "c1").sufficient
    assert original == review


def test_candidate_primary_quote_and_field_contract_are_preserved_in_input(tmp_path):
    paper, candidate, review = production_case(tmp_path, DECREASE, transition=True)
    candidate["comparison"] = [dict(block_id="p", quote=DECREASE.split(". ")[0] + ".", token="90%")]
    _, payload = capture(paper, candidate, review)
    assert payload["candidate_items"][0]["quote"] == DECREASE
    assert payload["candidate_items"][0]["comparison"][0]["quote"] != DECREASE
    assert payload["scope_field_contract"]["allowed_setting_keys"]["c1"] == list(
        paper[0].conditions[0].settings
    )
    defs = payload["output_schema"]["$defs"]
    assert "BARE" in defs["CatalogConditionScope"]["properties"]["setting_scopes"]["description"]
    assert "settings.KEY" in defs["SourceBridge"]["properties"]["condition_field"]["description"]


def test_condition_and_case_specific_choices_never_borrow_other_dataset_results(tmp_path):
    paper, _, _ = production_case(tmp_path)
    claim, materials, _ = paper
    claim.conditions.append(claim.conditions[0].model_copy(update={"id": "c2", "dataset": "E"}))
    append_block(materials, "e-results", COMPARE.replace("dataset D", "dataset E"))
    catalog = build_catalog(claim, materials)
    choices = _bound_prose_pair_choices(claim, materials, catalog)
    assert len(choices["c1"]) == len(choices["c2"]) == 1
    for condition in claim.conditions:
        pair = choices[condition.id][0]
        assert pair["case_id"] == catalog["conditions"][condition.id]["cases"][0]["id"]
        record = catalog["numbers"][pair["left"]["number_id"]]
        assert f"dataset {condition.dataset}," in record["sentence"]


def test_explicit_source_reference_for_another_condition_is_excluded(tmp_path):
    paper, _, _ = production_case(tmp_path)
    claim, materials, _ = paper
    claim.conditions.append(claim.conditions[0].model_copy(update={"id": "c2", "dataset": "E"}))
    claim.source_refs = [
        ClaimSourceRef(source_block_id="p", source_quote=COMPARE, loc=claim.loc, covered=["c2"])
    ]
    catalog = build_catalog(claim, materials)
    assert _bound_prose_pair_choices(claim, materials, catalog) == {"c1": [], "c2": []}


@pytest.mark.parametrize("secondary", [False, True])
def test_identical_source_ranges_can_explicitly_cover_two_conditions(tmp_path, secondary):
    paper, _, _ = production_case(tmp_path)
    claim, materials, _ = paper
    claim.conditions.append(claim.conditions[0].model_copy(update={"id": "c2"}))
    if secondary:
        append_block(materials, "shared-results", COMPARE)
    block = materials.blocks[-1] if secondary else materials.blocks[0]
    claim.source_refs = [
        ClaimSourceRef(source_block_id=block.id, source_quote=block.text, loc=block.loc, covered=[identifier])
        for identifier in ("c1", "c2")
    ]
    catalog = build_catalog(claim, materials)
    choices = _bound_prose_pair_choices(claim, materials, catalog)
    for identifier in ("c1", "c2"):
        assert any(
            catalog["numbers"][choice["left"]["number_id"]]["block_id"] == block.id
            for choice in choices[identifier]
        )


def test_overlapping_different_source_ranges_do_not_share_coverage(tmp_path):
    paper, _, _ = production_case(tmp_path)
    claim, materials, _ = paper
    claim.conditions.append(claim.conditions[0].model_copy(update={"id": "c2"}))
    block = materials.blocks[0]
    quote = "method A has 90% accuracy"
    start = block.text.index(quote)
    claim.source_refs = [
        ClaimSourceRef(source_block_id=block.id, source_quote=block.text, loc=block.loc, covered=["c1"]),
        ClaimSourceRef(
            source_block_id=block.id,
            source_quote=quote,
            loc=block.loc.model_copy(update={"char_start": start, "char_end": start + len(quote)}),
            covered=["c2"],
        ),
    ]
    choices = _bound_prose_pair_choices(claim, materials, build_catalog(claim, materials))
    assert choices == {"c1": [], "c2": []}


@pytest.mark.parametrize("change", ["record", "source", "location", "foreign_paper"])
def test_stale_or_foreign_numeric_occurrences_cannot_become_input_choices(tmp_path, change):
    paper, _, _ = production_case(tmp_path)
    claim, materials, catalog = paper
    if change == "record":
        next(iter(catalog["numbers"].values()))["token"] = "99%"
    elif change == "source":
        materials.blocks[0].text += " changed"
    elif change == "location":
        materials.blocks[0].loc.page = 9
    else:
        materials.paper_key = "foreign"
    assert _bound_prose_pair_choices(claim, materials, catalog) == {"c1": []}


def test_list_setting_cases_each_keep_their_own_actual_result_pair(tmp_path):
    paper, _, _ = production_case(tmp_path)
    claim, materials, _ = paper
    claim.conditions[0].settings["method"] = ["A", "C"]
    append_block(materials, "c-results", COMPARE.replace("method A", "method C"))
    catalog = build_catalog(claim, materials)
    choices = _bound_prose_pair_choices(claim, materials, catalog)["c1"]
    assert len(choices) == 2
    assert {row["case_id"] for row in choices} == {
        case["id"] for case in catalog["conditions"]["c1"]["cases"]
    }
    assert {row["subject"]["value"] for row in choices} == {"A", "C"}
    for row in choices:
        sentence = catalog["numbers"][row["left"]["number_id"]]["sentence"]
        assert f"method {row['subject']['value']} has" in sentence


def test_first_pass_partial_is_preserved_even_with_a_structural_choice(tmp_path):
    paper, candidate, review = production_case(tmp_path)
    candidate["fully_supported_conditions"] = []
    result, payload = capture(paper, candidate, review)
    assert len(payload["bound_prose_pair_choices"]["c1"]) == 1
    assert not evidence_for(result, "c1").sufficient
    assert candidate["fully_supported_conditions"] == []


def test_second_sentence_endpoint_ids_remain_rejected_despite_the_available_hint(tmp_path):
    paper, candidate, review = production_case(tmp_path, DECREASE, transition=True)
    rows = paper[2]["numbers"]
    comparison = review["items"][0]["comparisons"][0]
    for side, token in (("left", "80%"), ("right", "90%")):
        comparison[side]["number_id"] = next(
            key
            for key, row in rows.items()
            if row["token"] == token and row["sentence"].startswith("The recorded")
        )
    original = copy.deepcopy(review)
    result, payload = capture(paper, candidate, review)
    assert len(payload["bound_prose_pair_choices"]["c1"]) == 1
    assert not evidence_for(result, "c1").sufficient
    assert original == review


@pytest.mark.parametrize(
    "text,tokens",
    [
        (COMPARE.replace("90%", "80%"), ("80%", "80%")),
        (
            COMPARE.replace(
                "method A has 90% accuracy and method B has 80% accuracy",
                "method B has 80% accuracy and method A has 90% accuracy",
            ),
            ("90%", "80%"),
        ),
    ],
)
def test_equal_values_and_reversed_sentence_order_retain_distinct_actual_role_offsets(tmp_path, text, tokens):
    paper, _, _ = production_case(tmp_path, text, tokens=tokens)
    choices = _bound_prose_pair_choices(*paper[:2], paper[2])["c1"]
    assert len(choices) == 1
    choice = choices[0]
    assert choice["left"]["number_id"] != choice["right"]["number_id"]
    for side, actor in (("left", "A"), ("right", "B")):
        row = paper[2]["numbers"][choice[side]["number_id"]]
        assert text[: row["start"]].endswith(f"method {actor} has ")


def test_locatable_equal_value_pair_does_not_support_a_claimed_ten_point_gap(tmp_path):
    text = COMPARE.replace("90%", "80%") + " The recorded difference is 10 percentage points."
    paper, candidate, review = production_case(tmp_path, text, difference=True, tokens=("80%", "80%"))
    choice = _bound_prose_pair_choices(*paper[:2], paper[2])["c1"][0]
    comparison = review["items"][0]["comparisons"][0]
    comparison.update(left=choice["left"], right=choice["right"])
    original = copy.deepcopy(review)
    result, payload = capture(paper, candidate, review)
    assert len(payload["bound_prose_pair_choices"]["c1"]) == 1
    assert not evidence_for(result, "c1").sufficient
    assert any("difference" in issue.lower() for issue in result.issues)
    assert review == original
