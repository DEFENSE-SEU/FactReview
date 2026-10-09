"""Finite ordinary-pair templates retain original IDs and downstream support gates."""

import copy

import pytest

from llm.client import LLMConfig
from tests.test_experiments_bindings_v2 import evidence_for, response, run
from tests.test_prose_numbers_v2 import choose, inputs
from verification.experiment_catalog import build_catalog
from verification.experiments import _bound_prose_pair_choices
from verification.prose_numbers import _match_pair, bind_pair


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Nominal-pair tests must mock all external boundaries")

    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: LLMConfig("mock", "mock", None, None))
    monkeypatch.setattr("screening.checks.llm_json", forbidden)
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)


def case(tmp_path, text, *, left_token="84.0%", right_token="80.0%"):
    claim, materials, _ = inputs(tmp_path, text)
    claim.conditions[0].settings.update(method="R+G", comparator="R")
    catalog = build_catalog(claim, materials)
    left = choose(catalog["numbers"], left_token, occurrence=1 if left_token == right_token else 0)
    right = choose(catalog["numbers"], right_token)
    comparison = dict(
        case_id=catalog["conditions"]["c1"]["cases"][0]["id"],
        left=dict(kind="prose", number_id=left),
        right=dict(kind="prose", number_id=right),
        relation="gt",
    )
    paper = claim, materials, catalog
    candidate, review = response(paper, "p", [dict(comparison=comparison)])
    review["schema_version"] = "catalog-v2"
    review["conditions"][0].update(assertion="descriptive", matched_controls_required=False)
    return paper, candidate, review, left, right


@pytest.mark.parametrize("prefix", ["On the test split of dataset D, ", "On the D test set, "])
@pytest.mark.parametrize(
    "body",
    ["R has 80.0% accuracy, and R+G has 84.0% accuracy.", "R accuracy is 80.0%, and R+G accuracy is 84.0%."],
)
def test_complete_prefix_and_predicate_templates_preserve_exact_ids(tmp_path, prefix, body):
    paper, candidate, review, left, right = case(tmp_path, prefix + body)
    before = copy.deepcopy([paper[0].model_dump(), paper[1].model_dump(), candidate, review])
    choices = _bound_prose_pair_choices(*paper)
    assert len(choices["c1"]) == 1
    assert choices["c1"][0]["left"] == {"kind": "prose", "number_id": left}
    assert choices["c1"][0]["right"] == {"kind": "prose", "number_id": right}
    result = run(paper, candidate, review)
    assert evidence_for(result, "c1").sufficient, result.issues
    assert before == [paper[0].model_dump(), paper[1].model_dump(), candidate, review]


def test_reversed_sentence_order_keeps_actor_roles_and_rejects_swapped_ids(tmp_path):
    paper, candidate, review, left, right = case(
        tmp_path, "On the D test set, R+G accuracy is 84.0%, and R accuracy is 80.0%."
    )
    assert evidence_for(run(paper, candidate, review), "c1").sufficient
    claim, materials, catalog = paper
    with pytest.raises(ValueError, match="directly scoped"):
        bind_pair(
            materials,
            catalog["numbers"],
            right,
            left,
            dataset="D",
            metric="accuracy",
            settings=claim.conditions[0].settings,
            left_label="R+G",
            right_label="R",
        )


@pytest.mark.parametrize("stage", ["first", "scope"])
def test_structural_pair_does_not_upgrade_partial_semantic_flags(tmp_path, stage):
    paper, candidate, review, _, _ = case(
        tmp_path, "On the D test set, R accuracy is 80.0%, and R+G accuracy is 84.0%."
    )
    if stage == "first":
        candidate["fully_supported_conditions"] = []
    else:
        review["items"][0]["full_support"] = False
    assert _bound_prose_pair_choices(*paper)["c1"]
    assert not evidence_for(run(paper, candidate, review), "c1").sufficient


@pytest.mark.parametrize(
    "text,left_token,error",
    [
        ("On the D test set, R accuracy is 80.0%, and R+G accuracy is 0.84.", "0.84", "units"),
        ("On the D test set, R accuracy is 80.0%, and R+G accuracy is 80.0%.", "80.0%", "false"),
    ],
)
def test_valid_syntax_does_not_waive_units_or_strict_relation(tmp_path, text, left_token, error):
    paper, candidate, review, left, right = case(tmp_path, text, left_token=left_token)
    claim, materials, catalog = paper
    bind_pair(
        materials,
        catalog["numbers"],
        left,
        right,
        dataset="D",
        metric="accuracy",
        settings=claim.conditions[0].settings,
        left_label="R+G",
        right_label="R",
    )
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c1").sufficient
    assert any(error in issue for issue in result.issues)


@pytest.mark.parametrize(
    "text",
    [
        "On the D test set, method A has 90% accuracy in setting S1 and 80% accuracy in setting S2.",
        "On the test split of dataset D, method A accuracy is 90% in setting S1 and 80% in setting S2.",
    ],
)
def test_transition_keeps_its_existing_prefix_and_predicate_language(text):
    settings = {"split": "test", "method": "A", "from_setting": "S1", "to_setting": "S2"}
    assert _match_pair(text, "D", "accuracy", settings, "A", "A", transition=True) is None


def test_two_operands_cannot_mix_predicate_templates(tmp_path):
    paper, candidate, review, _, _ = case(
        tmp_path, "On the D test set, R accuracy is 80.0%, and R+G has 84.0% accuracy."
    )
    assert not _bound_prose_pair_choices(*paper)["c1"]
    assert not evidence_for(run(paper, candidate, review), "c1").sufficient


@pytest.mark.parametrize(
    "text",
    [
        "On the D test set, R accuracy is 80.0%, and on the D dev set R+G accuracy is 84.0%.",
        "On the D test set, R macro accuracy is 80.0%, and R+G accuracy is 84.0%.",
        "On the D test set, R accuracy is not 80.0%, and R+G accuracy is 84.0%.",
        "On the D test set, R accuracy may be 80.0%, and R+G accuracy is 84.0%.",
    ],
)
def test_new_prefix_does_not_absorb_foreign_scope_or_qualified_predicates(tmp_path, text):
    paper, candidate, review, _, _ = case(tmp_path, text)
    assert not _bound_prose_pair_choices(*paper)["c1"]
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c1").sufficient
    assert any("directly scoped subject-predicate" in issue for issue in result.issues)
