"""Prose occurrences bind explicit source roles, never mere word co-occurrence."""

import copy
from pathlib import Path

import pytest

from schemas.claim import Claim, ClaimLocation, Condition
from schemas.materials import MaterialBlock, SharedMaterials
from verification.prose_numbers import bind_pair, index_numbers, resolve_number, transition_endpoints

COMPARE = "On the test split of dataset D, method A has 90% accuracy and method B has 80% accuracy."
DECREASE = (
    "On the test split of dataset D, method A has 90% accuracy in setting S1 and 80% accuracy in setting S2. "
    "The recorded accuracy decreases from 90% in S1 to 80% in S2, a decrease of 10 percentage points."
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Prose number tests must not call an external service")

    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("screening.checks.llm_json", forbidden)


def inputs(tmp_path, text=COMPARE, *, transition=False):
    path = tmp_path / "paper.md"
    path.write_text(text, encoding="utf-8")
    loc = ClaimLocation(page=1, char_start=0, char_end=len(text))
    settings = (
        {"method": "A", "from_setting": "S1", "to_setting": "S2"}
        if transition
        else {"method": "A", "comparator": "B"}
    )
    condition = Condition(
        id="c1", dataset="D", metric="accuracy", settings={"split": "test", "unit": "percent", **settings}
    )
    claim = Claim(
        id="claim",
        text=text,
        loc=loc,
        source_block_id="p",
        source_quote=text,
        conditions=[condition],
        needs=["Experiments"],
    )
    materials = SharedMaterials(
        paper_key="prose",
        source_pdf="",
        markdown=text,
        markdown_path=str(path),
        content_list_path="",
        provider="fixture",
        blocks=[MaterialBlock(id="p", text=text, loc=loc)],
    )
    return claim, materials, index_numbers(materials)


def choose(numbers, token, *, occurrence=0):
    return [key for key, record in numbers.items() if record["token"] == token][occurrence]


def bind(claim, materials, numbers, *, transition=False, reverse=False):
    left, right = ("80%", "90%") if transition else ("90%", "80%")
    if reverse:
        left, right = right, left
    condition = claim.conditions[0]
    return bind_pair(
        materials,
        numbers,
        choose(numbers, left),
        choose(numbers, right),
        dataset=condition.dataset,
        metric=condition.metric,
        settings=condition.settings,
        left_label="A",
        right_label="A" if transition else "B",
        transition=transition,
    )


def test_shared_dataset_and_split_bind_the_direct_named_predicates(tmp_path):
    claim, materials, numbers = inputs(tmp_path)
    result = bind(claim, materials, numbers)
    assert result["left"]["token"] == "90%" and result["right"]["token"] == "80%"
    assert result["left"]["quote"] == COMPARE
    assert materials.blocks[0].text == COMPARE


def test_same_method_transition_uses_its_own_setting_and_original_endpoints(tmp_path):
    claim, materials, numbers = inputs(tmp_path, DECREASE, transition=True)
    assert transition_endpoints(claim, claim.conditions[0], materials) == ("80%", "90%")
    result = bind(claim, materials, numbers, transition=True)
    assert result["kind"] == "named_transition"
    delta = resolve_number(numbers, choose(numbers, "10"), materials)
    assert delta["token"] == "10" and delta["unit_suffix"] == "percentage points"
    assert "10 percentage points" in delta["quote"]


@pytest.mark.parametrize(
    "text",
    [
        "On the test split of dataset D we discuss the protocol; on dataset E, method A has 90% accuracy and method B has 80% accuracy.",
        "On the test split of dataset D we discuss the protocol; on the dev split of dataset D, method A has 90% accuracy and method B has 80% accuracy.",
        "On the test split of dataset D, unlike method A, method B has 90% accuracy, whereas unlike method B, method A has 80% accuracy.",
        "On the test split of dataset E, not dataset D, method A has 90% accuracy and method B has 80% accuracy.",
        "On the test split of dataset D or E, method A has 90% accuracy and method B has 80% accuracy.",
        "On the test split of dataset D. Method A has 90% accuracy and method B has 80% accuracy.",
        "On the test split of dataset D, method A may have 90% accuracy and method B may have 80% accuracy.",
        "On the test split of dataset D, method A has 90% accuracy and 85% accuracy, while method B has 80% accuracy.",
    ],
)
def test_reviewer_counterexamples_and_unknown_scopes_remain_unconfirmed(tmp_path, text):
    claim, materials, numbers = inputs(tmp_path, text)
    with pytest.raises(ValueError):
        bind(claim, materials, numbers)


def test_transition_method_mentioned_only_as_reference_cannot_supply_identity(tmp_path):
    text = "Method A is the reference; method B changes accuracy from 90% in S1 to 80% in S2."
    claim, materials, numbers = inputs(tmp_path, text, transition=True)
    assert transition_endpoints(claim, claim.conditions[0], materials) is None
    with pytest.raises(ValueError):
        bind(claim, materials, numbers, transition=True)


@pytest.mark.parametrize(
    "change", ["missing_fields", "swapped_fields", "wrong_method", "conflicting_source", "missing_transition"]
)
def test_transition_identity_requires_the_original_named_actor_settings_and_assertion(tmp_path, change):
    claim, materials, _ = inputs(tmp_path, DECREASE, transition=True)
    condition = claim.conditions[0]
    if change == "missing_fields":
        condition.settings.pop("from_setting")
    elif change == "swapped_fields":
        condition.settings.update(from_setting="S2", to_setting="S1")
    elif change == "wrong_method":
        condition.settings["method"] = "B"
    elif change == "conflicting_source":
        materials.blocks[0].text = materials.markdown = DECREASE.replace("80%", "70%")
        Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
        claim.source_quote = materials.markdown
    else:
        claim.text = claim.source_quote = DECREASE.split(". ")[0] + "."
    assert transition_endpoints(claim, condition, materials) is None


@pytest.mark.parametrize("transition", [False, True])
def test_number_occurrences_cannot_exchange_their_actual_roles(tmp_path, transition):
    claim, materials, numbers = inputs(tmp_path, DECREASE if transition else COMPARE, transition=transition)
    with pytest.raises(ValueError):
        bind(claim, materials, numbers, transition=transition, reverse=True)


@pytest.mark.parametrize("change", ["record", "source", "location", "foreign_paper"])
def test_occurrence_identity_rejects_changed_records_sources_and_locations(tmp_path, change):
    _, materials, numbers = inputs(tmp_path)
    identifier = choose(numbers, "90%")
    numbers = copy.deepcopy(numbers)
    if change == "record":
        numbers[identifier]["token"] = "80%"
    elif change == "source":
        materials.blocks[0].text += " extra"
    elif change == "location":
        materials.blocks[0].loc.page = 2
    else:
        materials.paper_key = "foreign"
    with pytest.raises(ValueError):
        resolve_number(numbers, identifier, materials)


def test_html_values_never_receive_a_prose_number_selector(tmp_path):
    _, _, numbers = inputs(tmp_path, "<table><tr><td>A</td><td>90%</td></tr></table>")
    assert numbers == {}


def test_different_setting_endpoints_cannot_be_borrowed_from_another_sentence(tmp_path):
    claim, materials, numbers = inputs(tmp_path, DECREASE, transition=True)
    with pytest.raises(ValueError):
        bind_pair(
            materials,
            numbers,
            choose(numbers, "80%", occurrence=1),
            choose(numbers, "90%"),
            dataset="D",
            metric="accuracy",
            settings=claim.conditions[0].settings,
            left_label="A",
            right_label="A",
            transition=True,
        )
