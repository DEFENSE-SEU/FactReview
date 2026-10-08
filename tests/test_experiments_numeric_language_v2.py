"""Exact scientific quantities allow lexical unit and spacing variants only."""

import pytest

from llm.client import LLMConfig
from schemas.claim import Claim, Condition
from schemas.materials import SharedMaterials
from tests.test_experiments_bindings_v2 import (
    append_block,
    beit042,
    evidence_for,
    prose_case,
    response,
    run,
    source,
)
from verification.experiment_catalog import build_catalog


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Numeric-language tests require mocked external boundaries")

    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: LLMConfig("mock", "mock", None, None))
    monkeypatch.setattr("screening.checks.llm_json", forbidden)
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)


@pytest.mark.parametrize("key", ["unit", "units"])
@pytest.mark.parametrize("value", ["percent", "percentage"])
def test_percent_condition_alias_binds_actual_percent_scaled_operands(tmp_path, key, value):
    paper, candidate, review = prose_case(tmp_path, "D test A accuracy 90%.", "D test B accuracy 80%.")
    claim, materials, _ = paper
    claim.conditions[0].settings[key] = value
    catalog = build_catalog(claim, materials)
    row = review["items"][0]["comparisons"][0]
    row.update(left_token="90%", right_token="80%", case_id=catalog["conditions"]["c1"]["cases"][0]["id"])
    result = run((claim, materials, catalog), candidate, review)
    assert evidence_for(result, "c1").sufficient, result.issues
    assert not result.issues


@pytest.mark.parametrize("value", ["fraction", "ms", "mystery-scale"])
def test_condition_unit_cannot_override_actual_percent_scale(tmp_path, value):
    paper, candidate, review = prose_case(tmp_path, "D test A accuracy 90%.", "D test B accuracy 80%.")
    claim, materials, _ = paper
    claim.conditions[0].settings["unit"] = value
    catalog = build_catalog(claim, materials)
    review["items"][0]["comparisons"][0].update(
        left_token="90%", right_token="80%", case_id=catalog["conditions"]["c1"]["cases"][0]["id"]
    )
    result = run((claim, materials, catalog), candidate, review)
    assert not evidence_for(result, "c1").sufficient and result.issues


def unit_case(tmp_path, *, metric, left_unit, right_unit, expected):
    materials = SharedMaterials(
        paper_key="explicit-units",
        source_pdf="",
        markdown="",
        markdown_path=str(tmp_path / "units.md"),
        content_list_path="",
        provider="mock",
    )
    assertion = append_block(materials, "claim", f"A {metric} is higher than B on D test.")
    left = append_block(materials, "left", f"D test A {metric} 90{left_unit}.")
    right = append_block(materials, "right", f"D test B {metric} 80{right_unit}.")
    claim = Claim(
        id="units",
        text=assertion.text,
        loc=assertion.loc,
        source_block_id=assertion.id,
        source_quote=assertion.text,
        needs=["Experiments"],
        conditions=[
            Condition(
                id="c1",
                dataset="D",
                metric=metric,
                settings={"model": "A", "comparison": "B", "split": "test", **expected},
            )
        ],
    )
    catalog = build_catalog(claim, materials)
    paper = claim, materials, catalog
    candidate, review = response(
        paper,
        assertion.id,
        [
            dict(
                comparison=dict(
                    case_id=catalog["conditions"]["c1"]["cases"][0]["id"],
                    left_source_id=source(catalog, "left"),
                    left_token="90%" if left_unit == "%" else "90",
                    left_value_context=left.text,
                    right_source_id=source(catalog, "right"),
                    right_token="80%" if right_unit == "%" else "80",
                    right_value_context=right.text,
                    relation="gt",
                )
            )
        ],
    )
    return paper, candidate, review


@pytest.mark.parametrize(
    "source_unit,expected",
    [(" s", {"unit": "seconds"}), (" seconds", {"units": "s"}), (" s", {"unit": "seconds", "units": "s"})],
)
def test_seconds_aliases_require_explicit_operand_scale(tmp_path, source_unit, expected):
    paper, candidate, review = unit_case(
        tmp_path, metric="latency", left_unit=source_unit, right_unit=source_unit, expected=expected
    )
    result = run(paper, candidate, review)
    assert evidence_for(result, "c1").sufficient, result.issues
    assert not result.issues


@pytest.mark.parametrize(
    "left_unit,right_unit,expected",
    [(" ms", " ms", {"unit": "seconds"}), (" ms", " s", {"unit": "ms"}), (" s", " ms", {"units": "seconds"})],
)
def test_milliseconds_and_seconds_are_not_converted_or_ignored(tmp_path, left_unit, right_unit, expected):
    paper, candidate, review = unit_case(
        tmp_path, metric="latency", left_unit=left_unit, right_unit=right_unit, expected=expected
    )
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c1").sufficient and result.issues


@pytest.mark.parametrize("key", ["unit", "units"])
def test_missing_explicit_scale_cannot_be_inferred_from_accuracy_or_condition(tmp_path, key):
    paper, candidate, review = unit_case(
        tmp_path, metric="accuracy", left_unit="", right_unit="", expected={key: "percent"}
    )
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c1").sufficient and result.issues


@pytest.mark.parametrize(
    "metric,source_unit,expected",
    [
        ("accuracy", "%", {"unit": "percent", "units": "fraction"}),
        ("accuracy", "%", {"unit": "fraction", "units": "percent"}),
        ("latency", " seconds", {"unit": "seconds", "units": "milliseconds"}),
    ],
)
def test_multiple_unit_fields_must_agree_with_each_other_and_each_operand(
    tmp_path, metric, source_unit, expected
):
    paper, candidate, review = unit_case(
        tmp_path, metric=metric, left_unit=source_unit, right_unit=source_unit, expected=expected
    )
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c1").sufficient and result.issues


@pytest.mark.parametrize("name", ["mIoU/ADE20K score", "mIoU  /  ADE20K score", "mIoU/\nADE20K score"])
def test_complete_metric_name_allows_only_formatting_whitespace(tmp_path, name):
    paper, candidate, review = beit042(tmp_path)
    review["items"][1]["comparisons"][0]["bridges"][0]["explanation"] = (
        f"The complete {name} quantity is the mIoU metric defined for ADE20K."
    )
    result = run(paper, candidate, review)
    assert evidence_for(result, "c1").sufficient
    assert evidence_for(result, "c2").sufficient, result.issues
    assert not result.issues


@pytest.mark.parametrize("name", ["IoU / ADE20K score", "mIoU / ImageNet score", "mIoU"])
def test_metric_explanation_must_preserve_complete_quantity_and_dataset(tmp_path, name):
    paper, candidate, review = beit042(tmp_path)
    review["items"][1]["comparisons"][0]["bridges"][0]["explanation"] = f"The measured quantity is {name}."
    result = run(paper, candidate, review)
    assert evidence_for(result, "c1").sufficient
    assert not evidence_for(result, "c2").sufficient and result.issues
