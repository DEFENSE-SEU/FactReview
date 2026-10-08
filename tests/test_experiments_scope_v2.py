"""Independent experiment review keeps applicable concerns and complete, bound support."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from common import run_stats
from llm.client import LLMConfig
from schemas.claim import Claim, ClaimLocation, ClaimSourceRef, Condition
from schemas.materials import MaterialBlock, SharedMaterials
from verification.experiments import verify_experiments

ASPECTS = ["correspondence", "fairness", "isolation", "stability", "consistency"]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: LLMConfig("mock", "mock", None, None))
    monkeypatch.setattr("screening.checks.llm_json", lambda **kw: pytest.fail("Unmocked model boundary"))
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **kw: pytest.fail("Network access"))
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: pytest.fail("External process"))


@pytest.fixture
def paper(tmp_path):
    text = "A accuracy is higher than B on D test."
    claim = Claim(
        id="c",
        text=text,
        loc=ClaimLocation(page=1),
        source_block_id="claim",
        source_quote=text,
        conditions=[Condition(id="c1", dataset="D", metric="accuracy", settings={"split": "test"})],
        needs=["Experiments"],
    )
    materials = SharedMaterials(
        paper_key="fixture",
        source_pdf="fixture.pdf",
        markdown="",
        markdown_path=str(tmp_path / "paper.md"),
        content_list_path="content.json",
        provider="mock",
        blocks=[],
    )
    append(materials, "claim", text)
    append(materials, "left", "D test A accuracy 90.")
    append(materials, "right", "D test B accuracy 80.")
    return claim, materials


def append(materials, block_id, text):
    start = len(materials.markdown)
    materials.markdown += text + "\n"
    materials.blocks.append(
        MaterialBlock(
            id=block_id,
            text=text,
            loc=ClaimLocation(page=1, char_start=start, char_end=start + len(text)),
        )
    )
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")


def item(materials, kind="paper_support", **changes):
    aspect = {
        "missing_ablation": "isolation",
        "missing_control": "fairness",
        "missing_statistic": "stability",
        "small_gap_without_statistics": "stability",
    }.get(kind, "correspondence")
    return {
        "aspect": aspect,
        "kind": kind,
        "block_id": "claim",
        "quote": materials.blocks[0].text,
        "covered": ["c1"],
        "fully_supported_conditions": ["c1"],
        "detail": "Candidate observation.",
        **changes,
    }


def number(materials, block_id, token):
    return {
        "block_id": block_id,
        "quote": next(b.text for b in materials.blocks if b.id == block_id),
        "token": token,
    }


def comparison(materials, **changes):
    return {
        "left": number(materials, "left", "90"),
        "right": number(materials, "right", "80"),
        "left_label": "A",
        "right_label": "B",
        "metric_label": "accuracy",
        "relation": "gt",
        "context": [{"block_id": b.id, "quote": b.text} for b in materials.blocks],
        **changes,
    }


def scope(claim, items, **condition_changes):
    return {
        "conditions": [
            {
                "condition_id": c.id,
                "claim_quote": claim.text,
                "assertion": "controlled_comparison",
                "matched_controls_required": True,
                "uncertainty_sensitive": True,
                "relation": "gt",
                "subject": "A",
                "comparator": "B",
                "rationale": "The claim compares the named systems.",
                **condition_changes,
            }
            for c in claim.conditions
        ],
        "items": [
            {
                "item_index": i,
                "condition_id": c,
                "applicability": "applicable",
                "grounds": [{"block_id": row["block_id"], "quote": row["quote"]}],
                "rationale": "Independent source review.",
                "full_support": True,
                "qualifiers_complete": True,
                "comparison_objects": "matched",
                "comparisons": [],
            }
            for i, row in enumerate(items)
            for c in row["covered"]
        ],
    }


def run(paper, items, review):
    claim, materials = paper
    calls = []

    def model(**kwargs):
        calls.append(kwargs["module"])
        assert kwargs["module"] == "verification.experiments"
        return {"checked_aspects": ASPECTS, "items": items, "plans": []}

    def reviewer(**kwargs):
        calls.append(kwargs["module"])
        assert kwargs["module"] == "verification.experiments.scope"
        if isinstance(review, Exception):
            raise review
        return review

    result = verify_experiments(claim, materials, call=model, scope_call=reviewer)
    assert calls == ["verification.experiments", "verification.experiments.scope"]
    return result


@pytest.mark.parametrize("kind", ["missing_ablation", "missing_statistic", "missing_control"])
def test_descriptive_resource_and_leaderboard_reports_do_not_require_causal_controls(paper, kind):
    claim, materials = paper
    claim.text = "A reports D test accuracy 90 using 4 accelerators over 4 days."
    items = [item(materials, kind)]
    review = scope(
        claim,
        items,
        assertion="descriptive",
        matched_controls_required=False,
        uncertainty_sensitive=False,
        relation="none",
        subject="",
        comparator="",
    )
    result = run(paper, items, review)
    assert len(result.evidence) == 1
    assert not result.evidence[0].sufficient and not result.evidence[0].concern
    assert not result.evidence[0].affects_claim and not result.questions
    assert result.issues


def test_causal_attribution_still_requires_component_ablation(paper):
    claim, materials = paper
    claim.text = "Component X causes A accuracy to exceed B on D test."
    items = [item(materials, "missing_ablation")]
    review = scope(claim, items, assertion="causal_attribution", credited_component="Component X")
    result = run(paper, items, review)
    assert result.evidence[0].sufficient and result.evidence[0].concern
    assert result.questions[0].claim_id == claim.id


def test_small_comparative_gap_still_requires_statistics(paper):
    claim, materials = paper
    items = [item(materials, "small_gap_without_statistics")]
    result = run(paper, items, scope(claim, items))
    assert result.evidence[0].sufficient and result.questions


def test_original_improves_sentence_without_baseline_cannot_be_sufficient(paper):
    claim, materials = paper
    claim.text = "The method improves MRR on A."
    claim.conditions[0].metric = "MRR"
    items = [item(materials)]
    review = scope(claim, items, subject="", comparator="")
    result = run(paper, items, review)
    assert not result.evidence[0].sufficient and not result.questions
    assert any("each required comparison case" in issue for issue in result.issues)


def test_grounded_numeric_comparison_can_fully_support_claim(paper):
    claim, materials = paper
    items = [item(materials)]
    review = scope(claim, items)
    review["items"][0]["comparisons"] = [comparison(materials)]
    result = run(paper, items, review)
    assert result.evidence[0].sufficient and not result.issues


@pytest.mark.parametrize(
    "change", ["wrong_comparator", "reversed_operands", "metric", "split", "units", "missing_relation"]
)
def test_comparative_support_rejects_unbound_or_changed_targets(paper, change):
    claim, materials = paper
    claim.conditions[0].settings.update(model="A", comparison="B")
    items = [item(materials)]
    review = scope(claim, items)
    comp = comparison(materials)
    if change == "wrong_comparator":
        review["items"][0]["comparison_objects"] = "unmatched"
    elif change == "reversed_operands":
        comp["left"], comp["right"] = comp["right"], comp["left"]
        comp["left_label"], comp["right_label"] = "B", "A"
        comp["relation"] = review["conditions"][0]["relation"] = "lt"
    elif change == "metric":
        comp["metric_label"] = "F1"
    elif change == "split":
        claim.conditions[0].settings["split"] = "validation"
    elif change == "units":
        append(materials, "percent", "D test A accuracy 90%.")
        comp["left"] = number(materials, "percent", "90%")
    else:
        del review["conditions"][0]["relation"]
    review["items"][0]["comparisons"] = [comp]
    result = run(paper, items, review)
    assert result.evidence and not result.evidence[0].sufficient and result.issues
    assert not result.questions


def test_swapped_scope_roles_cannot_override_structured_condition(paper):
    claim, materials = paper
    claim.conditions[0].settings.update(model="B", comparison="A")
    items = [item(materials)]
    review = scope(claim, items)
    review["items"][0]["comparisons"] = [comparison(materials)]
    assert not run(paper, items, review).evidence[0].sufficient


@pytest.mark.parametrize("right_split", ["test", "dev"])
@pytest.mark.parametrize("right_dataset", ["D", "Other"])
def test_each_operand_must_bind_its_own_dataset_and_split(paper, right_split, right_dataset):
    claim, materials = paper
    append(materials, "candidate", f"{right_dataset} {right_split} B accuracy 80.")
    items = [item(materials)]
    review = scope(claim, items)
    # The shared context still contains D/test in the claim and left operand.
    review["items"][0]["comparisons"] = [comparison(materials, right=number(materials, "candidate", "80"))]
    result = run(paper, items, review)
    assert result.evidence[0].sufficient is (right_split == "test" and right_dataset == "D")


@pytest.mark.parametrize("tie,missing", [(False, False), (True, False), (False, True)])
def test_every_task_in_native_html_table_must_satisfy_strict_relation(paper, tie, missing):
    claim, materials = paper
    claim.text = "A accuracy is lower than B across tasks T1, T2, T3 and T4 on D test."
    claim.conditions[0].settings["tasks"] = ["T1", "T2", "T3", "T4"]
    table = (
        "D test accuracy <table><tr><th>Model</th><th>T1</th><th>T2</th><th>T3</th><th>T4</th></tr>"
        f"<tr><td>A</td><td>{'82.1' if tie else '82.0'}</td><td>84.1</td><td>75.7</td><td>91.6</td></tr>"
        "<tr><td>B</td><td>82.1</td><td>84.3</td><td>77.5</td><td>92.1</td></tr></table>"
    )
    append(materials, "table", table)
    items = [item(materials)]
    review = scope(claim, items, relation="lt", required_cases=["T1", "T2", "T3", "T4"])
    lefts = ["82.1" if tie else "82.0", "84.1", "75.7", "91.6"]
    rights = ["82.1", "84.3", "77.5", "92.1"]
    review["items"][0]["comparisons"] = [
        comparison(
            materials,
            case=f"T{i + 1}",
            relation="lt",
            left=number(materials, "table", left),
            right=number(materials, "table", right),
            left_cell={"row": 1, "column": i + 1},
            right_cell={"row": 2, "column": i + 1},
        )
        for i, (left, right) in enumerate(zip(lefts, rights, strict=True))
    ]
    if missing:
        review["items"][0]["comparisons"].pop(0)
    result = run(paper, items, review)
    assert result.evidence[0].sufficient is (not tie and not missing)
    if tie:
        assert any("false for 82.1 and 82.1" in issue for issue in result.issues)


@pytest.mark.parametrize("wrong_column", [False, True])
def test_native_multimetric_table_binds_metric_to_cell_column(paper, wrong_column):
    claim, materials = paper
    claim.text = "A F1 is higher than B on D test."
    claim.conditions[0].metric = "F1"
    table = (
        "D test <table><tr><th rowspan='2'>Model</th><th colspan='2'>D test</th></tr>"
        "<tr><th>Accuracy</th><th>F1</th></tr><tr><td>A</td><td>90</td><td>91</td></tr>"
        "<tr><td>B</td><td>80</td><td>81</td></tr></table>"
    )
    append(materials, "table", table)
    items = [item(materials)]
    review = scope(claim, items)
    col = 1 if wrong_column else 2
    review["items"][0]["comparisons"] = [
        comparison(
            materials,
            metric_label="F1",
            left=number(materials, "table", "90" if wrong_column else "91"),
            right=number(materials, "table", "80" if wrong_column else "81"),
            left_cell={"row": 2, "column": col},
            right_cell={"row": 3, "column": col},
        )
    ]
    result = run(paper, items, review)
    assert result.evidence[0].sufficient is not wrong_column


@pytest.mark.parametrize("wrong_column", [False, True])
def test_common_caption_cannot_override_explicit_table_metric_column(paper, wrong_column):
    claim, materials = paper
    claim.text = "A accuracy is lower than B on D test T1."
    claim.conditions[0].settings["tasks"] = ["T1"]
    table = (
        "D test accuracy and F1 <table><tr><th rowspan='2'>Model</th><th colspan='2'>T1</th></tr>"
        "<tr><th>accuracy</th><th>F1</th></tr><tr><td>A</td><td>70</td><td>91</td></tr>"
        "<tr><td>B</td><td>80</td><td>81</td></tr></table>"
    )
    append(materials, "table", table)
    items = [item(materials)]
    relation = "gt" if wrong_column else "lt"
    review = scope(claim, items, relation=relation, required_cases=["T1"])
    col = 2 if wrong_column else 1
    review["items"][0]["comparisons"] = [
        comparison(
            materials,
            case="T1",
            relation=relation,
            left=number(materials, "table", "91" if wrong_column else "70"),
            right=number(materials, "table", "81" if wrong_column else "80"),
            left_cell={"row": 2, "column": col},
            right_cell={"row": 3, "column": col},
        )
    ]
    result = run(paper, items, review)
    assert result.evidence[0].sufficient is not wrong_column
    if wrong_column:
        assert any("different explicit metric header" in reason for reason in result.issues)


@pytest.mark.parametrize("axis", ["dataset", "split", "tasks"])
@pytest.mark.parametrize("wrong_column", [False, True])
def test_common_caption_cannot_override_dataset_or_setting_axis(paper, axis, wrong_column):
    claim, materials = paper
    target, other = {"dataset": ("D", "E"), "split": ("test", "dev"), "tasks": ("T1", "T2")}[axis]
    if axis == "tasks":
        claim.conditions[0].settings["tasks"] = ["T1"]
    caption = f"D test {target} and {other} results"
    table = (
        f"{caption} <table><tr><th rowspan='2'>Model</th><th>{target}</th><th>{other}</th></tr>"
        "<tr><th>accuracy</th><th>accuracy</th></tr><tr><td>A</td><td>70</td><td>91</td></tr>"
        "<tr><td>B</td><td>80</td><td>81</td></tr></table>"
    )
    append(materials, "table", table)
    items = [item(materials)]
    relation = "gt" if wrong_column else "lt"
    cases = ["T1"] if axis == "tasks" else []
    review = scope(claim, items, relation=relation, required_cases=cases)
    col = 2 if wrong_column else 1
    review["items"][0]["comparisons"] = [
        comparison(
            materials,
            case="T1" if axis == "tasks" else "",
            relation=relation,
            left=number(materials, "table", "91" if wrong_column else "70"),
            right=number(materials, "table", "81" if wrong_column else "80"),
            left_cell={"row": 2, "column": col},
            right_cell={"row": 3, "column": col},
        )
    ]
    result = run(paper, items, review)
    assert result.evidence[0].sufficient is not wrong_column
    if wrong_column:
        assert any(
            "own target dataset" in reason or "own condition setting" in reason for reason in result.issues
        )


@pytest.mark.parametrize("omit_combination", [False, True])
def test_two_list_dimensions_require_joint_coverage(paper, omit_combination):
    claim, materials = paper
    claim.conditions[0].settings.update(models=["A", "C"], tasks=["T1", "T2"])
    claim.text = "A and C accuracy exceed B for T1 and T2 on D test."
    items = [item(materials)]
    cases, comparisons = [], []
    for model in ("A", "C"):
        for task in ("T1", "T2"):
            case = model + task
            cases.append(case)
            append(materials, case + "left", f"D test {task} {model} accuracy 90.")
            append(materials, case + "right", f"D test {task} B accuracy 80.")
            comparisons.append(
                comparison(
                    materials,
                    case=case,
                    settings={"models": model, "tasks": task},
                    left_label=model,
                    left=number(materials, case + "left", "90"),
                    right=number(materials, case + "right", "80"),
                )
            )
    review = scope(claim, items, subject="", subject_setting="models", required_cases=cases)
    review["items"][0]["comparisons"] = comparisons[1:] if omit_combination else comparisons
    result = run(paper, items, review)
    assert result.evidence[0].sufficient is not omit_combination


@pytest.mark.parametrize(
    "left,right,expected,mode,metric,holds",
    [
        ("93.2", "91.7", "1.5", "absolute", "accuracy", True),
        ("91.8", "91.7", "1.3", "absolute", "accuracy", False),
        ("500 ms", "400 ms", "25%", "relative_percent", "latency", True),
        ("500 ms", "400 ms", "100%", "relative_percent", "latency", False),
        ("500 ms", "400 ms", "100%", "absolute", "latency", False),
        ("75%", "50%", "25 pp", "percentage_points", "accuracy", True),
        ("75%", "50%", "25%", "relative_percent", "accuracy", False),
        ("0.75", "0.50", "25 pp", "percentage_points", "accuracy", False),
    ],
)
def test_expected_difference_type_scale_and_arithmetic(paper, left, right, expected, mode, metric, holds):
    claim, materials = paper
    claim.text = f"A {metric} exceeds B on D test; difference {expected}."
    claim.source_quote = claim.text
    claim.source_block_id = "difference"
    claim.conditions[0].metric = metric
    append(materials, "difference", claim.text)
    append(materials, "value_a", f"D test A {metric} {left}.")
    append(materials, "value_b", f"D test B {metric} {right}.")
    items = [item(materials)]
    review = scope(claim, items, relation="difference")
    review["items"][0]["comparisons"] = [
        comparison(
            materials,
            relation="difference",
            metric_label=metric,
            left=number(materials, "value_a", left.split()[0]),
            right=number(materials, "value_b", right.split()[0]),
            difference={"block_id": "difference", "quote": expected, "token": expected.split()[0]},
            difference_mode=mode,
        )
    ]
    result = run(paper, items, review)
    assert result.evidence[0].sufficient is holds


@pytest.mark.parametrize(
    "subject,direction,quoted,holds",
    [
        ("300", "decrease", True, True),
        ("500", "decrease", True, False),
        ("300", "decrease", False, False),
        ("300", "signed", False, False),
    ],
)
def test_positive_reduction_magnitude_keeps_fixed_operands_and_grounded_direction(
    paper, subject, direction, quoted, holds
):
    claim, materials = paper
    claim.text = "A latency is 25% lower than B on D test."
    claim.source_quote, claim.source_block_id = claim.text, "decrease"
    claim.conditions[0].metric = "latency"
    append(materials, "decrease", claim.text)
    append(materials, "latency_a", f"D test A latency {subject} ms.")
    append(materials, "latency_b", "D test B latency 400 ms.")
    items = [item(materials)]
    review = scope(
        claim,
        items,
        relation="difference",
        difference_direction=direction,
        difference_direction_quote="25% lower" if quoted else "",
    )
    review["items"][0]["comparisons"] = [
        comparison(
            materials,
            relation="difference",
            metric_label="latency",
            left=number(materials, "latency_a", subject),
            right=number(materials, "latency_b", "400"),
            difference={"block_id": "decrease", "quote": "25%", "token": "25%"},
            difference_mode="relative_percent",
        )
    ]
    result = run(paper, items, review)
    assert result.evidence[0].sufficient is holds


@pytest.mark.parametrize("covered", [["c1"], ["other"]])
def test_difference_can_use_only_source_ref_covering_current_condition(paper, covered):
    claim, materials = paper
    claim.conditions.append(Condition(id="other", dataset="D", metric="accuracy"))
    append(materials, "supplement", "The A versus B accuracy difference is 10.")
    claim.source_refs = [
        ClaimSourceRef(
            source_block_id="supplement",
            source_quote=materials.blocks[-1].text,
            loc=ClaimLocation(page=1),
            covered=covered,
        )
    ]
    items = [item(materials)]
    review = scope(claim, items, relation="difference")
    review["items"][0]["comparisons"] = [
        comparison(
            materials,
            relation="difference",
            difference_mode="absolute",
            difference={"block_id": "supplement", "quote": "10", "token": "10"},
        )
    ]
    assert run(paper, items, review).evidence[0].sufficient is (covered == ["c1"])


@pytest.mark.parametrize("primary_coverage", [["c1"], ["c1", "other"]])
def test_explicit_primary_source_scope_cannot_leak_difference_to_another_condition(paper, primary_coverage):
    claim, materials = paper
    claim.text = "A accuracy exceeds B on D and E."
    claim.conditions.append(Condition(id="other", dataset="E", metric="accuracy", settings={"split": "test"}))
    append(materials, "primary", "D accuracy difference 10.")
    append(materials, "secondary", "E accuracy difference 5.")
    append(materials, "e_left", "E test A accuracy 90.")
    append(materials, "e_right", "E test B accuracy 80.")
    claim.source_block_id, claim.source_quote = "primary", "D accuracy difference 10."
    claim.source_refs = [
        ClaimSourceRef(
            source_block_id="primary",
            source_quote=claim.source_quote,
            loc=ClaimLocation(page=1),
            covered=primary_coverage,
        ),
        ClaimSourceRef(
            source_block_id="secondary",
            source_quote="E accuracy difference 5.",
            loc=ClaimLocation(page=1),
            covered=["other"],
        ),
    ]
    items = [item(materials, covered=["other"], fully_supported_conditions=["other"])]
    review = scope(claim, items, relation="difference")
    review["items"][0]["comparisons"] = [
        comparison(
            materials,
            relation="difference",
            difference_mode="absolute",
            left=number(materials, "e_left", "90"),
            right=number(materials, "e_right", "80"),
            difference={"block_id": "primary", "quote": "10", "token": "10"},
        )
    ]
    assert run(paper, items, review).evidence[0].sufficient is ("other" in primary_coverage)


@pytest.mark.parametrize(
    "invalid", ["failure", "malformed", "foreign", "duplicate", "missing_item", "ungrounded"]
)
def test_failed_or_incomplete_scope_review_cannot_create_decisive_evidence(paper, invalid):
    claim, materials = paper
    items = [item(materials), item(materials, "missing_statistic")]
    review = scope(claim, items)
    if invalid == "failure":
        review = RuntimeError("Provider failed")
    elif invalid == "malformed":
        review = {"unsupported": True}
    elif invalid == "foreign":
        review["conditions"][0]["condition_id"] = "foreign"
    elif invalid == "duplicate":
        review["items"].append(copy.deepcopy(review["items"][0]))
    elif invalid == "missing_item":
        review["items"].pop()
    else:
        review["items"][0]["grounds"][0]["quote"] = "Fabricated source"
    result = run(paper, items, review)
    assert len(result.evidence) == 2 and all(
        not row.sufficient and not row.concern for row in result.evidence
    )
    assert not result.questions and result.issues


def test_scope_audit_preserves_inputs_response_and_safe_filename(paper, tmp_path):
    claim, materials = paper
    claim.id = "../outside"
    items = [item(materials)]
    review = scope(claim, items)
    review["items"][0]["comparisons"] = [comparison(materials)]
    with run_stats.run_scope(tmp_path / "run" / "run_stats.json"):
        assert run(paper, items, review).evidence[0].sufficient
    paths = list((tmp_path / "run" / "experiment_scope").glob("*.json"))
    assert len(paths) == 1 and len(paths[0].stem) == 32
    audit = json.loads(paths[0].read_text(encoding="utf-8"))
    assert audit["claim_id"] == claim.id and audit["validated"] is True
    assert audit["input"]["candidate_items"][0]["quote"] == items[0]["quote"]
    assert audit["response"] == review


def test_sufficient_evidence_report_links_supplemental_numeric_sources_and_raw_audit(paper, tmp_path):
    from review.report.v2 import write_review
    from schemas.review import FinalReview

    claim, materials = paper
    items = [item(materials)]
    review = scope(claim, items)
    review["items"][0]["comparisons"] = [comparison(materials)]
    with run_stats.run_scope(tmp_path / "active" / "run_stats.json"):
        result = run(paper, items, review)
    claim.evidence = result.evidence
    assert claim.evidence[0].sufficient
    assert claim.evidence[0].pointer.quote == claim.text
    paths = list((tmp_path / "active" / "experiment_scope").glob("*.json"))
    assert len(paths) == 1
    audit = json.loads(paths[0].read_text(encoding="utf-8"))
    assert audit["item_locations"] == [
        {"candidate_index": 0, "condition_id": "c1", "response_pointer": "/response/items/0"}
    ]
    assert audit["response"]["items"][0]["comparisons"][0]["left"]["quote"] == "D test A accuracy 90."
    outputs = write_review(
        FinalReview(paper_key="trace", run_id="fixture", claims=[claim]),
        tmp_path / "report",
        render_pdf=False,
    )
    saved = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
    note = saved["claims"][0]["evidence"][0]["note"]
    assert f"scope_audit={paths[0]}" in note and "candidate_index=0" in note
    assert "block=left, page=1" in note and "block=right, page=1" in note
    markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
    assert paths[0].stem in markdown and "block=left, page=1" in markdown
    assert "D test A accuracy 90." not in note
