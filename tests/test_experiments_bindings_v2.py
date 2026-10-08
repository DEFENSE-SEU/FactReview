"""Production experiment support from unchanged real source text and stable catalog IDs."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from llm.client import LLMConfig
from schemas.claim import Claim, ClaimLocation, Condition
from schemas.materials import MaterialBlock, SharedMaterials
from verification.experiment_catalog import build_catalog
from verification.experiments import ExperimentsOutput, _decode_scope, verify_experiments

ASPECTS = ["correspondence", "fairness", "isolation", "stability", "consistency"]
FIXTURE = Path(__file__).parent / "fixtures" / "experiment_bindings_v2.json"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: LLMConfig("mock", "mock", None, None))
    monkeypatch.setattr("screening.checks.llm_json", lambda **kw: pytest.fail("Unmocked model call"))
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **kw: pytest.fail("Network call"))
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: pytest.fail("External process"))


def load_case(tmp_path, case_id):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    case = next(row for row in fixture["cases"] if row["id"] == case_id)
    claim = Claim.model_validate(case["claim"])
    materials = SharedMaterials(
        paper_key=case_id,
        source_pdf="fixture.pdf",
        markdown="",
        markdown_path=str(tmp_path / f"{case_id}.md"),
        content_list_path="fixture.json",
        provider="mock",
    )
    # Fixture text and its original provenance remain unchanged. Rebased test
    # offsets point at this exact local subset, which is the evidence artifact.
    for raw in case["blocks"]:
        block = MaterialBlock.model_validate(raw)
        start = len(materials.markdown)
        materials.markdown += block.text + "\n"
        block.loc = ClaimLocation(
            page=(block.loc.page if block.loc else 1), char_start=start, char_end=start + len(block.text)
        )
        materials.blocks.append(block)
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
    return claim, materials, build_catalog(claim, materials)


def source(catalog, block_id):
    return next(
        key
        for key, row in catalog["sources"].items()
        if row["kind"] == "paper_block" and row["block_id"] == block_id
    )


def cell(catalog, block_id, row, column):
    source_id = source(catalog, block_id)
    return next(
        key
        for key, value in catalog["cells"].items()
        if value["source_id"] == source_id and value["row"] == row and value["column"] == column
    )


def bridge(
    catalog,
    block_id,
    kind,
    condition_field,
    applies_to,
    blocks,
    paper_label="",
    explanation="Exact source and selected table binding.",
):
    table_id = catalog["cells"][cell(catalog, block_id, 1, 1)]["table_id"]
    return dict(
        kind=kind,
        condition_field=condition_field,
        applies_to=applies_to,
        table_id=table_id,
        source_ids=[source(catalog, b) for b in blocks],
        paper_label=paper_label,
        explanation=explanation,
    )


def comparison(catalog, condition_id, block_id, left, right, column, relation, bridges):
    return dict(
        case_id=catalog["conditions"][condition_id]["cases"][0]["id"],
        left_cell_id=cell(catalog, block_id, left, column),
        right_cell_id=cell(catalog, block_id, right, column),
        left_label_cell_id=cell(catalog, block_id, left, 0),
        right_label_cell_id=cell(catalog, block_id, right, 0),
        relation=relation,
        bridges=bridges,
    )


def response(paper, block_id, specs):
    claim, materials, catalog = paper
    candidate = dict(
        aspect="correspondence",
        kind="paper_support",
        block_id=block_id,
        quote=next(b.text for b in materials.blocks if b.id == block_id),
        covered=[c.id for c in claim.conditions],
        fully_supported_conditions=[c.id for c in claim.conditions],
        detail="Full comparison is assessed independently against selected sources.",
    )
    review = {"schema_version": "catalog-v1", "conditions": [], "items": []}
    for condition, spec in zip(claim.conditions, specs, strict=True):
        review["conditions"].append(
            dict(
                condition_id=condition.id,
                assertion="controlled_comparison",
                matched_controls_required=True,
                uncertainty_sensitive=False,
                relation=spec["comparison"]["relation"],
                setting_scopes=spec.get("setting_scopes", {}),
                rationale="Original asserted comparison with exact condition identity.",
            )
        )
        review["items"].append(
            dict(
                item_index=0,
                condition_id=condition.id,
                applicability="applicable",
                grounds_source_ids=[source(catalog, block_id)],
                rationale="Original table and explicitly linked definitions establish the values.",
                full_support=True,
                qualifiers_complete=True,
                comparison_objects="matched",
                comparisons=[spec["comparison"]],
            )
        )
    return candidate, review


def beit039(tmp_path):
    paper = load_case(tmp_path, "beit_039")
    catalog = paper[2]
    bridges = [
        bridge(catalog, "block_78", role, "settings.method", role, ["block_82"])
        for role in ("subject", "comparator")
    ]
    bridges += [bridge(catalog, "block_78", "metric", "metric", "shared", ["block_81", "block_82"], "mIoU")]
    bridges += [
        bridge(
            catalog,
            "block_78",
            "setting",
            f"settings.{key}",
            "subject",
            ["block_82"],
            "Intermediate Fine-Tuning",
        )
        for key in ("intermediate_dataset", "procedure")
    ]
    candidate, review = response(
        paper,
        "block_78",
        [
            dict(
                subject_setting="method",
                comparator_setting="method",
                setting_scopes={"intermediate_dataset": "subject", "procedure": "subject"},
                comparison=comparison(catalog, "c1", "block_78", 4, 3, 1, "gt", bridges),
            )
        ],
    )
    # The condition names only the method and treatment. Its baseline identity
    # is an explicit review assertion grounded in the existing BEIT source.
    review["conditions"][0]["comparator"] = "BEIT"
    return paper, candidate, review


def beit042(tmp_path):
    paper = load_case(tmp_path, "beit_042")
    catalog = paper[2]
    specs = []
    for condition_id, column, metric, definition in [
        ("c1", 1, "top-1 accuracy", "block_67"),
        ("c2", 2, "mIoU", "block_81"),
    ]:
        metric_bridge = bridge(
            catalog,
            "block_84",
            "metric",
            "metric",
            "shared",
            [definition, "block_88"],
            metric,
            explanation="The complete mIoU / ADE20K score name identifies the mIoU measure defined on ADE20K; ImageNet uses top-1 accuracy.",
        )
        specs.append(
            dict(
                subject_setting="ablation",
                comparator_setting="baseline",
                comparison=comparison(catalog, condition_id, "block_84", 3, 1, column, "lt", [metric_bridge]),
            )
        )
    candidate, review = response(paper, "block_84", specs)
    return paper, candidate, review


def evidence_for(result, condition_id):
    return next(e for e in result.evidence if condition_id in e.covered)


def legacy_review(paper, candidate, review):
    """Keep the older strict response contract's endpoint regressions exercised."""
    output = ExperimentsOutput(checked_aspects=ASPECTS, items=[candidate])
    conditions, items, errors, _ = _decode_scope(review, paper[0], paper[1], output, paper[2])
    assert not errors
    return {
        "conditions": [row.model_dump() for row in conditions.values()],
        "items": [row.model_dump() for row in items.values()],
    }


def run(paper, candidate, review):
    calls = []

    def model(**kwargs):
        calls.append(kwargs["module"])
        if kwargs["module"] == "verification.experiments":
            return dict(checked_aspects=ASPECTS, items=[candidate], plans=[])
        assert kwargs["module"] == "verification.experiments.scope"
        return review

    result = verify_experiments(paper[0], paper[1], call=model)
    assert calls == ["verification.experiments", "verification.experiments.scope"]
    return result


@pytest.mark.parametrize("builder,count", [(beit039, 1), (beit042, 2)])
def test_real_beit_table_and_definition_bindings_support_complete_claim(tmp_path, builder, count):
    paper, candidate, review = builder(tmp_path)
    result = run(paper, candidate, review)
    assert not result.issues
    assert sum(len(e.covered) for e in result.evidence) == count
    assert all(e.sufficient for e in result.evidence), [e.note for e in result.evidence]
    assert result.plans == []


def test_subject_only_procedure_does_not_apply_to_baseline(tmp_path):
    paper, candidate, review = beit039(tmp_path)
    result = run(paper, candidate, review)
    assert evidence_for(result, "c1").sufficient
    scoped = [b for b in review["items"][0]["comparisons"][0]["bridges"] if b["kind"] == "setting"]
    assert all(b["applies_to"] == "subject" for b in scoped)
    for binding in scoped:
        binding["applies_to"] = "comparator"
    assert not evidence_for(run(paper, candidate, review), "c1").sufficient


def test_subject_treatment_cannot_bind_to_plain_beit_vs_dino(tmp_path):
    paper, candidate, review = beit039(tmp_path)
    row = review["items"][0]["comparisons"][0]
    catalog = paper[2]
    row.update(
        left_cell_id=cell(catalog, "block_78", 3, 1),
        left_label_cell_id=cell(catalog, "block_78", 3, 0),
        right_cell_id=cell(catalog, "block_78", 2, 1),
        right_label_cell_id=cell(catalog, "block_78", 2, 0),
    )
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c1").sufficient
    assert any("treatment" in issue.lower() or "alternative table row" in issue for issue in result.issues)


def test_role_cooccurrence_cannot_alias_supervised_to_beit(tmp_path):
    paper, candidate, review = beit039(tmp_path)
    row = review["items"][0]["comparisons"][0]
    row.update(
        right_cell_id=cell(paper[2], "block_78", 1, 1), right_label_cell_id=cell(paper[2], "block_78", 1, 0)
    )
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c1").sufficient
    assert any("alternative table row" in issue for issue in result.issues)


def test_metric_bridge_cannot_override_explicit_other_dataset_column(tmp_path):
    paper, candidate, review = beit042(tmp_path)
    row = review["items"][1]["comparisons"][0]
    row.update(left_cell_id=cell(paper[2], "block_84", 3, 1), right_cell_id=cell(paper[2], "block_84", 1, 1))
    result = run(paper, candidate, review)
    assert evidence_for(result, "c1").sufficient
    assert not evidence_for(result, "c2").sufficient


def test_bridge_requires_actual_selected_table(tmp_path):
    paper, candidate, review = beit039(tmp_path)
    review["items"][0]["comparisons"][0]["bridges"][2]["table_id"] = "foreign-table"
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c1").sufficient
    assert any("another table" in issue for issue in result.issues)


def test_complete_composite_metric_alias_must_be_explained(tmp_path):
    paper, candidate, review = beit042(tmp_path)
    review["items"][1]["comparisons"][0]["bridges"][0]["explanation"] = "Use the usual metric."
    result = run(paper, candidate, review)
    assert evidence_for(result, "c1").sufficient
    assert not evidence_for(result, "c2").sufficient
    assert any("complete name" in issue for issue in result.issues)


@pytest.mark.parametrize("tamper", ["omit", "wrong", "swap"])
def test_expected_endpoints_cannot_be_self_selected_from_other_table_rows(tmp_path, tamper):
    paper, candidate, review = beit042(tmp_path)
    review = legacy_review(paper, candidate, review)
    row = review["items"][0]["comparisons"][0]
    if tamper == "swap":
        row["expected_left"]["token"] = "82.86"
        row["expected_right"]["token"] = "81.04"
    else:
        # Even with the exact role intact, an incorrect asserted endpoint cannot
        # replace the 81.04 -> 82.86 pair merely because it is in the source table.
        if tamper == "wrong":
            row["expected_left"]["token"] = "82.77"
        if tamper == "omit":
            row["expected_left"] = None
            paper[0].conditions[0].description = "Masked-pixel recovery: 80.50 vs baseline 82.86."
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c1").sufficient
    assert evidence_for(result, "c2").sufficient


def test_existing_ablation_axis_cannot_be_rebound_to_another_variant(tmp_path):
    paper, candidate, review = beit042(tmp_path)
    row = review["items"][0]["comparisons"][0]
    row.update(
        left_cell_id=cell(paper[2], "block_84", 2, 1),
        left_label_cell_id=cell(paper[2], "block_84", 2, 0),
    )
    row["bridges"].append(
        bridge(paper[2], "block_84", "subject", "settings.ablation", "subject", ["block_84"])
    )
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c1").sufficient
    assert any("alternative table row" in issue for issue in result.issues)


def test_conflicting_condition_description_cannot_replace_claims_explicit_endpoints(tmp_path):
    paper, candidate, review = beit042(tmp_path)
    claim, materials, _ = paper
    claim.conditions[0].description = "Combined ablation: 80.50 vs baseline 82.86."
    claim.conditions[0].settings["ablation"] = "- Visual tokens - Blockwise masking"
    catalog = build_catalog(claim, materials)
    paper = claim, materials, catalog
    # The wrong condition description and selected row agree with each other.
    # The full claim still explicitly asserts 81.04 and must also be honored.
    row = review["items"][0]["comparisons"][0]
    row.update(
        left_cell_id=cell(catalog, "block_84", 4, 1),
        left_label_cell_id=cell(catalog, "block_84", 4, 0),
        case_id=catalog["conditions"]["c1"]["cases"][0]["id"],
    )
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c1").sufficient
    assert evidence_for(result, "c2").sufficient
    assert any("conflicting endpoints" in issue for issue in result.issues)


def test_unbound_expected_tokens_cannot_be_supported_only_by_a_whole_source_table(tmp_path):
    paper, candidate, review = beit042(tmp_path)
    review = legacy_review(paper, candidate, review)
    claim = paper[0]
    claim.text = "The paper compares masked-pixel recovery with BEIT on ImageNet and ADE20K."
    claim.conditions[0].description = "A comparison is reported without an asserted endpoint pair."
    for condition in review["conditions"]:
        condition["claim_quote"] = claim.text
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c1").sufficient
    assert any("no uniquely bound asserted endpoint pair" in issue for issue in result.issues)


def test_invalid_source_for_one_condition_preserves_other_checked_condition(tmp_path):
    paper, candidate, review = beit042(tmp_path)
    review["items"][1]["grounds_source_ids"] = ["unknown-source"]
    result = run(paper, candidate, review)
    assert evidence_for(result, "c1").sufficient
    assert not evidence_for(result, "c2").sufficient
    assert result.issues


def test_real_bert_top_comparator_gap_stays_insufficient(tmp_path):
    paper = load_case(tmp_path, "bert_038")
    claim, _, catalog = paper
    specs = []
    for condition in claim.conditions:
        specs.append(
            dict(
                subject_setting="model",
                comparator_setting="comparison",
                comparison=comparison(catalog, condition.id, "block_82", 13, 4, 4, "gt", []),
            )
        )
    candidate, review = response(paper, "block_82", specs)
    review["items"][1].update(
        full_support=False,
        comparison_objects="unmatched",
        unresolved_qualifiers=["The +1.3 F1 gap is against the second-ranked ensemble; the top gap is +0.1."],
    )
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c2").sufficient
    assert "second-ranked" in evidence_for(result, "c2").note


def test_unresolved_variance_note_does_not_change_descriptive_report_coverage(tmp_path):
    paper, candidate, review = beit042(tmp_path)
    review["items"][1]["nonblocking_notes"] = [
        "No confidence interval reported; this observation only reports the original endpoints."
    ]
    result = run(paper, candidate, review)
    assert evidence_for(result, "c2").sufficient
    assert "No confidence interval" in evidence_for(result, "c2").note


@pytest.mark.parametrize("dataset,supported", [("SQuAD 1.1", False), ("SQuAD 2.0", True)])
def test_real_mixed_caption_prefix_only_binds_the_selected_tables_caption(tmp_path, dataset, supported):
    _, materials, _ = load_case(tmp_path, "bert_038")
    text = f"BERTLARGE(Single) has higher Test F1 than #1 Single - MIR-MRC (F-Net) on {dataset}."
    start = len(materials.markdown)
    materials.markdown += text
    loc = ClaimLocation(page=5, char_start=start, char_end=start + len(text))
    materials.blocks.append(MaterialBlock(id="caption_claim", text=text, loc=loc))
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
    claim = Claim(
        id="caption-claim",
        text=text,
        loc=loc,
        source_block_id="caption_claim",
        source_quote=text,
        needs=["Experiments"],
        conditions=[
            Condition(
                id="c1",
                dataset=dataset,
                metric="F1",
                settings={
                    "model": "BERTLARGE(Single)",
                    "comparison": "#1 Single - MIR-MRC (F-Net)",
                    "split": "Test",
                },
            )
        ],
    )
    catalog = build_catalog(claim, materials)
    paper = claim, materials, catalog
    candidate, review = response(
        paper,
        "block_83",
        [
            dict(
                subject_setting="model",
                comparator_setting="comparison",
                comparison=comparison(catalog, "c1", "block_83", 10, 4, 4, "gt", []),
            )
        ],
    )
    result = run(paper, candidate, review)
    assert evidence_for(result, "c1").sufficient is supported
    if not supported:
        assert any("own target dataset" in issue for issue in result.issues)


@pytest.mark.parametrize(
    "headers,issue",
    [
        ("<td>ADE20K mIoU</td><td>ADE20K F1</td>", "different explicit metric header"),
        ("<td>ADE20K mIoU (%)</td><td>ADE20K mIoU (fraction)</td>", "Comparison units/scales do not match"),
    ],
)
def test_explicit_metric_and_units_cannot_be_overridden_by_definition_bridge(tmp_path, headers, issue):
    paper, candidate, review = beit042(tmp_path)
    # Preserve original data and add an explicit metric axis. Both columns now
    # identify ADE20K, so the metric guard itself must reject the F1 selection.
    claim, materials, _ = paper
    table = next(b for b in materials.blocks if b.id == "block_84")
    table.text = table.text.replace("<td>ImageNet</td><td>ADE20K</td>", headers)
    assert headers in table.text
    claim.source_quote = table.text
    for ref in claim.source_refs:
        if ref.source_block_id == table.id:
            ref.source_quote = table.text
    materials.markdown = "\n".join(b.text for b in materials.blocks)
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
    catalog = build_catalog(claim, materials)
    paper = claim, materials, catalog
    candidate["quote"] = table.text
    for index, condition in enumerate(claim.conditions):
        review["items"][index]["grounds_source_ids"] = [source(catalog, "block_84")]
        old = review["items"][index]["comparisons"][0]
        b = old["bridges"][0]
        b["table_id"] = catalog["cells"][cell(catalog, "block_84", 1, 1)]["table_id"]
        old.update(
            left_cell_id=cell(catalog, "block_84", 3, index + 1),
            right_cell_id=cell(catalog, "block_84", 1, index + 1),
            left_label_cell_id=cell(catalog, "block_84", 3, 0),
            right_label_cell_id=cell(catalog, "block_84", 1, 0),
        )
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c2").sufficient
    assert any(issue in value for value in result.issues)


def test_catalog_response_audit_and_report_resolve_exact_cells_and_sources(tmp_path):
    from common import run_stats
    from review.report.v2 import write_review
    from schemas.review import FinalReview

    paper, candidate, review = beit042(tmp_path)
    with run_stats.run_scope(tmp_path / "run" / "run_stats.json"):
        result = run(paper, candidate, review)
    claim = paper[0]
    claim.evidence = result.evidence
    paths = list((tmp_path / "run" / "experiment_scope").glob("*.json"))
    assert len(paths) == 1
    audit = json.loads(paths[0].read_text(encoding="utf-8"))
    assert audit["validated"] is True and audit["response"] == review
    assert audit["resolved_items"][1]["comparisons"][0]["left"]["token"] == "41.38"
    assert audit["resolved_items"][1]["comparisons"][0]["left_cell"] == {"table": 0, "row": 3, "column": 2}
    assert audit["resolved_items"][1]["comparisons"][0]["expected_right"]["token"] == "44.65"
    assert {p["block_id"] for p in audit["resolved_items"][1]["comparisons"][0]["context"]} >= {
        "block_81",
        "block_84",
        "block_88",
    }
    outputs = write_review(
        FinalReview(paper_key="bindings", run_id="mock", claims=[claim]),
        tmp_path / "report",
        render_pdf=False,
    )
    saved = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
    note = saved["claims"][0]["evidence"][0]["note"]
    assert str(paths[0]) in note and "block=block_81" in note and "block=block_88" in note
    assert paths[0].stem in Path(outputs["markdown"]).read_text(encoding="utf-8")


def append_block(materials, block_id, text):
    start = len(materials.markdown)
    materials.markdown += "\n" + text
    block = MaterialBlock(
        id=block_id,
        text=text,
        loc=ClaimLocation(page=1, char_start=start + 1, char_end=start + 1 + len(text)),
    )
    materials.blocks.append(block)
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
    return block


def prose_case(tmp_path, left="D test A accuracy 90.", right="D test B accuracy 80."):
    materials = SharedMaterials(
        paper_key="prose",
        source_pdf="fixture.pdf",
        markdown="",
        markdown_path=str(tmp_path / "prose.md"),
        content_list_path="fixture.json",
        provider="mock",
    )
    assertion = append_block(materials, "claim", "A accuracy is higher than B on D test.")
    append_block(materials, "left", left)
    append_block(materials, "right", right)
    claim = Claim(
        id="prose",
        text=assertion.text,
        loc=assertion.loc,
        source_block_id="claim",
        source_quote=assertion.text,
        needs=["Experiments"],
        conditions=[
            Condition(
                id="c1",
                dataset="D",
                metric="accuracy",
                settings={"model": "A", "comparison": "B", "split": "test"},
            )
        ],
    )
    catalog = build_catalog(claim, materials)
    paper = claim, materials, catalog
    comp = dict(
        case_id=catalog["conditions"]["c1"]["cases"][0]["id"],
        left_source_id=source(catalog, "left"),
        left_token="90",
        left_value_context=left,
        right_source_id=source(catalog, "right"),
        right_token="80",
        right_value_context=right,
        relation="gt",
    )
    candidate, review = response(paper, "claim", [dict(comparison=comp)])
    return paper, candidate, review


def test_production_schema_offers_prose_selectors_and_derives_known_roles_and_endpoints(tmp_path):
    paper, candidate, review = prose_case(tmp_path)
    seen = []

    def model(**kwargs):
        if kwargs["module"] == "verification.experiments":
            return dict(checked_aspects=ASPECTS, items=[candidate], plans=[])
        payload = json.loads(kwargs["prompt"])
        seen.append(payload)
        schemas = payload["output_schema"]["$defs"]
        comparison_fields = schemas["CatalogComparison"]["properties"]
        assert {
            "left_source_id",
            "right_source_id",
            "left_token",
            "right_token",
            "left_value_context",
            "right_value_context",
        } <= comparison_fields.keys()
        assert not {"expected_left_token", "expected_right_token"} & comparison_fields.keys()
        assert (
            not {"endpoint_required", "subject_setting", "comparator_setting"}
            & schemas["CatalogConditionScope"]["properties"].keys()
        )
        return review

    result = verify_experiments(paper[0], paper[1], call=model)
    assert len(seen) == 1
    assert evidence_for(result, "c1").sufficient
    assert not result.issues


@pytest.mark.parametrize(
    "field,where,value",
    [
        ("expected_left_token", "comparison", "90"),
        ("expected_right_token", "comparison", "80"),
        ("endpoint_required", "condition", False),
        ("subject_setting", "condition", "model"),
        ("comparator_setting", "condition", "comparison"),
    ],
)
def test_removed_catalog_fields_are_rejected_instead_of_silently_ignored(tmp_path, field, where, value):
    paper, candidate, review = prose_case(tmp_path)
    target = review["items"][0]["comparisons"][0] if where == "comparison" else review["conditions"][0]
    target[field] = value
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c1").sufficient
    assert any(field in issue and "Extra inputs" in issue for issue in result.issues)


@pytest.mark.parametrize(
    "left,right,change",
    [
        ("D test B accuracy 90.", "D test A accuracy 80.", None),
        ("D test A accuracy 90.", "D dev B accuracy 80.", None),
        ("D test A accuracy 90.", "E test B accuracy 80.", None),
        ("D test A accuracy 90.", "D test B F1 80.", None),
        ("D test A accuracy 90 and 85.", "D test B accuracy 80.", None),
        ("D test A accuracy 90%.", "D test B accuracy 80.", {"left_token": "90%"}),
        ("D test A accuracy 90.", "D test B accuracy 90.", {"right_token": "90"}),
        ("D test A accuracy 90.", "D test B accuracy 80.", {"left_value_context": "D test A accuracy 91."}),
        (
            "D test A accuracy 90.",
            "D test B accuracy 80.",
            {"left_cell_id": "fake", "left_label_cell_id": "fake-label"},
        ),
    ],
)
def test_prose_selectors_preserve_role_metric_dataset_setting_unit_and_value_guards(
    tmp_path, left, right, change
):
    paper, candidate, review = prose_case(tmp_path, left, right)
    review["items"][0]["comparisons"][0].update(change or {})
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c1").sufficient
    assert result.issues


def test_prose_selector_cannot_flatten_html_table_and_bypass_cell_axis_checks(tmp_path):
    paper, candidate, review = beit042(tmp_path)
    comparison = review["items"][0]["comparisons"][0]
    for key in ("left_cell_id", "left_label_cell_id"):
        comparison.pop(key)
    comparison.update(
        left_source_id=source(paper[2], "block_84"), left_token="81.04", left_value_context="81.04"
    )
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c1").sufficient
    assert evidence_for(result, "c2").sufficient
    assert any("cannot replace HTML table coordinates" in issue for issue in result.issues)


def test_exact_named_role_fields_need_no_redundant_external_setup_definition(tmp_path):
    paper, candidate, review = beit042(tmp_path)
    for row in review["conditions"]:
        row["setting_scopes"] = {"ablation": "subject", "baseline": "comparator"}
    for row in review["items"]:
        row["comparisons"][0]["bridges"].append(
            bridge(
                paper[2],
                "block_84",
                "setting",
                "settings.baseline",
                "comparator",
                ["block_84"],
                "BEIT (300 Epochs)",
            )
        )
    result = run(paper, candidate, review)
    assert all(e.sufficient for e in result.evidence)
    assert not result.issues


def test_literal_method_axis_binding_is_derived_without_model_role_bridges(tmp_path):
    paper, candidate, review = beit039(tmp_path)
    comparison = review["items"][0]["comparisons"][0]
    comparison["bridges"] = [b for b in comparison["bridges"] if b["kind"] not in {"subject", "comparator"}]
    result = run(paper, candidate, review)
    assert evidence_for(result, "c1").sufficient
    assert not result.issues


@pytest.mark.parametrize(
    "reference,supported",
    [
        ("Table 4 reports the results of the ablations. The baseline follows Smith et al. (2020).", True),
        (
            "As shown in Table 4, the ablated models perform worse. Smith et al. (2020) provided the baseline.",
            True,
        ),
        ("Smith et al. (2020), Table 4, reports their model ablations.", False),
        ("Table 4 of Smith et al. (2020) shows their model ablations.", False),
        ("Table 4 reports results from Smith et al. (2020).", False),
        ("The discussion mentions Table 4 somewhere among related results.", False),
    ],
)
def test_table_reference_candidates_require_a_self_reference_sentence(tmp_path, reference, supported):
    paper, candidate, review = beit042(tmp_path)
    append_block(paper[1], "reference", reference)
    catalog = build_catalog(paper[0], paper[1])
    for row in review["items"]:
        metric = row["comparisons"][0]["bridges"][0]
        metric["source_ids"] = [
            value for value in metric["source_ids"] if value != source(catalog, "block_88")
        ]
        metric["source_ids"].append(source(catalog, "reference"))
    result = run(paper, candidate, review)
    assert all(e.sufficient is supported for e in result.evidence)
    if not supported:
        assert any("self-reference" in issue for issue in result.issues)


def test_dataset_axis_is_not_an_alias_for_the_measured_quantity(tmp_path):
    paper, candidate, review = beit042(tmp_path)
    review["items"][0]["comparisons"][0]["bridges"][0]["paper_label"] = "ImageNet"
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c1").sufficient
    assert evidence_for(result, "c2").sufficient
    assert any("does not identify the original quantity" in issue for issue in result.issues)


@pytest.mark.parametrize("with_setup", [False, True])
def test_missing_comparator_identity_cannot_default_to_the_subject_in_prose(tmp_path, with_setup):
    paper, candidate, review = prose_case(tmp_path, right="D test A accuracy 80.")
    paper[0].conditions[0].settings = {"method": "A", "split": "test"}
    if with_setup:
        paper[0].conditions[0].settings["procedure"] = "train X"
        review["conditions"][0]["setting_scopes"] = {"procedure": "subject"}
    catalog = build_catalog(paper[0], paper[1])
    review["items"][0]["comparisons"][0]["case_id"] = catalog["conditions"]["c1"]["cases"][0]["id"]
    result = run(paper, candidate, review)
    assert not evidence_for(result, "c1").sufficient
    assert result.issues


@pytest.mark.parametrize("collection", ["conditions", "items"])
@pytest.mark.parametrize("malformed", [False, True])
@pytest.mark.parametrize("duplicate_first", [False, True])
@pytest.mark.parametrize("third_row", [False, True])
def test_spaced_duplicate_scope_identity_invalidates_only_its_canonical_condition(
    tmp_path, collection, malformed, duplicate_first, third_row
):
    paper, candidate, review = beit042(tmp_path)
    canonical = copy.deepcopy(review[collection][0])
    duplicate = copy.deepcopy(canonical)
    duplicate["condition_id"] = " c1 "
    if malformed:
        duplicate["rationale"] = ""
    c1_rows = [duplicate, canonical] if duplicate_first else [canonical, duplicate]
    if third_row:
        # A later otherwise valid row, including its normalized ID alias, must
        # never restore a condition already invalidated by duplication.
        later = copy.deepcopy(canonical)
        later["condition_id"] = " c1 "
        c1_rows.append(later)
    review[collection] = [*c1_rows, review[collection][1]]
    result = run(paper, candidate, review)
    assert evidence_for(result, "c2").sufficient
    assert not evidence_for(result, "c1").sufficient
    assert result.issues
