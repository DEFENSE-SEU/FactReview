"""Public numerical verification keeps explicit units on the selected measurement axis."""

from pathlib import Path

import pytest

from tests.test_experiment_joint_sources_v2 import offline as offline
from tests.test_experiment_joint_sources_v2 import paper, review_for, run
from verification.experiment_catalog import TableGrid, build_catalog
from verification.experiment_sources import prepare_joint_candidate
from verification.experiments import _own_table_reference, verify_experiments


def configured(
    tmp_path, html, *, positions=((1, 1), (2, 1)), labels=((1, 0), (2, 0)), caption="Table 1: D test results."
):
    claim, materials, item, _, _ = paper(tmp_path)
    claim.conditions[0].settings["unit"] = "percent"
    materials.blocks[0].text = caption + "\n" + html
    materials.markdown = ""
    for block in materials.blocks:
        start = len(materials.markdown)
        materials.markdown += block.text + "\n"
        block.loc.char_start, block.loc.char_end = start, start + len(block.text)
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
    item.quote = materials.blocks[0].text
    catalog = build_catalog(claim, materials)
    bounded = prepare_joint_candidate(claim, materials, catalog, item, 0)

    # Reuse the semantic review while selecting the exact requested new axes.
    def cell(position):
        return next(
            key
            for key, value in bounded["cells"].items()
            if (value["row"], value["column"]) == position and value["origin_table_index"] == 0
        )

    # review_for normally expects the original 3x2 layout, so supply a temporary
    # coordinate-only reference to build the unchanged non-selector fields.
    original_path = tmp_path / "reference"
    original_path.mkdir()
    base = paper(original_path)
    review = review_for(base[4])
    members = {m["block_id"]: m["source_id"] for m in bounded["joint_view"]["members"]}
    mapping = {m["source_id"]: members[m["block_id"]] for m in base[4]["joint_view"]["members"]}
    row = review["items"][0]
    row["grounds_source_ids"] = [mapping[key] for key in row["grounds_source_ids"]]
    for use in row["source_uses"]:
        use["source_id"] = mapping[use["source_id"]]
    comp = row["comparisons"][0]
    comp["case_id"] = bounded["conditions"]["c1"]["cases"][0]["id"]
    for side, position, label in zip(("left", "right"), positions, labels, strict=True):
        comp[side] = {"kind": "cell", "cell_id": cell(position), "label_cell_id": cell(label)}
    for bridge in comp["bridges"]:
        bridge["table_id"] = bounded["cells"][cell(positions[0])]["table_id"]
        bridge["source_ids"] = [mapping[key] for key in bridge["source_ids"]]
    return (claim, materials, item, catalog, bounded), review


@pytest.mark.parametrize("tag", ["th", "td"])
def test_explicit_dataset_percent_header_with_metric_bridge(tmp_path, tag):
    html = f"<table><tr><{tag}>Method</{tag}><{tag}>D (%)</{tag}></tr><tr><td>A + Augment</td><td>90</td></tr><tr><td>B</td><td>80</td></tr></table>"
    p, review = configured(tmp_path, html)
    result, _ = run(p, review)
    assert result.evidence[0].sufficient, result.issues


@pytest.mark.parametrize("tag", ["th", "td"])
def test_transposed_metric_row_keeps_its_own_unit(tmp_path, tag):
    html = f"<table><tr><{tag}>Metric</{tag}><{tag}>A + Augment</{tag}><{tag}>B</{tag}></tr><tr><{tag}>D accuracy (%)</{tag}><td>90</td><td>80</td></tr></table>"
    p, review = configured(tmp_path, html, positions=((1, 1), (1, 2)), labels=((0, 1), (0, 2)))
    result, _ = run(p, review)
    assert result.evidence[0].sufficient, result.issues


def test_header_spans_and_standalone_unit_stay_on_selected_column(tmp_path):
    html = '<table><thead><tr><td rowspan="2">Method</td><td colspan="2">D</td></tr><tr><td>accuracy (%)</td><td>latency (ms)</td></tr></thead><tr><td>A + Augment</td><td>90</td><td>4</td></tr><tr><td>B</td><td>80</td><td>5</td></tr></table>'
    p, review = configured(tmp_path, html, positions=((2, 1), (3, 1)), labels=((2, 0), (3, 0)))
    result, _ = run(p, review)
    assert result.evidence[0].sufficient, result.issues
    parser = TableGrid(html)
    assert parser.headers[0][1, 1] and parser.origins[0][0, 2] == (0, 1)


@pytest.mark.parametrize(
    "shape", ["other_column", "other_table", "method", "prior_data", "unknown", "conflict"]
)
def test_percent_marker_cannot_be_borrowed_or_conflict_ignored(tmp_path, shape):
    header = "<tr><th>Method</th><th>D</th></tr>"
    rows = "<tr><td>A + Augment</td><td>90</td></tr><tr><td>B</td><td>80</td></tr>"
    positions, labels = ((1, 1), (2, 1)), ((1, 0), (2, 0))
    if shape == "other_column":
        header = "<tr><th>Method</th><th>D</th><th>F1 (%)</th></tr>"
        rows = "<tr><td>A + Augment</td><td>90</td><td>99</td></tr><tr><td>B</td><td>80</td><td>98</td></tr>"
    elif shape == "method":
        rows = rows.replace("A + Augment", "A + Augment (%)").replace(">B<", ">B (%)<")
    elif shape == "prior_data":
        rows = "<tr><td>C</td><td>90%</td></tr>" + rows
        positions, labels = ((2, 1), (3, 1)), ((2, 0), (3, 0))
    elif shape == "unknown":
        header = header.replace(">D<", ">D (widgets)<")
    elif shape == "conflict":
        header = header.replace(">D<", ">D (%)<")
        rows = rows.replace(">90<", ">90%<")
        header = header.replace("D (%)", "D (ms)")
    html = "<table>" + header + rows + "</table>"
    if shape == "other_table":
        html += "\nTable 2: D accuracy (%)\n<table><tr><td>Method</td><td>D accuracy (%)</td></tr><tr><td>C</td><td>99</td></tr></table>"
    p, review = configured(tmp_path, html, positions=positions, labels=labels)
    result, _ = run(p, review)
    assert result.issues and not result.evidence[0].sufficient


@pytest.mark.parametrize(
    "quote,expected",
    [
        ("Our Table 1 reports the exact results.", True),
        ("Our Table 1 reports the exact results. We also cite Smith et al. (2020).", True),
        ("Our Table 1 from Smith et al. (2020) reports their results.", False),
        ("Our Table 1 reports results from Smith et al. (2020).", False),
        ("Smith et al. (2020), Table 1, reports their results.", False),
        ("Our Table 10 reports the exact results.", False),
    ],
)
def test_bounded_our_table_self_reference(quote, expected):
    assert _own_table_reference(quote, "Table 1") is expected


@pytest.mark.parametrize("value", ["0", "1", "invalid"])
def test_invalid_joint_budget_is_rejected_before_model_call(tmp_path, monkeypatch, value):
    p = paper(tmp_path)
    monkeypatch.setenv("EXPERIMENT_JOINT_MAX_SOURCES", value)
    with pytest.raises(ValueError):
        verify_experiments(p[0], p[1], call=lambda **kw: pytest.fail("Invalid configuration reached model"))


def test_configured_budget_is_in_first_request(tmp_path, monkeypatch):
    import json

    p = paper(tmp_path)
    monkeypatch.setenv("EXPERIMENT_JOINT_MAX_SOURCES", "4")
    result, calls = run(p)
    assert result.evidence[0].sufficient
    assert json.loads(calls[0]["prompt"])["joint_source_limit"] == 4


@pytest.mark.parametrize("tag", ["th", "td"])
def test_generic_parent_header_unit_governs_exact_selected_child(tmp_path, tag):
    html = f'<table><tr><{tag} rowspan="2">Method</{tag}><{tag} colspan="2">Performance (%)</{tag}></tr><tr><{tag}>D</{tag}><{tag}>Other</{tag}></tr><tr><td>A + Augment</td><td>90</td><td>99</td></tr><tr><td>B</td><td>80</td><td>89</td></tr></table>'
    p, review = configured(tmp_path, html, positions=((2, 1), (3, 1)), labels=((2, 0), (3, 0)))
    result, _ = run(p, review)
    assert result.evidence[0].sufficient, result.issues


@pytest.mark.parametrize(
    "header",
    [
        "D (%) (ms)",
        "D (%)/s",
        "D (% / s)",
        "D (%) per second",
        "D (%) / elapsed (%)",
        "D (%) × elapsed (%)",
        "D (%) per elapsed (%)",
    ],
)
def test_entire_unit_expression_must_be_consumed(tmp_path, header):
    html = f"<table><tr><th>Method</th><th>{header}</th></tr><tr><td>A + Augment</td><td>90</td></tr><tr><td>B</td><td>80</td></tr></table>"
    p, review = configured(tmp_path, html)
    result, _ = run(p, review)
    assert not result.evidence[0].sufficient and result.issues


def test_adjacent_synonymous_unit_annotations_are_preserved(tmp_path):
    html = "<table><tr><th>Method</th><th>D (%) (percent)</th></tr><tr><td>A + Augment</td><td>90</td></tr><tr><td>B</td><td>80</td></tr></table>"
    p, review = configured(tmp_path, html)
    result, _ = run(p, review)
    assert result.evidence[0].sufficient, result.issues


def test_group_rowspan_cannot_turn_text_data_into_unit_header(tmp_path):
    html = '<table><tr><td rowspan="4">Group</td><td>Method</td><td>D</td></tr><tr><td>Earlier method</td><td>D (%)</td></tr><tr><td>A + Augment</td><td>90</td></tr><tr><td>B</td><td>80</td></tr></table>'
    p, review = configured(tmp_path, html, positions=((2, 2), (3, 2)), labels=((2, 1), (3, 1)))
    result, _ = run(p, review)
    assert not result.evidence[0].sufficient and result.issues


def test_colspan_group_closes_after_its_leaf_headers(tmp_path):
    html = '<table><tr><td colspan="3">Results</td></tr><tr><td>Method</td><td>D</td><td>Other</td></tr><tr><td>Earlier method</td><td>D (%)</td><td>note</td></tr><tr><td>A + Augment</td><td>90</td><td>99</td></tr><tr><td>B</td><td>80</td><td>89</td></tr></table>'
    p, review = configured(tmp_path, html, positions=((3, 1), (4, 1)), labels=((3, 0), (4, 0)))
    result, _ = run(p, review)
    assert not result.evidence[0].sufficient and result.issues


@pytest.mark.parametrize("tag,scope", [("td", ""), ("th", ' scope="row"')])
def test_previous_text_data_row_does_not_become_header(tmp_path, tag, scope):
    html = f"<table><tr><td>Method</td><td>D</td></tr><tr><td>Earlier method</td><{tag}{scope}>D (%)</{tag}></tr><tr><td>A + Augment</td><td>90</td></tr><tr><td>B</td><td>80</td></tr></table>"
    p, review = configured(tmp_path, html, positions=((2, 1), (3, 1)), labels=((2, 0), (3, 0)))
    result, _ = run(p, review)
    assert not result.evidence[0].sufficient and result.issues


@pytest.mark.parametrize("tag,scope", [("td", ""), ("th", ' scope="col"')])
def test_previous_transposed_method_data_does_not_become_measurement_stub(tmp_path, tag, scope):
    html = f"<table><tr><th>Metric</th><th>Earlier method</th><th>A + Augment</th><th>B</th></tr><tr><th>D accuracy</th><{tag}{scope}>D (%)</{tag}><td>90</td><td>80</td></tr></table>"
    p, review = configured(tmp_path, html, positions=((1, 2), (1, 3)), labels=((0, 2), (0, 3)))
    result, _ = run(p, review)
    assert not result.evidence[0].sufficient and result.issues


@pytest.mark.parametrize(
    "caption,expected",
    [
        ("Table 1: D test accuracy (%).", True),
        ("Table 1: accuracy (%).", True),
        ("Table 1: Results: accuracy (%).", True),
        ("Table 1: E test accuracy (%).", False),
        ("Table 0: D test accuracy (%).\nTable 1: D test results.", False),
        ("Table 1: D test F1 (%).", False),
        ("Table 1: D test accuracy (%) (ms).", False),
        ("Table 1: D test accuracy (%)/s.", False),
        ("Table 1: D test accuracy (%) per second.", False),
    ],
)
def test_only_selected_caption_exact_quantity_can_supply_unit(tmp_path, caption, expected):
    # The split stays on its own measurement axis when the caption only names
    # the common quantity/unit; this test does not relax setting grounding.
    html = "<table><tr><th>Method</th><th>D test</th></tr><tr><td>A + Augment</td><td>90</td></tr><tr><td>B</td><td>80</td></tr></table>"
    p, review = configured(tmp_path, html, caption=caption)
    result, _ = run(p, review)
    assert result.evidence[0].sufficient is expected, result.issues


def test_selected_caption_units_remain_compatible_with_single_source_path(tmp_path):
    html = "<table><tr><th>Method</th><th>D</th></tr><tr><td>A + Augment</td><td>90</td></tr><tr><td>B</td><td>80</td></tr></table>"
    p, review = configured(tmp_path, html, caption="Table 1: D test accuracy (%).")
    bounded, original = p[4], p[3]
    mapping = {key: value["parent_cell_id"] for key, value in bounded["cells"].items()}
    mapping.update({key: value["parent_table_id"] for key, value in bounded["tables"].items()})
    for key, value in bounded["sources"].items():
        mapping[key] = next(
            k
            for k, v in original["sources"].items()
            if v.get("block_id") == value["block_id"] and v["kind"] == "paper_block"
        )

    def remap(value):
        if isinstance(value, str):
            return mapping.get(value, value)
        if isinstance(value, list):
            return [remap(v) for v in value]
        if isinstance(value, dict):
            return {k: remap(v) for k, v in value.items()}
        return value

    review["items"][0].pop("source_uses")
    review = remap(review)
    candidate = p[2].model_dump()
    candidate["additional_sources"] = []
    result, _ = run(p, review, [candidate])
    assert result.evidence[0].sufficient, result.issues
    assert not result.evidence[0].additional_pointers
