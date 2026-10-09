"""Exact-source catalog choices avoid model transcription and table-index errors."""

import copy
import json
from pathlib import Path

import pytest

from schemas.claim import Claim, ClaimLocation, ClaimSourceRef, Condition
from schemas.materials import MaterialBlock, SharedMaterials
from verification.experiment_catalog import (
    TableGrid,
    build_catalog,
    catalog_prompt,
    resolve_case,
    resolve_cell,
    resolve_source,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Catalog construction/resolution must not call an external service or process")

    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("screening.checks.ask", forbidden)


@pytest.fixture
def paper():
    text = "Intermediate tuning improves A on D."
    table = (
        "Table 3: D results.\n"
        "<table><tr><th>Models</th><th>D</th></tr>"
        "<tr><td>Supervised</td><td>45.3</td></tr>"
        "<tr><td>Other</td><td>44.1</td></tr>"
        "<tr><td>A (ours)</td><td>45.6</td></tr>"
        "<tr><td>A + Intermediate Tuning (ours)</td><td>47.7</td></tr></table>"
    )
    claim = Claim(
        id="claim",
        text=text,
        loc=ClaimLocation(page=1),
        source_block_id="statement",
        source_quote=text,
        conditions=[Condition(id="c1", dataset="D", metric="mIoU", settings={"method": "A"})],
        needs=["Experiments"],
    )
    materials = SharedMaterials(
        paper_key="catalog-test",
        source_pdf="paper.pdf",
        markdown=text + "\n" + table,
        markdown_path="paper.md",
        content_list_path="content.json",
        provider="mock",
        blocks=[
            MaterialBlock(id="statement", text=text, loc=ClaimLocation(page=1)),
            MaterialBlock(id="table", text=table, loc=ClaimLocation(page=2)),
        ],
    )
    return claim, materials


def choice(catalog, token, *, row=None, column=None, table=0):
    matches = [
        key
        for key, item in catalog["cells"].items()
        if item["token"] == token
        and item["table"] == table
        and (row is None or item["row"] == row)
        and (column is None or item["column"] == column)
    ]
    assert len(matches) == 1
    return matches[0]


def test_table_choices_resolve_real_coordinates_and_exact_source_without_copied_html(paper):
    claim, materials = paper
    catalog = build_catalog(claim, materials)
    left = resolve_cell(catalog, choice(catalog, "47.7"), materials)
    right = resolve_cell(catalog, choice(catalog, "45.6"), materials)
    assert (left["row"], right["row"], left["column"]) == (4, 3, 1)
    assert left["quote"] == right["quote"] == materials.blocks[1].text
    assert left["block_id"] == "table" and left["table"] == 0
    assert left["row_labels"] == "A + Intermediate Tuning (ours)"
    assert left["column_labels"] == "D"
    assert left["caption"] == "Table 3: D results.\n"
    assert not left["caption_ambiguous"]
    assert "<table" not in json.dumps(catalog)
    assert all("quote" not in item for item in catalog["cells"].values())


def test_label_cell_choices_preserve_literal_names_and_entities(paper):
    claim, materials = paper
    materials.blocks[1].text = materials.blocks[1].text.replace("A (ours)", "A &amp; B (ours)")
    catalog = build_catalog(claim, materials)
    label = resolve_cell(catalog, choice(catalog, "A & B (ours)"), materials)
    assert label["cell_type"] == "label" and label["row"] == 3 and label["column"] == 0
    assert "A &amp; B (ours)" in label["quote"]


def test_numeric_cells_do_not_guess_mixed_values_or_decode_absent_numeric_tokens(paper):
    claim, materials = paper
    materials.blocks[1].text = (
        "<table><tr><td>82.1 ± 0.3</td><td>nan</td><td>inf</td>"
        "<td>9&#49;.2</td><td>1e999</td><td>-25%</td></tr></table>"
    )
    catalog = build_catalog(claim, materials)
    numbers = [cell["token"] for cell in catalog["cells"].values() if cell["cell_type"] == "number"]
    assert numbers == ["-25%"]
    assert resolve_cell(catalog, choice(catalog, "-25%"), materials)["token"] == "-25%"


def test_expanded_rowspan_colspan_and_transposed_axes_keep_exact_coordinates(paper):
    claim, materials = paper
    materials.blocks[1].text = (
        "<table><tr><th rowspan='2'>Model</th><th colspan='2'>Test</th></tr>"
        "<tr><th>F1</th><th>EM</th></tr>"
        "<tr><td>A</td><td>90</td><td>80</td></tr></table>"
    )
    catalog = build_catalog(claim, materials)
    cell = resolve_cell(catalog, choice(catalog, "90"), materials)
    assert (cell["row"], cell["column"]) == (2, 1)
    assert cell["row_labels"] == "A" and cell["column_labels"] == "Test | F1"
    assert choice(catalog, "Model", row=0, column=0) != choice(catalog, "Model", row=1, column=0)
    assert choice(catalog, "Test", row=0, column=1) != choice(catalog, "Test", row=0, column=2)
    assert TableGrid(materials.blocks[1].text).tables[0][1, 0] == "Model"
    materials.blocks[1].text = (
        "<table><tr><th>Metric</th><th>A</th><th>B</th></tr>"
        "<tr><th>F1</th><td>90</td><td>80</td></tr></table>"
    )
    catalog = build_catalog(claim, materials)
    cell = resolve_cell(catalog, choice(catalog, "90"), materials)
    assert cell["row_labels"] == "F1" and cell["column_labels"] == "A"


def test_multiple_tables_and_repeated_values_have_distinct_ids(paper):
    claim, materials = paper
    materials.blocks[1].text = (
        "Table 1: first.\n<table><tr><td>A</td><td>10</td><td>10</td></tr></table>\n"
        "Table 2: second.\n<table><tr><td>A</td><td>10</td></tr></table>"
    )
    catalog = build_catalog(claim, materials)
    ids = [
        choice(catalog, "10", table=0, column=1),
        choice(catalog, "10", table=0, column=2),
        choice(catalog, "10", table=1),
    ]
    assert len(set(ids)) == 3
    second = resolve_cell(catalog, ids[-1], materials)
    assert second["table"] == 1 and second["caption"] == "\nTable 2: second.\n"
    assert second["quote"] == materials.blocks[1].text


def test_mixed_caption_prefix_is_explicitly_ambiguous_with_exact_line_choices(paper):
    claim, materials = paper
    materials.blocks[1].text = "Table 2: Different task.\n" + materials.blocks[1].text
    catalog = build_catalog(claim, materials)
    cell = resolve_cell(catalog, choice(catalog, "47.7"), materials)
    assert cell["caption_ambiguous"]
    assert "Table 2" in cell["caption"] and "Table 3" in cell["caption"]
    lines = [
        resolve_source(catalog, key, materials)["quote"]
        for key, src in catalog["sources"].items()
        if src["kind"] == "paper_line"
    ]
    assert lines == ["Table 2: Different task.", "Table 3: D results."]


def test_table_reference_candidates_are_exact_located_mentions_in_other_blocks(paper):
    claim, materials = paper
    originals = {
        "matching": "Table 3 reports results. We discuss Table 3 again.",
        "wrong_number": "Table 30 reports different results.",
        "wrong_suffix": "Table 3a and Table 3.1 report different results.",
        "case_and_space": "See table  3 for model comparisons.",
    }
    materials.blocks.extend(
        MaterialBlock(id=key, text=value, loc=ClaimLocation(page=3)) for key, value in originals.items()
    )
    materials.blocks.append(MaterialBlock(id="unlocated", text="Table 3 reports results."))
    catalog = build_catalog(claim, materials)
    table = next(iter(catalog["tables"].values()))
    resolved = [resolve_source(catalog, key, materials) for key in table["reference_source_ids"]]
    assert {source["block_id"] for source in resolved} == {"matching", "case_and_space"}
    assert len(resolved) == 2
    assert all(source["quote"] == originals[source["block_id"]] for source in resolved)
    assert all(source["kind"] == "paper_block" and not source["covered"] for source in resolved)
    materials.blocks.reverse()
    assert build_catalog(claim, materials) == catalog


def test_mixed_caption_references_follow_only_the_selected_tables_last_caption(paper):
    claim, materials = paper
    materials.blocks[1].text = "Table 2: Different task.\n" + materials.blocks[1].text
    for number in (2, 3):
        materials.blocks.append(
            MaterialBlock(id=f"reference_{number}", text=f"Table {number} reports results.", loc=claim.loc)
        )
    catalog = build_catalog(claim, materials)
    table = next(iter(catalog["tables"].values()))
    assert table["caption_ambiguous"]
    assert [resolve_source(catalog, key, materials)["block_id"] for key in table["reference_source_ids"]] == [
        "reference_3"
    ]


@pytest.mark.parametrize("caption", ["", "Results in Table 3 are listed below.\n", "Table 3.1: Results.\n"])
def test_table_reference_index_does_not_invent_a_caption_number(paper, caption):
    claim, materials = paper
    materials.blocks[1].text = caption + materials.blocks[1].text.split("\n", 1)[1]
    materials.blocks.append(MaterialBlock(id="reference", text="See Table 3.", loc=claim.loc))
    catalog = build_catalog(claim, materials)
    assert next(iter(catalog["tables"].values()))["reference_source_ids"] == []


def test_cited_external_table_is_only_a_candidate_with_no_automatic_coverage(paper):
    claim, materials = paper
    external = MaterialBlock(
        id="external", text="Smith et al. (2020), Table 3, reports their results.", loc=ClaimLocation(page=3)
    )
    materials.blocks.append(external)
    catalog = build_catalog(claim, materials)
    table_id, table = next(iter(catalog["tables"].items()))
    (source_id,) = table["reference_source_ids"]
    resolved = resolve_source(catalog, source_id, materials)
    assert resolved["quote"] == external.text and resolved["covered"] == []
    assert "sufficient" not in table and "bridges" not in table
    payload = catalog_prompt(catalog)
    assert payload["tables"][table_id]["reference_source_ids"] == [source_id]
    assert external.text not in json.dumps(payload)
    external.text = external.text.replace("Smith", "Jones")
    with pytest.raises(ValueError, match="source changed"):
        resolve_source(catalog, source_id, materials)


def test_real_beit_table_four_indexes_original_external_reference_block_88():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "experiment_bindings_v2.json").read_text(encoding="utf-8")
    )
    case = next(row for row in fixture["cases"] if row["id"] == "beit_042")
    claim = Claim.model_validate(case["claim"])
    materials = SharedMaterials(
        paper_key="beit_042",
        source_pdf="fixture.pdf",
        markdown="\n".join(block["text"] for block in case["blocks"]),
        markdown_path="fixture.md",
        content_list_path="fixture.json",
        provider="mock",
        blocks=[MaterialBlock.model_validate(block) for block in case["blocks"]],
    )
    catalog = build_catalog(claim, materials)
    table = next(
        table
        for table in catalog["tables"].values()
        if catalog["sources"][table["source_id"]]["block_id"] == "block_84"
    )
    resolved = [resolve_source(catalog, key, materials) for key in table["reference_source_ids"]]
    block_88 = next(block for block in case["blocks"] if block["id"] == "block_88")
    assert any(
        source["block_id"] == "block_88" and source["quote"] == block_88["text"] for source in resolved
    )
    assert all(source["block_id"] != "block_84" for source in resolved)
    # Frozen v1-of-this-fixture predates optional report-stage advice. Its
    # original fields remain byte-for-value identical after catalog building.
    assert "advice" not in case["claim"]
    assert claim.model_dump(mode="json") == {**case["claim"], "advice": None}


@pytest.mark.parametrize("change", ["text", "page", "block_id", "paper_key", "source_pdf"])
def test_resolvers_reject_changed_original_materials(paper, change):
    claim, materials = paper
    catalog = build_catalog(claim, materials)
    key = choice(catalog, "47.7")
    if change == "text":
        materials.blocks[1].text = materials.blocks[1].text.replace("47.7", "48.7")
    elif change == "page":
        materials.blocks[1].loc.page = 3
    elif change == "block_id":
        materials.blocks[1].id = "replacement"
    else:
        setattr(materials, change, "different")
    with pytest.raises(ValueError, match=r"changed|different|unavailable"):
        resolve_cell(catalog, key, materials)


def test_unknown_ids_and_ids_from_different_source_snapshot_fail(paper):
    claim, materials = paper
    catalog = build_catalog(claim, materials)
    old_id = choice(catalog, "47.7")
    with pytest.raises(ValueError, match="Unknown catalog source"):
        resolve_source(catalog, "unknown", materials)
    with pytest.raises(ValueError, match="Unknown catalog cell"):
        resolve_cell(catalog, "unknown", materials)
    materials.blocks[1].text += " changed"
    rebuilt = build_catalog(claim, materials)
    with pytest.raises(ValueError, match="Unknown catalog cell"):
        resolve_cell(rebuilt, old_id, materials)


@pytest.mark.parametrize("field,value", [("row", 5), ("token", "45.6"), ("source_id", "unknown")])
def test_cell_record_cannot_change_selected_number_or_coordinates(paper, field, value):
    claim, materials = paper
    catalog = build_catalog(claim, materials)
    key = choice(catalog, "47.7")
    catalog["cells"][key][field] = value
    with pytest.raises(ValueError, match=r"changed|Unknown"):
        resolve_cell(catalog, key, materials)


def test_source_spans_and_stable_ids_survive_block_order_and_return_independent_values(paper):
    claim, materials = paper
    catalog = build_catalog(claim, materials)
    materials.blocks.reverse()
    assert catalog == build_catalog(claim, materials)
    src = next(key for key, value in catalog["sources"].items() if value["kind"] == "claim_source")
    resolved = resolve_source(catalog, src, materials)
    assert resolved["quote"] == claim.source_quote
    resolved["covered"].append("foreign")
    assert resolve_source(catalog, src, materials)["covered"] == ["c1"]
    catalog["sources"][src]["end"] -= 1
    with pytest.raises(ValueError, match="source record changed"):
        resolve_source(catalog, src, materials)


def test_identical_text_in_different_blocks_does_not_collide(paper):
    claim, materials = paper
    block = materials.blocks[1].model_copy(deep=True)
    block.id = "another_table"
    materials.blocks.append(block)
    catalog = build_catalog(claim, materials)
    ids = [key for key, cell in catalog["cells"].items() if cell["token"] == "47.7"]
    assert len(ids) == len(set(ids)) == 2
    assert {resolve_cell(catalog, key, materials)["block_id"] for key in ids} == {"table", "another_table"}


def test_colliding_source_ids_raise_instead_of_overwriting(paper, monkeypatch):
    claim, materials = paper
    monkeypatch.setattr(
        "verification.experiment_catalog._stable_id", lambda kind, *parts: kind + "_collision"
    )
    with pytest.raises(ValueError, match="ID collision"):
        build_catalog(claim, materials)


def test_scalar_and_list_condition_cases_use_only_allowed_inputs(paper):
    claim, materials = paper
    scalar = build_catalog(claim, materials)["conditions"]["c1"]
    assert scalar["field_values"] == {"dataset": "D", "metric": "mIoU", "settings": {"method": "A"}}
    assert len(scalar["cases"]) == 1 and scalar["cases"][0]["settings"] == {}
    claim.conditions[0].settings.update(tasks=["T1", "T2"], seeds=[1, 2])
    catalog = build_catalog(claim, materials)
    cases = catalog["conditions"]["c1"]["cases"]
    assert len(cases) == 4 and len({case["id"] for case in cases}) == 4
    assert {tuple(case["settings"].items()) for case in cases} == {
        (("tasks", task), ("seeds", seed)) for task in ("T1", "T2") for seed in ("1", "2")
    }
    assert resolve_case(catalog, "c1", cases[0]["id"]) == cases[0]
    with pytest.raises(ValueError, match="Unknown catalog condition"):
        resolve_case(catalog, "foreign", cases[0]["id"])
    with pytest.raises(ValueError, match="Unknown case ID"):
        resolve_case(catalog, "c1", "unknown")
    assert claim.conditions[0].settings["seeds"] == [1, 2]


@pytest.mark.parametrize(
    "values,limit", [(["T1", "T1"], 256), ([], 256), ([{"x": 1}], 256), (["T1", "T2"], 1)]
)
def test_ambiguous_or_excessive_cases_are_unavailable_with_visible_issue(paper, values, limit):
    claim, materials = paper
    claim.conditions[0].settings["tasks"] = values
    catalog = build_catalog(claim, materials, max_cases=limit)
    assert not catalog["conditions"]["c1"]["cases"]
    assert catalog["issues"]


def test_primary_source_scope_remains_explicit_and_paper_blocks_do_not_claim_coverage(paper):
    claim, materials = paper
    claim.conditions.append(Condition(id="c2", dataset="E", metric="mIoU"))
    claim.source_refs = [
        ClaimSourceRef(
            source_block_id="statement", source_quote=claim.source_quote, loc=claim.loc, covered=["c1"]
        )
    ]
    catalog = build_catalog(claim, materials)
    primary = [item for item in catalog["sources"].values() if item["kind"] == "claim_source"]
    assert len(primary) == 1 and primary[0]["covered"] == ["c1"]
    assert all(not item["covered"] for item in catalog["sources"].values() if item["kind"] == "paper_block")


@pytest.mark.parametrize("source", ["unknown", "duplicate_quote"])
def test_invalid_claim_source_is_not_guessed(paper, source):
    claim, materials = paper
    if source == "unknown":
        claim.source_block_id = "unknown"
    else:
        materials.blocks[0].text += claim.source_quote
    with pytest.raises(ValueError, match="one exact original"):
        build_catalog(claim, materials)


@pytest.mark.parametrize(
    "table",
    [
        "<table><tr><td>1</td></tr>",
        "<table><tr><td><table><tr><td>1</td></tr></table></td></tr></table>",
        "<table><tr><td rowspan='0'>1</td></tr></table>",
        "<table><tr><td>1</td><td rowspan='2'>2</td></tr><tr><td colspan='2'>3</td></tr></table>",
    ],
)
def test_malformed_tables_produce_no_numeric_choices_and_visible_issue(paper, table):
    claim, materials = paper
    materials.blocks[1].text = table
    catalog = build_catalog(claim, materials)
    assert not catalog["cells"] and catalog["issues"]


def test_duplicate_material_ids_are_rejected(paper):
    claim, materials = paper
    materials.blocks.append(copy.deepcopy(materials.blocks[1]))
    with pytest.raises(ValueError, match="unique material block"):
        build_catalog(claim, materials)


def test_prompt_projection_is_compact_and_resolves_using_untouched_local_catalog(paper):
    claim, materials = paper
    catalog = build_catalog(claim, materials)
    payload = catalog_prompt(catalog)
    assert "block_sha256" not in json.dumps(payload) and "<table" not in json.dumps(payload)
    assert len(json.dumps(payload)) < len(json.dumps(catalog)) * 0.7
    key = choice(catalog, "47.7")
    fields = dict(zip(payload["cell_fields"], payload["cells"][key], strict=True))
    assert fields["token"] == "47.7" and fields["row"] == 4
    assert payload["axes"][fields["row_axis_id"]] == "A + Intermediate Tuning (ours)"
    payload["cells"][key][3] = "changed"
    assert resolve_cell(catalog, key, materials)["token"] == "47.7"


def test_case_hash_collision_is_visible_and_does_not_lose_a_required_case(paper, monkeypatch):
    from verification import experiment_catalog

    original = experiment_catalog._stable_id
    monkeypatch.setattr(
        experiment_catalog,
        "_stable_id",
        lambda kind, *parts: "case_collision" if kind == "case" else original(kind, *parts),
    )
    claim, materials = paper
    claim.conditions[0].settings["tasks"] = ["T1", "T2"]
    catalog = build_catalog(claim, materials)
    assert not catalog["conditions"]["c1"]["cases"]
    assert any("colliding" in issue for issue in catalog["issues"])


def test_prompt_axis_sharing_has_a_measured_size_bound(paper, tmp_path):
    claim, materials = paper
    materials.blocks[1].text = (
        "<table><tr><th>Model</th>"
        + "".join(f"<th>Task-{column} accuracy</th>" for column in range(12))
        + "</tr>"
        + "".join(
            f"<tr><th>Model-{row} (shared evaluation protocol)</th>"
            + "".join(f"<td>{row + column}.5</td>" for column in range(12))
            + "</tr>"
            for row in range(20)
        )
        + "</table>"
    )
    catalog = build_catalog(claim, materials)
    payload = catalog_prompt(catalog)

    def size(value):
        return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

    counts = {
        "cells": len(catalog["cells"]),
        "shared_axes": len(payload["axes"]),
        "local_bytes": size(catalog),
        "prompt_bytes": size(payload),
    }
    (tmp_path / "catalog_sizes.json").write_text(json.dumps(counts), encoding="utf-8")
    assert counts["cells"] == 273 and counts["shared_axes"] < 60
    assert counts["prompt_bytes"] < counts["local_bytes"] * 0.6
    assert counts["prompt_bytes"] < 50000


def test_numeric_occurrences_share_sentence_text_without_losing_choices(paper):
    claim, materials = paper
    text = "On the test split of dataset D, method A has 90% accuracy and method B has 90% accuracy."
    materials.blocks[0].text = text
    claim.text = claim.source_quote = text
    catalog = build_catalog(claim, materials)
    payload = catalog_prompt(catalog)
    assert len(payload["numbers"]) == len(catalog["numbers"]) == 2
    assert len(payload["sentences"]) == 1
    choices = [dict(zip(payload["number_fields"], row, strict=True)) for row in payload["numbers"].values()]
    assert [row["ordinal"] for row in choices] == [1, 2]
    assert all(payload["sentences"][row["sentence_id"]] == text for row in choices)
    assert choices[0]["token"] == choices[1]["token"] == "90%"
    assert not {"start", "end", "sentence_start", "sentence_end"} & set(payload["number_fields"])
    assert all("start" in record and "block_sha256" in record for record in catalog["numbers"].values())
