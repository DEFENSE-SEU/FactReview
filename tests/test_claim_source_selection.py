"""Whole-block repair selectors preserve strict original-source contracts offline."""

import copy
import json

import pytest

from common import run_stats
from llm.client import LLMConfig
from schemas.claim import ClaimLocation
from schemas.materials import MaterialBlock, SharedMaterials
from screening import claims as module


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(module, "resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None))
    monkeypatch.setattr(module, "llm_json", lambda **_: pytest.fail("Unmocked LLM"))


def paper(*texts):
    markdown = "\n\n".join(texts)
    blocks, offset = [], 0
    for i, text in enumerate(texts, 1):
        blocks.append(
            MaterialBlock(
                id=f"b{i}",
                text=text,
                loc=ClaimLocation(page=i, char_start=offset, char_end=offset + len(text)),
            )
        )
        offset += len(text) + 2
    return SharedMaterials(
        paper_key="selectors",
        source_pdf="paper.pdf",
        markdown=markdown,
        markdown_path="paper.md",
        content_list_path="",
        provider="fixture",
        blocks=blocks,
    )


def candidate(quote="Missing", text="A reported result.", refs=None):
    return {
        "text": text,
        "source_block_id": "b1",
        "source_quote": quote,
        "source_refs": refs or [],
        "conditions": [{"id": "c1", "description": "As reported"}],
        "needs": ["Experiments"],
        "importance": "core",
    }


def selected(context, index=1, source_id="default", refs=None):
    if source_id == "default":
        source_id = context["source_choices"][0]["source_id"]
    return {"index": index, "source_id": source_id, "source_ref_ids": refs}


def response(*rows):
    return {"schema_version": "source-block-v1", "repairs": list(rows)}


def model_for(raw, repair):
    def call(**request):
        if request["module"] == "screening.claims":
            return copy.deepcopy(raw)
        return repair(json.loads(request["prompt"]))

    return call


@pytest.mark.parametrize(
    "text",
    [
        "In\nterms of performance.",
        "In\r\nterms of performance.",
        "Preﬁx-tuning works.",
        "<table><tr><td>0.75</td></tr></table>",
        r"For $x \\in X$, $f(x)=1$.",
        "Repeated. Repeated.",
    ],
)
def test_explicit_id_restores_exact_whole_block_with_no_semantic_promotion(text, tmp_path):
    materials = paper(text)
    raw = {"status": "ok", "claims": [candidate()]}

    def repair(ctx):
        assert ctx["output_schema"]["properties"]["schema_version"]["const"] == "source-block-v1"
        assert "source_quote" not in ctx["output_schema"]["$defs"]["SourceBlockRepair"]["properties"]
        return response(selected(ctx))

    with run_stats.run_scope(tmp_path / "run_stats.json"):
        result = module.extract_claims(materials, call=model_for(raw, repair), max_source_repairs=3)
    assert len(result) == 1 and result[0].source_quote == text
    assert materials.markdown[result[0].loc.char_start : result[0].loc.char_end] == text
    assert result[0].text == raw["claims"][0]["text"] and result[0].needs == ["Experiments"]
    assert result[0].status == "unverified" and not result[0].evidence
    audit = json.loads(next((tmp_path / "claim_extraction").glob("*.json")).read_text(encoding="utf-8"))
    attempt = audit["attempts"][1]
    assert attempt["repair_format"] == "source-block-v1"
    assert attempt["source_catalog"][0]["granularity"] == "whole_block"
    assert attempt["resolved_bindings"][0]["source_quote"] == text
    assert attempt["response"]["repairs"][0]["source_id"] == attempt["source_catalog"][0]["source_id"]
    assert raw["claims"][0]["source_quote"] == "Missing"


def test_same_id_can_repair_two_candidates_while_valid_neighbor_stays_exact():
    materials = paper("In\nterms of performance. A valid phrase.")
    raw = {
        "status": "ok",
        "claims": [candidate(text="One."), candidate(text="Two."), candidate("A valid phrase.", "Neighbor.")],
    }
    original = copy.deepcopy(raw)
    result = module.extract_claims(
        materials, call=model_for(raw, lambda c: response(selected(c), selected(c, 2))), max_source_repairs=3
    )
    assert len(result) == 3
    for claim, old in zip(result, raw["claims"], strict=True):
        for key in ("text", "conditions", "needs", "importance"):
            if key == "conditions":
                assert claim.conditions[0].description == old[key][0]["description"]
            else:
                assert getattr(claim, key) == old[key]
    assert result[0].source_quote == result[1].source_quote == materials.blocks[0].text
    assert result[2].source_quote == "A valid phrase."
    assert raw == original


def test_null_keeps_valid_primary_and_ref_slots_and_repair_retains_coverage():
    materials = paper("Narrow primary. More context.", "Old narrow. More text.", "Correct\nreference.")
    refs = [
        {"source_block_id": "b2", "source_quote": "Old narrow.", "covered": ["c1"]},
        {"source_block_id": "b3", "source_quote": "Correct reference.", "covered": ["c1"]},
    ]
    raw = {"status": "ok", "claims": [candidate("Narrow primary.", refs=refs)]}

    def repair(ctx):
        selector = next(x["source_id"] for x in ctx["source_choices"] if x["source_block_id"] == "b3")
        return response(selected(ctx, source_id=None, refs=[None, selector]))

    result = module.extract_claims(materials, call=model_for(raw, repair), max_source_repairs=3)[0]
    assert result.source_quote == "Narrow primary."
    assert result.source_refs[0].source_quote == "Old narrow."
    assert [r.source_block_id for r in result.source_refs] == ["b2", "b3"]
    assert [r.covered for r in result.source_refs] == [["c1"], ["c1"]]
    assert result.source_refs[1].source_quote == materials.blocks[2].text


def test_unresolved_null_preserves_error_and_uses_only_three_repair_rounds(tmp_path):
    materials = paper("Actual source.")
    raw = {"status": "ok", "claims": [candidate()]}
    calls = []

    def repair(ctx):
        calls.append(1)
        return response(selected(ctx, source_id=None))

    with (
        run_stats.run_scope(tmp_path / "run_stats.json"),
        pytest.raises(module.ClaimExtractionError, match="does not occur"),
    ):
        module.extract_claims(materials, call=model_for(raw, repair), max_source_repairs=3)
    assert len(calls) == 3
    audit = json.loads(next((tmp_path / "claim_extraction").glob("*.json")).read_text(encoding="utf-8"))
    assert audit["status"] == "failed" and len(audit["attempts"]) == 4
    assert all(a["source_errors"] for a in audit["attempts"])


@pytest.mark.parametrize("index", [True, False, "1", 1.0, 0, 2])
def test_selector_indices_are_strict_and_must_equal_invalid_candidates(index):
    with pytest.raises(module.ClaimExtractionError):
        module.extract_claims(
            paper("Actual."),
            call=model_for({"status": "ok", "claims": [candidate()]}, lambda c: response(selected(c, index))),
            max_source_repairs=1,
        )


@pytest.mark.parametrize(
    "change",
    [
        "unknown",
        "padded",
        "version",
        "mixed",
        "missing_version",
        "missing_field",
        "duplicate",
        "missing_candidate",
    ],
)
def test_malformed_selection_cannot_fall_back_to_legacy(change):
    def repair(ctx):
        row = selected(ctx)
        raw = response(row)
        if change == "unknown":
            row["source_id"] = "unknown"
        elif change == "padded":
            row["source_id"] = " " + row["source_id"] + " "
        elif change == "version":
            raw["schema_version"] = "source-block-future"
        elif change == "mixed":
            row.update(source_block_id="b1", source_quote="Actual.")
        elif change == "missing_version":
            raw.pop("schema_version")
        elif change == "missing_field":
            row.pop("source_ref_ids")
        elif change == "duplicate":
            raw["repairs"].append(copy.deepcopy(row))
        elif change == "missing_candidate":
            raw["repairs"] = []
        return raw

    with pytest.raises(module.ClaimExtractionError):
        module.extract_claims(
            paper("Actual."),
            call=model_for({"status": "ok", "claims": [candidate()]}, repair),
            max_source_repairs=1,
        )


@pytest.mark.parametrize("mode", ["text", "loc", "markdown", "replace", "replace_equal"])
def test_material_mutation_after_catalog_is_not_rebound_to_a_stale_id(mode):
    materials = paper("Actual.")

    def repair(ctx):
        result = response(selected(ctx))
        if mode == "text":
            materials.blocks[0].text = "Changed"
        elif mode == "loc":
            materials.blocks[0].loc.page = 9
        elif mode == "markdown":
            materials.markdown = "Changed"
        elif mode == "replace":
            materials.blocks[0] = materials.blocks[0].model_copy(update={"text": "Changed"})
        else:
            materials.blocks[0] = materials.blocks[0].model_copy(deep=True)
        return result

    with pytest.raises(module.ClaimExtractionError, match="changed"):
        module.extract_claims(
            materials, call=model_for({"status": "ok", "claims": [candidate()]}, repair), max_source_repairs=1
        )


@pytest.mark.parametrize("text", [" Leading", "Trailing\n", "  "])
def test_outer_whitespace_is_unavailable_without_trimming(text):
    def repair(ctx):
        assert ctx["source_choices"] == []
        assert ctx["unavailable_sources"]
        return response(selected(ctx, source_id=None))

    with pytest.raises(module.ClaimExtractionError):
        module.extract_claims(
            paper("Actual.", text),
            call=model_for({"status": "ok", "claims": [{**candidate(), "source_block_id": "b2"}]}, repair),
            max_source_repairs=1,
        )


def test_mixed_good_bad_selection_is_atomic_for_original_candidates(monkeypatch):
    seen = []
    original_ground = module._ground_claims

    def observe(output, *args):
        seen.append(output)
        return original_ground(output, *args)

    monkeypatch.setattr(module, "_ground_claims", observe)
    raw = {"status": "ok", "claims": [candidate(text="One."), candidate(text="Two.")]}
    with pytest.raises(module.ClaimExtractionError):
        module.extract_claims(
            paper("Actual."),
            call=model_for(raw, lambda c: response(selected(c), selected(c, 2, source_id="unknown"))),
            max_source_repairs=1,
        )
    assert len(seen) == 1
    assert [x.source_quote for x in seen[0].claims] == ["Missing", "Missing"]


def test_duplicate_refs_after_whole_block_resolution_still_fail():
    refs = [
        {"source_block_id": "b1", "source_quote": "Bad one", "covered": ["c1"]},
        {"source_block_id": "b1", "source_quote": "Bad two", "covered": ["c1"]},
    ]
    raw = {"status": "ok", "claims": [candidate(refs=refs)]}

    def repair(ctx):
        sid = ctx["source_choices"][0]["source_id"]
        return response(selected(ctx, refs=[sid, sid]))

    with pytest.raises(module.ClaimExtractionError, match="duplicated"):
        module.extract_claims(paper("Actual."), call=model_for(raw, repair), max_source_repairs=1)


@pytest.mark.parametrize("refs", [[], [None, None]])
def test_ref_slot_count_cannot_change(refs):
    raw = {
        "status": "ok",
        "claims": [
            candidate(refs=[{"source_block_id": "b1", "source_quote": "Missing ref", "covered": ["c1"]}])
        ],
    }
    with pytest.raises(module.ClaimExtractionError, match="reference"):
        module.extract_claims(
            paper("Actual."),
            call=model_for(raw, lambda c: response(selected(c, refs=refs))),
            max_source_repairs=1,
        )


def test_legacy_non_verbatim_quote_stays_invalid_despite_one_obvious_block():
    raw = {"status": "ok", "claims": [candidate("In terms.")]}

    def repair(_):
        return {"repairs": [{"index": 1, "source_block_id": "b1", "source_quote": "In terms."}]}

    with pytest.raises(module.ClaimExtractionError, match="does not occur"):
        module.extract_claims(paper("In\nterms."), call=model_for(raw, repair), max_source_repairs=3)


def test_id_from_another_repair_context_cannot_select_an_unsupplied_block():
    materials = paper("First original block.", "Second original block.")
    saved_ids = []

    def collect(ctx):
        assert [block["id"] for block in ctx["blocks"]] == ["b2"]
        saved_ids.append(ctx["source_choices"][0]["source_id"])
        return response(selected(ctx))

    second = {**candidate(), "source_block_id": "b2"}
    module.extract_claims(
        materials, call=model_for({"status": "ok", "claims": [second]}, collect), max_source_repairs=1
    )

    def misuse(ctx):
        assert [block["id"] for block in ctx["blocks"]] == ["b1"]
        assert saved_ids[0] not in [item["source_id"] for item in ctx["source_choices"]]
        return response(selected(ctx, source_id=saved_ids[0]))

    with pytest.raises(module.ClaimExtractionError, match="outside the provided repair context"):
        module.extract_claims(
            materials,
            call=model_for({"status": "ok", "claims": [candidate()]}, misuse),
            max_source_repairs=1,
        )


@pytest.mark.parametrize("location", [None, ClaimLocation(page=2, char_start=0, char_end=2)])
def test_unlocated_or_inconsistent_full_block_cannot_be_selected(location):
    materials = paper("Valid neighbor.", "Second original block.")
    materials.blocks[1].loc = location

    def repair(ctx):
        assert ctx["source_choices"] == []
        assert [item["source_block_id"] for item in ctx["unavailable_sources"]] == ["b2"]
        return response(selected(ctx, source_id=None))

    with pytest.raises(module.ClaimExtractionError):
        module.extract_claims(
            materials,
            call=model_for({"status": "ok", "claims": [{**candidate(), "source_block_id": "b2"}]}, repair),
            max_source_repairs=1,
        )


def test_selector_cannot_supply_replacement_coverage():
    raw = {"status": "ok", "claims": [candidate()]}

    def repair(ctx):
        row = selected(ctx)
        row["covered"] = ["different-condition"]
        return response(row)

    with pytest.raises(module.ClaimExtractionError, match="covered"):
        module.extract_claims(paper("Actual."), call=model_for(raw, repair), max_source_repairs=1)
