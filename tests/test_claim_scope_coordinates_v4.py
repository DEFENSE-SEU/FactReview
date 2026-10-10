"""Necessary source-coordinate/carrier controls; no model/service/Docker calls."""
import copy
import json

import pytest

from schemas.materials import MaterialBlock
from screening import claim_coverage as m
from screening.claim_scope import ScopeSpan
from tests.test_claim_coverage_v4 import preserved_row


def directory(budget):
    prefix, text, tail = "Heading\n", "A🙂中e\u0301", "tail"
    markdown = prefix + text + "\n" + tail
    blocks = {
        "u": MaterialBlock(id="u", text=text, loc={"page": 1, "char_start": len(prefix), "char_end": len(prefix + text)}),
        "t": MaterialBlock(id="t", text=tail, loc={"page": 1, "char_start": len(prefix + text + "\n"), "char_end": len(markdown)}),
    }
    registry = [{"claim_id": "claim", "digest": "mock", "conditions": [{"id": "c1"}],
                 "source_block_id": "u", "source_refs": [{"source_block_id": "t", "covered": ["c1"]}]}]
    context, extras = m._scope_context(registry, [{"claim_id": "claim"}], ["u"], blocks, markdown, budget)
    return context[0], extras, blocks


def test_visible_directory_coordinates_are_python_codepoints_and_global_span_stays_global():
    context, extras, blocks = directory(100)
    for row in context["sources"]:
        assert row["block_local_whole_span"] == {"start": 0, "end": len(blocks[row["block_id"]].text)}
    row = next(r for r in context["sources"] if r["block_id"] == "u")
    assert row["block_local_whole_span"] == {"start": 0, "end": 5}
    assert row["trusted_span"] == [8, 13]
    assert row["availability"] == "in_window" and extras[0]["block"]["id"] == "t"
    schema = ScopeSpan.model_json_schema()["properties"]
    assert "block-relative" in schema["start"]["description"]
    assert "exclusive" in schema["end"]["description"]


def test_unavailable_directory_source_never_gets_canonical_visible_span():
    context, extras, _blocks = directory(0)
    row = next(r for r in context["sources"] if r["block_id"] == "t")
    assert row["availability"] == "unavailable" and row["unavailable_reason"] == "budget"
    assert row["block_local_whole_span"] is None and extras == []


def test_visible_span_map_covers_actual_bodies_only_with_ordered_deduplication():
    context, extras, blocks = directory(100)
    supplied = [b.model_dump(mode="json") for b in blocks.values()]
    payload = {"blocks": supplied, "supplemental_sources": [*extras, {"block": supplied[0]}],
               "scope_context": [context], "unloaded_block_ids": ["unloaded"]}
    before = copy.deepcopy(payload)
    assert m._visible_source_spans(payload) == [
        {"block_id": "u", "block_local_whole_span": {"start": 0, "end": 5}},
        {"block_id": "t", "block_local_whole_span": {"start": 0, "end": 4}},
    ]
    assert payload == before


@pytest.mark.parametrize("select_item", [False, True])
def test_list_carrier_is_rejected_with_scalar_diagnostic_and_actual_item_is_accepted(select_item):
    row, claim, context, blocks = preserved_row()
    values = ["configuration", "four predictions"]
    claim["conditions"][0]["settings"] = {"distributed_artifacts": values}
    row["preserved_qualifiers"][0].update(
        claim_path="/conditions/0/settings/distributed_artifacts" + ("/0" if select_item else ""),
        claim_value=values[0] if select_item else values,
    )
    checked = m.CurrentClaimReviewV4.model_validate(row)
    if select_item:
        m._check_scope(checked, claim, context, blocks)
        assert checked.preserved_qualifiers[0].claim_value == "configuration"
    else:
        with pytest.raises(ValueError, match=r"nonempty scalar.*containers forbidden"):
            m._check_scope(checked, claim, context, blocks)


def test_source_and_carrier_instructions_name_canonical_offsets_and_item_leaf():
    assert "block_local_whole_span" in m._SCOPE_SYSTEM_V4
    assert "trusted_span" in m._SCOPE_SYSTEM_V4 and "Markdown-global" in m._SCOPE_SYSTEM_V4
    description = m.PreservedQualifier.model_json_schema()["properties"]["claim_path"]["description"]
    assert "nonempty scalar" in description and "array item" in description


def test_native_first_and_third_bind_visible_body_metadata_in_context_without_extra_call(monkeypatch, tmp_path):
    from tests.test_claim_coverage_v2 import CFG, review, validation
    from tests.test_claim_coverage_v3 import case

    paper, claims = case()
    before = paper.model_dump(mode="json")
    original_claims = [c.model_dump(mode="json") for c in claims]
    monkeypatch.setattr(m, "resolve_llm_config", lambda: CFG)
    monkeypatch.setattr(m, "llm_json", lambda **_: pytest.fail("Unexpected external model call"))
    modules = []

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"].split("DATA_JSON:\n", 1)[1])
        modules.append(kwargs["module"])
        assert payload["context_id"] == m._digest({k: v for k, v in payload.items() if k != "context_id"})
        assert payload["visible_source_spans"] == m._visible_source_spans(payload)
        for context in payload["scope_context"]:
            for source in context["sources"]:
                assert "block_local_whole_span" in source
        if kwargs["module"] == "screening.claims.coverage":
            return review(payload, [])
        assert kwargs["module"] == "screening.claims.coverage_validation"
        return validation(payload)

    result = m.review_claim_coverage(paper, claims, call=call, output_dir=tmp_path,
                                    max_review_calls=1, max_followup_calls=0, max_validation_calls=1)
    assert modules == ["screening.claims.coverage", "screening.claims.coverage_validation"]
    assert result.coverage["status"] == "complete"
    assert paper.model_dump(mode="json") == before
    assert [c.model_dump(mode="json") for c in result.claims] == original_claims
