"""Distinct page identities and credential-safe audit keys survive path aliases."""

import copy
import json

import pytest

from common import run_stats
from schemas.materials import FigureMaterial
from screening.checks import check_writing
from screening.visual_audit import redacted_record, redaction_scope
from tests.test_writing_sections_v2 import candidate, confirmations, make_materials
from tests.test_writing_source_privacy_v2 import offline as offline


@pytest.mark.parametrize("mutate", [False, True])
def test_shared_image_path_still_guards_each_target_page(tmp_path, mutate):
    materials = make_materials(
        tmp_path, [("a", "Figure 1 demonstrates the procedure."), ("b", "The procedure is clear.")]
    )
    materials.pages[1].path = materials.pages[0].path
    materials.figures.append(
        FigureMaterial(id="f", anchor="1", caption="Figure 1: Procedure.", loc=materials.blocks[1].loc)
    )
    calls, records, target_pages = [], [], []

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"])
        calls.append(kwargs["module"])
        if kwargs["module"] == "screening_writing":
            block = payload["blocks"][0]
            return {
                "findings": [
                    candidate(
                        block,
                        category="cross_reference",
                        target_kind="figure",
                        target_label="1",
                        reference_problem="inconsistent_target",
                    )
                ]
                if block["id"] == "b1"
                else []
            }
        target_pages.extend(payload["additional_target_pages"])
        assert len(kwargs["images"]) == 1
        if mutate:
            materials.pages[1].dpi += 1
        return confirmations(payload, kind="cross_reference")

    findings = check_writing(materials, call=call, records=records, recover_errors=True)
    assert target_pages == [2]
    assert len(findings) == (0 if mutate else 1)
    assert records[0].status == ("failed" if mutate else "checked")
    assert records[1].status == "checked"
    assert calls == ["screening_writing", "screening_writing.validation", "screening_writing"]


@pytest.mark.parametrize("stage", ["first", "validation"])
def test_raw_audit_nested_keys_are_redacted_without_mutating_response(tmp_path, offline, stage):
    text, vision = offline
    materials = make_materials(tmp_path, [("a", "These results is fixed.")])
    originals, returned, records = [], [], []

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"])
        first = kwargs["module"] == "screening_writing"
        result = {"findings": [candidate(payload["blocks"][0])]} if first else confirmations(payload)
        if first == (stage == "first"):
            # The first-pass envelope preserves extras; the strict visual schema rejects them.
            result["audit_extra"] = {"nested": [{text.api_key: "text-value", vision.api_key: "vision-value"}]}
        originals.append(copy.deepcopy(result))
        returned.append(result)
        return result

    with run_stats.run_scope(tmp_path / "run_stats.json"):
        findings = check_writing(materials, call=call, records=records, recover_errors=True)
    assert originals == returned
    assert len(findings) == (1 if stage == "first" else 0)
    assert records[0].status == ("checked" if stage == "first" else "failed")
    audits = [p for folder in ("writing_calls", "visual_calls") for p in (tmp_path / folder).glob("*.json")]
    serialized = "\n".join(p.read_text("utf-8") for p in audits)
    assert text.api_key not in serialized and vision.api_key not in serialized
    assert "text-value" in serialized and "vision-value" in serialized
    assert "dictionary keys collided after credential redaction" in serialized
    assert len(returned) == 2


def test_audit_key_collision_preserves_all_values_and_scope_restores(offline):
    text, vision = offline
    original = {"nested": ({text.api_key: "one", vision.api_key: "two", "[redacted]": "three"},)}
    before = copy.deepcopy(original)
    with redaction_scope((text, vision)):
        safe = redacted_record(original, text)
    assert original == before
    collision = safe["nested"][0]
    assert collision["_audit_redaction"] == "dictionary keys collided after credential redaction"
    assert collision["entries"] == [
        {"key": "[redacted]", "value": "one"},
        {"key": "[redacted]", "value": "two"},
        {"key": "[redacted]", "value": "three"},
    ]
    assert text.api_key not in json.dumps(safe) and vision.api_key not in json.dumps(safe)
    # Scope reset leaves only the explicit text config active.
    assert vision.api_key in redacted_record({vision.api_key: "visible"}, text)
