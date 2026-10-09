"""Freeze notation pixels before the first model response, without requiring unused pages."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from llm.client import LLMConfig
from schemas.materials import PageImage
from screening import checks
from tests.test_theory_derivations_v2 import paper
from verification.theory import verify_theory


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*a, **kw):
        raise AssertionError("Only mocked model/image boundaries are permitted")

    cfg = LLMConfig("mock", "notation-source-fixture", None, None)
    monkeypatch.setattr(checks, "resolve_llm_config", lambda: cfg)
    monkeypatch.setattr(checks, "resolve_vlm_config", lambda **kw: cfg)
    monkeypatch.setattr(checks, "llm_json", blocked)
    monkeypatch.setattr("httpx.Client.send", blocked)
    monkeypatch.setattr("httpx.AsyncClient.send", blocked)
    monkeypatch.setattr("requests.sessions.Session.request", blocked)
    monkeypatch.setattr("subprocess.run", blocked)


def context(tmp_path, *, notation=True):
    claim, materials, response = paper(tmp_path)
    image = tmp_path / "original.png"
    alternate = tmp_path / "alternate.png"
    Image.new("RGB", (32, 32), "white").save(image)
    Image.new("RGB", (32, 32), "black").save(alternate)
    materials.pages = [PageImage(page=1, path=str(image), width_points=32, height_points=32)]
    if notation:
        response["items"][0].update(kind="notation", direction="flaw")
    return claim, materials, response, image, alternate


def outside_scope(kwargs):
    """The original valid real identity has no demonstrated notation contradiction."""
    payload = json.loads(kwargs["prompt"])
    return {
        "schema_version": "theory-concern-v1",
        "items": [
            {
                **pair,
                "disposition": "outside_scope",
                "target_sources": [{"block_id": "b1", "quote": "For real x and y"}],
                "trace_step_ids": ["s2"],
                "trace_gap_indices": [],
                "scope_reason": "The source and the trace establish the same real identity; the visual label supplies no contrary mathematical step.",
                "resolution": "The printed observation does not undermine the stated real identity.",
            }
            for pair in payload["allowed_pairs"]
        ],
    }


@pytest.mark.parametrize("phase", ["verification.theory", "verification.theory.notation"])
@pytest.mark.parametrize("mutation", ["bytes", "path", "replacement", "metadata", "duplicate"])
def test_consumed_image_cannot_change_before_or_during_notation(tmp_path, phase, mutation):
    claim, materials, response, image, alternate = context(tmp_path)
    calls = []

    def model(**kw):
        calls.append(kw["module"])
        if kw["module"] == phase:
            if mutation == "bytes":
                image.write_bytes(alternate.read_bytes())
            elif mutation == "path":
                materials.pages[0].path = str(alternate)
            elif mutation == "replacement":
                materials.pages = [materials.pages[0].model_copy(update={"path": str(alternate)})]
            elif mutation == "metadata":
                materials.pages[0].width_points = 99
            else:
                materials.pages.append(materials.pages[0].model_copy())
        if kw["module"] == "verification.theory.notation":
            return {"classification": "manuscript_issue", "explanation": "Fixed visual confirmation."}
        return copy.deepcopy(response)

    result = verify_theory(claim, materials, call=model)
    assert result.evidence == []
    assert any("page" in issue.lower() and "failed" in issue.lower() for issue in result.issues)
    assert len(calls) == (1 if phase == "verification.theory" else 2)


def test_late_image_creation_cannot_become_a_new_first_response_baseline(tmp_path):
    claim, materials, response, image, alternate = context(tmp_path)
    image.unlink()
    calls = []

    def model(**kw):
        calls.append(kw["module"])
        image.write_bytes(alternate.read_bytes())
        return response

    result = verify_theory(claim, materials, call=model)
    assert not result.evidence and calls == ["verification.theory"]
    assert result.issues


def test_healthy_consumed_image_hash_is_retained_for_later_advice(tmp_path):
    claim, materials, response, image, _ = context(tmp_path)
    expected = hashlib.sha256(image.read_bytes()).hexdigest()

    def model(**kw):
        if kw["module"] == "verification.theory.concern_scope":
            return outside_scope(kw)
        if kw["module"] == "verification.theory.notation":
            return {"classification": "manuscript_issue", "explanation": "Fixed visual confirmation."}
        return response

    result = verify_theory(claim, materials, call=model)
    assert not result.evidence[0].sufficient and result.evidence[0].direction == "flaw"
    assert not result.evidence[0].affects_claim
    assert "original PDF page 1 confirmed" in result.evidence[0].note
    assert result.theory_derivations[0].concern_reviews[0].decision.disposition == "outside_scope"
    assert result.theory_derivations[0].source_hashes[str(image)] == expected


@pytest.mark.parametrize("notation", [True, False])
def test_missing_unconsumed_page_has_no_effect_on_healthy_item(tmp_path, notation):
    claim, materials, response, _, _ = context(tmp_path, notation=notation)
    materials.pages.append(
        PageImage(page=2, path=str(tmp_path / "absent.png"), width_points=32, height_points=32)
    )

    def model(**kw):
        if kw["module"] == "verification.theory.concern_scope":
            return outside_scope(kw)
        if kw["module"] == "verification.theory.notation":
            return {"classification": "manuscript_issue", "explanation": "Fixed visual confirmation."}
        # Even changes to an unused image cannot affect this item's source identity.
        materials.pages[1].path = str(tmp_path / "still-absent.png")
        return response

    result = verify_theory(claim, materials, call=model)
    assert result.evidence[0].sufficient is (not notation)
    assert result.evidence[0].affects_claim is (not notation)
    if notation:
        assert result.theory_derivations[0].concern_reviews[0].decision.disposition == "outside_scope"


def test_missing_image_without_notation_does_not_block_derivation(tmp_path):
    claim, materials, response, image, _ = context(tmp_path, notation=False)
    image.unlink()
    result = verify_theory(claim, materials, call=lambda **kw: response)
    assert result.evidence[0].sufficient
    assert str(image) not in result.theory_derivations[0].source_hashes


def two_notation_items(tmp_path, *, different_pages):
    claim, materials, response, image, alternate = context(tmp_path)
    second_item = copy.deepcopy(response["items"][0])
    second_trace = copy.deepcopy(response["derivations"][0])
    second_trace["item_index"] = 1
    if different_pages:
        block = materials.blocks[0].model_copy(deep=True)
        block.id = "b2"
        block.loc.page = 2
        block.loc.char_start = len(materials.markdown) + 2
        materials.markdown += "\n\n" + block.text
        block.loc.char_end = len(materials.markdown)
        Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
        materials.blocks.append(block)
        second_image = tmp_path / "second.png"
        Image.new("RGB", (32, 32), "white").save(second_image)
        materials.pages.append(PageImage(page=2, path=str(second_image), width_points=32, height_points=32))
        second_item["block_id"] = block.id
        for part in [*second_trace["trace"]["assumptions"], *second_trace["trace"]["steps"]]:
            for source in part["sources"]:
                source["block_id"] = block.id
    response["items"].append(second_item)
    response["derivations"].append(second_trace)
    return claim, materials, response, image, alternate


@pytest.mark.parametrize("different_pages", [False, True])
@pytest.mark.parametrize(
    "later_outcome", ["manuscript_issue", "parser_artifact", "uncertain", "model_error", "malformed"]
)
def test_later_notation_cannot_leave_earlier_changed_page_evidence(tmp_path, different_pages, later_outcome):
    claim, materials, response, image, alternate = two_notation_items(
        tmp_path, different_pages=different_pages
    )
    visual_calls = 0

    def model(**kw):
        nonlocal visual_calls
        if kw["module"] != "verification.theory.notation":
            return response
        visual_calls += 1
        if visual_calls == 2:
            image.write_bytes(alternate.read_bytes())
            if later_outcome == "model_error":
                raise TimeoutError("fixed injected failure after changing an earlier page")
            if later_outcome == "malformed":
                return {"classification": "unknown", "explanation": "Fixed invalid classification."}
        return {
            "classification": "manuscript_issue" if visual_calls == 1 else later_outcome,
            "explanation": "Fixed visual observation.",
        }

    with pytest.raises(ValueError, match="notation page"):
        verify_theory(claim, materials, call=model)
    assert visual_calls == 2


@pytest.mark.parametrize("different_pages", [False, True])
def test_later_notation_cannot_relocate_an_earlier_consumed_page(tmp_path, different_pages):
    claim, materials, response, _, alternate = two_notation_items(tmp_path, different_pages=different_pages)
    visual_calls = 0

    def model(**kw):
        nonlocal visual_calls
        if kw["module"] != "verification.theory.notation":
            return response
        visual_calls += 1
        if visual_calls == 2:
            materials.pages[0] = materials.pages[0].model_copy(update={"path": str(alternate)})
        return {"classification": "manuscript_issue", "explanation": "Fixed visual observation."}

    with pytest.raises(ValueError, match="notation page"):
        verify_theory(claim, materials, call=model)
    assert visual_calls == 2


@pytest.mark.parametrize("different_pages", [False, True])
def test_two_healthy_notation_items_retain_their_actual_consumed_hashes(tmp_path, different_pages):
    claim, materials, response, _, _ = two_notation_items(tmp_path, different_pages=different_pages)

    def model(**kw):
        if kw["module"] == "verification.theory.concern_scope":
            return outside_scope(kw)
        if kw["module"] == "verification.theory.notation":
            return {"classification": "manuscript_issue", "explanation": "Fixed visual observation."}
        return response

    result = verify_theory(claim, materials, call=model)
    assert len(result.evidence) == 2 and all(
        not e.sufficient and not e.affects_claim for e in result.evidence
    )
    for index, record in enumerate(result.theory_derivations):
        assert record.concern_reviews[0].decision.disposition == "outside_scope"
        page = materials.pages[index if different_pages else 0]
        assert record.source_hashes[page.path] == hashlib.sha256(Path(page.path).read_bytes()).hexdigest()
