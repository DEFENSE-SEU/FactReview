"""Original-page heading association controls; model and all external tools are mocked."""

import copy
import hashlib
import json
from pathlib import Path

import fitz
import pytest

from assessment import assess_claim
from schemas.claim import ClaimLocation, TheoryDerivationRecord
from schemas.materials import MaterialBlock
from tests.test_theory_partial_visual_v2 import partial_case
from tests.test_theory_visual_recheck_v2 import offline, recovered
from verification.theory import TheoryItem, verify_theory
from verification.theory_derivations import SourceContext
from verification.theory_visual import recheck

__all__ = ["offline"]


def heading_case(tmp_path, mode="healthy"):
    claim, materials, partial = partial_case(tmp_path)
    heading = MaterialBlock(
        id="heading",
        text="Appendix A. Proof of Theorem 1",
        kind="heading",
        loc=ClaimLocation(page=1, section="Appendix A"),
    )
    main, proof = materials.blocks
    proof.loc.section = "Appendix A. Proof of Theorem 1"
    blocks = [main, heading, proof]
    if mode == "wrong_theorem":
        heading.text = "Appendix A. Proof of Theorem 2"
    elif mode == "extra_text":
        heading.text += " and another theorem"
    elif mode == "cross_page":
        heading.loc.page = 2
    elif mode == "after":
        blocks = [main, proof, heading]
    elif mode in {"sibling", "unloaded_sibling"}:
        sibling = heading.model_copy(deep=True)
        sibling.id, sibling.text = "sibling", "A.2 Proof of Theorem 2"
        blocks.insert(2, sibling)
    elif mode == "duplicate":
        duplicate = heading.model_copy(deep=True)
        duplicate.id = "duplicate"
        blocks.insert(2, duplicate)
    elif mode == "not_heading":
        heading.kind = "text"
    elif mode == "unloaded_tail":
        tail = heading.model_copy(deep=True)
        tail.id, tail.text = "tail", "Appendix B. Details"
        blocks.append(tail)
    markdown = "\n\n".join(b.text for b in blocks)
    offset = 0
    for block in blocks:
        block.loc.char_start, block.loc.char_end = offset, offset + len(block.text)
        offset = block.loc.char_end + 2
    if mode == "bad_span":
        heading.loc.char_start += 1
    elif mode == "no_span":
        heading.loc.char_start = heading.loc.char_end = None
    materials.blocks, materials.markdown = blocks, markdown
    Path(materials.markdown_path).write_text(markdown, "utf-8")
    pdf = tmp_path / "heading-paper.pdf"
    with fitz.open() as doc:
        page = doc.new_page()
        for i, block in enumerate(blocks):
            page.insert_text((30, 40 + i * 50), block.text)
        page.get_pixmap().save(materials.pages[0].path)
        doc.new_page()
        doc.save(pdf)
    materials.source_pdf = str(pdf)
    claim.loc = main.loc.model_copy(deep=True)
    return claim, materials, partial, heading


def direct(tmp_path, mode="healthy", *, mutate=None, printed=None, partial_result=False):
    from screening.checks import grounded_paper_pointer

    claim, materials, partial, heading = heading_case(tmp_path, mode)
    loaded = [
        b
        for b in materials.blocks
        if not (mode == "unloaded" and b.id == "heading")
        and not (mode == "unloaded_sibling" and b.id == "sibling")
        and not (mode == "unloaded_tail" and b.id == "tail")
    ]
    context = SourceContext.capture(claim, materials, loaded)
    proof = next(b for b in materials.blocks if b.id == "b2")
    pointer = grounded_paper_pointer(materials, proof, proof.text)
    item = TheoryItem.model_validate(partial["items"][0])
    previous = TheoryDerivationRecord(
        claim_id=claim.id,
        item_index=0,
        phase="appendix",
        adopted=True,
        covered=["c1"],
        state="legacy_unavailable",
        source_pointer=pointer,
    )
    targets = [
        dict(
            target_id="target:test",
            trigger="partial_proof",
            condition_id="c1",
            original_record_index=0,
            item=item,
            record=previous,
            pointer=pointer,
            anchors=[("b1", claim.source_quote)],
            printed_identity="theorem 1",
            notation=None,
        )
    ]
    responses = []

    def model(**kw):
        assert kw["module"] == "verification.theory.visual_recheck"
        payload = json.loads(kw["prompt"])
        raw = recovered(payload)
        raw["visual_sources"][0]["printed_anchor"] = heading.text if printed is None else printed
        if partial_result:
            raw["items"][0]["fully_supported"] = False
            trace = raw["items"][0]["trace"]
            trace["outcome"] = "partial"
            trace["gaps"] = [
                {
                    "at": "goal",
                    "reason": "An independent qualifier is not derived.",
                    "needed": "A valid derivation of the original qualifier.",
                    "sources": [{"source_kind": "visual", "visual_source_index": 0}],
                }
            ]
        if mutate:
            mutate(claim, materials, heading)
        responses.append(copy.deepcopy(raw))
        return raw

    results = recheck(claim, materials, targets, context=context, call=model, output_dir=tmp_path / "audit")
    assert len(responses) == 1
    return results[0], responses[0], claim, materials


def test_loaded_original_heading_has_structured_audit_binding(tmp_path):
    (_, item, record), raw, _, materials = direct(tmp_path)
    assert item is not None and record.state == "validated", record.issues
    audit = json.loads(Path(record.audit_pointer).read_text("utf-8"))
    assert audit["response"] == raw
    bindings = audit["heading_bindings"]
    assert len(bindings) == 1
    source = bindings[0]["source"]
    assert source["block_id"] == "heading"
    assert source["pointer"]["quote"] == "Appendix A. Proof of Theorem 1"
    assert source["pointer"]["page"] == 1
    assert (
        source["artifact_sha256"]
        == hashlib.sha256(Path(source["pointer"]["locator"]).read_bytes()).hexdigest()
    )
    assert source["block_sha256"] == hashlib.sha256(materials.blocks[1].text.encode()).hexdigest()
    assert record.audit_sha256 == hashlib.sha256(Path(record.audit_pointer).read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "mode",
    [
        "wrong_theorem",
        "cross_page",
        "after",
        "sibling",
        "unloaded_sibling",
        "duplicate",
        "extra_text",
        "not_heading",
        "bad_span",
        "no_span",
        "unloaded",
    ],
)
def test_heading_fallback_cannot_borrow_wrong_or_unlocated_region(tmp_path, mode):
    (_, item, record), raw, _, _ = direct(tmp_path, mode)
    assert item is None and record.state == "invalid"
    assert record.trace is None and record.issues
    assert json.loads(Path(record.audit_pointer).read_text("utf-8"))["response"] == raw


@pytest.mark.parametrize("printed", ["Theorem 1", "Proof of Theorem 1", "Derivation for Theorem 1"])
def test_original_direct_match_still_works_without_loaded_heading(tmp_path, printed):
    (_, item, record), _, _, _ = direct(tmp_path, "unloaded", printed=printed)
    assert item is not None and record.state == "validated"
    assert not json.loads(Path(record.audit_pointer).read_text("utf-8")).get("heading_bindings")


@pytest.mark.parametrize(
    "printed",
    ["Appendix A. Proof of Theorem 1 ", "appendix A. Proof of Theorem 1", "Appendix B. Proof of Theorem 1"],
)
def test_fallback_requires_full_original_printed_text(tmp_path, printed):
    (_, item, record), _, _, _ = direct(tmp_path, printed=printed)
    assert item is None and record.state == "invalid"


@pytest.mark.parametrize("mutation", ["text", "loc", "replacement", "markdown_file"])
def test_callback_cannot_change_heading_source(tmp_path, mutation):
    def mutate(claim, materials, heading):
        if mutation == "text":
            heading.text += " changed"
        elif mutation == "loc":
            heading.loc.page = 2
        elif mutation == "replacement":
            i = next(i for i, b in enumerate(materials.blocks) if b.id == "heading")
            materials.blocks[i] = heading.model_copy(update={"text": "changed"})
        else:
            Path(materials.markdown_path).write_text("changed", "utf-8")

    with pytest.raises(ValueError, match="changed"):
        direct(tmp_path, mutate=mutate)


@pytest.mark.parametrize("mutation", ["kind", "delete", "loc", "reorder"])
def test_callback_cannot_change_unloaded_layout_dependency(tmp_path, mutation):
    def mutate(claim, materials, heading):
        sibling = next(b for b in materials.blocks if b.id == "sibling")
        if mutation == "kind":
            sibling.kind = "text"
        elif mutation == "delete":
            materials.blocks.remove(sibling)
        elif mutation == "loc":
            sibling.loc.char_start = 0
        else:
            materials.blocks.remove(sibling)
            materials.blocks.insert(0, sibling)

    with pytest.raises(ValueError, match="layout changed"):
        direct(tmp_path, "unloaded_sibling", mutate=mutate)


def test_final_guard_rechecks_layout_after_successful_heading_consumption(tmp_path, monkeypatch):
    original = SourceContext.source

    def consume_then_change(self, quote, claim, materials):
        source = original(self, quote, claim, materials)
        if quote.block_id == "heading":
            next(b for b in materials.blocks if b.id == "tail").kind = "text"
        return source

    monkeypatch.setattr(SourceContext, "source", consume_then_change)
    with pytest.raises(ValueError, match="layout changed"):
        direct(tmp_path, "unloaded_tail")


def test_binding_a_heading_does_not_close_independent_gap(tmp_path):
    (_, item, record), raw, _, _ = direct(tmp_path, partial_result=True)
    assert item is not None and record.state == "validated"
    assert not item.fully_supported
    assert record.trace.outcome == "partial" and len(record.trace.gaps) == 1
    assert json.loads(Path(record.audit_pointer).read_text("utf-8"))["response"] == raw


@pytest.mark.parametrize("fully_supported", [True, False])
def test_verify_public_loaded_appendix_heading_retains_visual_judgment(tmp_path, fully_supported):
    claim, materials, partial, heading = heading_case(tmp_path)
    calls, originals = [], []

    def model(**kw):
        calls.append(kw["module"])
        if kw["module"] == "verification.theory":
            return {
                "schema_version": "theory-derivation-v1",
                "appendix_block_ids": ["heading", "b2"],
                "items": [],
                "derivations": [],
            }
        if kw["module"] == "verification.theory.appendix":
            return copy.deepcopy(partial)
        assert kw["module"] == "verification.theory.visual_recheck"
        raw = recovered(json.loads(kw["prompt"]))
        raw["visual_sources"][0]["printed_anchor"] = heading.text
        if not fully_supported:
            raw["items"][0]["fully_supported"] = False
            raw["items"][0]["trace"]["outcome"] = "partial"
            raw["items"][0]["trace"]["gaps"] = [
                {
                    "at": "goal",
                    "reason": "Unproved independent qualifier.",
                    "needed": "A derivation.",
                    "sources": [{"source_kind": "visual", "visual_source_index": 0}],
                }
            ]
        originals.append(copy.deepcopy(raw))
        return raw

    result = verify_theory(
        claim, materials, call=model, visual_recheck_rounds=1, output_dir=tmp_path / "audit"
    )
    assert calls == [
        "verification.theory",
        "verification.theory.appendix",
        "verification.theory.visual_recheck",
    ]
    record = result.theory_derivations[-1]
    assert record.state == "validated", record.issues
    assert json.loads(Path(record.audit_pointer).read_text("utf-8"))["response"] == originals[0]
    claim.evidence.extend(result.evidence)
    assert assess_claim(claim).status == ("supported" if fully_supported else "unverified")
