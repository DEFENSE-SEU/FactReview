"""Theory section roles preserve original sources and reading-phase boundaries."""

import copy
import json

import pytest

from llm.client import LLMConfig
from schemas.claim import Claim, ClaimLocation, Condition
from schemas.materials import MaterialBlock, SharedMaterials
from verification.theory import verify_theory
from verification.theory_sections import partition_theory_sections


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail("This test must not call external services or execute code")

    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: LLMConfig("mock", "mock", None, None))
    monkeypatch.setattr("screening.checks.llm_json", forbidden)
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)


def paper(tmp_path, rows):
    markdown = ""
    blocks = []
    section = None
    for identifier, kind, text in rows:
        if kind == "heading":
            section = text
            markdown += "## "
        start = len(markdown)
        markdown += text + "\n\n"
        blocks.append(
            MaterialBlock(
                id=identifier,
                kind=kind,
                text=text,
                loc=ClaimLocation(page=1, section=section, char_start=start, char_end=start + len(text)),
            )
        )
    path = tmp_path / "paper.md"
    path.write_text(markdown, encoding="utf-8")
    return SharedMaterials(
        paper_key="partition",
        source_pdf=str(tmp_path / "missing.pdf"),
        markdown=markdown,
        markdown_path=str(path),
        content_list_path="",
        provider="fixture",
        blocks=blocks,
    )


def rows(*, references=True):
    result = [
        ("h1", "heading", "1 Theory"),
        ("main", "text", "Theorem 1: n is even. See appendix B.1 and appendix C."),
    ]
    if references:
        result += [("refs", "heading", "References"), ("ref", "text", "[1] Author. Source title.")]
    return [
        *result,
        ("a", "heading", "A Algorithm"),
        ("algorithm", "text", "x = y."),
        ("b", "heading", "B Proofs"),
        ("b1", "heading", "B.1 Proof of Theorem 1"),
        ("proof", "text", "Proof of Theorem 1. n = 2k; therefore n is even."),
        ("c", "heading", "C More results"),
        ("other", "text", "Proof of Theorem 2. x = y."),
    ]


def target(materials):
    block = next(b for b in materials.blocks if b.id == "main")
    return Claim(
        id="claim",
        text="Every stated n is even.",
        loc=block.loc,
        source_block_id=block.id,
        source_quote=block.text,
        conditions=[Condition(id="c1", description="Every stated n is even")],
        needs=["Theory"],
    )


def item(block, *, kind="derivation", main="main"):
    return {
        "block_id": block.id,
        "quote": block.text,
        "covered": ["c1"],
        "fully_supported_conditions": ["c1"] if kind == "derivation" else [],
        "kind": kind,
        "direction": "support" if kind == "derivation" else "flaw",
        "detail": "Exact mock statement for source-contract testing.",
        "step_quote": "n = 2k" if kind == "derivation" else "",
        "main_block_id": main,
    }


@pytest.mark.parametrize("references", [True, False])
def test_cross_referenced_terminal_letter_chain_and_numbered_children(tmp_path, references):
    materials = paper(tmp_path, rows(references=references))
    before = materials.model_dump(mode="json")
    result = partition_theory_sections(materials)
    assert result.is_main("main")
    assert all(result.is_appendix(key) for key in ["a", "algorithm", "b", "b1", "proof", "c", "other"])
    if references:
        assert result.roles["ref"] == result.roles["refs"] == "references"
    assert next(r for r in result.regions if r["heading_id"] == "b1")["parent_label"] == "B"
    assert materials.model_dump(mode="json") == before


def test_explicit_appendix_inherits_child_without_references(tmp_path):
    materials = paper(
        tmp_path,
        [
            ("main", "text", "Theorem 1: n is even."),
            ("a", "heading", "Appendix A Proofs"),
            ("child", "heading", "A.1 Details"),
            ("proof", "text", "Proof of Theorem 1. n = 2k."),
        ],
    )
    result = partition_theory_sections(materials)
    assert result.is_appendix("proof") and result.is_appendix("child")
    assert result.is_main("main")


@pytest.mark.parametrize("heading", ["Appendix Proofs", "Appendix: Proofs", "Appendix Derivations"])
def test_explicit_unlabelled_proof_appendix_is_separate_from_main(tmp_path, heading):
    materials = paper(
        tmp_path,
        [
            ("main", "text", "Theorem 1: n is even."),
            ("h", "heading", heading),
            ("proof", "text", "Proof of Theorem 1. n = 2k."),
        ],
    )
    roles = partition_theory_sections(materials)
    assert roles.is_appendix("proof") and roles.is_main("main")


@pytest.mark.parametrize("change", ["unlocated", "changed_text"])
def test_unverifiable_author_reference_cannot_license_appendix_chain(tmp_path, change):
    materials = paper(tmp_path, rows())
    ref = materials.blocks[1]
    if change == "unlocated":
        ref.loc.char_start = None
    else:
        ref.text += " A changed passage."
    roles = partition_theory_sections(materials)
    assert roles.roles["proof"] == "unknown" and not roles.references


@pytest.mark.parametrize(
    "heading", ["A Algorithm", "Appendix detection in NLP", "2 Appendix detection in NLP"]
)
def test_letter_or_topic_title_alone_does_not_create_appendix(tmp_path, heading):
    materials = paper(
        tmp_path,
        [("h", "heading", heading), ("p", "text", "A UPPERCASE LINE. We discuss appendix C as an example.")],
    )
    result = partition_theory_sections(materials)
    assert result.is_main("h") and result.is_main("p")


@pytest.mark.parametrize("change", ["duplicate", "missing_ref", "unlocated", "interleaved"])
def test_ambiguous_letter_region_cannot_acquire_main_or_appendix_qualification(tmp_path, change):
    data = rows()
    if change == "duplicate":
        data += [("duplicate", "heading", "B Other proofs"), ("late", "text", "x = y.")]
    elif change == "missing_ref":
        data[1] = ("main", "text", "Theorem 1: n is even. See appendix Z.")
    elif change == "interleaved":
        data += [("numbered", "heading", "2 Main results")]
    materials = paper(tmp_path, data)
    if change == "unlocated":
        next(b for b in materials.blocks if b.id == "b").loc.char_start = None
    partition = partition_theory_sections(materials)
    assert partition.roles["proof"] == "unknown"
    assert not partition.is_main("proof") and not partition.is_appendix("proof")
    assert partition.issues


def test_public_entry_requires_explicit_second_phase_and_preserves_claim(tmp_path):
    materials = paper(tmp_path, rows())
    claim = target(materials)
    before = claim.model_dump(mode="json")
    proof = next(b for b in materials.blocks if b.id == "proof")
    calls = []

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"])
        calls.append(payload)
        assert "source_partition" in payload
        if len(calls) == 1:
            assert {b["id"] for b in payload["main_text"]} == {"h1", "main"}
            assert "proof" not in payload["allowed_theory_source_block_ids"]
            return {"items": [], "appendix_block_ids": ["proof"]}
        assert {b["id"] for b in payload["appendix_proofs"]} == {"proof"}
        return {"items": [item(proof)]}

    result = verify_theory(claim, materials, call=call, output_dir=tmp_path / "audits")
    # Legacy exact proof remains readable; its old eligibility contract is retained.
    assert len(calls) == 2 and result.evidence[0].sufficient
    assert claim.model_dump(mode="json") == before
    audits = [json.loads(p.read_text("utf-8")) for p in (tmp_path / "audits").glob("*.json")]
    assert all(a["request"]["source_partition"]["roles"]["proof"] == "appendix" for a in audits)


@pytest.mark.parametrize("which", ["first_appendix", "references", "unrequested_sibling", "unknown_request"])
def test_stage_boundaries_reject_old_unrequested_sources(tmp_path, which):
    materials = paper(tmp_path, rows())
    claim = target(materials)
    blocks = {b.id: b for b in materials.blocks}
    calls = []

    def call(**kwargs):
        calls.append(kwargs["module"])
        if which == "unknown_request":
            return {"items": [], "appendix_block_ids": ["not_in_materials"]}
        if which == "first_appendix":
            return {"items": [item(blocks["proof"])]}
        if which == "references":
            return {"items": [item(blocks["ref"], kind="no_proof")]}
        if len(calls) == 1:
            return {"items": [], "appendix_block_ids": ["proof"]}
        return {"items": [item(blocks["other"], kind="no_proof")]}

    with pytest.raises(ValueError, match=r"main-text|unknown appendix|outside"):
        verify_theory(claim, materials, call=call)
    assert len(calls) <= 2


@pytest.mark.parametrize("read", [False, True])
def test_no_proof_requires_reading_related_appendix_before_author_question(tmp_path, read):
    materials = paper(
        tmp_path,
        [
            ("main", "text", "Theorem 1: n is even. See appendix A."),
            ("a", "heading", "Appendix A Proofs"),
            ("proof", "text", "An author assertion repeats that n is even; no derivation is supplied."),
        ],
    )
    claim = target(materials)
    calls = []

    def call(**kwargs):
        calls.append(kwargs["module"])
        if read and len(calls) == 1:
            return {"items": [], "appendix_block_ids": ["a", "proof"]}
        return {"items": [item(materials.blocks[0], kind="no_proof")]}

    result = verify_theory(claim, materials, call=call)
    assert not result.evidence and result.issues
    assert bool(result.questions) == read
    assert bool(result.verification_limitations) == (not read)
    if not read:
        limitation = result.verification_limitations[0]
        assert limitation.kind == "source_context_unavailable" and limitation.condition_ids == ["c1"]
        assert limitation.responsibility == "system"


def test_unrelated_appendix_does_not_block_known_main_no_proof(tmp_path):
    materials = paper(
        tmp_path,
        [
            ("main", "text", "Theorem 1: n is even."),
            ("a", "heading", "Appendix A Proofs"),
            ("proof", "text", "Proof of Theorem 2. m = 2j."),
        ],
    )
    result = verify_theory(
        target(materials),
        materials,
        call=lambda **kw: {"items": [item(materials.blocks[0], kind="no_proof")]},
    )
    assert result.questions and not result.verification_limitations


def test_legacy_explicit_parent_section_keeps_unmarked_child_in_appendix(tmp_path):
    materials = paper(
        tmp_path,
        [
            ("main", "text", "Theorem 1: n is even."),
            ("a", "text", "Proof introduction."),
            ("proof", "text", "Proof of Theorem 1. n = 2k."),
        ],
    )
    materials.blocks[1].loc.section = "Appendix A"
    materials.blocks[2].loc.section = "A.1 Details"
    sections = partition_theory_sections(materials)
    assert sections.is_appendix("a") and sections.is_appendix("proof")
    assert sections.is_main("main")


@pytest.mark.parametrize("reference", ["appendix B.99", "supplementary material"])
def test_unresolved_exact_subsection_or_supplement_ref_cannot_establish_absence(tmp_path, reference):
    data = rows()
    data[1] = ("main", "text", f"Theorem 1: n is even. See {reference}.")
    materials = paper(tmp_path, data)
    result = verify_theory(
        target(materials),
        materials,
        call=lambda **kw: {"items": [item(materials.blocks[1], kind="no_proof")]},
    )
    assert not result.questions and result.verification_limitations


def test_orphan_numbered_child_cannot_borrow_another_parent_role(tmp_path):
    data = rows()
    data.extend(
        [("orphan", "heading", "Z.1 Extra proof"), ("orphan_proof", "text", "Proof of Theorem 9. x = y.")]
    )
    materials = paper(tmp_path, data)
    result = partition_theory_sections(materials)
    assert result.roles["orphan_proof"] == "unknown"
    assert result.is_appendix("proof")


@pytest.mark.parametrize(
    "reference",
    [
        "Their appendix B.1 and appendix C of [1] discuss this.",
        "Smith’s appendix B.1 and their appendix C discuss this.",
    ],
)
def test_explicit_external_reference_does_not_classify_local_letter_regions(tmp_path, reference):
    data = rows()
    data[1] = ("main", "text", reference)
    result = partition_theory_sections(paper(tmp_path, data))
    assert result.roles["proof"] == "unknown"
    assert not result.references


def test_bibliography_title_is_not_an_author_internal_appendix_reference(tmp_path):
    data = rows()
    data[1] = ("main", "text", "Theorem 1: n is even.")
    data[3] = ("ref", "text", "[1] Someone. See appendix B.1 and appendix C.")
    result = partition_theory_sections(paper(tmp_path, data))
    assert result.roles["ref"] == "references"
    assert result.roles["proof"] == "unknown"
    assert not result.references


def test_paper_title_starting_a_does_not_hide_terminal_appendices(tmp_path):
    materials = paper(tmp_path, [("title", "heading", "A Study of Learning"), *rows()])
    roles = partition_theory_sections(materials)
    assert roles.is_main("title") and roles.is_main("main")
    assert roles.is_appendix("proof") and roles.is_appendix("other")
    seen = []

    def call(**kwargs):
        data = json.loads(kwargs["prompt"])
        seen.extend(b["id"] for b in data["main_text"])
        return {"items": []}

    verify_theory(target(materials), materials, call=call)
    assert "proof" not in seen and "title" in seen


@pytest.mark.parametrize("entry_heading", ["Smith et al. 2020", "1 Smith et al."])
def test_reference_entry_heading_cannot_turn_citation_text_into_main(tmp_path, entry_heading):
    materials = paper(
        tmp_path,
        [
            ("main", "text", "Theorem 1: n is even."),
            ("refs", "heading", "References"),
            ("entry", "heading", entry_heading),
            ("body", "text", "Proof of Theorem 1. n = 2k."),
        ],
    )
    roles = partition_theory_sections(materials)
    assert not roles.is_main("body") and not roles.is_appendix("body")

    def call(**kw):
        assert "body" not in json.loads(kw["prompt"])["allowed_theory_source_block_ids"]
        return {"items": [item(materials.blocks[3])]}

    with pytest.raises(ValueError, match="main-text"):
        verify_theory(target(materials), materials, call=call)


@pytest.mark.parametrize(
    "heading", ["Appendices A and B", "Appendix A:Proofs", "Appendix: Additional experiment details"]
)
def test_unrecognized_explicit_appendix_signal_is_unknown_not_main(tmp_path, heading):
    materials = paper(
        tmp_path,
        [
            ("main", "text", "Theorem 1: n is even."),
            ("h", "heading", heading),
            ("proof", "text", "Proof of Theorem 1. n = 2k."),
        ],
    )
    roles = partition_theory_sections(materials)
    assert roles.roles["proof"] == "unknown"


def test_unknown_relevant_region_is_system_limitation_and_no_author_question(tmp_path):
    materials = paper(
        tmp_path,
        [
            ("main", "text", "Theorem 1: n is even. See appendix A."),
            ("a", "heading", "A Details"),
            ("proof", "text", "Proof of Theorem 1. n = 2k."),
        ],
    )
    result = verify_theory(
        target(materials),
        materials,
        call=lambda **kw: {"items": [item(materials.blocks[0], kind="no_proof")]},
    )
    assert not result.questions and not result.evidence
    assert result.verification_limitations[0].kind == "source_context_unavailable"


def test_no_proof_uses_only_current_condition_source_refs(tmp_path):
    from schemas.claim import ClaimSourceRef

    materials = paper(
        tmp_path,
        [
            ("main", "text", "Theorem 1: n is even. See appendix A."),
            ("second", "text", "Theorem 2: m is finite."),
            ("a", "heading", "Appendix A Proofs"),
            ("proof", "text", "Proof of Theorem 1. n = 2k."),
        ],
    )
    claim = target(materials)
    claim.conditions.append(Condition(id="c2", description="m is finite"))
    claim.source_refs = [
        ClaimSourceRef(source_block_id=block.id, source_quote=block.text, loc=block.loc, covered=[cid])
        for block, cid in zip(materials.blocks[:2], ["c1", "c2"], strict=True)
    ]
    response = item(materials.blocks[1], kind="no_proof")
    response["covered"] = ["c2"]
    result = verify_theory(claim, materials, call=lambda **kw: {"items": [response]})
    assert result.questions and not result.verification_limitations


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("read_body", [False, True])
def test_reading_proof_heading_requires_its_body_and_numbered_descendants(tmp_path, nested, read_body):
    data = [
        ("main", "text", "Theorem 1: n is even."),
        ("appendix", "heading", "Appendix A Proofs"),
        ("proof_heading", "heading", "A.1 Proof of Theorem 1"),
    ]
    if nested:
        data += [("detail_heading", "heading", "A.1.1 Details")]
    data += [
        ("proof_body", "text", "The statement is repeated here without a derivation."),
        ("sibling", "heading", "A.2 Proof of Theorem 2"),
        ("sibling_body", "text", "Proof of Theorem 2. m = 2j."),
    ]
    materials = paper(tmp_path, data)
    claim = target(materials)
    count = 0

    def call(**kwargs):
        nonlocal count
        count += 1
        if count == 1:
            selected = ["proof_heading"]
            if read_body:
                selected += ["proof_body", *(["detail_heading"] if nested else [])]
            return {"items": [], "appendix_block_ids": selected}
        return {"items": [item(materials.blocks[0], kind="no_proof")]}

    result = verify_theory(claim, materials, call=call)
    assert not result.evidence
    assert bool(result.questions) == read_body
    assert bool(result.verification_limitations) == (not read_body)
    if not read_body:
        assert "proof_body" in result.verification_limitations[0].reason
        assert "sibling" not in result.verification_limitations[0].reason


def test_appendix_claim_does_not_gain_a_fabricated_main_anchor(tmp_path):
    materials = paper(tmp_path, rows())
    claim = target(materials)
    proof = next(b for b in materials.blocks if b.id == "proof")
    claim.source_block_id, claim.source_quote, claim.loc = proof.id, proof.text, proof.loc
    before = copy.deepcopy(claim.model_dump(mode="json"))
    count = 0

    def call(**kw):
        nonlocal count
        count += 1
        return {"items": [], "appendix_block_ids": ["proof"]} if count == 1 else {"items": [item(proof)]}

    result = verify_theory(claim, materials, call=call)
    assert result.evidence and not result.evidence[0].sufficient
    assert "no verifiable target-claim main-text anchor" in result.issues[0]
    assert claim.model_dump(mode="json") == before


@pytest.mark.parametrize(
    "action,expected", [("verification_followup", "generated"), ("author_question", "unavailable")]
)
def test_new_limitation_roundtrips_and_keeps_advice_system_responsibility(
    tmp_path, monkeypatch, action, expected
):
    from review.report.advice import generate_advice
    from review.report.v2 import render_markdown
    from schemas.limitations import VerificationLimitation
    from schemas.review import FinalReview
    from tests.test_report_advice_v2 import payload, valid_response

    materials = paper(tmp_path, [("main", "text", "Theorem 1: n is even.")])
    claim = target(materials)
    claim.verification_limitations = [
        VerificationLimitation(
            claim_id=claim.id,
            condition_ids=["c1"],
            stage="Theory",
            kind="source_context_unavailable",
            reason="Related appendix A has not been read.",
        )
    ]
    review = FinalReview.model_validate_json(
        FinalReview(paper_key="test", run_id="test", claims=[claim]).model_dump_json()
    )
    monkeypatch.setattr(
        "review.report.advice.resolve_llm_config", lambda: LLMConfig("mock", "mock", None, None)
    )

    def call(**kwargs):
        response = valid_response(payload(kwargs))
        response["items"][0].update(action=action, text="Read the related appendix and retry verification.")
        response["items"][0]["basis_refs"].append("/verification_limitations/0")
        return response

    result = generate_advice(review, tmp_path / "advice", call=call)
    assert result.review.claims[0].advice.state == expected
    assert result.review.claims[0].status == "unverified"
    assert "System verification limitations" in render_markdown(result.review)
