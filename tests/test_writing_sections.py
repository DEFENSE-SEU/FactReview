import json
from pathlib import Path

import fitz
import pytest

from schemas.claim import ClaimLocation
from schemas.materials import MaterialBlock, PageImage, SharedMaterials, TableMaterial
from screening.checks import check_writing


def make_materials(tmp_path, paragraphs):
    markdown = "\n".join(text for _section, text in paragraphs)
    md = tmp_path / "paper.md"
    md.write_text(markdown, encoding="utf-8")
    blocks, pages, offset = [], [], 0
    pdf_path = tmp_path / "paper.pdf"
    with fitz.open() as pdf:
        for number, (section, text) in enumerate(paragraphs, 1):
            page = pdf.new_page()
            page.insert_textbox(fitz.Rect(30, 30, 570, 700), text)
            image_path = tmp_path / f"page-{number}.png"
            page.get_pixmap().save(image_path)
            pages.append(PageImage(page=number, path=str(image_path), width_points=595, height_points=842))
            blocks.append(
                MaterialBlock(
                    id=f"b{number}",
                    text=text,
                    loc=ClaimLocation(
                        page=number, section=section, char_start=offset, char_end=offset + len(text)
                    ),
                )
            )
            offset += len(text) + 1
        pdf.save(pdf_path)
    return SharedMaterials(
        paper_key="writing",
        source_pdf=str(pdf_path),
        markdown=markdown,
        markdown_path=str(md),
        content_list_path="",
        provider="mock",
        blocks=blocks,
        pages=pages,
    )


def candidate(block, **extra):
    return {
        "block_id": block["id"],
        "quote": block["text"],
        "text": "Correct the identified agreement error.",
        "level": "definite_error",
        **extra,
    }


def confirmation(
    candidate_id,
    *,
    decision="accept",
    kind="grammar",
    reason="visible_defect",
    explanation="The original page confirms the stated, concrete issue.",
):
    return {
        "candidate_id": candidate_id,
        "decision": decision,
        "confirmed_kind": kind,
        "reason": reason,
        "explanation": explanation,
    }


def confirmations(payload, kind="grammar", **decision_fields):
    return {
        "version": "writing-decision-v1",
        "results": [
            confirmation(row["candidate_id"], kind=kind, **decision_fields) for row in payload["candidates"]
        ],
    }


def test_three_sections_isolate_failed_request_and_keep_neighbors(tmp_path):
    materials = make_materials(
        tmp_path,
        [
            ("intro", "Our methods is fast."),
            ("methods", "The algorithms is clear."),
            ("results", "These scores is fixed."),
        ],
    )
    calls, records, issues = [], [], []

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"])
        calls.append((kwargs["module"], payload))
        if kwargs["module"] == "screening_writing.validation":
            return confirmations(payload)
        assert len(payload["blocks"]) == 1
        if payload["blocks"][0]["id"] == "b2":
            raise RuntimeError("section temporarily unavailable")
        return {"findings": [candidate(payload["blocks"][0])]}

    findings = check_writing(materials, call=call, records=records, issues=issues, recover_errors=True)
    assert [f.loc.section for f in findings] == ["intro", "results"]
    assert [r.status for r in records] == ["checked", "failed", "checked"]
    assert [r.block_ids for r in records] == [["b1"], ["b2"], ["b3"]]
    assert len({r.section_id for r in records}) == 3
    assert [r.confirmed_count for r in records] == [1, 0, 1]
    assert any("temporarily unavailable" in issue for issue in issues)
    assert len(calls) == 5


def test_foreign_section_quote_rejects_entire_candidate_batch(tmp_path):
    materials = make_materials(tmp_path, [("a", "These tests is fixed."), ("b", "Those values is fixed.")])
    records, visual_pages = [], []

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"])
        if kwargs["module"] == "screening_writing.validation":
            visual_pages.append(payload["page"])
            return confirmations(payload)
        block = payload["blocks"][0]
        if block["id"] == "b1":
            return {"findings": [candidate(block), candidate(materials.blocks[1].model_dump())]}
        return {"findings": [candidate(block)]}

    findings = check_writing(materials, call=call, records=records, recover_errors=True)
    assert [f.loc.section for f in findings] == ["b"]
    assert visual_pages == [2]
    assert [r.status for r in records] == ["failed", "checked"]


def test_unsectioned_group_and_stable_coverage_with_no_findings(tmp_path):
    materials = make_materials(tmp_path, [(None, "A clear sentence."), (None, "Another clear sentence.")])
    first, second = [], []
    for records in [first, second]:
        assert check_writing(materials, call=lambda **_: {"findings": []}, records=records) == []
    assert first == second
    assert len(first) == 1 and first[0].block_ids == ["b1", "b2"]
    assert first[0].section is None and first[0].status == "checked"


def test_empty_materials_are_unavailable_without_request(tmp_path):
    materials = SharedMaterials(
        paper_key="empty", source_pdf="", markdown="", markdown_path="", content_list_path="", provider="mock"
    )
    records, issues = [], []

    def call(**kwargs):
        pytest.fail("Empty input must not produce a model request")

    assert check_writing(materials, call=call, records=records, issues=issues) == []
    assert len(records) == 1 and records[0].status == "unavailable"
    assert "no" in issues[0].lower()


@pytest.mark.parametrize("policy", ["unspecified", "not_required", "required"])
def test_anonymity_explicit_policy_controls_same_source(tmp_path, policy):
    materials = make_materials(tmp_path, [("authors", "This manuscript is authored by Alice Example.")])
    records, calls = [], []

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"])
        calls.append(kwargs["module"])
        assert payload["anonymity_policy"] == policy
        if kwargs["module"] == "screening_writing.validation":
            return confirmations(payload, "anonymity")
        return {
            "findings": [
                candidate(
                    payload["blocks"][0],
                    category="anonymity",
                    text="The manuscript explicitly identifies its author.",
                )
            ]
        }

    findings = check_writing(materials, call=call, records=records, anonymity_policy=policy)
    assert bool(findings) == (policy == "required")
    assert len(calls) == (2 if policy == "required" else 1)
    assert records[0].anonymity_policy == policy


def test_invalid_policy_fails_before_model(tmp_path):
    materials = make_materials(tmp_path, [("intro", "Clear sentence.")])
    with pytest.raises(ValueError, match="anonymity_policy"):
        check_writing(materials, anonymity_policy="venue-guessed", call=lambda **_: pytest.fail("request"))


def test_cited_author_name_is_not_a_confirmed_anonymity_violation(tmp_path):
    materials = make_materials(tmp_path, [("related", "Smith et al. introduced the baseline [1].")])

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"])
        if kwargs["module"] == "screening_writing.validation":
            return confirmations(payload, None, decision="uncertain", reason="insufficient_context")
        return {
            "findings": [
                candidate(
                    payload["blocks"][0],
                    category="anonymity",
                    text="Check whether the cited author identifies this submission.",
                )
            ]
        }

    records = []
    assert check_writing(materials, call=call, anonymity_policy="required", records=records) == []
    assert records[0].status == "unavailable"


def test_full_directory_distinguishes_cross_section_and_namespaces(tmp_path):
    materials = make_materials(
        tmp_path, [("intro", "The full result is in Table 10."), ("appendix", "Table 10. The full results.")]
    )
    materials.blocks[1].kind = "table"
    materials.tables = [
        TableMaterial(
            id="table_10",
            block_id="b2",
            anchor="10",
            loc=materials.blocks[1].loc,
            caption=materials.blocks[1].text,
        )
    ]
    calls = []

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"])
        calls.append(payload)
        entries = payload["cross_reference_index"]["entries"]
        assert any(e["kind"] == "table" and e["label"] == "10" and e["loc"]["page"] == 2 for e in entries)
        assert not any(e["kind"] == "figure" and e["label"] == "10" for e in entries)
        return {"findings": []}

    assert check_writing(materials, call=call) == []
    assert len(calls) == 1


@pytest.mark.parametrize(
    "problem, text, label, expected",
    [
        ("unresolved_placeholder", "The result is shown in Table ??.", "??", True),
        ("missing_target", "The result is shown in Table 17.", "17", False),
    ],
)
def test_placeholder_can_be_confirmed_but_catalog_absence_is_not_proof(
    tmp_path, problem, text, label, expected
):
    materials = make_materials(tmp_path, [("intro", text)])

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"])
        if kwargs["module"] == "screening_writing.validation":
            return confirmations(payload, "cross_reference")
        return {
            "findings": [
                candidate(
                    payload["blocks"][0],
                    category="cross_reference",
                    target_kind="table",
                    target_label=label,
                    reference_problem=problem,
                    text="The printed internal reference is unresolved.",
                )
            ]
        }

    findings = check_writing(materials, call=call)
    assert bool(findings) is expected


def test_dataset_phrase_style_is_excluded_while_agreement_error_survives(tmp_path):
    materials = make_materials(
        tmp_path,
        [("evaluation", "On DeltaSet validation, model A has accuracy 0.8. These predictions is fixed.")],
    )
    quotes = ["On DeltaSet validation, model A has accuracy 0.8.", "These predictions is fixed."]

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"])
        if kwargs["module"] == "screening_writing.validation":
            return {
                "version": "writing-decision-v1",
                "results": [
                    {
                        "candidate_id": c["candidate_id"],
                        "decision": "reject" if c["quote"] == quotes[0] else "accept",
                        "confirmed_kind": None if c["quote"] == quotes[0] else "grammar",
                        "reason": "style" if c["quote"] == quotes[0] else "visible_defect",
                        "explanation": "Optional article convention."
                        if c["quote"] == quotes[0]
                        else "Plural subject conflicts with singular verb.",
                    }
                    for c in payload["candidates"]
                ],
            }
        return {"findings": [candidate({"id": "b1", "text": q}) for q in quotes]}

    findings = check_writing(materials, call=call)
    assert [f.evidence[0].pointer.quote for f in findings] == [quotes[1]]
    assert Path(materials.markdown_path).read_text(encoding="utf-8") == materials.markdown


def test_cross_reference_conflict_uses_both_original_pages(tmp_path):
    materials = make_materials(
        tmp_path,
        [("intro", "Table 10 reports a latency of 4 seconds."), ("appendix", "Table 10. Accuracy is 0.9.")],
    )
    materials.blocks[1].kind = "table"
    records = []

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"])
        if kwargs["module"] == "screening_writing.validation":
            assert kwargs["images"] == [p.path for p in materials.pages]
            assert payload["additional_target_pages"] == [2]
            assert payload["candidates"][0]["targets"][0]["quote"] == materials.blocks[1].text
            return confirmations(payload, "cross_reference")
        return {
            "findings": [
                candidate(
                    payload["blocks"][0],
                    category="cross_reference",
                    target_kind="table",
                    target_label="10",
                    reference_problem="inconsistent_target",
                    text="The table reports accuracy; the reference sentence describes latency.",
                )
            ]
        }

    findings = check_writing(materials, call=call, records=records)
    assert len(findings) == 1 and records[0].status == "checked"
    assert "Table 10. Accuracy is 0.9." in findings[0].evidence[0].note


@pytest.mark.parametrize("change", [{"target_label": "1"}, {"target_kind": "figure"}, {"target_label": "11"}])
def test_cross_reference_cannot_borrow_wrong_label_or_namespace(tmp_path, change):
    materials = make_materials(tmp_path, [("intro", "Table 10 reports the result.")])
    calls, records = [], []

    def call(**kwargs):
        calls.append(kwargs["module"])
        return {
            "findings": [
                candidate(
                    materials.blocks[0].model_dump(),
                    category="cross_reference",
                    **{
                        "target_kind": "table",
                        "target_label": "10",
                        "reference_problem": "inconsistent_target",
                        **change,
                    },
                )
            ]
        }

    assert check_writing(materials, call=call, records=records, recover_errors=True) == []
    assert records[0].status == "failed" and calls == ["screening_writing"]


@pytest.mark.parametrize(
    "text, label", [("Why?? Table 1 describes a result.", "1"), ("See \\ref{sec:actual}.", "sec:other")]
)
def test_placeholder_is_bound_to_exact_reference_occurrence(tmp_path, text, label):
    materials = make_materials(tmp_path, [("intro", text)])
    records = []

    def call(**kwargs):
        assert kwargs["module"] == "screening_writing"
        return {
            "findings": [
                candidate(
                    materials.blocks[0].model_dump(),
                    category="cross_reference",
                    target_kind="section",
                    target_label=label,
                    reference_problem="unresolved_placeholder",
                )
            ]
        }

    assert check_writing(materials, call=call, records=records, recover_errors=True) == []
    assert records[0].status == "failed"


def test_required_policy_includes_original_author_block_only_for_policy(tmp_path):
    materials = make_materials(tmp_path, [("front", "Alice Example, Example Institute.")])
    materials.blocks[0].kind = "author"

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"])
        if kwargs["module"] == "screening_writing.validation":
            return confirmations(payload, "anonymity")
        assert payload["policy_only_block_ids"] == ["b1"]
        return {
            "findings": [
                candidate(
                    payload["blocks"][0],
                    category="anonymity",
                    text="The author line identifies this submission.",
                )
            ]
        }

    assert len(check_writing(materials, call=call, anonymity_policy="required")) == 1
    records = []
    assert check_writing(materials, call=lambda **_: pytest.fail("not applicable"), records=records) == []
    assert records[0].status == "unavailable"


def test_visual_failure_rolls_back_section_and_next_section_continues(tmp_path):
    materials = make_materials(
        tmp_path,
        [
            ("same", "These tests is fixed."),
            ("same", "The examples is short."),
            ("next", "The authors is careful."),
        ],
    )
    records = []

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"])
        if kwargs["module"] == "screening_writing.validation":
            return (
                {"status": "error", "error": "visual offline"}
                if payload["page"] == 2
                else confirmations(payload)
            )
        return {"findings": [candidate(b) for b in payload["blocks"]]}

    findings = check_writing(materials, call=call, records=records, recover_errors=True)
    assert [f.loc.page for f in findings] == [3]
    assert [r.status for r in records] == ["failed", "checked"]
    assert [r.confirmed_count for r in records] == [0, 1]


def test_unlocated_text_does_not_claim_complete_coverage(tmp_path):
    materials = make_materials(tmp_path, [("intro", "A clear sentence.")])
    materials.blocks[0].loc = None
    records = []
    assert check_writing(materials, call=lambda **_: {"findings": []}, records=records) == []
    assert records[0].status == "unavailable" and records[0].issues


def test_positive_classification_cannot_change_candidate_category(tmp_path):
    materials = make_materials(tmp_path, [("intro", "A clear sentence.")])

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"])
        if kwargs["module"] == "screening_writing.validation":
            return confirmations(payload, "anonymity")
        return {"findings": [candidate(payload["blocks"][0])]}

    records = []
    assert check_writing(materials, call=call, records=records) == []
    assert records[0].status == "failed"


def test_cross_reference_missing_original_target_page_is_unavailable(tmp_path):
    materials = make_materials(
        tmp_path, [("intro", "Table 10 reports latency."), ("appendix", "Table 10. Accuracy is 0.9.")]
    )
    materials.blocks[1].kind = "table"
    Path(materials.pages[1].path).unlink()
    records = []

    def call(**kwargs):
        assert kwargs["module"] == "screening_writing"
        return {
            "findings": [
                candidate(
                    materials.blocks[0].model_dump(),
                    category="cross_reference",
                    target_kind="table",
                    target_label="10",
                    reference_problem="inconsistent_target",
                )
            ]
        }

    assert check_writing(materials, call=call, records=records) == []
    assert records[0].status == "unavailable"
    assert any("target page 2 unavailable" in issue for issue in records[0].issues)


def test_repeated_sentence_is_not_an_exact_location(tmp_path):
    materials = make_materials(tmp_path, [("intro", "These values is fixed. These values is fixed.")])
    records = []

    def call(**kwargs):
        assert kwargs["module"] == "screening_writing"
        return {"findings": [candidate({"id": "b1", "text": "These values is fixed."})]}

    assert check_writing(materials, call=call, records=records, recover_errors=True) == []
    assert records[0].status == "failed"


def test_unexpanded_latex_reference_retains_exact_target_key(tmp_path):
    text = "See \\ref{sec:method} for the procedure."
    materials = make_materials(tmp_path, [("intro", text)])

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"])
        if kwargs["module"] == "screening_writing.validation":
            return confirmations(payload, "cross_reference")
        return {
            "findings": [
                candidate(
                    payload["blocks"][0],
                    category="cross_reference",
                    target_kind="section",
                    target_label="sec:method",
                    reference_problem="unresolved_placeholder",
                )
            ]
        }

    findings = check_writing(materials, call=call)
    assert len(findings) == 1 and findings[0].evidence[0].pointer.quote == text
