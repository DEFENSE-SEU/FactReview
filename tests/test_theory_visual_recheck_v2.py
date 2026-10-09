"""Mocked mathematics/visual readings test protocol guards, not model accuracy."""

import copy
import hashlib
import json
from pathlib import Path

import fitz
import pytest

from assessment import assess_claim
from llm.client import LLMConfig
from schemas.claim import Claim, ClaimLocation, Condition
from schemas.materials import MaterialBlock, PageImage, SharedMaterials
from verification.theory import verify_theory


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    cfg = LLMConfig("mock", "visual-recovery", None, None)
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: cfg)
    monkeypatch.setattr("screening.checks.resolve_vlm_config", lambda **kw: cfg)

    def blocked(*a, **kw):
        pytest.fail("Unmocked external boundary")

    monkeypatch.setattr("screening.checks.llm_json", blocked)
    monkeypatch.setattr("httpx.Client.send", blocked)
    monkeypatch.setattr("httpx.AsyncClient.send", blocked)
    monkeypatch.setattr("requests.sessions.Session.request", blocked)
    monkeypatch.setattr("subprocess.run", blocked)


def case(tmp_path, *, conditions=1):
    texts = [
        "Theorem 1. For every real x, x*x is nonnegative.",
        "Proof of Theorem 1. For real x, x^{star}x >= 0 by the two sign cases.",
    ]
    markdown = "\n\n".join(texts)
    md = tmp_path / "paper.md"
    md.write_text(markdown, encoding="utf-8")
    pdf = tmp_path / "paper.pdf"
    image = tmp_path / "page.png"
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((30, 40), texts[0])
        page.insert_text(
            (30, 70), "Proof of Theorem 1. For x >= 0, x*x >= 0. For x < 0, y=-x > 0 and x*x=y*y >= 0."
        )
        doc.save(pdf)
        page.get_pixmap().save(image)
    blocks = [
        MaterialBlock(
            id="b1",
            text=texts[0],
            loc=ClaimLocation(page=1, section="Theory", char_start=0, char_end=len(texts[0])),
        ),
        MaterialBlock(
            id="b2",
            text=texts[1],
            loc=ClaimLocation(
                page=1, section="Proof of Theorem 1", char_start=len(texts[0]) + 2, char_end=len(markdown)
            ),
        ),
    ]
    materials = SharedMaterials(
        paper_key="visual-recovery",
        source_pdf=str(pdf),
        markdown=markdown,
        markdown_path=str(md),
        content_list_path="",
        provider="mock",
        blocks=blocks,
        pages=[PageImage(page=1, path=str(image), width_points=595, height_points=842)],
    )
    claim = Claim(
        id="square",
        text=texts[0],
        source_block_id="b1",
        source_quote=texts[0],
        loc=blocks[0].loc,
        conditions=[Condition(id=f"c{i + 1}", description="For every real x") for i in range(conditions)],
        needs=["Theory"],
    )
    initial = {
        "schema_version": "theory-derivation-v1",
        "appendix_block_ids": [],
        "items": [
            {
                "block_id": "b2",
                "quote": texts[1],
                "step_quote": "x^{star}x >= 0",
                "covered": [c.id for c in claim.conditions],
                "fully_supported_conditions": [],
                "kind": "notation",
                "direction": "flaw",
                "detail": "The parsed superscript is ambiguous.",
                "main_block_id": "b1",
            }
        ],
        "derivations": [
            {
                "item_index": 0,
                "trace": {
                    "goal": claim.text,
                    "assumptions": [],
                    "steps": [],
                    "gaps": [
                        {
                            "at": "goal",
                            "reason": "The parsed operator is ambiguous.",
                            "needed": "Read original pixels.",
                            "sources": [{"block_id": "b2", "quote": texts[1]}],
                        }
                    ],
                    "outcome": "unable",
                    "completion_reason": "Parsed symbols prevent this proof review.",
                },
            }
        ],
    }
    return claim, materials, initial


def recovered(payload):
    readings, items = [], []
    for target in payload["targets"]:
        page = next(page_id for page_id, blocks in target["page_anchors"].items() if "b2" in blocks)
        readings.append(
            {
                "target_id": target["target_id"],
                "page_id": page,
                "anchor_block_id": "b2",
                "printed_anchor": "Theorem 1",
                "transcription": "Proof of Theorem 1. For x >= 0, x*x >= 0. For x < 0, y=-x > 0 and x*x=y*y >= 0.",
            }
        )
        items.append(
            {
                "target_id": target["target_id"],
                "direction": "support",
                "fully_supported": True,
                "detail": "Both sign cases cover all real numbers, including zero.",
                "trace": {
                    "goal": payload["claim"]["text"],
                    "assumptions": [
                        {
                            "id": "a1",
                            "text": "x is real",
                            "status": "paper_explicit",
                            "sources": [
                                {"source_kind": "paper", "block_id": "b1", "quote": "For every real x"}
                            ],
                        }
                    ],
                    "steps": [
                        {
                            "id": "s1",
                            "statement": "In both sign cases x*x >= 0.",
                            "reason": "Multiply nonnegative factors; for negative x substitute y=-x.",
                            "assumption_ids": ["a1"],
                            "previous_step_ids": [],
                            "sources": [{"source_kind": "visual", "visual_source_index": len(readings) - 1}],
                        }
                    ],
                    "gaps": [],
                    "outcome": "completed",
                    "completion_reason": "The cases include every real x.",
                },
            }
        )
    return {"schema_version": "theory-visual-derivation-v1", "visual_sources": readings, "items": items}


def run(tmp_path, change=None, *, conditions=1, rounds=None):
    claim, materials, initial = case(tmp_path, conditions=conditions)
    originals = copy.deepcopy((claim.model_dump(mode="json"), materials.model_dump(mode="json"), initial))
    calls, responses = [], []

    def model(**kw):
        calls.append(kw["module"])
        if kw["module"] == "verification.theory":
            return copy.deepcopy(initial)
        if kw["module"] == "verification.theory.notation":
            return {
                "classification": "parser_artifact",
                "explanation": "The original page prints ordinary multiplication.",
            }
        if kw["module"] == "verification.theory.visual_recheck":
            payload = json.loads(kw["prompt"])
            assert payload["claim"] == originals[0]
            assert kw["images"] == [materials.pages[0].path]
            value = recovered(payload)
            if change:
                change(value, payload, materials)
            responses.append(copy.deepcopy(value))
            return value
        if kw["module"] == "verification.theory.concern_scope":
            payload = json.loads(kw["prompt"])
            return {
                "schema_version": "theory-concern-v1",
                "items": [
                    {
                        **pair,
                        "disposition": "outside_scope",
                        "target_sources": [{"block_id": "b1", "quote": "For every real x"}],
                        "trace_step_ids": ["s1"],
                        "trace_gap_indices": [],
                        "scope_reason": "The recovered trace demonstrates nonnegative square, not a contradiction.",
                        "resolution": "This is outside the stated claim's concerns.",
                    }
                    for pair in payload["allowed_pairs"]
                ],
            }
        pytest.fail(kw["module"])

    result = verify_theory(
        claim, materials, call=model, output_dir=tmp_path / "audit", visual_recheck_rounds=rounds
    )
    assert (claim.model_dump(mode="json"), materials.model_dump(mode="json"), initial) == originals
    enriched = claim.model_copy(deep=True)
    enriched.evidence.extend(result.evidence)
    enriched.theory_derivations.extend(result.theory_derivations)
    enriched.verification_limitations.extend(result.verification_limitations)
    return result, assess_claim(enriched), calls, responses


def test_default_once_complete_visual_proof_and_round_trip(tmp_path):
    result, claim, calls, responses = run(tmp_path)
    assert claim.status == "supported" and not result.verification_limitations
    assert calls == [
        "verification.theory",
        "verification.theory.notation",
        "verification.theory.visual_recheck",
    ]
    assert result.theory_derivations[0].trace.outcome == "unable"
    record = result.theory_derivations[-1]
    assert record.schema_version == "theory-visual-derivation-v1" and record.state == "validated"
    source = record.trace.steps[0].sources[0]
    assert source.origin == "model_transcribed_from_original_pixels"
    assert "x^{star}" in record.source_pointer.quote and "x^{star}" not in source.pointer.quote
    assert json.loads(Path(record.audit_pointer).read_text())["response"] == responses[0]
    assert hashlib.sha256(Path(record.audit_pointer).read_bytes()).hexdigest() == record.audit_sha256
    assert Claim.model_validate_json(claim.model_dump_json()).model_dump() == claim.model_dump()


@pytest.mark.parametrize(
    "change",
    [
        "foreign_page",
        "foreign_anchor",
        "wrong_label",
        "foreign_target",
        "unknown_index",
        "bool_index",
        "future_dependency",
        "gap",
        "missing_visual",
        "paper_transcription",
        "extra_field",
        "wrong_version",
        "duplicate",
        "missing_target",
    ],
)
def test_invalid_recovery_never_upgrades_original_parser_observation(tmp_path, change):
    def mutate(raw, payload, materials):
        source = raw["visual_sources"][0]
        item = raw["items"][0]
        step = item["trace"]["steps"][0]
        if change == "foreign_page":
            source["page_id"] = "foreign"
        elif change == "foreign_anchor":
            source["anchor_block_id"] = "unread_appendix"
        elif change == "wrong_label":
            source["printed_anchor"] = "Theorem 2"
        elif change == "foreign_target":
            source["target_id"] = "foreign"
        elif change == "unknown_index":
            step["sources"][0]["visual_source_index"] = 10
        elif change == "bool_index":
            step["sources"][0]["visual_source_index"] = True
        elif change == "future_dependency":
            step["previous_step_ids"] = ["s2"]
        elif change == "gap":
            item["trace"]["gaps"] = [{"at": "goal", "reason": "A gap", "needed": "More proof", "sources": []}]
        elif change == "missing_visual":
            step["sources"] = []
        elif change == "paper_transcription":
            step["sources"] = [{"source_kind": "paper", "block_id": "b2", "quote": source["transcription"]}]
        elif change == "extra_field":
            source["locator"] = "forged.pdf"
        elif change == "wrong_version":
            raw["schema_version"] = "theory-derivation-v1"
        elif change == "duplicate":
            raw["items"].extend([copy.deepcopy(item), copy.deepcopy(item)])
        elif change == "missing_target":
            raw["items"] = []

    result, claim, calls, _ = run(tmp_path, mutate)
    assert claim.status == "unverified" and not result.evidence
    assert result.theory_derivations[-1].state == "invalid"
    assert any(x.kind == "source_context_unavailable" for x in result.verification_limitations)
    assert calls.count("verification.theory.visual_recheck") == 1


def test_pair_local_bad_trace_preserves_other_target(tmp_path):
    def mutate(raw, *a):
        raw["items"][0]["trace"]["steps"][0]["previous_step_ids"] = ["future"]

    result, claim, calls, _ = run(tmp_path, mutate, conditions=2)
    assert claim.status == "unverified"
    assert (
        len(result.evidence) == 1 and result.evidence[0].covered == ["c2"] and result.evidence[0].sufficient
    )
    assert calls.count("verification.theory.visual_recheck") == 1


def test_visual_full_flag_remains_independent_of_completed_trace(tmp_path):
    result, claim, _, _ = run(tmp_path, lambda raw, *a: raw["items"][0].update(fully_supported=False))
    assert claim.status == "unverified" and not result.evidence[0].sufficient
    assert result.theory_derivations[-1].state == "validated"


def test_visual_flaw_requires_original_condition_scope(tmp_path):
    result, claim, calls, _ = run(tmp_path, lambda raw, *a: raw["items"][0].update(direction="flaw"))
    assert claim.status == "unverified" and not result.evidence[0].affects_claim
    assert calls[-1] == "verification.theory.concern_scope"
    record = result.theory_derivations[-1]
    assert record.concern_reviews[0].decision.disposition == "outside_scope"
    assert record.concern_reviews[0].item_index == record.item_index


def test_explicit_zero_preserves_v1_no_recovery(tmp_path):
    result, claim, calls, _ = run(tmp_path, rounds=0)
    assert claim.status == "unverified" and calls == ["verification.theory", "verification.theory.notation"]
    assert len(result.theory_derivations) == 1
    assert "target_id" not in result.theory_derivations[0].model_dump()


@pytest.mark.parametrize("value", [True, False, -1, 2, 1.0, "1"])
def test_direct_budget_is_strict_before_first_call(tmp_path, value):
    claim, materials, _ = case(tmp_path)
    with pytest.raises(ValueError, match="rounds"):
        verify_theory(
            claim, materials, call=lambda **kw: pytest.fail("No request"), visual_recheck_rounds=value
        )


@pytest.mark.parametrize("mutation", ["pdf", "image", "markdown", "metadata"])
def test_original_source_mutation_during_recovery_is_rejected(tmp_path, mutation):
    def mutate(raw, payload, materials):
        if mutation == "metadata":
            materials.pages[0].width_points += 1
        else:
            path = {
                "pdf": materials.source_pdf,
                "image": materials.pages[0].path,
                "markdown": materials.markdown_path,
            }[mutation]
            with Path(path).open("ab") as stream:
                stream.write(b"changed")

    with pytest.raises(ValueError, match="changed"):
        run(tmp_path, mutate)


@pytest.mark.parametrize("artifact", ["audit", "image", "pdf"])
def test_advice_and_report_recheck_visual_file_bytes(tmp_path, artifact):
    from review.report.advice import theory_source_integrity

    result, claim, _, _ = run(tmp_path)
    record = result.theory_derivations[-1]
    source = record.trace.steps[0].sources[0]
    assert theory_source_integrity(claim)
    path = {"audit": record.audit_pointer, "image": source.image_path, "pdf": source.pointer.locator}[
        artifact
    ]
    with Path(path).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        theory_source_integrity(claim)


def test_report_and_pdf_keep_parsed_and_visual_sources_distinct(tmp_path):
    from pypdf import PdfReader

    from review.report.v2 import _text, write_review
    from tests.test_report_v2 import review

    result, claim, _, _ = run(tmp_path)
    document = review()
    document.claims = [claim]
    document.findings = []
    outputs = write_review(document, tmp_path / "delivery")
    saved = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
    assert saved["claims"][0] == claim.model_dump(mode="json")
    markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
    assert "model-transcribed original pixels" in markdown
    assert "Model-transcribed original-page source" in markdown
    assert _text("x^{star}") in markdown and _text("x*x=y*y") in markdown
    pdf = "\n".join(page.extract_text() for page in PdfReader(outputs["pdf"]).pages)
    assert "Model-transcribed original-page source" in pdf
    assert result.theory_derivations[-1].target_id in markdown


def test_failed_visual_request_is_preserved_and_never_retried(tmp_path):
    def failure(*args):
        raise TimeoutError("Explicit mock timeout")

    result, claim, calls, _ = run(tmp_path, failure)
    assert claim.status == "unverified" and not result.evidence
    assert calls.count("verification.theory.visual_recheck") == 1
    record = result.theory_derivations[-1]
    audit = json.loads(Path(record.audit_pointer).read_text())
    assert audit["status"] == "failed" and "timeout" in audit["error"]
    assert record.state == "invalid" and record.audit_sha256


@pytest.mark.parametrize("classification", ["uncertain", "manuscript_issue"])
def test_only_parser_artifact_triggers_recovery(tmp_path, classification):

    claim, materials, initial = case(tmp_path)
    calls = []

    def model(**kw):
        calls.append(kw["module"])
        if kw["module"] == "verification.theory":
            return copy.deepcopy(initial)
        if kw["module"] == "verification.theory.notation":
            return {"classification": classification, "explanation": "Explicit mock classification."}
        if kw["module"] == "verification.theory.concern_scope":
            payload = json.loads(kw["prompt"])
            return {
                "schema_version": "theory-concern-v1",
                "items": [
                    {
                        **pair,
                        "disposition": "unresolved",
                        "target_sources": [],
                        "trace_step_ids": [],
                        "trace_gap_indices": [],
                        "scope_reason": "Cannot establish applicability.",
                        "resolution": "More evidence needed.",
                    }
                    for pair in payload["allowed_pairs"]
                ],
            }
        pytest.fail("No visual recovery authorized")

    result = verify_theory(claim, materials, call=model, output_dir=tmp_path / "audit")
    assert "verification.theory.visual_recheck" not in calls
    assert all(r.schema_version == "theory-derivation-v1" for r in result.theory_derivations)


def test_healthy_support_prevents_unnecessary_visual_recovery(tmp_path):
    claim, materials, initial = case(tmp_path)
    healthy = copy.deepcopy(initial["items"][0])
    healthy.update(kind="derivation", direction="support", fully_supported_conditions=["c1"])
    initial["items"].append(healthy)
    trace = {
        "goal": claim.text,
        "assumptions": [],
        "steps": [
            {
                "id": "s1",
                "statement": "Both cases cover real x.",
                "reason": "Fixed mock proof.",
                "assumption_ids": [],
                "previous_step_ids": [],
                "sources": [{"block_id": "b2", "quote": materials.blocks[1].text}],
            }
        ],
        "gaps": [],
        "outcome": "completed",
        "completion_reason": "Explicit positive control.",
    }
    initial["derivations"].append({"item_index": 1, "trace": trace})
    calls = []

    def model(**kw):
        calls.append(kw["module"])
        if kw["module"] == "verification.theory":
            return copy.deepcopy(initial)
        assert kw["module"] == "verification.theory.notation"
        return {"classification": "parser_artifact", "explanation": "Explicit mock reading."}

    result = verify_theory(claim, materials, call=model, output_dir=tmp_path / "audit")
    assert result.evidence[0].sufficient and not result.verification_limitations
    assert calls == ["verification.theory", "verification.theory.notation"]


def test_audit_write_failure_preserves_healthy_neighbour(tmp_path, monkeypatch):
    original = Path.write_text

    def fail_audit(path, *a, **kw):
        if path.name.startswith("visual-"):
            raise OSError("Explicit audit failure")
        return original(path, *a, **kw)

    monkeypatch.setattr(Path, "write_text", fail_audit)
    result, claim, calls, _ = run(tmp_path)
    assert claim.status == "unverified" and not result.evidence
    assert "verification.theory.visual_recheck" not in calls
    assert result.theory_derivations[-1].state == "invalid"


def test_duplicate_target_tombstone_does_not_discard_healthy_pair(tmp_path):
    def mutate(raw, *a):
        raw["items"].extend([copy.deepcopy(raw["items"][0]), copy.deepcopy(raw["items"][0])])

    result, claim, calls, _ = run(tmp_path, mutate, conditions=2)
    assert len(result.evidence) == 1 and result.evidence[0].covered == ["c2"]
    assert result.evidence[0].sufficient and claim.status == "unverified"
    assert calls.count("verification.theory.visual_recheck") == 1


@pytest.mark.parametrize("value", [0, 1, "0", "1", 2, True, 1.0, "yes"])
def test_setting_has_one_closed_zero_or_one_control(value):
    from common.config import Settings

    if type(value) in (int, str) and value in (0, 1, "0", "1"):
        assert Settings(theory_visual_recheck_rounds=value).theory_visual_recheck_rounds == int(value)
    else:
        with pytest.raises(ValueError, match="rounds"):
            Settings(theory_visual_recheck_rounds=value)


@pytest.mark.parametrize("mutation", ["claim", "block"])
def test_notation_mutation_is_blocked_before_new_visual_request(tmp_path, mutation):
    claim, materials, initial = case(tmp_path)
    calls = []

    def model(**kw):
        calls.append(kw["module"])
        if kw["module"] == "verification.theory":
            return copy.deepcopy(initial)
        assert kw["module"] == "verification.theory.notation"
        if mutation == "claim":
            claim.text += " altered"
        else:
            materials.blocks[0].text += " altered"
        return {"classification": "parser_artifact", "explanation": "Explicit mock reading."}

    with pytest.raises(ValueError, match="changed"):
        verify_theory(claim, materials, call=model, output_dir=tmp_path / "audit")
    assert calls == ["verification.theory", "verification.theory.notation"]


def test_missing_main_anchor_page_pixels_do_not_block_two_proof_targets(tmp_path):
    claim, materials, initial = case(tmp_path, conditions=2)
    # The located main statement is on page 2, which has no rendered image.
    # Both conditions' affected proof is on the original rendered page 1.
    with fitz.open(materials.source_pdf) as old:
        original_pdf = old.tobytes()
    with fitz.open(stream=original_pdf, filetype="pdf") as doc:
        page = doc.new_page()
        page.insert_text((30, 40), materials.blocks[0].text)
        doc.save(materials.source_pdf)
    materials.blocks[0].loc.page = 2
    claim.loc.page = 2
    calls = []

    def model(**kw):
        calls.append(kw["module"])
        if kw["module"] == "verification.theory":
            return copy.deepcopy(initial)
        if kw["module"] == "verification.theory.notation":
            return {"classification": "parser_artifact", "explanation": "Explicit mock."}
        payload = json.loads(kw["prompt"])
        assert [page["page"] for page in payload["pages"]] == [1]
        assert len(payload["targets"]) == 2
        assert all(
            source["block_id"] == "b1"
            for target in payload["targets"]
            for source in target["condition_sources"]
        )
        return recovered(payload)

    result = verify_theory(claim, materials, call=model, output_dir=tmp_path / "audit")
    assert len(result.evidence) == 2 and all(e.sufficient for e in result.evidence)
    assert not result.verification_limitations
    assert calls.count("verification.theory.visual_recheck") == 1


@pytest.mark.parametrize(
    "title,valid",
    [
        ("Proof of Theorem 1", True),
        ("Derivation for Theorem 1:", True),
        ("Theorem 1", True),
        ("Proof of Theorem 2", False),
        ("Theorem 1 and Theorem 2", False),
        ("Proof of Theorem 1. Theorem 2", False),
    ],
)
def test_printed_title_preserved_with_finite_identity_matching(tmp_path, title, valid):
    result, claim, _, _ = run(tmp_path, lambda raw, *a: raw["visual_sources"][0].update(printed_anchor=title))
    assert (claim.status == "supported") == valid
    if valid:
        assert result.theory_derivations[-1].trace.steps[0].sources[0].printed_anchor == title


@pytest.mark.parametrize("field", ["audit_pointer", "audit_sha256"])
def test_new_validated_records_cannot_strip_audit_identity(tmp_path, field):
    from review.report.advice import theory_source_integrity
    from schemas.claim import TheoryVisualRecord
    from verification.contracts import BranchResult

    result, claim, _, _ = run(tmp_path)
    assert BranchResult.model_validate_json(result.model_dump_json()).model_dump() == result.model_dump()
    record = claim.theory_derivations[-1]
    value = record.model_dump(mode="json")
    value[field] = None
    with pytest.raises(ValueError, match="audit"):
        TheoryVisualRecord.model_validate(value)
    setattr(record, field, None)
    with pytest.raises(ValueError, match="audit"):
        theory_source_integrity(claim)


@pytest.mark.parametrize("failure", ["missing", "unreadable"])
def test_audit_hash_read_failure_cannot_deliver_validated_proof(tmp_path, monkeypatch, failure):
    from verification import theory_derivations

    original = theory_derivations._file_hash

    def fail(path):
        if Path(path).name.startswith("visual-"):
            if failure == "unreadable":
                raise OSError("Explicit read failure")
            Path(path).unlink(missing_ok=True)
            return None
        return original(path)

    monkeypatch.setattr(theory_derivations, "_file_hash", fail)
    result, claim, _, _ = run(tmp_path)
    assert claim.status == "unverified" and not result.evidence
    assert result.theory_derivations[-1].state == "invalid"
    assert result.theory_derivations[-1].audit_sha256 is None


@pytest.mark.parametrize("reflected", ["text", "vision", "both"])
def test_successful_visual_detail_is_safe_for_both_provider_configs(tmp_path, monkeypatch, reflected):
    from review.report.v2 import write_review
    from tests.test_report_v2 import review

    text_cfg = LLMConfig("mock", "text-model", None, "synthetic-text-key-f51d67")
    vision_cfg = LLMConfig("mock", "vision-model", None, "synthetic-vision-key-f39a16")
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: text_cfg)
    monkeypatch.setattr("screening.checks.resolve_vlm_config", lambda **kw: vision_cfg)
    reflected_keys = (
        [text_cfg.api_key]
        if reflected == "text"
        else [vision_cfg.api_key]
        if reflected == "vision"
        else [text_cfg.api_key, vision_cfg.api_key]
    )

    def mutate(raw, *args):
        raw["items"][0]["detail"] += " " + " ".join(reflected_keys)

    result, claim, _, raw = run(tmp_path, mutate)
    assert claim.status == "supported" and result.evidence[0].sufficient
    assert all(key in raw[0]["items"][0]["detail"] for key in reflected_keys)
    assert all(key not in claim.model_dump_json() for key in (text_cfg.api_key, vision_cfg.api_key))
    document = review()
    document.claims = [claim]
    document.findings = []
    outputs = write_review(document, tmp_path / "safe-report", render_pdf=False)
    for field in ("json", "markdown"):
        text = Path(outputs[field]).read_text(encoding="utf-8")
        assert text_cfg.api_key not in text and vision_cfg.api_key not in text


@pytest.mark.parametrize("mode", ["success", "invalid_keys", "failed_keys", "exception"])
def test_all_visual_audit_paths_copy_redact_both_configs(tmp_path, monkeypatch, mode):
    from common import run_stats

    text_cfg = LLMConfig("mock", "text", None, "synthetic-text-private-a112")
    vision_cfg = LLMConfig("mock", "vision", None, "synthetic-vision-private-b113")
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: text_cfg)
    monkeypatch.setattr("screening.checks.resolve_vlm_config", lambda **kw: vision_cfg)
    claim, materials, initial = case(tmp_path)
    raw_responses = []

    def model(**kw):
        if kw["module"] == "verification.theory":
            return copy.deepcopy(initial)
        if kw["module"] == "verification.theory.notation":
            return {"classification": "parser_artifact", "explanation": "Explicit mock."}
        if mode == "exception":
            raise RuntimeError(text_cfg.api_key + " " + vision_cfg.api_key)
        raw = recovered(json.loads(kw["prompt"]))
        raw["items"][0]["detail"] += " " + text_cfg.api_key + " " + vision_cfg.api_key
        if mode in {"invalid_keys", "failed_keys"}:
            raw[text_cfg.api_key] = "first retained value"
            raw[vision_cfg.api_key] = "second retained value"
        if mode == "failed_keys":
            raw["error"] = "explicit mock provider failure"
        raw_responses.append(raw)
        return raw

    with run_stats.run_scope(tmp_path / "run_stats.json"):
        result = verify_theory(claim, materials, call=model, output_dir=tmp_path / "audit")
    assert bool(result.evidence) == (mode == "success")
    audit_paths = list((tmp_path / "audit").glob("visual-*.json")) + list(
        (tmp_path / "visual_calls").glob("verification.theory.visual_recheck*.json")
    )
    assert len(audit_paths) == 2
    for path in audit_paths:
        saved = path.read_text(encoding="utf-8")
        assert text_cfg.api_key not in saved and vision_cfg.api_key not in saved
        if mode in {"invalid_keys", "failed_keys"}:
            assert "first retained value" in saved and "second retained value" in saved
            if "response" in json.loads(saved):
                assert "dictionary keys collided" in saved
    if raw_responses:
        assert text_cfg.api_key in raw_responses[0]["items"][0]["detail"]
        assert vision_cfg.api_key in raw_responses[0]["items"][0]["detail"]
        if mode in {"invalid_keys", "failed_keys"}:
            assert text_cfg.api_key in raw_responses[0] and vision_cfg.api_key in raw_responses[0]
    assert text_cfg.api_key not in result.model_dump_json()
    assert vision_cfg.api_key not in result.model_dump_json()


@pytest.mark.parametrize("field", ["transcription", "printed_anchor", "step", "assumption"])
@pytest.mark.parametrize("provider", ["text", "vision"])
def test_visual_source_and_trace_never_publish_provider_credentials(tmp_path, monkeypatch, field, provider):
    text_cfg = LLMConfig("mock", "text", None, "synthetic-text-private-c114")
    vision_cfg = LLMConfig("mock", "vision", None, "synthetic-vision-private-d115")
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: text_cfg)
    monkeypatch.setattr("screening.checks.resolve_vlm_config", lambda **kw: vision_cfg)
    reflected = text_cfg.api_key if provider == "text" else vision_cfg.api_key

    def mutate(raw, *args):
        if field in {"transcription", "printed_anchor"}:
            raw["visual_sources"][0][field] += " " + reflected
        elif field == "step":
            raw["items"][0]["trace"]["steps"][0]["statement"] += " " + reflected
        else:
            raw["items"][0]["trace"]["assumptions"][0]["text"] += " " + reflected

    result, claim, _, raw = run(tmp_path, mutate)
    assert reflected in json.dumps(raw)
    assert not result.evidence and claim.status == "unverified"
    assert result.theory_derivations[-1].state == "invalid"
    assert reflected not in result.model_dump_json()
