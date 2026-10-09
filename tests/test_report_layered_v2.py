"""Layered presentation retains canonical records and verifies real PDF navigation."""

import hashlib
import json
import re
from pathlib import Path

import pytest
from markdown_it import MarkdownIt
from pypdf import PdfReader

from review.report import v2
from schemas.claim import Claim, Condition, Evidence, EvidencePointer, Finding
from schemas.review import FinalReview


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Report presentation cannot call models, retrieval or execution")

    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("llm.client.llm_json", forbidden)
    monkeypatch.setattr("review.report.advice.llm_json", forbidden)


def fixture_review():
    evidence = []
    for index in range(5):
        evidence.append(
            Evidence(
                source="paper_internal",
                covered=["c1"],
                direction="support",
                sufficient=index == 3,
                pointer=EvidencePointer(
                    locator="original.pdf",
                    page=4,
                    key=f"result{index}",
                    quote=f"Result {index}: 90 versus 80.",
                ),
                note=f"Full diagnostic {index}. " * 25,
            )
        )
    evidence += [
        Evidence(
            source="paper_internal",
            covered=["c1"],
            direction="flaw",
            concern=True,
            sufficient=False,
            pointer=EvidencePointer(locator="original.pdf", page=5, quote="Conflicting conditions."),
        ),
        Evidence(
            source="paper_internal",
            covered=["c2"],
            direction="support",
            sufficient=False,
            pointer=EvidencePointer(locator="prior.pdf", page=2, quote="Prior scope."),
            additional_pointers=[
                EvidencePointer(locator="protocol.md", key="settings", quote="Test, single crop.")
            ],
        ),
        Evidence(
            source="paper_internal",
            covered=[],
            direction="flaw",
            affects_claim=False,
            sufficient=False,
            pointer=EvidencePointer(locator="original.pdf", page=7, quote="Unassigned observation."),
        ),
    ]
    first = Claim(
        id="z-original",
        text="A reaches 90 on D under the original evaluation conditions.",
        status="questioned",
        loc={"page": 4, "section": "Results"},
        importance="core",
        source_block_id="b4",
        source_quote="Exact primary assertion, including its qualifier.",
        source_refs=[
            {
                "source_block_id": "b5",
                "source_quote": "Dataset-specific choices may differ.",
                "loc": {"page": 5},
                "covered": ["c2"],
            }
        ],
        conditions=[
            Condition(id="c1", dataset="D", metric="accuracy", settings={"split": "test"}),
            Condition(id="c2", dataset="E", metric="accuracy", settings={"crops": 1}),
        ],
        needs=["Experiments", "Literature"],
        evidence=evidence,
        questions=[
            {
                "text": "Which settings produced this observation?",
                "reason": "Both directions remain recorded.",
            }
        ],
        notes=[
            "Raw candidate diagnostics remain complete. " * 200,
            "Second note with full trailing qualifier.",
        ],
    )
    other = Claim(
        id="a-original",
        text="The remaining condition has no assessed evidence.",
        status="unverified",
        loc={"page": 8},
        conditions=[Condition(id="q", description="Entire qualifier retained")],
        needs=["Code"],
    )
    finding = Finding(
        kind="reference",
        level="metadata_candidate",
        loc={"page": 9},
        text="Venue metadata differs; identity is unconfirmed.",
        evidence=[
            Evidence(
                source="paper_internal",
                direction="flaw",
                sufficient=False,
                covered=[],
                affects_claim=False,
                pointer=EvidencePointer(locator="check.json", key="ref2", quote="Original raw metadata."),
            )
        ],
    )
    return FinalReview(
        paper_key="layered offline",
        run_id="offline",
        claims=[first, other],
        findings=[finding],
        run_status="partial",
        incomplete_stages=["verification"],
        execution_requested=True,
        ledger=[
            {
                "plan": {"id": "p1", "claim_id": first.id, "run_mode": "evaluation"},
                "approval_mode": "auto",
                "approved": True,
                "training_budget": 0,
                "attempts": [{"status": "complete", "returncode": 0}],
                "alignment": [
                    {
                        "condition_id": "c1",
                        "aligned": True,
                        "resource_mode": "released_predictions",
                        "model_inference_performed": False,
                        "expected": 0.9,
                        "observed": 0.8,
                        "consistent": False,
                        "measurement": {"unit": "fraction", "sample_count": 10},
                    }
                ],
            }
        ],
    )


def read_outputs(outputs):
    assert not any(key.endswith("_error") for key in outputs), outputs
    return {
        key: Path(outputs[key]).read_text("utf-8")
        for key in ("markdown", "appendix_markdown", "bundle_markdown")
    }


def markdown_navigation(markdown):
    anchors, links = [], []
    for token in MarkdownIt("gfm-like", {"linkify": False}).parse(markdown):
        assert token.type != "html_block", token.content
        for child in token.children or []:
            assert child.type != "image"
            if child.type == "html_inline" and child.content != "</a>":
                match = re.fullmatch(
                    r'<a id="(factreview-(?:(?:source|record|main)-[a-f0-9]{64}|evidence-\d+(?:-source-\d+)?))">',
                    child.content,
                )
                assert match, child.content
                anchors.append(match.group(1))
            if child.type == "link_open":
                links.append(child.attrGet("href"))
    assert len(anchors) == len(set(anchors))
    return anchors, links


def pdf_navigation(path):
    pdf = PdfReader(path)
    pages = {p.indirect_reference.idnum: i + 1 for i, p in enumerate(pdf.pages)}
    links = []
    for origin, page in enumerate(pdf.pages, 1):
        for ref in page.get("/Annots", []):
            annotation = ref.get_object()
            if annotation.get("/Subtype") != "/Link":
                continue
            if "/A" in annotation:
                assert annotation["/A"]["/S"] == "/GoTo"
                dest = annotation["/A"]["/D"]
            else:
                dest = annotation["/Dest"]
            assert dest[0].idnum in pages
            assert dest[1] == "/XYZ" and 0 <= float(dest[3]) <= float(page.mediabox.height)
            links.append((origin, pages[dest[0].idnum]))
    return pdf, links


def test_layered_complete_records_selection_context_and_exact_source_links(tmp_path, monkeypatch):
    original = fixture_review()
    before = original.model_dump(mode="json")
    calls = []
    checked = v2._checked_report
    monkeypatch.setattr(v2, "_checked_report", lambda value: (calls.append(value), checked(value))[1])
    outputs = v2.write_review(
        original,
        tmp_path,
        presentation="layered",
        render_pdf=False,
        issues=["Raw provider diagnostic. " * 90],
        figure_coverage={"total": 3, "checked": 1, "failed": 2},
        token_usage={"requests": 5, "failed_requests": 2, "unavailable_usage_requests": 2},
    )
    texts = read_outputs(outputs)
    saved = json.loads(Path(outputs["json"]).read_text("utf-8"))
    manifest = json.loads(Path(outputs["manifest"]).read_text("utf-8"))
    assert len(calls) == 1 and original.model_dump(mode="json") == before
    assert {k: v for k, v in saved.items() if k != "review_markdown"} == {
        k: v for k, v in before.items() if k != "review_markdown"
    }
    assert saved["review_markdown"] == texts["markdown"]
    assert manifest["checked_records_equal_input"] and manifest["records_equal_saved_json"]
    assert manifest["counts"] == {
        "claims": 2,
        "conditions": 3,
        "findings": 1,
        "evidence": 9,
        "pointer_usages": 10,
        "notes": 2,
        "questions": 1,
        "ledger": 1,
        "unique_passage_sources": 10,
        "unique_pointer_sources": 10,
    }
    selected = manifest["selection"]["/claims/0"]["selected"]
    assert set(selected) == {"0", "3", "5", "6", "7"}
    assert manifest["selection"]["/claims/0"]["groups"][0]["count"] == 5
    main, appendix = texts["markdown"], texts["appendix_markdown"]
    for claim in original.claims:
        assert v2._text(claim.text) in main
        for condition in claim.conditions:
            assert condition.id in main
        for note in claim.notes:
            assert v2._text(note) in appendix and v2._text(note) not in main
        for item in claim.evidence:
            if item.note:
                assert v2._text(item.note) in appendix and v2._text(item.note) not in main
    assert v2._text(original.claims[0].source_quote) in appendix
    assert "Which settings produced this observation" in main
    assert "reference source; discrepancy unconfirmed" in main
    assert "unavailable for 2 call attempt" in main and "2 failed" in main
    assert (
        v2._text("released_predictions") in main
        and v2._text("model_inference_performed") in main
        and "false" in main
    )
    assert manifest["delivery_context"]["issues"][0] not in main
    assert len(main) < len(appendix)
    for text in texts.values():
        anchors, links = markdown_navigation(text)
        if text is texts["bundle_markdown"]:
            assert all(link.startswith("#") and link[1:] in anchors for link in links)
    appendix_anchors = set(markdown_navigation(appendix)[0])
    main_anchors = set(markdown_navigation(main)[0])
    for record in manifest["records"].values():
        assert record["appendix_anchor"] in appendix_anchors
        assert record["main_anchor"] in main_anchors
    for usage in manifest["source_usages"]:
        assert usage["usage_anchor"] in appendix_anchors
        assert usage["source_anchor"] in appendix_anchors
    for record in manifest["artifacts"].values():
        assert hashlib.sha256((tmp_path / record["name"]).read_bytes()).hexdigest() == record["sha256"]


def test_pdf_real_bidirectional_goto_and_standalone_page_entry(tmp_path):
    review = fixture_review()
    output = v2.write_review(review, tmp_path, presentation="layered")
    read_outputs(output)
    manifest = json.loads(Path(output["manifest"]).read_text("utf-8"))
    bundle, links = pdf_navigation(output["bundle_pdf"])
    main, _ = pdf_navigation(output["pdf"])
    appendix, _ = pdf_navigation(output["appendix_pdf"])
    boundary = manifest["pdf_targets"]["bundle"][manifest["records"][""]["appendix_anchor"]]
    assert len(bundle.pages) > 3 and len(appendix.pages) > 1
    assert any(a < boundary <= b for a, b in links)
    assert any(b < boundary <= a for a, b in links)
    for record in manifest["records"].values():
        assert record["main_pdf_page"] and record["appendix_pdf_page"]
        assert record["bundle_main_page"] and record["bundle_appendix_page"]
    main_text = "\n".join(p.extract_text() or "" for p in main.pages)
    claim_record = manifest["records"]["/claims/0/source_quote"]
    assert f"technical_appendix.pdf, page {claim_record['appendix_pdf_page']}" in main_text
    assert not any("/URI" in str(ref.get_object()) for p in bundle.pages for ref in p.get("/Annots", []))


def test_advice_checked_once_valid_text_retained_invalid_basis_unavailable(tmp_path, monkeypatch):
    from llm.client import LLMConfig
    from tests.test_report_advice_v2 import generate, review

    monkeypatch.setattr(
        "review.report.advice.resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None)
    )
    result = generate(review(tmp_path), tmp_path).review
    result.claims[1].advice.items[0].basis_refs = ["/evidence/999"]
    before = result.model_dump(mode="json")
    output = v2.write_review(result, tmp_path / "layered", presentation="layered", render_pdf=False)
    text = read_outputs(output)["markdown"]
    saved = FinalReview.model_validate_json(Path(output["json"]).read_text("utf-8"))
    assert saved.claims[1].advice.state == "unavailable" and saved.claims[1].advice.items == []
    for i in (0, 2, 3):
        assert saved.claims[i].advice == result.claims[i].advice
        assert v2._text(result.claims[i].advice.items[0].text) in text
    assert "Advice unavailable" in text and result.model_dump(mode="json") == before
    manifest = json.loads(Path(output["manifest"]).read_text("utf-8"))
    assert manifest["checked_changes"] == {"advice_claim_indices": [1], "reference_finding_indices": []}


def test_malicious_text_cannot_create_anchors_links_or_images(tmp_path):
    original = fixture_review()
    attack = (
        '<a id="factreview-main-'
        + "0" * 64
        + '"></a> [go](https://invalid.example) ![image](remote.png)\n# injected'
    )
    original.claims[0].text = attack
    original.claims[0].notes.append(attack)
    original.claims[0].evidence[0].pointer.quote = attack
    result = v2.write_review(original, tmp_path, presentation="layered")
    texts = read_outputs(result)
    for text in texts.values():
        anchors, links = markdown_navigation(text)
        assert "factreview-main-" + "0" * 64 not in anchors
        assert all("https:" not in link and "remote.png" not in link for link in links)
    pdf_navigation(result["bundle_pdf"])


def test_unknown_presentation_fails_without_creating_outputs(tmp_path):
    with pytest.raises(ValueError, match="presentation"):
        v2.write_review(fixture_review(), tmp_path / "absent", presentation="automatic")
    assert not (tmp_path / "absent").exists()


def test_pdf_failure_visible_complete_json_md_still_delivered(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("render failed")

    monkeypatch.setattr("review.report.compact._build_pdf", fail)
    output = v2.write_review(fixture_review(), tmp_path, presentation="layered")
    assert set(k for k in output if k.endswith("_error")) == {
        "pdf_error",
        "appendix_pdf_error",
        "bundle_pdf_error",
    }
    manifest = json.loads(Path(output["manifest"]).read_text("utf-8"))
    assert manifest["records_equal_saved_json"] and manifest["render_errors"]
    assert all(value["appendix_pdf_page"] is None for value in manifest["records"].values())


def test_empty_partial_review_and_full_default_compatibility(tmp_path):
    review = FinalReview(
        paper_key="partial",
        run_id="offline",
        run_status="partial",
        incomplete_stages=["screening", "execution"],
    )
    default = v2.write_review(review, tmp_path / "default", render_pdf=False)
    full = v2.write_review(review, tmp_path / "full", render_pdf=False, presentation="full")
    assert set(default) == {"markdown", "json"}
    assert Path(default["markdown"]).read_bytes() == Path(full["markdown"]).read_bytes()
    layered = v2.write_review(review, tmp_path / "layered", render_pdf=False, presentation="layered")
    text = read_outputs(layered)["markdown"]
    assert "partial" in text and "No complete execution records" in text
    assert json.loads(Path(layered["manifest"]).read_text("utf-8"))["counts"]["claims"] == 0


@pytest.mark.parametrize(
    "failed_title,key",
    [
        ("Technical appendix", "appendix_pdf"),
        ("Reading report", "pdf"),
        ("Reading report and technical appendix", "bundle_pdf"),
    ],
)
def test_individual_pdf_failure_is_visible_without_cross_file_fake_links(
    tmp_path, monkeypatch, failed_title, key
):
    from review.report import compact

    real = compact._build_pdf

    def selectively_fail(review, markdown, context, targets, title):
        if title == failed_title:
            raise RuntimeError("selected export failed")
        return real(review, markdown, context, targets, title)

    monkeypatch.setattr(compact, "_build_pdf", selectively_fail)
    outputs = v2.write_review(fixture_review(), tmp_path, presentation="layered")
    assert key not in outputs and key + "_error" in outputs
    manifest = json.loads(Path(outputs["manifest"]).read_text("utf-8"))
    assert list(manifest["render_errors"]) == [key + "_error"]
    for present in {"pdf", "appendix_pdf", "bundle_pdf"} - {key}:
        pdf_navigation(outputs[present])
    if key == "appendix_pdf":
        assert all(v["appendix_pdf_page"] is None for v in manifest["records"].values())
        pdf = PdfReader(outputs["pdf"])
        assert "technical_appendix.pdf, page" not in " ".join(p.extract_text() or "" for p in pdf.pages)


def test_existing_pdf_prevents_stale_artifact_delivery(tmp_path):
    stale = tmp_path / "review_bundle.pdf"
    stale.write_bytes(b"previous export retained")
    with pytest.raises(FileExistsError, match="fresh output directory"):
        v2.write_review(fixture_review(), tmp_path, presentation="layered")
    assert stale.read_bytes() == b"previous export retained"
    assert not (tmp_path / "report_manifest.json").exists()


def test_long_cjk_claim_and_spanned_table_keep_sources_values_and_pdf_targets(tmp_path):
    review = fixture_review()
    review.claims[0].text = "逐项验证原始限定和证据来源。" * 100
    table = '<table><tr><th rowspan="2">Model</th><th colspan="2">D test</th></tr><tr><th>accuracy (%)</th><th>latency (ms)</th></tr>'
    table += (
        "".join(f"<tr><td>A{i}</td><td>{90 + i / 100:.2f}</td><td>{100 + i}</td></tr>" for i in range(80))
        + "</table>"
    )
    review.claims[0].evidence[0].pointer.quote = table
    output = v2.write_review(review, tmp_path, presentation="layered")
    text = read_outputs(output)["appendix_markdown"]
    assert "<td>" not in text and v2._text("90.79") in text and "179" in text
    pdf, links = pdf_navigation(output["bundle_pdf"])
    assert links and len(pdf.pages) > 5
    full = "\n".join(p.extract_text() or "" for p in pdf.pages)
    assert "90.79" in full and "latency" in full and "accuracy" in full
    manifest = json.loads(Path(output["manifest"]).read_text("utf-8"))
    assert all(v["bundle_appendix_page"] for v in manifest["records"].values())


def test_all_structured_limitations_and_requested_author_questions_stay_in_main(tmp_path):
    from schemas.limitations import VerificationLimitation

    review = fixture_review()
    review.claims[0].verification_limitations = [
        VerificationLimitation(
            claim_id=review.claims[0].id,
            condition_ids=["c2"],
            stage="verification",
            kind="branch_failed",
            reason="The returned trace did not complete; this remains a system failure.",
        )
    ]
    output = v2.write_review(review, tmp_path, presentation="layered", render_pdf=False)
    main = read_outputs(output)["markdown"]
    for value in review.claims[0].verification_limitations[0].model_dump(mode="json").values():
        if isinstance(value, str):
            assert v2._text(value) in main


def test_duplicate_and_empty_passages_keep_distinct_usage_navigation(tmp_path):
    review = fixture_review()
    review.claims[0].evidence[1].pointer = review.claims[0].evidence[0].pointer.model_copy(deep=True)
    review.claims[0].evidence[2].pointer.quote = ""
    output = v2.write_review(review, tmp_path, presentation="layered")
    read_outputs(output)
    manifest = json.loads(Path(output["manifest"]).read_text("utf-8"))
    assert manifest["counts"]["pointer_usages"] == 10
    assert manifest["counts"]["unique_pointer_sources"] == 9
    assert manifest["counts"]["unique_passage_sources"] == 8
    usages = {x["json_pointer"]: x for x in manifest["source_usages"]}
    first, repeated = (usages[f"/claims/0/evidence/{i}/pointer"] for i in (0, 1))
    assert first["source_anchor"] == repeated["source_anchor"]
    assert first["usage_anchor"] != repeated["usage_anchor"]
    assert usages["/claims/0/evidence/2/pointer"]["source_anchor"] is None
    for row in usages.values():
        assert row["usage_anchor"] in manifest["pdf_targets"]["appendix"]
    pdf_navigation(output["bundle_pdf"])


def test_opposed_sufficient_records_remain_visible_without_reassessment(tmp_path):
    review = fixture_review()
    flaw = review.claims[0].evidence[5]
    flaw.sufficient, flaw.overturnable, flaw.concern = True, False, False
    before = review.model_dump(mode="json", exclude={"review_markdown"})
    output = v2.write_review(review, tmp_path, presentation="layered", render_pdf=False)
    manifest = json.loads(Path(output["manifest"]).read_text("utf-8"))
    selected = manifest["selection"]["/claims/0"]["selected"]
    assert "recorded sufficient and affects_claim" in selected["3"]
    assert "recorded sufficient and affects_claim" in selected["5"]
    saved = json.loads(Path(output["json"]).read_text("utf-8"))
    assert {k: v for k, v in saved.items() if k != "review_markdown"} == before
    assert saved["claims"][0]["status"] == "questioned"
