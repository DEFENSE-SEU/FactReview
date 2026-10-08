"""Reference parser mismatches need original-PDF confirmation before findings."""

import copy
import json
from pathlib import Path

import pytest
from PIL import Image

from common import run_stats
from llm.client import LLMConfig
from schemas.claim import ClaimLocation
from schemas.materials import MaterialBlock, PageImage, SharedMaterials
from screening.references import check_bibliography


@pytest.fixture
def reference_input(tmp_path, monkeypatch):
    text = "Ankur P Parikh, Oscar Tackstr¨ om, Dipanjan Das, and¨ Jakob Uszkoreit. 2016. A decomposable attention model for natural language inference."
    text += " doi:10.18653/v1/D16-1244."
    markdown = tmp_path / "paper.md"
    markdown.write_text(text, encoding="utf-8")
    image = tmp_path / "original-page.png"
    Image.new("RGB", (600, 800), "white").save(image)
    block = MaterialBlock(id="ref1", text=text, loc=ClaimLocation(page=4, char_start=0, char_end=len(text)))
    materials = SharedMaterials(
        paper_key="references",
        source_pdf="paper.pdf",
        markdown=text,
        markdown_path=str(markdown),
        content_list_path="content.json",
        provider="fixture",
        blocks=[block],
        bibliography=[block],
        pages=[PageImage(page=4, path=str(image), width_points=600, height_points=800)],
    )
    row = {
        "severity": "error",
        "type": "hallucination::author_mismatch",
        "reference_title": "A decomposable attention model for natural language inference",
        "cited_url": "https://doi.org/10.18653/v1/D16-1244",
        "verified_url": "https://doi.org/10.18653/v1/D16-1244",
        "details": "Retrieved authors: Ankur P. Parikh, Oscar Täckström, Dipanjan Das, Jakob Uszkoreit",
        "raw_reference": text,
    }
    backend = {"ok": True, "total_refs": 1, "issues": [row]}
    monkeypatch.setattr(
        "screening.checks.resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None)
    )
    monkeypatch.setattr("screening.checks.resolve_vlm_config", lambda **kwargs: kwargs["fallback"])
    return materials, backend


def decision(**changes):
    return {
        "candidate_id": "reference_1",
        "page": 4,
        "classification": "manuscript_error",
        "printed_quote": "Oscar Other",
        "comparison_quote": "Oscar Täckström",
        "mismatch_kind": "wrong_author",
        "reason": "The printed author surname differs from the retrieved author.",
        **changes,
    }


def set_original_identifier(materials, backend, original_identifier):
    block = materials.bibliography[0]
    block.text = block.text.split(" doi:", 1)[0] + original_identifier
    block.loc.char_end = len(block.text)
    materials.markdown = block.text
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
    backend["issues"][0]["raw_reference"] = block.text


@pytest.mark.parametrize("classification", ["parser_artifact", "uncertain"])
def test_parser_accent_mismatch_never_becomes_manuscript_error(reference_input, tmp_path, classification):
    materials, backend = reference_input
    original = copy.deepcopy(backend)

    def model(**request):
        assert request["images"] == [materials.pages[0].path]
        data = json.loads(request["prompt"])
        assert data["page"] == 4
        assert data["candidates"][0]["parsed_reference"] == materials.bibliography[0].text
        assert "Oscar Täckström" in data["candidates"][0]["comparison_metadata"]
        return {
            "results": [
                decision(
                    classification=classification,
                    printed_quote="Oscar Täckström",
                    reason="Original accents agree; parser displaced them.",
                )
            ]
        }

    with run_stats.run_scope(tmp_path / "stats.json"):
        findings, issues = check_bibliography(
            materials, tmp_path / "out", checker=lambda **kw: backend, call=model
        )
    assert findings == [] and any(classification in issue for issue in issues)
    assert json.loads((tmp_path / "out/reference_check.json").read_text(encoding="utf-8")) == original
    assert backend == original
    assert materials.markdown == original["issues"][0]["raw_reference"]
    audit = json.loads((tmp_path / "out/reference_validation.json").read_text(encoding="utf-8"))
    assert audit[0]["status"] == classification
    assert audit[0]["decision"]["printed_quote"] == "Oscar Täckström"
    assert len(list((tmp_path / "visual_calls").glob("*.json"))) == 1


@pytest.mark.parametrize("kind", ["author_mismatch", "title_mismatch"])
def test_confirmed_reference_mismatch_retains_pdf_pixels_and_comparison(reference_input, tmp_path, kind):
    materials, backend = reference_input
    backend["issues"][0]["type"] = "hallucination::" + kind
    expected = decision()
    if kind == "title_mismatch":
        canonical = "A decomposable attention model for natural language inference"
        backend["issues"][0]["details"] = "Retrieved title: " + canonical
        expected = decision(
            printed_quote="A decomposable learning model for natural language inference",
            comparison_quote=canonical,
            mismatch_kind="title",
            reason="The printed title uses learning where the retrieved title uses attention.",
        )
    findings, issues = check_bibliography(
        materials,
        tmp_path,
        checker=lambda **kw: backend,
        call=lambda **kw: {"results": [expected]},
    )
    assert issues == [] and len(findings) == 1
    finding = findings[0]
    assert finding.kind == "reference" and finding.level == "error"
    assert "reference_check.json#issues.0" in finding.text
    assert finding.evidence[-1].pointer.locator == materials.pages[0].path
    assert finding.evidence[-1].pointer.quote == expected["printed_quote"]
    assert finding.evidence[-1].pointer.page == 4
    assert expected["comparison_quote"] in finding.evidence[-1].note
    assert finding.evidence[-2].pointer.locator == backend["issues"][0]["verified_url"]
    assert "doi:10.18653/v1/d16-1244" in finding.evidence[-2].note
    assert "external accuracy was not independently rechecked" in finding.text
    assert all(not evidence.sufficient and not evidence.affects_claim for evidence in finding.evidence)


@pytest.mark.parametrize(
    "change",
    [
        {"printed_quote": ""},
        {"comparison_quote": ""},
        {"comparison_quote": "Invented canonical author"},
        {"printed_quote": "Oscar Täckström"},
    ],
)
def test_classification_alone_cannot_confirm_reference_error(reference_input, tmp_path, change):
    materials, backend = reference_input
    findings, issues = check_bibliography(
        materials,
        tmp_path,
        checker=lambda **kw: backend,
        call=lambda **kw: {"results": [decision(**change)]},
    )
    assert findings == [] and any("uncertain" in issue for issue in issues)


@pytest.mark.parametrize(
    "printed", ["Ankur P Parikh", "ANKUR P. PARIKH", "Ankur P.\nParikh", "Ａｎｋｕｒ Ｐ． Ｐａｒｉｋｈ"]
)
def test_format_only_difference_cannot_be_upgraded_by_model_classification(
    reference_input, tmp_path, printed
):
    materials, backend = reference_input
    findings, issues = check_bibliography(
        materials,
        tmp_path,
        checker=lambda **kw: backend,
        call=lambda **kw: {"results": [decision(printed_quote=printed, comparison_quote="Ankur P. Parikh")]},
    )
    assert findings == [] and any("uncertain" in issue for issue in issues)
    audit = json.loads((tmp_path / "reference_validation.json").read_text(encoding="utf-8"))
    assert audit[0]["status"] == "uncertain"


@pytest.mark.parametrize(
    "response",
    [
        {"results": []},
        {"results": [decision(candidate_id="foreign")]},
        {"results": [decision(), decision()]},
        {"results": [decision(page=5)]},
        {"status": "error", "error": "vision unavailable"},
    ],
)
def test_failed_or_malformed_reference_confirmation_stays_unconfirmed(reference_input, tmp_path, response):
    materials, backend = reference_input
    findings, issues = check_bibliography(
        materials, tmp_path, checker=lambda **kw: backend, call=lambda **kw: response
    )
    assert findings == [] and any("PDF reference validation failed" in issue for issue in issues)
    audit = json.loads((tmp_path / "reference_validation.json").read_text(encoding="utf-8"))
    assert audit[0]["status"] == "failed"


def test_missing_original_page_retains_unavailable_reason_without_vlm(reference_input, tmp_path):
    materials, backend = reference_input
    Path(materials.pages[0].path).unlink()
    findings, issues = check_bibliography(
        materials,
        tmp_path,
        checker=lambda **kw: backend,
        call=lambda **kw: pytest.fail("No page pixels to inspect"),
    )
    assert findings == [] and any("page image unavailable" in issue for issue in issues)
    audit = json.loads((tmp_path / "reference_validation.json").read_text(encoding="utf-8"))
    assert audit[0]["status"] == "unavailable"


@pytest.mark.parametrize(
    "cited_url,verified_url",
    [
        ("", ""),
        ("", "https://doi.org/10.18653/v1/D16-1244"),
        ("https://doi.org/10.18653/v1/D16-1244", ""),
        ("https://doi.org/10.18653/v1/D16-1244", "https://doi.org/10.18653/v1/D16-1111"),
        ("https://arxiv.org/abs/1606.01933v1", "https://arxiv.org/abs/1606.01933v2"),
        ("https://arxiv.org/abs/1606.01933", "https://arxiv.org/abs/1606.01933v1"),
        ("https://arxiv.org/abs/1606.01933", "https://arxiv.org/abs/1606.01933"),
        ("https://arxiv.org/abs/1606.01933v1", "https://arxiv.org/abs/1606.01934v1"),
        ("https://doi.org/10.18653/v1/D16-1244", "https://example.com/10.18653/v1/D16-1244"),
        ("https://doi.org/10.18653/v1/D16-1244", "https://arxiv.org/search?doi=10.18653/v1/D16-1244"),
        ("https://doi.org/10.18653/v1/D16-1244", "https://doi.org/search?doi=10.18653/v1/D16-1244"),
    ],
)
def test_model_cannot_upgrade_missing_or_different_work_version_identity(
    reference_input, tmp_path, cited_url, verified_url
):
    materials, backend = reference_input
    backend["issues"][0].update(cited_url=cited_url, verified_url=verified_url)
    set_original_identifier(materials, backend, " " + cited_url if cited_url else "")
    findings, issues = check_bibliography(
        materials, tmp_path, checker=lambda **kw: backend, call=lambda **kw: {"results": [decision()]}
    )
    assert findings == [] and any("Same-work/version identity unavailable" in issue for issue in issues)
    audit = json.loads((tmp_path / "reference_validation.json").read_text(encoding="utf-8"))
    assert audit[0]["status"] == "uncertain" and audit[0]["identity"]["qualified"] is False
    assert audit[0]["decision"]["classification"] == "manuscript_error"


@pytest.mark.parametrize(
    "suffix,cited_url,verified_url,identifier",
    [
        (
            " DOI: 10.18653/v1/D16-1244.",
            "",
            "https://doi.org/10.18653/v1/d16-1244",
            "doi:10.18653/v1/d16-1244",
        ),
        (" (doi:10.1000/AbC(def)).", "", "https://doi.org/10.1000/abc%28def%29", "doi:10.1000/abc(def)"),
        (
            "",
            "https://DX.DOI.ORG/10.18653%2Fv1%2FD16-1244",
            "https://doi.org/10.18653/v1/d16-1244",
            "doi:10.18653/v1/d16-1244",
        ),
        (" arXiv:1606.01933v1.", "", "https://arxiv.org/pdf/1606.01933v1.pdf", "arxiv:1606.01933v1"),
        (" (arXiv:cs/9901001v2).", "", "https://arxiv.org/abs/cs/9901001v2", "arxiv:cs/9901001v2"),
    ],
)
def test_identifiers_preserve_version_and_normalize_citation_punctuation(
    reference_input, tmp_path, suffix, cited_url, verified_url, identifier
):
    materials, backend = reference_input
    set_original_identifier(materials, backend, suffix or " " + cited_url)
    backend["issues"][0].update(cited_url=cited_url, verified_url=verified_url)
    findings, issues = check_bibliography(
        materials, tmp_path, checker=lambda **kw: backend, call=lambda **kw: {"results": [decision()]}
    )
    assert not issues and len(findings) == 1
    audit = json.loads((tmp_path / "reference_validation.json").read_text(encoding="utf-8"))
    assert audit[0]["identity"]["qualified"] and audit[0]["identity"]["identifier"] == identifier


@pytest.mark.parametrize(
    "kind,details,printed,comparison",
    [
        (
            "title",
            "No retrieved record has the cited title. (Closest match: The Sixth PASCAL Recognizing Textual Entailment Challenge)",
            "The fifth PASCAL recognizing textual entailment challenge.",
            "The Sixth PASCAL Recognizing Textual Entailment Challenge",
        ),
        (
            "title",
            "No retrieved record has the cited title. (Closest match: Gaussian Error Linear Units (GELUs))",
            "Bridging nonlinearities and stochastic regularizers with gaussian error linear units.",
            "Gaussian Error Linear Units (GELUs)",
        ),
        (
            "author_order",
            "A paper with this title exists, but no retrieved version has the cited author list (names and order). (Retrieved authors: P. Brown, V. D. Della Pietra, P. DeSouza, Jenifer C. Lai, R. Mercer)",
            "Peter F Brown, Peter V Desouza, Robert L Mercer, Vincent J Della Pietra, and Jenifer C Lai.",
            "P. Brown, V. D. Della Pietra, P. DeSouza, Jenifer C. Lai, R. Mercer",
        ),
    ],
)
def test_real_export_shape_without_identity_retains_visible_difference_as_uncertain(
    reference_input, tmp_path, kind, details, printed, comparison
):
    materials, backend = reference_input
    set_original_identifier(materials, backend, "")
    backend["issues"][0].update(
        type="hallucination::" + ("title_mismatch" if kind == "title" else "author_mismatch"),
        reference_year="2016",
        cited_url="",
        verified_url="",
        details=details,
        corrected_plaintext="",
        corrected_bibtex="",
        corrected_bibitem="",
    )
    findings, issues = check_bibliography(
        materials,
        tmp_path,
        checker=lambda **kw: backend,
        call=lambda **kw: {
            "results": [decision(mismatch_kind=kind, printed_quote=printed, comparison_quote=comparison)]
        },
    )
    assert not findings and any("uncertain" in issue for issue in issues)
    audit = json.loads((tmp_path / "reference_validation.json").read_text(encoding="utf-8"))
    assert audit[0]["decision"]["mismatch_kind"] == kind
    assert audit[0]["decision"]["printed_quote"] == printed


@pytest.mark.parametrize("kind", ["wrong_author", "author_order", "unknown", "title"])
def test_bound_author_difference_requires_explicit_difference_kind(reference_input, tmp_path, kind):
    materials, backend = reference_input
    expected = decision(mismatch_kind=kind)
    if kind == "author_order":
        expected.update(
            printed_quote="Oscar Täckström, Ankur P. Parikh, Dipanjan Das, Jakob Uszkoreit",
            comparison_quote="Ankur P. Parikh, Oscar Täckström, Dipanjan Das, Jakob Uszkoreit",
            reason="The first two authors are in reversed order in the printed citation.",
        )
    findings, issues = check_bibliography(
        materials,
        tmp_path,
        checker=lambda **kw: backend,
        call=lambda **kw: {"results": [expected]},
    )
    if kind in {"wrong_author", "author_order"}:
        assert len(findings) == 1 and not issues
        assert findings[0].text.startswith(kind + ":")
    else:
        assert not findings and any("uncertain" in issue for issue in issues)


def test_extractor_cited_url_cannot_supply_an_identifier_absent_from_original(reference_input, tmp_path):
    materials, backend = reference_input
    set_original_identifier(materials, backend, "")
    assert backend["issues"][0]["cited_url"] == backend["issues"][0]["verified_url"]
    findings, issues = check_bibliography(
        materials, tmp_path, checker=lambda **kw: backend, call=lambda **kw: {"results": [decision()]}
    )
    assert not findings and any("Same-work/version identity unavailable" in issue for issue in issues)
    audit = json.loads((tmp_path / "reference_validation.json").read_text(encoding="utf-8"))
    assert audit[0]["identity"]["cited_identifiers"] == []
    assert audit[0]["identity"]["qualified"] is False
