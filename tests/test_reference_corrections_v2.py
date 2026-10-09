"""Reference replacement identity and provenance are deterministic, local contracts."""

import json

import pytest

# Bootstrap the vendored public API through its production wrapper.
# isort: off
from fact_generation.refcheck.refcheck import reference_records
from refcopilot.report import to_factreview_dict

# isort: on
from schemas.claim import ClaimLocation
from schemas.materials import MaterialBlock, SharedMaterials
from screening.reference_corrections import bound_record, build_correction
from screening.references import check_bibliography
from tests.test_reference_records_v2 import offline, report  # noqa: F401


def files(tmp_path, value):
    payload = to_factreview_dict(value)
    text = value.checked[0].reference.raw
    (tmp_path / "bibliography.txt").write_text(text, "utf-8")
    records = reference_records(value, payload, __import__("hashlib").sha256(text.encode()).hexdigest())
    result_path = tmp_path / "reference_check.json"
    records_path = tmp_path / "reference_records.json"
    result_path.write_text(json.dumps(payload), "utf-8")
    records_path.write_text(json.dumps(records), "utf-8")
    block = MaterialBlock(id="r1", text=text, loc=ClaimLocation(page=1, char_start=0, char_end=len(text)))
    return payload, records, block, result_path, records_path


def correction(data):
    payload, _, block, result_path, records_path = data
    return build_correction(payload["issues"][0], block, 0, result_path, records_path)


def test_same_doi_public_metadata_binds_every_field_with_original_bibtex(tmp_path):
    data = files(tmp_path, report())
    result = correction(data)
    assert result.state == "metadata_candidate"
    assert result.identity_identifier == "doi:10.1000/one"
    assert result.corrected_bibtex == data[1]["bindings"][0]["corrected_bibtex"]
    assert "\n" in result.corrected_bibtex
    assert {x.field for x in result.fields} == {"title", "authors", "year", "venue", "doi", "url"}
    assert all(x.record_id == "record-1" for x in result.fields)
    assert result.verified_url.startswith("https://www.semanticscholar.org/")


@pytest.mark.parametrize(
    "change",
    [
        "wrong_doi",
        "missing_original_id",
        "unknown_provenance",
        "wrong_field",
        "duplicate_backend",
        "closest_match",
        "retracted",
        "missing_sources",
        "changed_input",
        "tampered_sidecar",
        "tampered_bibtex",
        "missing_sidecar",
        "legacy_dict",
    ],
)
def test_invalid_identity_or_provenance_is_not_a_replacement(tmp_path, change):
    value = report()
    c = value.checked[0]
    if change == "wrong_doi":
        c.merged.sources[0].doi = "10.1000/other"
    if change == "missing_original_id":
        c.reference.raw = "Ada Author. 2019. A grounded paper."
    if change == "unknown_provenance":
        c.merged.provenance.pop("year")
    if change == "wrong_field":
        c.merged.sources[0].year = 2021
    if change == "duplicate_backend":
        c.merged.sources.append(c.merged.sources[0].model_copy(update={"record_id": "other"}))
    if change == "closest_match":
        c.issues[0].message = "Closest match: A grounded paper"
    if change == "retracted":
        c.merged.is_retracted = True
    if change == "missing_sources":
        c.merged.sources = []
    data = files(tmp_path, value)
    if change == "changed_input":
        data[2].text += " Changed."
    if change == "tampered_sidecar":
        data[1]["report"]["checked"][0]["merged"]["year"] = 2022
        data[4].write_text(json.dumps(data[1]), "utf-8")
    if change == "tampered_bibtex":
        data[1]["bindings"][0]["corrected_bibtex"] += "\n@article{injected,doi={10.1000/wrong}}"
        data[4].write_text(json.dumps(data[1]), "utf-8")
    if change == "missing_sidecar":
        data[4].unlink()
    if change == "legacy_dict":
        data = (*data[:4], None)
    result = correction(data)
    assert result.state == "unavailable" and result.reason
    assert not result.corrected_bibtex and not result.fields


@pytest.mark.parametrize(
    "original_version,source_version,expected", [("v1", "v1", True), ("v1", "v2", False), ("", "v1", False)]
)
def test_arxiv_version_is_not_guessed_from_latest(original_version, source_version, expected, tmp_path):
    value = report()
    c = value.checked[0]
    source = c.merged.sources[0]
    c.reference.raw = f"Ada Author. A grounded paper. arXiv:2401.12345{original_version}."
    c.reference.doi = None
    c.reference.arxiv_id = "2401.12345"
    c.merged.doi = None
    c.merged.arxiv_id = "2401.12345"
    c.merged.url = f"https://arxiv.org/abs/2401.12345{source_version}"
    source.doi = None
    source.arxiv_id = "2401.12345"
    source.url = c.merged.url
    c.merged.provenance.pop("doi")
    c.merged.provenance["arxiv_id"] = source.backend
    result = correction(files(tmp_path, value))
    assert (result.state == "metadata_candidate") is expected


def test_public_url_can_have_uniquely_recorded_source_without_merge_provenance(tmp_path):
    value = report()
    value.checked[0].merged.provenance.pop("url")
    result = correction(files(tmp_path, value))
    assert result.state == "metadata_candidate"
    assert next(x for x in result.fields if x.field == "url").source_field == "url"


def test_bound_record_rejects_wrong_export_index(tmp_path):
    data = files(tmp_path, report())
    with pytest.raises(ValueError):
        bound_record(data[0]["issues"][0], data[2], 1, data[3], data[4])


def test_actual_arxiv_published_shape_with_printed_venue_is_unconfirmed(tmp_path):
    text = "[1] Ashish Vaswani et al. 2017. Attention Is All You Need. Advances in Neural Information Processing Systems 30. https://arxiv.org/abs/1706.03762"
    block = MaterialBlock(id="r1", text=text, loc=ClaimLocation(page=2, char_start=0, char_end=len(text)))
    paper = tmp_path / "paper.md"
    paper.write_text(text, "utf-8")
    materials = SharedMaterials(
        paper_key="fixture",
        source_pdf="paper.pdf",
        markdown=text,
        markdown_path=str(paper),
        content_list_path="",
        provider="mock",
        blocks=[block],
        bibliography=[block],
    )
    row = {
        "type": "outdated::arxiv_published",
        "severity": "warning",
        "reference_title": "Attention Is All You Need",
        "raw_reference": text,
        "details": "Paper was published at venue 'Neural Information Processing Systems' but cited as an arXiv preprint.",
        "corrected_bibtex": "@article{suggestion,year={2017}}",
        "verified_url": "https://arxiv.org/abs/1706.03762v7",
    }
    findings, issues = check_bibliography(
        materials, tmp_path / "check", checker=lambda **kw: {"ok": True, "total_refs": 1, "issues": [row]}
    )
    assert findings == []
    assert any("already" in i and "venue" in i for i in issues)
    assert json.loads((tmp_path / "check/reference_check.json").read_text("utf-8"))["issues"] == [row]


def test_screening_keeps_raw_export_and_serializes_independent_multiline_correction(tmp_path):
    from fact_generation.refcheck.refcheck import ReferenceCheckBundle
    from schemas.claim import Finding

    value = report()
    payload, records, block, _, _ = files(tmp_path, value)
    paper = tmp_path / "paper.md"
    paper.write_text(block.text, "utf-8")
    materials = SharedMaterials(
        paper_key="fixture",
        source_pdf="paper.pdf",
        markdown=block.text,
        markdown_path=str(paper),
        content_list_path="",
        provider="mock",
        blocks=[block],
        bibliography=[block],
    )
    findings, issues = check_bibliography(
        materials, tmp_path / "screen", checker=lambda **kw: ReferenceCheckBundle(payload, records)
    )
    assert not issues and len(findings) == 1
    serialized = findings[0].model_dump(mode="json")
    assert serialized["reference_correction"]["state"] == "metadata_candidate"
    assert (
        serialized["reference_correction"]["corrected_bibtex"] == records["bindings"][0]["corrected_bibtex"]
    )
    assert "@article" not in findings[0].text
    assert Finding.model_validate(serialized).reference_correction == findings[0].reference_correction
    assert json.loads((tmp_path / "screen/reference_check.json").read_text("utf-8")) == payload
    assert json.loads((tmp_path / "screen/reference_records.json").read_text("utf-8")) == records


def test_same_title_references_bind_raw_entries_not_title_substrings(tmp_path):
    from fact_generation.refcheck.refcheck import ReferenceCheckBundle

    value = report()
    value.checked.append(report(doi="10.1000/two").checked[0])
    value.summary.total_refs = 2
    payload = to_factreview_dict(value)
    text = "\n\n".join(c.reference.raw for c in value.checked)
    records = reference_records(value, payload, __import__("hashlib").sha256(text.encode()).hexdigest())
    paper = tmp_path / "paper.md"
    paper.write_text(text, "utf-8")
    blocks = [
        MaterialBlock(id=f"r{i}", text=c.reference.raw, loc=ClaimLocation(page=i + 1))
        for i, c in enumerate(value.checked)
    ]
    materials = SharedMaterials(
        paper_key="fixture",
        source_pdf="paper.pdf",
        markdown=text,
        markdown_path=str(paper),
        content_list_path="",
        provider="mock",
        blocks=blocks,
        bibliography=blocks,
    )
    findings, issues = check_bibliography(
        materials, tmp_path / "screen", checker=lambda **kw: ReferenceCheckBundle(payload, records)
    )
    assert not issues and len(findings) == 2
    assert [f.reference_correction.identity_identifier for f in findings] == [
        "doi:10.1000/one",
        "doi:10.1000/two",
    ]


def test_missing_original_pdf_preserves_metadata_candidate_without_error_upgrade(tmp_path):
    from fact_generation.refcheck.refcheck import ReferenceCheckBundle

    value = report()
    value.checked[0].issues[0].code = "title_mismatch"
    payload, records, block, _, _ = files(tmp_path, value)
    paper = tmp_path / "paper.md"
    paper.write_text(block.text, "utf-8")
    materials = SharedMaterials(
        paper_key="fixture",
        source_pdf="paper.pdf",
        markdown=block.text,
        markdown_path=str(paper),
        content_list_path="",
        provider="mock",
        blocks=[block],
        bibliography=[block],
    )
    findings, issues = check_bibliography(
        materials, tmp_path / "screen", checker=lambda **kw: ReferenceCheckBundle(payload, records)
    )
    assert len(findings) == 1 and findings[0].level == "metadata_candidate"
    assert "not confirmed" in findings[0].text
    assert any("original PDF" in item for item in issues)
    assert all(not e.sufficient and not e.affects_claim for e in findings[0].evidence)


def test_full_long_correction_is_usable_and_legacy_truncation_unchanged(tmp_path):
    data = files(tmp_path, report(title="A " + "long title " * 500))
    result = correction(data)
    assert result.state == "metadata_candidate", result.reason
    assert len(result.corrected_bibtex) > 4000
    assert "...(truncated)" in data[0]["issues"][0]["corrected_bibtex"]


@pytest.mark.parametrize(
    "change", ["none", "bibtex", "field", "hash", "pointer", "index", "missing_records", "input"]
)
def test_reopened_candidate_rebuilds_all_fields_from_local_protected_records(tmp_path, change):
    from screening.reference_corrections import checked_correction

    data = files(tmp_path, report())
    original = correction(data)
    if change == "bibtex":
        original.corrected_bibtex = original.corrected_bibtex.replace("2020", "2030")
    elif change == "field":
        original.fields[0].record_id = "another-record"
    elif change == "hash":
        original.records_sha256 = "0" * 64
    elif change == "pointer":
        original.records_pointer = original.records_pointer.replace("checked.0", "checked.1")
    elif change == "index":
        original.raw_result_pointer = original.raw_result_pointer.replace("issues.0", "issues.1")
    elif change == "missing_records":
        data[4].unlink()
    elif change == "input":
        (tmp_path / "bibliography.txt").write_text("Changed bibliography", "utf-8")
    before = original.model_dump(mode="json")
    checked = checked_correction(original)
    assert original.model_dump(mode="json") == before
    assert (checked.state == "metadata_candidate") is (change == "none")
    if change != "none":
        assert checked.reason and not checked.corrected_bibtex and not checked.fields


def test_metadata_candidate_never_counts_as_claim_flaw(tmp_path):
    from review.report.v2 import render_markdown
    from schemas.claim import Claim, Condition, Finding
    from schemas.review import FinalReview
    from screening.checks import paper_finding

    data = files(tmp_path, report())
    paper = tmp_path / "paper.md"
    paper.write_text(data[2].text, "utf-8")
    materials = SharedMaterials(
        paper_key="fixture",
        source_pdf="paper.pdf",
        markdown=data[2].text,
        markdown_path=str(paper),
        content_list_path="",
        provider="mock",
        blocks=[data[2]],
        bibliography=[data[2]],
    )
    finding = paper_finding(
        materials,
        data[2],
        quote=data[2].text,
        kind="reference",
        level="metadata_candidate",
        text="Retrieved metadata candidate; original error unconfirmed.",
    )
    finding = Finding.model_validate(
        {**finding.model_dump(), "reference_correction": correction(data).model_dump()}
    )
    claim = Claim(
        id="c",
        text="An unresolved scientific claim",
        loc={"page": 1},
        conditions=[Condition(id="a", description="Source evidence required")],
        needs=["Literature"],
    )
    output = render_markdown(
        FinalReview(paper_key="fixture", run_id="test", claims=[claim], findings=[finding])
    )
    assert "| flawed | 0 |" in output and "| unverified | 1 |" in output
    assert not claim.evidence and all(not e.affects_claim for e in finding.evidence)
