"""Publication metadata may be shown without inventing workshop promotion."""

import hashlib
import json
from copy import deepcopy

import pytest

# Bootstrap the public vendored API through its production wrapper.
# isort: off
from fact_generation.refcheck.refcheck import ReferenceCheckBundle, reference_records
from refcopilot.models import Issue, IssueCategory, Severity
from refcopilot.report import to_factreview_dict
# isort: on

from schemas.claim import ClaimLocation
from schemas.materials import MaterialBlock, SharedMaterials
from screening.references import check_bibliography
from tests.test_reference_records_v2 import offline, report  # noqa: F401


def publication_report():
    value = report()
    checked = value.checked[0]
    checked.reference.raw = (
        "[1] Ada Author. A grounded paper. Workshop on Synthetic Models, 2019. doi:10.1000/one"
    )
    checked.reference.venue = "Workshop on Synthetic Models"
    checked.merged.venue = "International Conference on Synthetic Models"
    checked.merged.sources[0].venue = checked.merged.venue
    checked.merged.sources[0].publication_venue = checked.merged.venue
    checked.issues = [
        Issue(
            severity=Severity.WARNING,
            category=IssueCategory.OUTDATED,
            code="workshop_promoted",
            message="Cited workshop version, but a full version appears at the conference.",
            suggestion="Cite the full version at the conference.",
        )
    ]
    return value


def invoke(tmp_path, value, *, legacy=False, original_suffix=""):
    payload = to_factreview_dict(value)
    texts = [c.reference.raw + (original_suffix if i == 0 else "") for i, c in enumerate(value.checked)]
    original = "\n\n".join(texts)
    records = reference_records(value, payload, hashlib.sha256(original.encode()).hexdigest())
    saved = deepcopy((payload, records))
    path = tmp_path / "paper.md"
    path.write_text(original, "utf-8")
    blocks = [
        MaterialBlock(id=f"ref-{i}", text=text, loc=ClaimLocation(page=1)) for i, text in enumerate(texts)
    ]
    materials = SharedMaterials(
        paper_key="mock",
        source_pdf="paper.pdf",
        markdown=original,
        markdown_path=str(path),
        content_list_path="",
        provider="mock",
        blocks=blocks,
        bibliography=blocks,
    )
    calls = []

    def checker(**kwargs):
        calls.append(kwargs)
        return payload if legacy else ReferenceCheckBundle(payload, records)

    def no_model(**kwargs):
        pytest.fail("Publication metadata gate must not call an additional model")

    findings, issues = check_bibliography(materials, tmp_path / "screen", checker=checker, call=no_model)
    assert len(calls) == 1
    assert (payload, records) == saved
    assert json.loads((tmp_path / "screen/reference_check.json").read_text("utf-8")) == payload
    if not legacy:
        assert json.loads((tmp_path / "screen/reference_records.json").read_text("utf-8")) == records
    return findings, issues


@pytest.mark.parametrize("identifier", ["doi", "explicit_arxiv_version"])
def test_bound_complete_publication_is_metadata_without_historical_promotion(tmp_path, identifier):
    value = publication_report()
    if identifier == "explicit_arxiv_version":
        c = value.checked[0]
        source = c.merged.sources[0]
        c.reference.raw = c.reference.raw.replace("doi:10.1000/one", "arXiv:2401.12345v1")
        c.reference.doi = c.merged.doi = source.doi = None
        c.reference.arxiv_id = c.merged.arxiv_id = source.arxiv_id = "2401.12345"
        c.merged.url = source.url = "https://arxiv.org/abs/2401.12345v1"
        c.merged.provenance.pop("doi")
        c.merged.provenance["arxiv_id"] = source.backend
    findings, issues = invoke(tmp_path, value)
    assert len(findings) == 1 and findings[0].level == "metadata_candidate"
    assert "International Conference on Synthetic Models" in findings[0].text
    assert "promotion" in findings[0].text.lower() and "unconfirmed" in findings[0].text.lower()
    assert "cite the full version" not in findings[0].text.lower()
    assert findings[0].reference_correction.state == "metadata_candidate"
    assert all(not e.sufficient and not e.affects_claim for e in findings[0].evidence)
    assert not issues


@pytest.mark.parametrize(
    "change",
    [
        "unknown_source",
        "no_identifier",
        "wrong_doi",
        "generic_volume",
        "generic_issue",
        "bare_container",
        "workshop_journal",
        "conflicting_journal",
        "unknown_provenance",
        "ambiguous_source",
        "missing_venue",
        "title_only_closest_match",
        "arxiv_no_version",
        "arxiv_wrong_version",
        "raw_backreference_suffix",
    ],
)
def test_unconfirmed_upgrade_is_issue_without_author_correction_finding(tmp_path, change):
    value = publication_report()
    c = value.checked[0]
    source = c.merged.sources[0]
    suffix = ""
    if change == "no_identifier":
        c.reference.raw = c.reference.raw.replace(" doi:10.1000/one", "")
    elif change == "wrong_doi":
        source.doi = "10.1000/other"
    elif change in {"generic_volume", "generic_issue"}:
        c.merged.venue = source.venue = source.publication_venue = (
            "Volume 1" if change == "generic_volume" else "Issue 2"
        )
    elif change == "bare_container":
        c.merged.venue = source.venue = source.publication_venue = "International Conference"
    elif change == "workshop_journal":
        source.journal = "Proceedings of the Workshop on Synthetic Models - Volume 1"
    elif change == "conflicting_journal":
        source.journal = "International Conference on Different Models"
    elif change == "unknown_provenance":
        c.merged.provenance.pop("venue")
    elif change == "ambiguous_source":
        c.merged.sources.append(source.model_copy(update={"record_id": "record-other"}))
    elif change == "missing_venue":
        c.merged.venue = source.venue = source.publication_venue = None
    elif change == "title_only_closest_match":
        c.issues[0].message = "Closest match: another work at a full conference."
    elif change.startswith("arxiv_"):
        original_version = "" if change == "arxiv_no_version" else "v1"
        c.reference.raw = c.reference.raw.replace("doi:10.1000/one", f"arXiv:2401.12345{original_version}")
        c.reference.doi = c.merged.doi = source.doi = None
        c.reference.arxiv_id = c.merged.arxiv_id = source.arxiv_id = "2401.12345"
        c.merged.url = source.url = "https://arxiv.org/abs/2401.12345v2"
        c.merged.provenance.pop("doi")
        c.merged.provenance["arxiv_id"] = source.backend
    elif change == "raw_backreference_suffix":
        suffix = " 1, 4, 5"
    findings, issues = invoke(tmp_path, value, legacy=change == "unknown_source", original_suffix=suffix)
    assert findings == []
    assert any(
        "workshop publication warning unconfirmed" in issue.lower() and "#issues.0" in issue
        for issue in issues
    )


def test_original_fixmatch_workshop_volume_shape_keeps_full_record_and_rejects_suggestion(tmp_path):
    value = publication_report()
    c = value.checked[0]
    c.reference.raw = (
        "[44] C. Rosenberg, M. Hebert, and H. Schneiderman. Semi-supervised self-training of object "
        "detection models. In Proceedings of the Seventh IEEE Workshops on Application of Computer Vision, 2005."
    )
    c.reference.title = "Semi-supervised self-training of object detection models"
    c.reference.venue = "Proceedings of the Seventh IEEE Workshops on Application of Computer Vision"
    c.reference.doi = None
    c.merged.venue = "Volume 1"
    c.merged.sources[0].venue = c.merged.sources[0].publication_venue = "Volume 1"
    c.merged.sources[
        0
    ].journal = "2005 Seventh IEEE Workshops on Applications of Computer Vision (WACV/MOTION'05) - Volume 1"
    c.issues[0].message = "Cited workshop version, but a full version appears at 'Volume 1'."
    c.issues[0].suggestion = "Cite the full version at Volume 1."
    findings, issues = invoke(tmp_path, value, original_suffix=" 1, 4, 5")
    assert findings == []
    assert any("workshop publication warning unconfirmed" in issue.lower() for issue in issues)


def test_exact_original_arxiv_venue_warning_stays_unconfirmed(tmp_path):
    value = publication_report()
    c = value.checked[0]
    c.reference.raw = "[1] Ashish Vaswani et al. 2017. Attention Is All You Need. Advances in Neural Information Processing Systems 30. https://arxiv.org/abs/1706.03762"
    c.reference.title = "Attention Is All You Need"
    c.reference.venue = "Advances in Neural Information Processing Systems 30"
    c.issues[0].code = "arxiv_published"
    c.issues[
        0
    ].message = (
        "Paper was published at venue 'Neural Information Processing Systems' but cited as an arXiv preprint."
    )
    findings, issues = invoke(tmp_path, value)
    assert findings == []
    assert any("already names publication venue" in issue for issue in issues)


def test_unconfirmed_publication_preserves_healthy_neighbor_warning(tmp_path):
    value = publication_report()
    value.checked[0].reference.raw = value.checked[0].reference.raw.replace(" doi:10.1000/one", "")
    value.checked.append(report(doi="10.1000/two").checked[0])
    value.summary.total_refs = value.summary.warnings = 2
    findings, issues = invoke(tmp_path, value)
    assert len(findings) == 1 and "Retrieved year is 2020" in findings[0].text
    assert findings[0].reference_correction.state == "metadata_candidate"
    assert any("workshop publication warning unconfirmed" in issue.lower() for issue in issues)
