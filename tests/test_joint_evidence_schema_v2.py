import pytest

from schemas.claim import Evidence, EvidencePointer


def primary(**kwargs):
    return EvidencePointer(locator="paper.md", quote="Exact original.", key="chars:0-15", **kwargs)


def test_legacy_default_and_round_trip_with_ordered_additional_sources():
    evidence = Evidence(source="paper_internal", direction="support", pointer=primary(), covered=["c1"])
    assert evidence.additional_pointers == []
    raw = evidence.model_dump()
    raw.pop("additional_pointers")
    assert Evidence.model_validate(raw) == evidence
    extras = [
        EvidencePointer(locator="paper.md", quote="Same words.", key=key)
        for key in ("chars:20-31", "chars:40-51")
    ]
    evidence.additional_pointers = extras
    assert Evidence.model_validate_json(evidence.model_dump_json()).additional_pointers == extras


@pytest.mark.parametrize(
    "case", ["primary", "duplicate", "empty_quote", "no_location", "foreign", "flaw", "none"]
)
def test_additional_source_contract_rejects_invalid_pointer_sets(case):
    additional = [EvidencePointer(locator="paper.md", quote="Another exact source.", page=2)]
    kwargs = dict(source="paper_internal", direction="support", pointer=primary(), covered=["c1"])
    if case == "primary":
        additional = [primary()]
    elif case == "duplicate":
        additional *= 2
    elif case == "empty_quote":
        additional[0].quote = " "
    elif case == "no_location":
        additional[0].page = None
    elif case == "foreign":
        kwargs["source"] = "theory"
    elif case == "flaw":
        kwargs["direction"] = "flaw"
    else:
        additional = None
    with pytest.raises(ValueError):
        Evidence(**kwargs, additional_pointers=additional)


def test_distinct_overlapping_ranges_and_different_locations_preserve_identity():
    evidence = Evidence(
        source="paper_internal",
        direction="support",
        pointer=primary(),
        covered=["c1"],
        sufficient=True,
        additional_pointers=[
            EvidencePointer(locator="paper.md", quote="Exact", key="chars:0-5"),
            EvidencePointer(locator="paper.md", quote="Exact original.", key="chars:40-55"),
        ],
    )
    assert len(evidence.additional_pointers) == 2 and evidence.covered == ["c1"]
