from unittest.mock import AsyncMock

import pytest

from util.submission_cutoff import arxiv_identifier, resolve_arxiv_first_submission


@pytest.mark.parametrize(
    "source,expected",
    [
        ("https://arxiv.org/abs/2201.01234v3", "2201.01234"),
        ("https://arxiv.org/pdf/2201.01234.pdf?download=1", "2201.01234"),
        ("arXiv:hep-th/9901001v2", "hep-th/9901001"),
    ],
)
def test_cutoff_identifier_preserves_identity_without_revision(source, expected):
    assert arxiv_identifier(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "https://evil-arxiv.org/abs/2201.01234",
        "https://arxiv.org.example.com/abs/2201.01234",
        "https://user:password@arxiv.org/abs/2201.01234",
        "C:/papers/2201.01234.pdf",
        "paper.pdf",
    ],
)
def test_cutoff_does_not_guess_identity_from_unrelated_urls_or_filenames(source):
    with pytest.raises(ValueError, match="arXiv"):
        arxiv_identifier(source)


@pytest.mark.asyncio
async def test_first_submission_uses_published_and_records_provenance():
    lookup = AsyncMock(
        return_value={
            "success": True,
            "paper": {
                "arxiv_id": "2201.01234v4",
                "published": "2022-01-07T12:30:00Z",
                "updated": "2025-05-04T00:00:00Z",
            },
        }
    )
    cutoff, provenance = await resolve_arxiv_first_submission(
        "https://arxiv.org/abs/2201.01234v2", lookup=lookup
    )
    lookup.assert_awaited_once_with(identifier="2201.01234")
    assert cutoff.to_string() == "2022-01-07"
    assert provenance["source"] == "arxiv_first_submission"
    assert provenance["venue_deadline_known"] is False
    assert provenance["published"] == "2022-01-07T12:30:00Z"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "paper",
    [
        {"arxiv_id": "2201.09999", "published": "2022-01-07"},
        {"arxiv_id": "2201.01234", "updated": "2022-01-07"},
        {"arxiv_id": "2201.01234", "published": "2022-01"},
        {"arxiv_id": "2201.01234", "published": "2022-02-30"},
    ],
)
async def test_unresolved_metadata_never_produces_a_guessed_cutoff(paper):
    with pytest.raises(ValueError):
        await resolve_arxiv_first_submission(
            "2201.01234", lookup=AsyncMock(return_value={"success": True, "paper": paper})
        )


@pytest.mark.asyncio
async def test_metadata_service_failure_remains_explicit():
    with pytest.raises(ValueError, match="could not be resolved"):
        await resolve_arxiv_first_submission("2201.01234", lookup=AsyncMock(return_value={"success": False}))


@pytest.mark.asyncio
async def test_legacy_arxiv_feed_identity_survives_metadata_resolution(monkeypatch):
    from fact_generation.positioning.paper_search import PaperSearchAdapter

    async def fetch(self, identifier):
        return self._parse_arxiv_feed("""<feed xmlns="http://www.w3.org/2005/Atom"><entry>
            <id>http://arxiv.org/abs/hep-th/9901001v2</id><title>A theory paper</title>
            <published>1999-01-04T09:00:00Z</published><updated>2000-01-05T10:00:00Z</updated>
            </entry></feed>""")[0]

    monkeypatch.setattr(PaperSearchAdapter, "_arxiv_fetch_single", fetch)
    cutoff, provenance = await resolve_arxiv_first_submission("hep-th/9901001v2")
    assert cutoff.to_string() == "1999-01-04"
    assert provenance["arxiv_id"] == "hep-th/9901001"
