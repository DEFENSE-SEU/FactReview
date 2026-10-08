from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from fact_generation.positioning import paper_search as paper_search_mod
from fact_generation.positioning.paper_search import PaperReadConfig, PaperSearchAdapter, PaperSearchConfig


def _adapter() -> PaperSearchAdapter:
    return PaperSearchAdapter(
        search_cfg=PaperSearchConfig(
            enabled=False,
            provider="arxiv",
            base_url=None,
            api_key=None,
            endpoint="/pasa/search",
            timeout_seconds=120,
            health_endpoint="/health",
            health_timeout_seconds=5,
        ),
        read_cfg=PaperReadConfig(
            base_url=None,
            api_key=None,
            endpoint="/read",
            timeout_seconds=180,
        ),
    )


def _arxiv_detail() -> dict:
    return {
        "title": "Composition Methods for Relational Graphs",
        "abstract": "This abstract is only the fallback evidence.",
        "arxiv_id": "1911.03082",
        "pdf_url": "https://arxiv.org/pdf/1911.03082.pdf",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested", "returned", "success"),
    [
        ("1911.03082", "1911.03082v2", True),
        ("arXiv:1911.03082", "1911.03082", True),
        ("1911.03082v1", "1911.03082v1", True),
        ("1911.03082v1", "1911.03082v2", False),
        ("1911.03082", "1911.09999", False),
    ],
)
async def test_metadata_lookup_checks_identifier_without_full_text(monkeypatch, requested, returned, success):
    adapter = _adapter()
    fetch = AsyncMock(return_value={**_arxiv_detail(), "arxiv_id": returned})
    download = AsyncMock(side_effect=AssertionError("metadata must not download a PDF"))
    monkeypatch.setattr(adapter, "_arxiv_fetch_single", fetch)
    monkeypatch.setattr(adapter, "_download_pdf", download)
    result = await adapter.lookup_metadata(identifier=requested)
    assert result["success"] is success
    fetch.assert_awaited_once_with(requested.removeprefix("arXiv:"))
    download.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "identifier", ["10.1000/example", "https://openreview.net/forum?id=x", "author:Smith", ""]
)
async def test_metadata_lookup_rejects_unsupported_identifiers_without_network(monkeypatch, identifier):
    adapter = _adapter()
    fetch = AsyncMock(side_effect=AssertionError("unsupported ID must not reach network"))
    monkeypatch.setattr(adapter, "_arxiv_fetch_single", fetch)
    result = await adapter.lookup_metadata(identifier=identifier)
    assert result["success"] is False
    assert result["error"] == "unsupported_metadata_identifier"
    fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_paper_uses_arxiv_full_text_fallback_when_pdf_parses(monkeypatch) -> None:
    adapter = _adapter()

    async def fake_fetch_single(arxiv_id: str) -> dict:
        assert arxiv_id == "1911.03082"
        return _arxiv_detail()

    async def fake_download_pdf(url: str) -> bytes:
        assert url.endswith("1911.03082.pdf")
        return b"%PDF fake bytes"

    def fake_parse_pdf_locally(pdf_bytes: bytes) -> SimpleNamespace:
        assert pdf_bytes.startswith(b"%PDF")
        return SimpleNamespace(
            pages=[
                "Unrelated introduction text.",
                (
                    "The method studies relation composition and demonstrates "
                    "generalization across relational graph benchmarks."
                ),
            ]
        )

    monkeypatch.setattr(adapter, "_arxiv_fetch_single", fake_fetch_single)
    monkeypatch.setattr(adapter, "_download_pdf", fake_download_pdf)
    monkeypatch.setattr(paper_search_mod, "parse_pdf_locally", fake_parse_pdf_locally)

    result = await adapter.read_papers(
        items=[{"id": "1911.03082", "question": "Does relation composition generalize?"}]
    )

    item = result["items"][0]
    assert item["success"] is True
    assert item["reader_provider"] == "arxiv_full_text_fallback"
    assert item["evidence"][0]["page"] == 2
    assert "relation composition" in item["answer"]


@pytest.mark.asyncio
async def test_read_paper_falls_back_to_arxiv_abstract_when_pdf_read_fails(monkeypatch) -> None:
    adapter = _adapter()

    async def fake_fetch_single(_arxiv_id: str) -> dict:
        return _arxiv_detail()

    async def fake_download_pdf(_url: str) -> bytes:
        raise RuntimeError("download failed")

    monkeypatch.setattr(adapter, "_arxiv_fetch_single", fake_fetch_single)
    monkeypatch.setattr(adapter, "_download_pdf", fake_download_pdf)

    result = await adapter.read_papers(items=[{"id": "1911.03082", "question": "What is the evidence?"}])

    item = result["items"][0]
    assert item["success"] is True
    assert item["reader_provider"] == "arxiv_abstract_fallback"
    assert "abstract-level" in item["answer"]


@pytest.mark.asyncio
async def test_arxiv_provider_starts_without_remote_base_url(monkeypatch) -> None:
    adapter = PaperSearchAdapter(
        search_cfg=PaperSearchConfig(
            enabled=True,
            provider="arxiv",
            base_url=None,
            api_key=None,
            endpoint="/pasa/search",
            timeout_seconds=120,
            health_endpoint="/health",
            health_timeout_seconds=5,
        ),
        read_cfg=PaperReadConfig(
            base_url=None,
            api_key=None,
            endpoint="/read",
            timeout_seconds=180,
        ),
    )

    async def fake_arxiv_query(question: str, *, max_results: int) -> list[dict]:
        assert question == "relation composition"
        return [{"title": "T", "arxiv_id": "1911.03082", "url": "https://arxiv.org/abs/1911.03082"}]

    monkeypatch.setattr(adapter, "_arxiv_query", fake_arxiv_query)

    state = await adapter.get_search_runtime_state()
    result = await adapter.search(query="relation composition")

    assert state.started is True
    assert state.provider == "arxiv"
    assert result["provider"] == "arxiv_fallback"
    assert result["count"] == 1


def test_semantic_scholar_normalization_exposes_arxiv_id() -> None:
    adapter = _adapter()

    item = adapter._normalize_semantic_scholar_item(
        {
            "paperId": "s2id",
            "title": "A Paper",
            "abstract": "Abstract.",
            "url": "https://www.semanticscholar.org/paper/s2id",
            "year": 2024,
            "externalIds": {"ArXiv": "2401.12345"},
            "openAccessPdf": {},
            "authors": [{"name": "Ada"}],
            "citationCount": 7,
        }
    )

    assert item["arxiv_id"] == "2401.12345"
    assert item["pdf_url"] == "https://arxiv.org/pdf/2401.12345.pdf"
    assert item["source"] == "semantic_scholar"


def test_openalex_normalization_reconstructs_abstract_and_arxiv_pdf() -> None:
    adapter = _adapter()

    item = adapter._normalize_openalex_item(
        {
            "id": "https://openalex.org/W1",
            "display_name": "Open Paper",
            "publication_year": 2024,
            "abstract_inverted_index": {"hello": [0], "world": [1]},
            "ids": {"openalex": "https://openalex.org/W1"},
            "best_oa_location": {"landing_page_url": "https://arxiv.org/abs/2401.12345"},
            "authorships": [{"author": {"display_name": "Grace"}}],
            "cited_by_count": 3,
        }
    )

    assert item["abstract"] == "hello world"
    assert item["arxiv_id"] == "2401.12345"
    assert item["pdf_url"] == "https://arxiv.org/pdf/2401.12345.pdf"
    assert item["source"] == "openalex"
