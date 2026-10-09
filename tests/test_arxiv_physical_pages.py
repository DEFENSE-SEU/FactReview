from io import BytesIO
from unittest.mock import AsyncMock

import pytest
from reportlab.pdfgen import canvas

from fact_generation.positioning import paper_search
from fact_generation.positioning.paper_search import PaperReadConfig, PaperSearchAdapter, PaperSearchConfig


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "page_texts",
    [
        [
            "",
            "Mechanism uses additive aggregation over all input feature vectors.",
            "Evaluation uses exact equality for all released predictions and labels.",
        ],
        [
            "Mechanism uses additive aggregation over all input feature vectors.",
            "",
            "Evaluation uses exact equality for all released predictions and labels.",
        ],
        [
            "",
            "Mechanism uses additive aggregation over all input feature vectors.",
            "",
            "Evaluation uses exact equality for all released predictions and labels.",
        ],
        [
            "Mechanism uses additive aggregation over all input feature vectors.",
            "Evaluation uses exact equality for all released predictions and labels.",
        ],
        ["", "", ""],
    ],
    ids=["leading-blank", "middle-blank", "both-blanks", "no-blanks", "all-blank"],
)
async def test_reader_preserves_physical_pdf_page_numbers(monkeypatch, page_texts):
    def unexpected_service(*args, **kwargs):
        raise AssertionError("This test must not call external services")

    monkeypatch.setattr(paper_search.httpx, "AsyncClient", unexpected_service)
    adapter = PaperSearchAdapter(
        search_cfg=PaperSearchConfig(
            enabled=False,
            provider="arxiv",
            base_url=None,
            api_key=None,
            endpoint="/search",
            timeout_seconds=1,
            health_endpoint="/health",
            health_timeout_seconds=1,
        ),
        read_cfg=PaperReadConfig(base_url=None, api_key=None, endpoint="/read", timeout_seconds=1),
    )
    data = BytesIO()
    pdf = canvas.Canvas(data)
    for text in page_texts:
        if text:
            pdf.drawString(72, 720, text)
        pdf.showPage()
    pdf.save()
    assert paper_search.parse_pdf_locally(data.getvalue()).pages == page_texts
    detail = {
        "arxiv_id": "2401.00001",
        "title": "Synthetic physical-page fixture",
        "abstract": "Only an abstract fallback.",
        "pdf_url": "https://arxiv.org/pdf/2401.00001",
    }
    fetch = AsyncMock(return_value=detail)
    download = AsyncMock(return_value=data.getvalue())
    monkeypatch.setattr(adapter, "_arxiv_fetch_single", fetch)
    monkeypatch.setattr(adapter, "_download_pdf", download)

    result = await adapter.read_papers(
        items=[{"id": detail["arxiv_id"], "question": "Describe the mechanism and evaluation."}]
    )

    fetch.assert_awaited_once_with(detail["arxiv_id"])
    download.assert_awaited_once_with(detail["pdf_url"])
    assert result["success"] is True
    item = result["items"][0]
    if any(page_texts):
        assert item["reader_provider"] == "arxiv_full_text_fallback"
        assert {row["text"]: row["page"] for row in item["evidence"]} == {
            text: page for page, text in enumerate(page_texts, 1) if text
        }
        for evidence in item["evidence"]:
            assert f"Page {evidence['page']}: {evidence['text']}" in item["answer"]
    else:
        assert item["reader_provider"] == "arxiv_abstract_fallback"
        assert "abstract-level" in item["answer"]
