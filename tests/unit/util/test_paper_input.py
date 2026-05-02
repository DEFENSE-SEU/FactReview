from __future__ import annotations

from pathlib import Path

import pytest


def test_materialize_local_pdf_copies_to_run_inputs(tmp_path: Path) -> None:
    from util.paper_input import infer_paper_key, materialize_paper_pdf

    source = tmp_path / "CompGCN Paper.pdf"
    source.write_bytes(b"%PDF-1.4\ncontent")

    assert infer_paper_key(str(source)) == "CompGCN Paper"

    result = materialize_paper_pdf(source, tmp_path / "run" / "inputs" / "source_pdf")

    assert result.source_type == "path"
    assert result.downloaded is False
    assert result.path != source
    assert result.path.name == "CompGCN_Paper.pdf"
    assert result.path.read_bytes() == source.read_bytes()


def test_materialize_arxiv_abs_url_downloads_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import util.paper_input as paper_input

    seen_urls: list[str] = []

    class FakeResponse:
        def __init__(self) -> None:
            self._data = b"%PDF-1.7\narxiv"
            self._offset = 0

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, size: int = -1) -> bytes:
            if self._offset >= len(self._data):
                return b""
            if size < 0:
                size = len(self._data) - self._offset
            chunk = self._data[self._offset : self._offset + size]
            self._offset += len(chunk)
            return chunk

    def fake_urlopen(request: object, timeout: int) -> FakeResponse:
        assert timeout == 60
        seen_urls.append(str(getattr(request, "full_url")))
        return FakeResponse()

    monkeypatch.setattr(paper_input, "urlopen", fake_urlopen)

    result = paper_input.materialize_paper_pdf(
        "https://arxiv.org/abs/1911.03082",
        tmp_path / "inputs",
        paper_key="compgcn",
    )

    assert seen_urls == ["https://arxiv.org/pdf/1911.03082.pdf"]
    assert result.source_type == "url"
    assert result.downloaded is True
    assert result.path.name == "1911.03082.pdf"
    assert result.path.read_bytes() == b"%PDF-1.7\narxiv"


def test_materialize_rejects_non_pdf_local_file(tmp_path: Path) -> None:
    from util.paper_input import materialize_paper_pdf

    source = tmp_path / "not-a-pdf.pdf"
    source.write_text("html", encoding="utf-8")

    with pytest.raises(ValueError, match="not a valid PDF"):
        materialize_paper_pdf(source, tmp_path / "inputs")
