"""Test A.4 — URL / arXiv ID normalization."""

from __future__ import annotations

import pytest

from refcopilot.inputs import url


@pytest.mark.parametrize(
    ("spec", "expected_url", "expected_id", "expected_v"),
    [
        ("1706.03762", "https://arxiv.org/pdf/1706.03762.pdf", "1706.03762", None),
        ("arxiv:1706.03762", "https://arxiv.org/pdf/1706.03762.pdf", "1706.03762", None),
        ("1706.03762v5", "https://arxiv.org/pdf/1706.03762v5.pdf", "1706.03762", 5),
        ("arxiv:1706.03762v2", "https://arxiv.org/pdf/1706.03762v2.pdf", "1706.03762", 2),
        ("https://arxiv.org/abs/1706.03762", "https://arxiv.org/pdf/1706.03762.pdf", "1706.03762", None),
        ("https://arxiv.org/pdf/1706.03762v3", "https://arxiv.org/pdf/1706.03762v3.pdf", "1706.03762", 3),
        ("https://example.com/paper.pdf", "https://example.com/paper.pdf", None, None),
    ],
)
def test_normalize(spec, expected_url, expected_id, expected_v):
    out_url, aid, v = url.normalize(spec)
    assert out_url == expected_url
    assert aid == expected_id
    assert v == expected_v


def test_download_uses_cached_file(tmp_path, monkeypatch):
    cached = tmp_path / "arxiv_1706.03762.pdf"
    cached.write_bytes(b"%PDF-1.4 fake")

    called = {"count": 0}

    class _StubClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, _url):
            called["count"] += 1
            raise AssertionError("network should not be touched when cache is fresh")

    monkeypatch.setattr(url.httpx, "Client", _StubClient)

    out = url.download("1706.03762", tmp_path)
    assert out == cached
    assert called["count"] == 0


def test_download_writes_pdf(tmp_path, monkeypatch):
    class _Resp:
        content = b"%PDF-1.4 mocked"

        def raise_for_status(self):
            return None

    class _StubClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, _url):
            return _Resp()

    monkeypatch.setattr(url.httpx, "Client", _StubClient)
    out = url.download("https://example.com/paper.pdf", tmp_path)
    assert out.exists()
    assert out.read_bytes().startswith(b"%PDF-")
