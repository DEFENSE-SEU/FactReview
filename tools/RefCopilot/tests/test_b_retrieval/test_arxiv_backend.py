"""ArXiv search backend."""

from __future__ import annotations

from refcopilot.cache.disk_cache import DiskCache
from refcopilot.models import Backend, Reference, SourceFormat
from refcopilot.ratelimit.arxiv import ArxivRateLimiter
from refcopilot.search.arxiv import ArxivBackend


_ATTENTION_FEED = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <title>Attention Is All You Need</title>
    <published>2017-06-12T17:57:34Z</published>
    <summary>The dominant sequence transduction models...</summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <author><name>Niki Parmar</name></author>
    <arxiv:journal_ref>NeurIPS 2017</arxiv:journal_ref>
    <arxiv:doi>10.5555/3295222.3295349</arxiv:doi>
  </entry>
</feed>
"""

_WITHDRAWN_FEED = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/9999.99999v1</id>
    <title>Sample Title</title>
    <published>2099-01-01T00:00:00Z</published>
    <summary>This paper has been withdrawn by the author due to a critical flaw.</summary>
    <author><name>X. Author</name></author>
  </entry>
</feed>
"""


class _Resp:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text
        self.headers: dict[str, str] = {}


def _mock_http(canned_text: str):
    calls: list[dict] = []

    def get(url, params):
        calls.append({"url": url, "params": params})
        return _Resp(200, canned_text)

    return get, calls


def test_lookup_by_id_parses_fields(tmp_path):
    get, calls = _mock_http(_ATTENTION_FEED)
    backend = ArxivBackend(
        cache=DiskCache(tmp_path),
        rate_limiter=ArxivRateLimiter(min_interval_seconds=0.0),
        http_get=get,
    )
    rec = backend.lookup_by_id("1706.03762")
    assert rec is not None
    assert rec.backend == Backend.ARXIV
    assert rec.title == "Attention Is All You Need"
    assert "Ashish Vaswani" in rec.authors
    assert rec.year == 2017
    assert rec.arxiv_id == "1706.03762"
    assert rec.latest_arxiv_version == 7
    assert rec.doi == "10.5555/3295222.3295349"
    assert calls[0]["params"]["id_list"] == "1706.03762"


def test_lookup_strips_version_from_id(tmp_path):
    get, calls = _mock_http(_ATTENTION_FEED)
    backend = ArxivBackend(
        cache=DiskCache(tmp_path),
        rate_limiter=ArxivRateLimiter(min_interval_seconds=0.0),
        http_get=get,
    )
    backend.lookup_by_id("1706.03762v3")
    assert calls[0]["params"]["id_list"] == "1706.03762"


def test_search_by_title(tmp_path):
    get, calls = _mock_http(_ATTENTION_FEED)
    backend = ArxivBackend(
        cache=DiskCache(tmp_path),
        rate_limiter=ArxivRateLimiter(min_interval_seconds=0.0),
        http_get=get,
    )
    results = backend.search_by_title("Attention Is All You Need", year=2017)
    assert len(results) == 1
    assert "Attention" in calls[0]["params"]["search_query"]


def test_lookup_via_reference(tmp_path):
    get, _ = _mock_http(_ATTENTION_FEED)
    backend = ArxivBackend(
        cache=DiskCache(tmp_path),
        rate_limiter=ArxivRateLimiter(min_interval_seconds=0.0),
        http_get=get,
    )
    ref = Reference(
        raw="x",
        source_format=SourceFormat.BIBTEX,
        title="Attention Is All You Need",
        arxiv_id="1706.03762",
        year=2017,
    )
    out = backend.lookup(ref)
    assert len(out) == 1
    assert out[0].arxiv_id == "1706.03762"


def test_handles_withdrawn(tmp_path):
    get, _ = _mock_http(_WITHDRAWN_FEED)
    backend = ArxivBackend(
        cache=DiskCache(tmp_path),
        rate_limiter=ArxivRateLimiter(min_interval_seconds=0.0),
        http_get=get,
    )
    rec = backend.lookup_by_id("9999.99999")
    assert rec is not None
    assert rec.withdrawn is True


def test_cache_hit_avoids_http(tmp_path):
    get, calls = _mock_http(_ATTENTION_FEED)
    backend = ArxivBackend(
        cache=DiskCache(tmp_path),
        rate_limiter=ArxivRateLimiter(min_interval_seconds=0.0),
        http_get=get,
    )
    backend.lookup_by_id("1706.03762")
    assert len(calls) == 1
    backend.lookup_by_id("1706.03762")
    assert len(calls) == 1, "second call should hit cache"
