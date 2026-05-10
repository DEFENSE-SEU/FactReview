"""Retrieval-layer tests.

Replaces the per-backend test_b_retrieval/* files. Covers the cache (TTL +
on-disk format wipe) and the multi-source merge (provenance ordering and
the cross-backend retraction OR). Live HTTP backends are exercised through
``RefCopilotPipeline`` integration tests in ``test_verify.py`` with stubbed
backends; we don't re-test their request shapes here.
"""

from __future__ import annotations

import os
import time

from refcopilot.cache.disk_cache import API_CACHE_VERSION, DiskCache
from refcopilot.merge import merge_records
from refcopilot.models import Backend, ExternalRecord


def test_disk_cache_set_get_and_ttl(tmp_path) -> None:
    c = DiskCache(tmp_path, ttl_days=30)
    c.set_api("semantic_scholar", "doi_10.x_y", {"hello": "world"})
    assert c.get_api("semantic_scholar", "doi_10.x_y") == {"hello": "world"}

    # An entry mtime'd past the TTL must be treated as a miss; otherwise stale
    # backend responses keep contradicting fresh ones.
    path = c._api_path("semantic_scholar", "doi_10.x_y")
    old = time.time() - (40 * 86400)
    os.utime(path, (old, old))
    assert c.get_api("semantic_scholar", "doi_10.x_y") is None


def test_disk_cache_wipes_on_version_mismatch(tmp_path) -> None:
    api_dir = tmp_path / "api_cache"
    (api_dir / "arxiv").mkdir(parents=True)
    stale = api_dir / "arxiv" / "id_old.json"
    stale.write_text('{"old": "format"}', encoding="utf-8")
    (api_dir / ".version").write_text(str(API_CACHE_VERSION - 1), encoding="utf-8")

    DiskCache(tmp_path)  # init re-reads the marker and wipes mismatches

    assert not stale.exists()
    assert (api_dir / ".version").read_text().strip() == str(API_CACHE_VERSION)


def _arxiv(**overrides) -> ExternalRecord:
    base = dict(
        backend=Backend.ARXIV,
        record_id="1706.03762",
        title="Attention Is All You Need (arXiv title)",
        authors=["A. Vaswani"],
        year=2017,
        arxiv_id="1706.03762",
    )
    base.update(overrides)
    return ExternalRecord(**base)


def _s2(**overrides) -> ExternalRecord:
    base = dict(
        backend=Backend.SEMANTIC_SCHOLAR,
        record_id="abc123",
        title="Attention Is All You Need (S2 title)",
        authors=["Vaswani A."],
        year=2017,
        venue="NeurIPS",
        publication_venue="NeurIPS",
        doi="10.5555/3295222.3295349",
        arxiv_id="1706.03762",
    )
    base.update(overrides)
    return ExternalRecord(**base)


def test_merge_uses_arxiv_for_title_authors_and_s2_for_venue_doi() -> None:
    merged = merge_records([_arxiv(), _s2()])
    assert merged is not None
    # arXiv is authoritative for title/authors/year (the canonical paper);
    # S2 wins venue/DOI (the published-record metadata).
    assert merged.title.endswith("(arXiv title)")
    assert merged.authors == ["A. Vaswani"]
    assert merged.provenance["title"] is Backend.ARXIV
    assert merged.venue == "NeurIPS"
    assert merged.doi == "10.5555/3295222.3295349"
    assert merged.provenance["doi"] is Backend.SEMANTIC_SCHOLAR


def test_merge_propagates_retraction_from_any_backend() -> None:
    # Cross-source retraction signal: even when the priority backend (arXiv)
    # says not retracted, an OpenAlex hit with is_retracted must flip the
    # merged record so the retraction guard fires.
    openalex_retracted = ExternalRecord(
        backend=Backend.OPENALEX,
        record_id="W123",
        title="Some retracted paper",
        authors=["X"],
        year=2020,
        doi="10.1109/access.2020.3018326",
        is_retracted=True,
    )
    merged = merge_records([_arxiv(is_retracted=False), openalex_retracted])
    assert merged is not None
    assert merged.is_retracted is True


def test_merge_empty_returns_none() -> None:
    assert merge_records([]) is None
