"""Test B.1 — disk cache key construction + TTL + prune."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from refcopilot.cache.disk_cache import DiskCache, cache_key_for_paper


@pytest.mark.parametrize(
    ("spec", "starts_with"),
    [
        ("arxiv:1706.03762", "arxiv_1706.03762"),
        ("1706.03762", "arxiv_1706.03762"),
        ("1706.03762v3", "arxiv_1706.03762v3"),
        ("https://arxiv.org/abs/1706.03762", "url_"),
        ("just some bibliography text", "spec_"),
    ],
)
def test_cache_key_for_paper(spec, starts_with):
    assert cache_key_for_paper(spec).startswith(starts_with)


def test_cache_key_for_local_file(tmp_path):
    p = tmp_path / "paper.pdf"
    p.write_bytes(b"%PDF-1.4 fake content")
    key = cache_key_for_paper(str(p))
    assert key.startswith("file_paper.pdf_")


def test_disk_cache_set_and_get(tmp_path):
    c = DiskCache(tmp_path, ttl_days=30)
    c.set_api("semantic_scholar", "doi_10.x_y", {"hello": "world"})
    out = c.get_api("semantic_scholar", "doi_10.x_y")
    assert out == {"hello": "world"}


def test_disk_cache_respects_ttl(tmp_path):
    c = DiskCache(tmp_path, ttl_days=30)
    c.set_api("arxiv", "k1", {"x": 1})
    path = c._api_path("arxiv", "k1")
    # Make it look ancient
    old = time.time() - (40 * 86400)
    os.utime(path, (old, old))
    assert c.get_api("arxiv", "k1") is None


def test_disk_cache_disabled(tmp_path):
    c = DiskCache(tmp_path, enabled=False)
    c.set_api("arxiv", "k1", {"x": 1})
    assert c.get_api("arxiv", "k1") is None


def test_disk_cache_prune_removes_stale_only(tmp_path):
    c = DiskCache(tmp_path, ttl_days=1)
    c.set_api("arxiv", "fresh", {"x": 1})
    c.set_api("arxiv", "stale", {"x": 1})
    stale_path = c._api_path("arxiv", "stale")
    old = time.time() - (5 * 86400)
    os.utime(stale_path, (old, old))

    removed = c.prune()
    assert removed == 1
    assert c._api_path("arxiv", "fresh").exists()
    assert not stale_path.exists()


def test_paper_dir_creates(tmp_path):
    c = DiskCache(tmp_path)
    d = c.paper_dir("arxiv:1706.03762")
    assert d.exists()
    assert d.name == "arxiv_1706.03762"
