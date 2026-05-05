"""Disk cache key construction + TTL + prune."""

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


def test_api_cache_writes_version_marker(tmp_path):
    """First-time use should drop a ``.version`` marker into ``api_cache/``
    so a future on-disk format change can detect and wipe stale entries."""
    from refcopilot.cache.disk_cache import API_CACHE_VERSION

    DiskCache(tmp_path)
    marker = tmp_path / "api_cache" / ".version"
    assert marker.exists()
    assert marker.read_text().strip() == str(API_CACHE_VERSION)


def test_api_cache_wipes_on_version_mismatch(tmp_path):
    """When the on-disk version disagrees with the running code's version,
    every cache entry is from an incompatible format and must be wiped."""
    from refcopilot.cache.disk_cache import API_CACHE_VERSION

    # Pretend we have an old cache from a previous version.
    api_dir = tmp_path / "api_cache"
    (api_dir / "arxiv").mkdir(parents=True)
    stale = api_dir / "arxiv" / "id_old.json"
    stale.write_text('{"this": "is the old format"}')
    (api_dir / ".version").write_text(str(API_CACHE_VERSION - 1))

    DiskCache(tmp_path)  # must wipe on init

    assert not stale.exists()
    # Marker has been rewritten to the current version.
    assert (api_dir / ".version").read_text().strip() == str(API_CACHE_VERSION)


def test_api_cache_preserves_when_version_matches(tmp_path):
    """Same-version cache entries survive the compatibility check."""
    from refcopilot.cache.disk_cache import API_CACHE_VERSION

    DiskCache(tmp_path)  # writes the marker
    c = DiskCache(tmp_path)
    c.set_api("arxiv", "id_keep", {"x": 1})

    DiskCache(tmp_path)  # fresh handle, same version → must NOT wipe
    assert c.get_api("arxiv", "id_keep") == {"x": 1}
    assert (tmp_path / "api_cache" / ".version").read_text().strip() == str(
        API_CACHE_VERSION
    )
