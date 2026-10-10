"""Large source identities and dependency previews keep bounded read sizes."""

import hashlib
from pathlib import Path

import pytest

from fact_generation.execution.tools import docker
from verification import code_sources


def test_full_identity_hash_and_dependency_prefix_use_bounded_reads(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Streaming source control forbids external services and processes")

    for name in ("llm.client.llm_json", "subprocess.Popen", "subprocess.run",
                 "requests.sessions.Session.request", "httpx.Client.send", "httpx.AsyncClient.send"):
        monkeypatch.setattr(name, forbidden)

    paper = tmp_path / "paper.pdf"
    preview = tmp_path / "requirements.txt"
    paper.touch()
    preview.touch()
    payload = b"a" * (1024 * 1024) + b"different-tail"
    text = b"ab\xe4\xb8\xadXYZ"
    reads = {paper: [], preview: []}

    class Stream:
        def __init__(self, path, data, maximum):
            self.path, self.data, self.maximum, self.position = path, data, maximum, 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size=-1):
            assert 0 < size <= self.maximum, "source read must have a bounded size"
            reads[self.path].append(size)
            chunk = self.data[self.position:self.position + size]
            self.position += len(chunk)
            return chunk

    original_open, original_read_bytes = Path.open, Path.read_bytes

    def bounded_open(path, mode="r", *args, **kwargs):
        if path in reads:
            assert mode == "rb"
            return Stream(path, payload if path == paper else text, 1024 * 1024 if path == paper else 4)
        return original_open(path, mode, *args, **kwargs)

    def no_whole_file_read(path):
        if path in reads:
            raise AssertionError("full-file byte allocation is forbidden")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "open", bounded_open)
    monkeypatch.setattr(Path, "read_bytes", no_whole_file_read)
    hash_error, digest = None, None
    try:
        digest = code_sources._hash(paper)
    except AssertionError as exc:
        hash_error = exc
    prefix = docker._read_text_limited(preview, max_bytes=4)
    assert hash_error is None
    assert digest == hashlib.sha256(payload).hexdigest()
    assert prefix == text[:4].decode("utf-8", errors="ignore")
    assert reads[paper] == [1024 * 1024] * 3
    assert reads[preview] == [4]
    assert code_sources._hash(tmp_path / "missing.pdf") is None
