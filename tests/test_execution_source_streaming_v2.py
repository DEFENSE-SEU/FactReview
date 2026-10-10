"""Execution checkpoint identity consumes full bytes with bounded memory."""
import hashlib
from pathlib import Path

import pytest

from fact_generation.execution import v2


def test_execution_source_hash_reads_complete_tail_without_whole_file_allocation(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Execution hashing control forbids external services and processes")

    for name in ("llm.client.llm_json", "subprocess.Popen", "subprocess.run",
                 "requests.sessions.Session.request", "httpx.Client.send", "httpx.AsyncClient.send"):
        monkeypatch.setattr(name, forbidden)

    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.touch()
    payload = b"a" * (1024 * 1024) + b"different-final-parameter"
    sizes = []

    class Stream:
        position = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size=-1):
            assert 0 < size <= 1024 * 1024
            sizes.append(size)
            result = payload[self.position:self.position + size]
            self.position += len(result)
            return result

    original_open, original_read = Path.open, Path.read_bytes

    def bounded_open(path, mode="r", *args, **kwargs):
        if path == checkpoint:
            assert mode == "rb"
            return Stream()
        return original_open(path, mode, *args, **kwargs)

    def no_whole_file_read(path):
        if path == checkpoint:
            raise AssertionError("Execution must not allocate the complete checkpoint for its hash")
        return original_read(path)

    monkeypatch.setattr(Path, "open", bounded_open)
    monkeypatch.setattr(Path, "read_bytes", no_whole_file_read)
    assert v2._sha(checkpoint) == hashlib.sha256(payload).hexdigest()
    assert sizes
    with pytest.raises(FileNotFoundError):
        v2._sha(tmp_path / "absent-checkpoint.bin")
