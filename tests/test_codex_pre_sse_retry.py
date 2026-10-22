"""Bounded connection recovery, with no network and unchanged provider terminals."""

import errno
import io
import json
import ssl
import urllib.error

import pytest

from common import run_stats
from llm import client, codex_client
from llm.codex_auth import CodexAuth

USAGE = {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
SECRET = "fixture-codex-token"


class Stream:
    def __init__(self, lines, error=None):
        self.lines = lines
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __iter__(self):
        yield from self.lines
        if self.error:
            raise self.error


def event(value):
    return ("data: " + json.dumps(value) + "\n").encode()


def completed():
    return Stream(
        [
            event(
                {
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "output_text": '{"value": 42}',
                        "usage": USAGE,
                    },
                }
            )
        ]
    )


def eof():
    return urllib.error.URLError(ssl.SSLEOFError(8, "UNEXPECTED_EOF_WHILE_READING " + SECRET))


def run(monkeypatch, tmp_path, outcomes, *, image=False):
    requests, waits, records = [], [], []
    monkeypatch.setattr(client, "get_codex_auth", lambda **kw: CodexAuth(SECRET))
    monkeypatch.setattr(codex_client, "load_codex_instructions", lambda: "Frozen fixture instructions")
    monkeypatch.setattr(client.time, "sleep", waits.append)

    def post(request, *, timeout):
        requests.append((request.full_url, request.data, dict(request.headers), timeout))
        assert len(requests) <= len(outcomes), "unexpected additional provider attempt"
        outcome = outcomes[len(requests) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    original_record = run_stats.record_llm_call

    def record(**kwargs):
        records.append(kwargs.copy())
        return original_record(**kwargs)

    monkeypatch.setattr(codex_client.urllib.request, "urlopen", post)
    monkeypatch.setattr(run_stats, "record_llm_call", record)
    images = []
    if image:
        # Transport encoding does not decode pixels; keep an exact local byte source.
        picture = tmp_path / "picture.png"
        picture.write_bytes(b"fixed image transport bytes")
        images.append(str(picture))
    with run_stats.run_scope(tmp_path / "stats.json"):
        result = client.llm_json(
            "Frozen prompt",
            "Frozen system",
            client.LLMConfig("openai-codex", "fixture-model", "https://example.invalid", None),
            module="analysis",
            images=images,
        )
        stats = run_stats.read()["modules"]["analysis"]
    return result, stats, requests, waits, records


@pytest.mark.parametrize(
    "failure,image", [(eof(), False), (ConnectionResetError(errno.ECONNRESET, "reset"), True)]
)
def test_pre_sse_transient_retries_once_without_hiding_unknown_usage(monkeypatch, tmp_path, failure, image):
    result, stats, requests, waits, records = run(monkeypatch, tmp_path, [failure, completed()], image=image)
    assert result == {"value": 42}
    assert len(requests) == 2 and requests[0] == requests[1]
    assert waits == [1.0]
    assert len(records) == 2 and records[0]["failed"] and records[0]["usage"] == {}
    assert records[1]["usage"] == USAGE and not records[1].get("failed", False)
    assert SECRET not in records[0]["warning"]
    assert stats["token_usage"]["requests"] == 2
    assert stats["token_usage"]["total_tokens"] == 10
    assert stats["token_usage"]["estimated_requests"] == 0
    assert stats["failed_requests"] == stats["unavailable_usage_requests"] == 1
    assert stats["image_count"] == 2 * int(image)


def test_two_eofs_stop_without_a_third_attempt(monkeypatch, tmp_path):
    result, stats, requests, waits, records = run(monkeypatch, tmp_path, [eof(), eof()])
    assert result["status"] == "error" and SECRET not in str(result)
    assert len(requests) == len(records) == 2 and waits == [1.0]
    assert stats["failed_requests"] == stats["unavailable_usage_requests"] == 2
    assert stats["token_usage"]["requests"] == 2
    assert stats["token_usage"]["total_tokens"] == stats["token_usage"]["estimated_requests"] == 0


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.URLError(ssl.SSLCertVerificationError(1, "bad certificate")),
        urllib.error.HTTPError("https://example.invalid", 401, "unauthorized", {}, io.BytesIO(b"denied")),
        urllib.error.URLError(TimeoutError("timeout")),
        urllib.error.URLError("UNEXPECTED_EOF_WHILE_READING"),
    ],
)
def test_noneligible_transport_errors_are_not_retried(monkeypatch, tmp_path, failure):
    result, stats, requests, waits, records = run(monkeypatch, tmp_path, [failure])
    assert result["status"] == "error"
    assert len(requests) == len(records) == 1 and waits == []
    assert stats["failed_requests"] == stats["unavailable_usage_requests"] == 1


@pytest.mark.parametrize(
    "lines,measured",
    [
        ([b": keep alive\n"], False),
        ([b"data: malformed\n"], False),
        (
            [
                event({"type": "response.in_progress", "usage": USAGE}),
                event({"type": "response.output_text.delta", "delta": '{"value": 1}'}),
            ],
            True,
        ),
    ],
)
def test_any_response_line_closes_retry_window(monkeypatch, tmp_path, lines, measured):
    result, stats, requests, waits, records = run(monkeypatch, tmp_path, [Stream(lines, eof())])
    assert result["status"] == "error" and "value" not in result
    assert len(requests) == len(records) == 1 and waits == []
    assert stats["failed_requests"] == 1
    assert stats["unavailable_usage_requests"] == int(not measured)
    assert stats["token_usage"]["total_tokens"] == (10 if measured else 0)
