"""Completed and interrupted SSE requests, with only in-memory provider streams."""

import json

import pytest

from common import run_stats
from llm import client, codex_client
from llm.codex_auth import CodexAuth

USAGE = {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}


class Stream:
    def __init__(self, events, *, fail_after=True):
        self.events = events
        self.fail_after = fail_after
        self.tail_read = False
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def __iter__(self):
        for event in self.events:
            yield ("data: " + json.dumps(event) + "\n\n").encode()
        self.tail_read = True
        if self.fail_after:
            raise TimeoutError("synthetic trailing timeout with fixture-oauth-secret")
        yield b"data: [DONE]\n"


def call(monkeypatch, tmp_path, events, *, fail_after=True):
    stream = Stream(events, fail_after=fail_after)
    monkeypatch.setattr(client, "get_codex_auth", lambda **kwargs: CodexAuth("fixture-oauth-secret"))
    monkeypatch.setattr(codex_client, "load_codex_instructions", lambda: "Fixture instructions")
    monkeypatch.setattr(codex_client.urllib.request, "urlopen", lambda *a, **kw: stream)
    with run_stats.run_scope(tmp_path / "stats.json"):
        result = client.llm_json(
            "Fixture prompt",
            "Fixture system",
            client.LLMConfig("openai-codex", "fixture-model", "https://example.invalid", None),
            module="analysis",
        )
        stats = run_stats.read()["modules"]["analysis"]
    return result, stats, stream


def test_valid_completed_stops_before_trailing_timeout_and_retains_measured_usage(monkeypatch, tmp_path):
    result, stats, stream = call(
        monkeypatch,
        tmp_path,
        [
            {"type": "response.output_text.delta", "delta": '{"value": 1}'},
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "output_text": '{"value": 42}',
                    "usage": USAGE,
                },
            },
        ],
    )
    assert result == {"value": 42}
    assert not stream.tail_read and stream.closed
    assert stats["token_usage"]["total_tokens"] == 10
    assert stats["failed_requests"] == stats["unavailable_usage_requests"] == 0


@pytest.mark.parametrize(
    "event_type", ["response.failed", "response.incomplete", "response.cancelled", "error"]
)
def test_failed_terminal_stops_with_measured_usage(monkeypatch, tmp_path, event_type):
    result, stats, stream = call(
        monkeypatch,
        tmp_path,
        [
            {"type": "response.output_text.delta", "delta": '{"value": 1}'},
            {"type": event_type, "usage": USAGE},
        ],
    )
    assert result["status"] == "error" and "value" not in result
    assert not stream.tail_read and stream.closed
    assert stats["token_usage"]["total_tokens"] == 10
    assert stats["failed_requests"] == 1 and stats["unavailable_usage_requests"] == 0


@pytest.mark.parametrize("fail_after", [False, True])
def test_unfinished_stream_never_accepts_partial_json_and_preserves_usage(monkeypatch, tmp_path, fail_after):
    result, stats, stream = call(
        monkeypatch,
        tmp_path,
        [
            {"type": "response.in_progress", "usage": USAGE},
            {"type": "response.output_text.delta", "delta": '{"value": 1}'},
        ],
        fail_after=fail_after,
    )
    assert result["status"] == "error" and "value" not in result
    assert "fixture-oauth-secret" not in str(result)
    assert stream.tail_read and stream.closed
    assert stats["failed_requests"] == 1 and stats["unavailable_usage_requests"] == 0
    assert stats["token_usage"]["total_tokens"] == 10


@pytest.mark.parametrize("response", [None, [], {}, {"status": "in_progress"}, {"status": "failed"}])
def test_malformed_completed_cannot_authorize_partial_answer(monkeypatch, tmp_path, response):
    result, stats, _ = call(
        monkeypatch,
        tmp_path,
        [
            {"type": "response.output_text.delta", "delta": '{"value": 1}'},
            {"type": "response.completed", "response": response, "usage": USAGE},
        ],
        fail_after=False,
    )
    assert result["status"] == "error" and "value" not in result
    assert stats["failed_requests"] == 1
    assert stats["token_usage"]["total_tokens"] == 10


def test_completed_without_text_is_failed_with_usage(monkeypatch, tmp_path):
    result, stats, stream = call(
        monkeypatch,
        tmp_path,
        [
            {"type": "response.completed", "response": {"status": "completed", "usage": USAGE}},
        ],
    )
    assert result["status"] == "error"
    assert stats["token_usage"]["total_tokens"] == 10
    assert stream.closed and not stream.tail_read


def test_completed_delta_only_text_remains_supported(monkeypatch, tmp_path):
    result, _, stream = call(
        monkeypatch,
        tmp_path,
        [
            {"type": "response.output_text.delta", "delta": '{"value": 42}'},
            {"type": "response.completed", "response": {"status": "completed", "usage": USAGE}},
        ],
    )
    assert result == {"value": 42}
    assert not stream.tail_read


@pytest.mark.parametrize(
    "extra",
    [
        {"output": "not-a-list"},
        {"output": ["not-an-item"]},
        {"output": [{"content": "not-content"}]},
        {"output": [{"content": [{"type": "output_text", "text": 42}]}]},
        {"output_text": 42},
        {"error": {"message": "failed response"}},
    ],
)
def test_malformed_terminal_body_does_not_reuse_partial_text(monkeypatch, tmp_path, extra):
    result, stats, _ = call(
        monkeypatch,
        tmp_path,
        [
            {"type": "response.output_text.delta", "delta": '{"value": 1}'},
            {"type": "response.completed", "response": {"status": "completed", "usage": USAGE, **extra}},
        ],
        fail_after=False,
    )
    assert result["status"] == "error" and "value" not in result
    assert stats["token_usage"]["total_tokens"] == 10


def test_explicit_terminal_refusal_rejects_prior_json_delta(monkeypatch, tmp_path):
    result, stats, stream = call(
        monkeypatch,
        tmp_path,
        [
            {"type": "response.output_text.delta", "delta": '{"value":1}'},
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "usage": USAGE,
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "refusal", "refusal": "Cannot complete this request."}],
                        }
                    ],
                },
            },
        ],
    )
    assert result["status"] == "error" and "value" not in result
    assert stats["failed_requests"] == 1 and stats["token_usage"]["total_tokens"] == 10
    assert stream.closed and not stream.tail_read


@pytest.mark.parametrize("status", ["incomplete", "in_progress", "failed", None])
def test_noncompleted_terminal_message_rejects_json(monkeypatch, tmp_path, status):
    result, stats, stream = call(
        monkeypatch,
        tmp_path,
        [
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "usage": USAGE,
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "status": status,
                            "content": [{"type": "output_text", "text": '{"value":1}'}],
                        }
                    ],
                },
            },
        ],
    )
    assert result["status"] == "error" and "value" not in result
    assert stats["failed_requests"] == 1 and stats["token_usage"]["total_tokens"] == 10
    assert stream.closed and not stream.tail_read
