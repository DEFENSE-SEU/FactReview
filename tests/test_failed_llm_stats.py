"""Call accounting must expose failed requests and unknown visual token costs."""

import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from common import run_stats
from llm.client import LLMConfig, llm_json


@pytest.fixture
def picture(tmp_path):
    path = tmp_path / "figure.png"
    Image.new("RGB", (96, 48), "white").save(path)
    return str(path)


def install_provider(monkeypatch, provider, create):
    if provider == "openai-codex":
        monkeypatch.setattr("llm.client.get_codex_auth", lambda **kwargs: object())
        monkeypatch.setattr("llm.client.invoke_codex", create)
    elif provider == "claude":
        monkeypatch.setitem(
            sys.modules,
            "anthropic",
            SimpleNamespace(
                Anthropic=lambda **kwargs: SimpleNamespace(messages=SimpleNamespace(create=create))
            ),
        )
    else:
        monkeypatch.setattr(
            "openai.OpenAI",
            lambda **kwargs: SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            ),
        )


@pytest.mark.parametrize("provider", ["openai", "claude", "openai-codex"])
def test_failed_provider_is_counted_with_unavailable_usage(monkeypatch, tmp_path, picture, provider):
    def fail(**kwargs):
        raise TimeoutError("fixture request timed out")

    install_provider(monkeypatch, provider, fail)
    path = tmp_path / "stats.json"
    with run_stats.run_scope(path):
        result = llm_json(
            "A prompt long enough to yield an estimate if incorrectly counted as successful.",
            "Inspect the pixels.",
            LLMConfig(provider, "fixture-model", "https://example.invalid", "fixture"),
            module="screening_figures",
            images=[picture],
        )
    assert result["status"] == "error"
    stats = run_stats.with_totals(run_stats.read(path))
    row = stats["modules"]["analysis"]
    assert row["failed_requests"] == 1
    assert row["unavailable_usage_requests"] == 1
    assert row["image_count"] == 1
    assert row["token_usage"]["requests"] == 1
    assert row["token_usage"]["total_tokens"] == 0
    assert row["token_usage"]["estimated_requests"] == 0
    assert row["estimated"] is False
    assert row["providers"] == {provider: 1}
    assert any("Token totals are incomplete" in warning for warning in row["warnings"])
    for key in ("failed_requests", "unavailable_usage_requests", "image_count"):
        assert stats["total"][key] == 1
    assert any(
        "failed=1" in line and "usage unavailable=1" in line for line in run_stats.format_summary_table(stats)
    )


def test_missing_local_image_records_attempt_without_contacting_provider(monkeypatch, tmp_path):
    def forbidden(**kwargs):
        pytest.fail("a missing local image cannot trigger a provider request")

    install_provider(monkeypatch, "openai", forbidden)
    path = tmp_path / "stats.json"
    with run_stats.run_scope(path):
        result = llm_json(
            "Inspect",
            "Pixels",
            LLMConfig("openai", "fixture-model", "https://example.invalid", "fixture"),
            module="screening_figures",
            images=[str(tmp_path / "missing.png")],
        )
    assert result["status"] == "error"
    total = run_stats.with_totals(run_stats.read(path))["total"]
    assert total["failed_requests"] == total["unavailable_usage_requests"] == total["image_count"] == 1
    assert total["token_usage"]["requests"] == 1
    assert total["token_usage"]["estimated_requests"] == 0


@pytest.mark.parametrize("provider", ["openai", "claude", "openai-codex"])
@pytest.mark.parametrize("exact", [True, False])
def test_success_records_images_and_distinguishes_exact_from_text_only_usage(
    monkeypatch, tmp_path, picture, provider, exact
):
    def create(**kwargs):
        if provider == "openai-codex":
            return '{"findings": []}', {
                "input_tokens": 100,
                "output_tokens": 5,
                "total_tokens": 105,
            } if exact else {}
        usage = (
            SimpleNamespace(
                input_tokens=100, output_tokens=5, prompt_tokens=100, completion_tokens=5, total_tokens=105
            )
            if exact
            else None
        )
        return SimpleNamespace(
            content=[SimpleNamespace(text='{"findings": []}')],
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"findings": []}'))],
            usage=usage,
        )

    install_provider(monkeypatch, provider, create)
    path = tmp_path / "stats.json"
    with run_stats.run_scope(path):
        result = llm_json(
            "Inspect both figures",
            "Use their pixels",
            LLMConfig(provider, "fixture-model", "https://example.invalid", "fixture"),
            module="screening_figures",
            images=[picture, picture],
        )
    assert result == {"findings": []}
    total = run_stats.with_totals(run_stats.read(path))["total"]
    assert total["failed_requests"] == 0
    assert total["image_count"] == 2
    assert total["unavailable_usage_requests"] == int(not exact)
    assert total["token_usage"]["requests"] == 1
    assert total["token_usage"]["estimated_requests"] == int(not exact)
    if exact:
        assert total["token_usage"]["total_tokens"] == 105
        assert not total["warnings"]
    else:
        assert total["token_usage"]["total_tokens"] > 0
        assert any("image token cost is unknown" in warning for warning in total["warnings"])


def test_old_token_keys_unchanged_and_reported_zero_usage_is_exact(tmp_path):
    path = tmp_path / "stats.json"
    with run_stats.run_scope(path):
        run_stats.record_llm_call(
            module="analysis", usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        )
    total = run_stats.with_totals(run_stats.read(path))["total"]
    assert set(total["token_usage"]) == {
        "requests",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "estimated_requests",
    }
    assert total["unavailable_usage_requests"] == 0
    assert total["estimated"] is False
    assert total["warnings"] == []
