"""Diagnostic redaction preserves readable content without contacting providers."""

import pytest

from llm.diagnostics import redact_provider_details


@pytest.mark.parametrize("password", ["p", "ab", "%70"])
def test_short_basic_auth_secret_preserves_words_and_safe_endpoint(password):
    endpoint = f"https://u:{password}@example.test:8443/v1?key=short-query"
    value = {
        "endpoint": endpoint,
        "message": f"paper paragraph at {endpoint}; password={password}; decoded password=p",
        "secret": password,
    }
    result = redact_provider_details(value, base_url=endpoint)
    assert result["endpoint"] == "https://example.test:8443/v1"
    assert result["message"].startswith("paper paragraph at https://example.test:8443/v1;")
    assert f"password={password};" not in result["message"]
    assert result["secret"] == "[redacted]"
    assert "short-query" not in str(result)
    assert value["endpoint"] == endpoint


def test_short_query_secret_and_nested_values_are_redacted_without_cascading():
    value = {"values": ("token=r", ["r", "report", "[redacted]"]), "text": "provider-password"}
    result = redact_provider_details(
        value, base_url="https://example.test/v1?token=r", secrets=("provider-password",)
    )
    assert result == {
        "values": ("token=[redacted]", ["[redacted]", "report", "[redacted]"]),
        "text": "[redacted]",
    }


def test_long_secret_inside_error_token_is_still_redacted():
    assert (
        redact_provider_details("prefix-fixture-secret-suffix", secrets=("fixture-secret",))
        == "prefix-[redacted]-suffix"
    )


def test_no_credentials_leaves_content_unchanged():
    assert redact_provider_details({"message": "paper response", "count": 2}) == {
        "message": "paper response",
        "count": 2,
    }
