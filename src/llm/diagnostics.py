"""Credential-safe provider diagnostics shared by transport and review callers."""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, unquote, urlsplit, urlunsplit


def sanitized_endpoint(base_url: str | None) -> str:
    endpoint = urlsplit(base_url or "")
    return urlunsplit((endpoint.scheme, endpoint.netloc.rsplit("@", 1)[-1], endpoint.path, "", ""))


def redact_provider_details(value, cfg=None, *, base_url=None, secrets=()):
    """Copy diagnostic data, removing configured credentials and endpoint secrets."""
    base_url = (cfg.base_url if cfg is not None else base_url) or ""
    endpoint = urlsplit(base_url)
    replacements = {}
    if base_url:
        replacements[base_url] = sanitized_endpoint(base_url)
    if "@" in endpoint.netloc:
        # Include the delimiter so a common username does not alter prose.
        userinfo = endpoint.netloc.rsplit("@", 1)[0] + "@"
        replacements[userinfo] = ""
        replacements[unquote(userinfo)] = ""
    credentials = [cfg.api_key if cfg is not None else None, endpoint.password, *secrets]
    credentials.extend(
        token
        for name, token in parse_qsl(endpoint.query)
        if re.search(r"key|token|secret|password|signature|credential|auth", name, re.IGNORECASE)
    )
    for credential in credentials:
        if credential:
            replacements[credential] = "[redacted]"
            replacements[unquote(credential)] = "[redacted]"

    def redact(item):
        if isinstance(item, str):
            for original in sorted(replacements, key=len, reverse=True):
                item = item.replace(original, replacements[original])
            return item
        if isinstance(item, dict):
            return {key: redact(content) for key, content in item.items()}
        if isinstance(item, list):
            return [redact(content) for content in item]
        if isinstance(item, tuple):
            return tuple(redact(content) for content in item)
        return item

    return redact(value)


