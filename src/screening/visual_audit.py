"""Per-call provenance for image-backed checks in an active review run."""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit, urlunsplit

from common import run_stats


def sanitized_endpoint(base_url: str | None) -> str:
    endpoint = urlsplit(base_url or "")
    return urlunsplit((endpoint.scheme, endpoint.netloc.rsplit("@", 1)[-1], endpoint.path, "", ""))


def redact_provider_details(value, cfg):
    """Copy diagnostic data, removing configured credentials and endpoint secrets."""
    base_url = cfg.base_url or ""
    endpoint = urlsplit(base_url)
    replacements = {}
    if base_url:
        replacements[base_url] = sanitized_endpoint(base_url)
    if "@" in endpoint.netloc:
        # Include the delimiter so a common username does not alter prose.
        userinfo = endpoint.netloc.rsplit("@", 1)[0] + "@"
        replacements[userinfo] = ""
        replacements[unquote(userinfo)] = ""
    credentials = [cfg.api_key, endpoint.password]
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


@contextmanager
def visual_call_audit(*, module, cfg, system, payload, images, injected=False):
    stats = run_stats.stats_path()
    if not images or stats is None:
        yield {}
        return
    from PIL import Image

    directory = stats.parent / "visual_calls"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (re.sub(r"[^a-zA-Z0-9_.-]", "_", module) + "_" + uuid.uuid4().hex + ".json")
    record = {
        "module": module,
        "provider": cfg.provider,
        "model": cfg.model,
        "endpoint": sanitized_endpoint(cfg.base_url),
        "transport": "injected" if injected else "live",
        "system": system,
        "payload": payload,
        "images": [],
        "status": "started",
    }
    start = time.perf_counter()

    def save():
        safe_record = dict(record)
        for key in ("response", "error"):
            if key in safe_record:
                safe_record[key] = redact_provider_details(safe_record[key], cfg)
        path.write_text(json.dumps(safe_record, ensure_ascii=False, indent=2), encoding="utf-8")

    save()
    try:
        for source in images:
            image_path = Path(source).resolve()
            content = image_path.read_bytes()
            with Image.open(image_path) as pixels:
                metadata = {"width": pixels.width, "height": pixels.height, "format": pixels.format}
            record["images"].append(
                {"path": str(image_path), "sha256": hashlib.sha256(content).hexdigest(), **metadata}
            )
        save()
        yield record
        record["status"] = "ok"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["duration_seconds"] = time.perf_counter() - start
        save()
