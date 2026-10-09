"""Preserve complete writing-candidate responses before source binding checks."""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager

from common import run_stats
from llm.diagnostics import redact_provider_details, sanitized_endpoint


@contextmanager
def writing_call_audit(*, module, cfg, system, payload, injected=False):
    stats = run_stats.stats_path()
    if module != "screening_writing" or stats is None:
        yield {}
        return
    directory = stats.parent / "writing_calls"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (uuid.uuid4().hex + ".json")
    record = {
        "version": "writing-candidates-v1",
        "module": module,
        "provider": cfg.provider,
        "model": cfg.model,
        "endpoint": sanitized_endpoint(cfg.base_url),
        "transport": "injected" if injected else "live",
        "system": system,
        "payload": payload,
        "status": "started",
        "binding_status": "not_evaluated",
    }

    def save():
        path.write_text(
            json.dumps(redact_provider_details(record, cfg), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    started = time.perf_counter()
    save()
    try:
        yield record
        record["status"] = "returned"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["duration_seconds"] = time.perf_counter() - started
        save()
