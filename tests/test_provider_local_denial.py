"""Current issuer identity is required before excluding a predelegate denial."""
import contextlib
import contextvars
import dataclasses
import json
import ssl
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from common import run_stats
from llm import client, codex_client
from llm.codex_auth import CodexAuth


def test_current_single_use_local_denial_keeps_first_eof_unknown(tmp_path, monkeypatch):
    external = []

    def forbidden(*args, **kwargs):
        external.append(True)
        pytest.fail("Real provider, retrieval, network, Docker or process forbidden")

    for name in ("requests.sessions.Session.request", "httpx.Client.send", "httpx.AsyncClient.send",
                 "socket.create_connection", "subprocess.Popen", "subprocess.run"):
        monkeypatch.setattr(name, forbidden)
    monkeypatch.setitem(sys.modules, "docker", SimpleNamespace(from_env=forbidden, DockerClient=forbidden))
    monkeypatch.setattr(client, "get_codex_auth", lambda **kwargs: CodexAuth("public-mock-token"))
    monkeypatch.setattr(codex_client, "load_codex_instructions", lambda: "Public frozen instructions")
    monkeypatch.setattr(client.time, "sleep", lambda seconds: None)
    # Import after the external boundaries are forbidden; the old source lacks this contract.
    from common import provider_attempts as attempts

    original_record = run_stats.record_llm_call
    original_invoke = codex_client.invoke_codex
    config = client.LLMConfig("openai-codex", "public-model", "https://example.invalid", None)
    scope_args = {"module": "analysis", "provider": config.provider, "model": config.model}

    for mode in ("typed", "plain", "attribute", "copied", "malformed", "unregistered", "stale", "cross", "subclass", "duplicate"):
        issuer = attempts.PhysicalAttemptIssuer()
        transport, physical, usage_rows, local_rows, receipts, acks = [], [], [], [], [], []
        known_tokens = 0
        missing_usage = 0
        blocked = 0
        foreign_context = foreign_manager = None
        foreign_receipt = None
        if mode in {"stale", "cross"}:
            foreign_context = contextvars.Context()
            foreign_manager = issuer.logical_scope(99 if mode == "cross" else 0, **scope_args)
            foreign_context.run(foreign_manager.__enter__)
            foreign_context.run(issuer.begin_attempt, 1)
            foreign_receipt = foreign_context.run(issuer.deny_attempt, 1, "per_logical_physical_limit").receipt
            if mode == "stale":
                foreign_context.run(foreign_manager.__exit__, None, None, None)
                foreign_manager = None

        def post(request, *, timeout):
            transport.append((request.get_method(), timeout))
            assert len(transport) == 1
            raise urllib.error.URLError(ssl.SSLEOFError(8, "UNEXPECTED_EOF_WHILE_READING"))

        def invoke(**kwargs):
            nonlocal blocked
            index = len(physical)
            issuer.begin_attempt(index)
            row = {"index": index, "logical_index": 0, "status": "pending"}
            physical.append(row)
            if index:
                row["status"] = "blocked_before_provider"
                blocked += 1
                if mode in {"typed", "copied", "malformed", "duplicate", "subclass"}:
                    error = issuer.deny_attempt(index, "per_logical_physical_limit")
                    receipts.append(error.receipt)
                    if mode == "copied":
                        error = attempts.PhysicalAttemptNotStarted(dataclasses.replace(error.receipt))
                    elif mode == "malformed":
                        error = attempts.PhysicalAttemptNotStarted(dataclasses.replace(error.receipt, attempt_index=[]))
                    elif mode == "subclass":
                        class PretendDenial(attempts.PhysicalAttemptNotStarted):
                            pass
                        error = PretendDenial(error.receipt)
                    raise error
                if mode in {"stale", "cross"}:
                    raise attempts.PhysicalAttemptNotStarted(foreign_receipt)
                if mode == "unregistered":
                    forged = attempts.DenialReceipt("0" * 32, 0, index, "per_logical_physical_limit", **scope_args)
                    raise attempts.PhysicalAttemptNotStarted(forged)
                error = RuntimeError("Declared physical boundary blocked: physical_attempt_not_started")
                if mode == "attribute":
                    error.physical_not_started = True
                    error.receipt = {"logical_index": 0, "attempt_index": index}
                raise error
            issuer.enter_delegate(index)
            row.update(status="running", admitted_number=1)
            try:
                return original_invoke(**kwargs)
            except codex_client.CodexResponseError as exc:
                row.update(status="raised", pre_sse_transient=exc.pre_sse_transient)
                raise

        def record(**kwargs):
            nonlocal missing_usage, known_tokens
            ack = original_record(**kwargs)
            if attempts.is_denial_acknowledgement(ack, receipt=kwargs.get("local_denial")):
                receipt = ack.receipt
                assert receipt.logical_index == 0 and receipt.attempt_index == 1
                assert physical[receipt.attempt_index]["status"] == "blocked_before_provider"
                assert "admitted_number" not in physical[receipt.attempt_index]
                assert receipt.receipt_id not in {row.receipt_id for row in local_rows}
                local_rows.append(receipt)
                acks.append(ack)
            else:
                usage_rows.append(kwargs)
                missing_usage += int(not kwargs.get("usage"))
                known_tokens += int((kwargs.get("usage") or {}).get("total_tokens", 0))
            return ack

        monkeypatch.setattr(codex_client.urllib.request, "urlopen", post)
        monkeypatch.setattr(client, "invoke_codex", invoke)
        monkeypatch.setattr(run_stats, "record_llm_call", record)
        stats_path = tmp_path / (mode + ".json")
        try:
            with run_stats.run_scope(stats_path), issuer.logical_scope(0, **scope_args):
                result = client.llm_json("Public prompt", "Public system", config, module="analysis")
                assert result["status"] == "error"
                assert len(transport) == 1 and transport[0] == ("POST", 120)
                assert len(physical) == 2 and physical[0]["status"] == "raised"
                assert physical[0]["pre_sse_transient"] is True and blocked == 1
                assert sum("admitted_number" in row for row in physical) == 1
                assert usage_rows[0]["failed"] is True and usage_rows[0]["usage"] == {}
                if mode == "duplicate":
                    assert len(acks) == 1
                    assert attempts.is_denial_acknowledgement(acks[0], receipt=receipts[0])
                    assert attempts.is_denial_acknowledgement(acks[0], receipt=receipts[0])
                    # Wrapper verification is read-only; native is the sole consume.
                    assert record(**{**scope_args, "usage": {}, "failed": True, "local_denial": receipts[0]}) is None
                stats = run_stats.read_initialized()["modules"]["analysis"]
                classified = mode in {"typed", "duplicate"}
                expected_unknown = 2 if mode == "duplicate" or not classified else 1
                assert stats["token_usage"]["requests"] == expected_unknown
                assert stats["failed_requests"] == stats["unavailable_usage_requests"] == expected_unknown
                assert stats["token_usage"]["total_tokens"] == stats["token_usage"]["estimated_requests"] == 0
                assert stats["estimated"] is False and known_tokens == 0
                assert len(stats["local_denials"]) == len(local_rows) == int(classified)
                assert len(usage_rows) == missing_usage == expected_unknown
                assert stats["providers"] == {config.provider: expected_unknown}
                assert stats["models"] == {config.model: expected_unknown}
                if classified:
                    saved = stats["local_denials"][0]
                    assert saved["receipt_id"] == receipts[0].receipt_id
                    assert saved["logical_index"] == 0 and saved["attempt_index"] == 1
                    assert saved["reason_code"] == "per_logical_physical_limit" and saved["failed"] is True
                    assert not ({"usage", "tokens", "admitted_number", "prompt", "response", "error"} & set(saved))
                    assert attempts.is_denial_acknowledgement(acks[0], receipt=dataclasses.replace(receipts[0])) is False
                    assert attempts.is_denial_acknowledgement(attempts.DenialAcknowledgement(receipts[0]), receipt=receipts[0]) is False
                assert stats["warnings"] and any("Token totals are incomplete" in item for item in stats["warnings"])
                totals = run_stats.with_totals(run_stats.read())["total"]
                assert totals["local_denial_count"] == int(classified)
                assert totals["unavailable_usage_requests"] == expected_unknown
                # The public current wrapper's prior_problem still blocks the next logical call.
                prior_problem = "prior unmetered or missing usage" if missing_usage else ""
                with pytest.raises(RuntimeError, match="prior unmetered or missing usage"):
                    if prior_problem:
                        blocked += 1
                        raise RuntimeError(prior_problem)
                    pytest.fail("unknown usage cannot admit another logical request")
                closed = not blocked and not missing_usage and all(row["status"] == "returned" for row in physical)
                assert closed is False and len(transport) == 1 and blocked == 2
            for ack in acks:
                assert attempts.is_denial_acknowledgement(ack, receipt=ack.receipt) is False
        finally:
            if foreign_manager is not None:
                foreign_context.run(foreign_manager.__exit__, None, None, None)

    # Already-delegated attempts cannot issue a denial. Persistence failure cannot yield/reuse an ack.
    issuer = attempts.PhysicalAttemptIssuer()
    with run_stats.run_scope(tmp_path / "persistence.json"), issuer.logical_scope(0, **scope_args):
        issuer.begin_attempt(0)
        issuer.enter_delegate(0)
        with pytest.raises(ValueError, match="denial_attempt_not_pending"):
            issuer.deny_attempt(0, "per_logical_physical_limit")
        issuer.begin_attempt(1)
        error = issuer.deny_attempt(1, "per_logical_physical_limit")
        update = Path.replace
        def fail_persistence(path, target):
            raise OSError("public persistence failure")
        monkeypatch.setattr(Path, "replace", fail_persistence)
        with pytest.raises(OSError, match="public persistence failure"):
            original_record(**scope_args, failed=True, usage={}, local_denial=error.receipt)
        monkeypatch.setattr(Path, "replace", update)
        assert original_record(**scope_args, failed=True, usage={}, local_denial=error.receipt) is None
        saved = run_stats.read_initialized()["modules"]["analysis"]
        assert saved["local_denials"] == [] and saved["unavailable_usage_requests"] == 1

    # A transaction or atomic replacement that silently returns without commit cannot acknowledge.
    for mode in ("transaction_no_commit", "replace_no_commit"):
        issuer = attempts.PhysicalAttemptIssuer()
        path = tmp_path / (mode + ".json")
        with run_stats.run_scope(path), issuer.logical_scope(0, **scope_args):
            issuer.begin_attempt(0)
            error = issuer.deny_attempt(0, "physical_call_budget")
            with monkeypatch.context() as patch:
                if mode == "transaction_no_commit":
                    @contextlib.contextmanager
                    def no_commit(target):
                        yield run_stats.read_initialized(target)
                    patch.setattr(run_stats, "_file_locked_payload", no_commit)
                else:
                    patch.setattr(Path, "replace", lambda self, target: Path(target))
                with pytest.raises(RuntimeError, match="local_denial_not_committed"):
                    original_record(**scope_args, failed=True, usage={}, local_denial=error.receipt)
            assert run_stats.read_initialized(path)["modules"]["analysis"]["local_denials"] == []
            assert attempts.is_denial_acknowledgement(
                attempts.DenialAcknowledgement(error.receipt), receipt=error.receipt) is False
            assert original_record(**scope_args, failed=True, usage={}, local_denial=error.receipt) is None
            saved = run_stats.read_initialized(path)["modules"]["analysis"]
            assert saved["local_denials"] == [] and saved["unavailable_usage_requests"] == 1

    # The denial commits to the captured target even if a later legacy path lookup would change.
    for mode in ("target_withdrawn", "target_redirected"):
        issuer = attempts.PhysicalAttemptIssuer()
        path, other = tmp_path / (mode + ".json"), tmp_path / (mode + "-other.json")
        run_stats.initialize(other, activate=False)
        with run_stats.run_scope(path), issuer.logical_scope(0, **scope_args):
            issuer.begin_attempt(0)
            error = issuer.deny_attempt(0, "physical_call_budget")
            path_calls = []
            def unstable_target():
                path_calls.append(True)
                return path if len(path_calls) == 1 else (None if mode == "target_withdrawn" else other)
            with monkeypatch.context() as patch:
                patch.setattr(run_stats, "stats_path", unstable_target)
                ack = original_record(**scope_args, failed=True, usage={}, local_denial=error.receipt)
                assert attempts.is_denial_acknowledgement(ack, receipt=error.receipt)
                assert path_calls == [True]
            complete = {"version": "predelegate-local-denial-v1", **dataclasses.asdict(error.receipt),
                        "duration_sec": 0.0, "image_count": 0, "failed": True}
            assert run_stats.read_initialized(path)["modules"]["analysis"]["local_denials"] == [complete]
            assert run_stats.read_initialized(other)["modules"]["analysis"]["local_denials"] == []

    # Old version-1 stats remain initialized without the optional new collection.
    legacy_path = tmp_path / "legacy.json"
    with run_stats.run_scope(legacy_path):
        payload = run_stats.read_initialized()
        for row in payload["modules"].values():
            row.pop("local_denials", None)
        legacy_path.write_text(json.dumps(payload), encoding="utf-8")
        assert run_stats.read_initialized() == payload
        assert run_stats.with_totals(run_stats.read())["total"]["local_denial_count"] == 0
        payload["modules"]["analysis"]["local_denials"] = "invalid"
        legacy_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="usage_local_denials_invalid"):
            run_stats.read_initialized()
        payload["modules"]["analysis"].pop("local_denials")
        payload["modules"]["analysis"]["unavailable_usage_requests"] = "unknown"
        legacy_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="usage_counts_invalid"):
            run_stats.read_initialized()
    assert external == []
