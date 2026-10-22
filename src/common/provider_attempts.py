"""Single-use program receipts for a registered physical delegate not entered."""
from __future__ import annotations

import contextlib
import contextvars
import math
import threading
import uuid
from dataclasses import dataclass

REASON_CODES = frozenset({
    "per_logical_physical_limit", "physical_call_budget", "known_token_budget",
    "prior_usage_incomplete", "application_import_integrity",
})
_CURRENT = contextvars.ContextVar("factreview_physical_attempt_scope", default=None)


@dataclass(frozen=True, slots=True)
class DenialReceipt:
    receipt_id: str
    logical_index: int
    attempt_index: int
    reason_code: str
    module: str
    provider: str
    model: str


@dataclass(frozen=True, slots=True)
class DenialAcknowledgement:
    receipt: DenialReceipt


class PhysicalAttemptNotStarted(RuntimeError):
    def __init__(self, receipt):
        super().__init__("physical_attempt_not_started")
        self.receipt = receipt


@dataclass(frozen=True, slots=True)
class _Scope:
    issuer: PhysicalAttemptIssuer
    nonce: object
    logical_index: int
    module: str
    provider: str
    model: str


class PhysicalAttemptIssuer:
    """Used only by a program boundary before it calls its physical delegate."""

    def __init__(self):
        self._lock = threading.RLock()
        self._open_scopes = set()
        self._attempts = {}
        self._receipt_ids = set()

    @contextlib.contextmanager
    def logical_scope(self, logical_index, *, module, provider, model):
        if (_CURRENT.get() is not None or type(logical_index) is not int or not 0 <= logical_index < 2**31
                or type(module) is not str or not 0 < len(module) <= 64
                or provider != "openai-codex" or type(model) is not str or not 0 < len(model) <= 256):
            raise ValueError("denial_logical_scope_invalid")
        scope = _Scope(self, object(), logical_index, module, provider, model)
        with self._lock:
            self._open_scopes.add(scope.nonce)
        token = _CURRENT.set(scope)
        try:
            yield
        finally:
            with self._lock:
                self._open_scopes.remove(scope.nonce)
            _CURRENT.reset(token)

    def _scope(self):
        scope = _CURRENT.get()
        if scope is None or scope.issuer is not self or scope.nonce not in self._open_scopes:
            raise ValueError("denial_logical_scope_unbound")
        return scope

    def begin_attempt(self, attempt_index):
        with self._lock:
            scope = self._scope()
            if type(attempt_index) is not int or not 0 <= attempt_index < 2**31:
                raise ValueError("denial_attempt_index_invalid")
            key = (scope.nonce, attempt_index)
            if key in self._attempts:
                raise ValueError("denial_attempt_duplicate")
            self._attempts[key] = {"state": "pending", "receipt": None, "ack": None}

    def enter_delegate(self, attempt_index):
        with self._lock:
            scope = self._scope()
            row = self._attempts.get((scope.nonce, attempt_index))
            if row is None or row["state"] != "pending":
                raise ValueError("denial_attempt_not_pending")
            row["state"] = "delegated"

    def deny_attempt(self, attempt_index, reason_code):
        with self._lock:
            scope = self._scope()
            row = self._attempts.get((scope.nonce, attempt_index))
            if row is None or row["state"] != "pending":
                raise ValueError("denial_attempt_not_pending")
            if type(reason_code) is not str or reason_code not in REASON_CODES:
                raise ValueError("denial_reason_invalid")
            receipt_id = uuid.uuid4().hex
            if receipt_id in self._receipt_ids:
                raise ValueError("denial_receipt_duplicate")
            self._receipt_ids.add(receipt_id)
            receipt = DenialReceipt(receipt_id, scope.logical_index, attempt_index, reason_code,
                                    scope.module, scope.provider, scope.model)
            row.update(state="denied", receipt=receipt)
            return PhysicalAttemptNotStarted(receipt)


def _bound_row(receipt, scope):
    if (type(receipt) is not DenialReceipt or type(receipt.attempt_index) is not int
            or scope.nonce not in scope.issuer._open_scopes):
        return None
    row = scope.issuer._attempts.get((scope.nonce, receipt.attempt_index))
    if (row is None or row["receipt"] is not receipt or receipt.logical_index != scope.logical_index
            or (receipt.module, receipt.provider, receipt.model) != (scope.module, scope.provider, scope.model)):
        return None
    return row


def record_denial(receipt, *, module, provider, model, duration_sec, image_count, persist):
    """Native accounting alone reserves, persists and commits this exact receipt."""
    scope = _CURRENT.get()
    if scope is None or (module, provider, model) != (scope.module, scope.provider, scope.model):
        return None
    with scope.issuer._lock:
        row = _bound_row(receipt, scope)
        if row is None or row["state"] != "denied":
            return None
        record = {"version": "predelegate-local-denial-v1", **{
            key: getattr(receipt, key) for key in DenialReceipt.__dataclass_fields__},
            "duration_sec": duration_sec, "image_count": image_count, "failed": True}
        if not valid_denial_record(record, module=module):
            return None
        row["state"] = "recording"
        try:
            persist(record)
        except BaseException:
            row["state"] = "recording_failed"
            raise
        ack = DenialAcknowledgement(receipt)
        row.update(state="recorded", ack=ack)
        return ack


def is_denial_acknowledgement(value, *, receipt):
    """Read-only wrapper verification; this never consumes the receipt again."""
    scope = _CURRENT.get()
    if scope is None or type(value) is not DenialAcknowledgement or value.receipt is not receipt:
        return False
    with scope.issuer._lock:
        row = _bound_row(receipt, scope)
        return row is not None and row["state"] == "recorded" and row["ack"] is value


def valid_denial_record(row, *, module):
    fields = set(DenialReceipt.__dataclass_fields__) | {"version", "duration_sec", "image_count", "failed"}
    return (type(row) is dict and set(row) == fields
            and row["version"] == "predelegate-local-denial-v1" and row["failed"] is True
            and type(row["receipt_id"]) is str and len(row["receipt_id"]) == 32
            and all(char in "0123456789abcdef" for char in row["receipt_id"])
            and all(type(row[key]) is int and 0 <= row[key] < 2**31 for key in ("logical_index", "attempt_index"))
            and type(row["reason_code"]) is str and row["reason_code"] in REASON_CODES and row["module"] == module
            and row["provider"] == "openai-codex" and type(row["model"]) is str and 0 < len(row["model"]) <= 256
            and type(row["duration_sec"]) in (int, float) and 0 <= row["duration_sec"] <= 1.7976931348623157e308
            and math.isfinite(row["duration_sec"])
            and type(row["image_count"]) is int and row["image_count"] >= 0)
