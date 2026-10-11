"""One all-mock differential control for finite tags retained before diagnostic IO."""
from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from verification import rejection_diagnostics as diagnostics


def test_finite_joint_subguard_memory_preserves_writer_contract(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("External LLM/retrieval/Docker/process forbidden")

    monkeypatch.setattr("llm.client.llm_json", forbidden)
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr("httpx.AsyncClient.send", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)
    before = ModuleType("joint_subguard_before")
    source = Path(__file__).resolve().parent / "fixtures/v2_original_sources/before-joint-subguard.py"
    exec(compile(source.read_bytes(), str(source), "exec", dont_inherit=True), before.__dict__)
    material = SimpleNamespace(source_pdf=None, markdown_path=None, content_list_path=None,
                               pages=[], figures=[], tables=[], repository=None)
    code = "experiments_joint_final_validation_failed"
    open_original = Path.open
    opens = []
    failure = [False]

    def controlled_open(path, *args, **kwargs):
        if path.name.startswith("verification_rejection_") and args and args[0] == "x":
            opens.append((str(path.parent), args, kwargs))
            if failure[0]:
                raise OSError("Explicit injected IO failure")
        return open_original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", controlled_open)

    def run(module, name, tag, *, enabled=True, io_fail=True, branch="Experiments", selected_code=code):
        directory = tmp_path / name / "diagnostics"
        failure[0] = io_fail
        initial_opens = len(opens)
        with module.write_outcome_scope(enabled=enabled) as collector:
            writer = module.FirstRejectionDiagnostic(directory, material, branch)
            returned = writer.record(selected_code, first_subguard=tag)
            writer.record(code, first_subguard="source_count")
            captured = None if collector is None else collector.close_snapshot()
            writer.record(code, first_subguard="member_lookup")
            assert collector is None or captured == collector.close_snapshot()
        assert writer.attempted is True and returned is None
        paths = list(directory.glob("verification_rejection_*.json"))
        payloads = [p.read_bytes() for p in paths]
        return captured, payloads, len(opens) - initial_opens

    captured, payloads, count = run(diagnostics, "first-failed", "scope_pair_missing")
    tag_rows = [r for r in captured["events"] if r["outcome"].startswith("joint_subguard_")]
    assert tag_rows == [{"branch": "Experiments", "code": code, "stage": "writer",
                         "outcome": "joint_subguard_scope_pair_missing", "count": 1}]
    assert not payloads and count == 1
    assert any(r["outcome"] == "io_failed" and r["count"] == 1 for r in captured["events"])
    assert any(r["outcome"] == "already_attempted" and r["count"] == 1 for r in captured["events"])
    assert set(captured) == {"schema", "audit_only", "adoption_authorized", "coverage", "closed", "overflow", "unavailable", "events"}
    assert captured["schema"] == "verification-write-outcomes-v1"
    assert captured["audit_only"] is True and captured["adoption_authorized"] is False
    assert captured["coverage"] == "captured_scope_events_only" and captured["closed"] is True
    assert captured["overflow"] is captured["unavailable"] is False
    assert all(set(r) == {"branch", "code", "stage", "outcome", "count"} for r in captured["events"])

    class StrSubclass(str):
        pass

    class Untouchable:
        def __str__(self):
            pytest.fail("Opaque argument string conversion forbidden")

        def __hash__(self):
            pytest.fail("Opaque argument hashing forbidden")

        def __eq__(self, other):
            pytest.fail("Opaque argument comparison forbidden")

    cases = [(tag, tag) for tag in sorted(diagnostics._JOINT_SUBGUARDS)]
    cases += [("foreign-body", "unknown"), (StrSubclass("scope_pair_missing"), "unknown"),
              (Untouchable(), "unknown"), (True, "unknown"), (None, None)]
    for i, (tag, expected) in enumerate(cases):
        for io_fail in (True, False):
            current, payloads, count = run(diagnostics, f"new-{i}-{io_fail}", tag, io_fail=io_fail)
            original, old_payloads, old_count = run(before, f"old-{i}-{io_fail}", tag, io_fail=io_fail)
            rows = [r for r in current["events"] if r["outcome"].startswith("joint_subguard_")]
            assert [r["outcome"] for r in rows] == ([] if expected is None else [f"joint_subguard_{expected}"])
            assert all(r["count"] == 1 for r in rows)
            assert {**current, "events": [r for r in current["events"] if r not in rows]} == original
            assert payloads == old_payloads and count == old_count == 1
            if payloads:
                assert len(payloads) == 1 and len(payloads[0]) <= 512
                value = json.loads(payloads[0])
                assert set(value) == ({"schema", "branch", "scope", "code"} if expected is None
                                      else {"schema", "branch", "scope", "code", "first_subguard"})
                assert value.get("first_subguard") == expected

    for i, (branch, selected_code) in enumerate([
        ("Theory", "theory_no_proof_trace_invalid"), ("Experiments", "experiments_choice_privacy_blocked"),
        (StrSubclass("Experiments"), code), ("Experiments", StrSubclass(code)),
    ]):
        current, payloads, count = run(diagnostics, f"foreign-new-{i}", "scope_pair_missing",
                                       branch=branch, selected_code=selected_code, io_fail=False)
        original, old_payloads, old_count = run(before, f"foreign-old-{i}", "scope_pair_missing",
                                                branch=branch, selected_code=selected_code, io_fail=False)
        assert current == original and payloads == old_payloads and count == old_count
        assert not any(r["outcome"].startswith("joint_subguard_") for r in current["events"])

    for i, enabled in enumerate((False, None, 1)):
        current, payloads, count = run(diagnostics, f"off-new-{i}", "scope_pair_missing", enabled=enabled, io_fail=False)
        original, old_payloads, old_count = run(before, f"off-old-{i}", "scope_pair_missing", enabled=enabled, io_fail=False)
        assert current is original is None and payloads == old_payloads and count == old_count == 1
    with diagnostics.write_outcome_scope() as disabled:
        assert disabled is None and diagnostics._OUTCOME_CONTEXT.get() is None
    assert diagnostics._OUTCOME_CONTEXT.get() is None
    assert diagnostics._OUTCOME_MAX_ROWS == before._OUTCOME_MAX_ROWS == 128
    assert diagnostics._OUTCOME_MAX_COUNT == before._OUTCOME_MAX_COUNT == 1000000
    assert diagnostics._OUTCOMES == before._OUTCOMES | {f"joint_subguard_{tag}" for tag in before._JOINT_SUBGUARDS}
    assert vars(material) == {"source_pdf": None, "markdown_path": None, "content_list_path": None,
                              "pages": [], "figures": [], "tables": [], "repository": None}
