"""Three bounded local-Python controls; all external transports are forbidden."""

import json

import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Observer controls cannot launch external services or processes")

    for name in (
        "llm.client.llm_json", "requests.sessions.Session.request", "httpx.Client.send",
        "httpx.AsyncClient.send", "subprocess.run", "subprocess.Popen",
        "fact_generation.execution.v2.docker_runner",
    ):
        monkeypatch.setattr(name, forbidden)


def fixture(tmp_path, text):
    cwd = tmp_path / "released"
    cwd.mkdir()
    entry = cwd / "eval.py"
    entry.write_text(text, encoding="utf-8")
    return cwd, entry


def test_exact_code_identity_and_native_script_semantics(tmp_path, monkeypatch):
    from fact_generation.execution.runtime_observer import prepare_site, run_observed

    cwd, entry = fixture(tmp_path, '''import json, os, sys
def inner(value):
    return value + 1
def outer(value, *, split):
    return inner(value)
assert __name__ == "__main__"
assert sys.argv == ["eval.py", "native-arg"]
assert sys.path[0] == os.getcwd()
assert __file__ == os.path.abspath("eval.py")
assert __spec__ is None
print(outer(2, split="test"))
''')
    sites = [prepare_site(str(entry), "outer", 4), prepare_site(str(entry), "inner", 2)]
    original = entry.read_bytes()
    out = tmp_path / "observed.json"
    report = run_observed("eval.py", ["native-arg"], str(cwd), sites, str(out))
    assert report["execution"]["status"] == "completed"
    assert report["unresolved"] == []
    assert [row["event"] for row in report["events"]] == ["call", "call", "return", "return"]
    assert report["events"][1]["parent_invocation"] == report["events"][0]["invocation"]
    assert report["events"][0]["arguments"]["split"] == "test"
    assert report["events"][2]["value"] == 3
    assert json.loads(out.read_text()) == report and entry.read_bytes() == original
    assert report["source_hashes_after"][str(entry)] == report["entry"]["sha256"]
    assert not {"aligned", "sufficient", "complete", "weights_used"} & report.keys()
    bad = [dict(sites[0], code_sha256="0" * 64)]
    rejected = run_observed("eval.py", ["native-arg"], str(cwd), bad, str(tmp_path / "bad.json"))
    assert rejected["execution"]["status"] == "not_started"
    assert rejected["unresolved"] and rejected["events"] == []
    assert json.loads((tmp_path / "bad.json").read_text()) == rejected
    from fact_generation.execution import runtime_observer
    selector = {key: value for key, value in sites[0].items() if key != "code_sha256"}
    cli_request = dict(entry="eval.py", args=["native-arg"], cwd=str(cwd), sites=[selector], output=str(tmp_path / "cli.json"))
    cli_config = tmp_path / "config.json"
    cli_config.write_text(json.dumps(cli_request), encoding="utf-8")
    hooks = []
    monkeypatch.setattr("sys.addaudithook", hooks.append)
    monkeypatch.setattr("sys.argv", ["observer", str(cli_config)])
    runtime_observer.main()
    assert len(hooks) == 1
    cli_report = json.loads((tmp_path / "cli.json").read_text())
    assert cli_report["execution"]["status"] == "completed"
    assert cli_report["sites"][0] == sites[0]
    entry.write_text('''import marshal, types
def outer(value, *, split):
    return value + 1
outer.__code__ = outer.__code__.replace(co_consts=(None, 99))
outer(2, split="test")
''', encoding="utf-8")
    modified_site = prepare_site(str(entry), "outer", 2)
    changed = run_observed("eval.py", [], str(cwd), [modified_site], str(tmp_path / "changed-code.json"))
    assert "runtime_code_identity_mismatch" in changed["unresolved"]
    assert changed["events"] == []


def test_exact_builtin_snapshots_capacity_and_missing_events(tmp_path):
    from fact_generation.execution.runtime_observer import prepare_site, run_observed

    cwd, entry = fixture(tmp_path, '''def echo(value):
    return value
class Poison:
    def __repr__(self): raise RuntimeError("repr accessed")
    def __getattribute__(self, name): raise RuntimeError("getter accessed")
class SubList(list):
    def __iter__(self): raise RuntimeError("iteration accessed")
echo((1, {"letters": ["α", True, None]}))
echo(Poison())
echo(SubList([1]))
''')
    site = prepare_site(str(entry), "echo", 1)
    result = run_observed("eval.py", [], str(cwd), [site], str(tmp_path / "snapshots.json"))
    assert result["execution"]["status"] == "completed"
    assert result["events"][0]["arguments"]["value"]["type"] == "tuple"
    assert "unsupported_snapshot_type" in result["unresolved"]
    assert not any("repr accessed" in value or "getter accessed" in value for value in result["unresolved"])
    small = run_observed("eval.py", [], str(cwd), [site], str(tmp_path / "limited.json"), max_events=1)
    assert "event_capacity_exceeded" in small["unresolved"]
    capacity = run_observed("eval.py", [], str(cwd), [site], str(tmp_path / "snapshot-cap.json"), max_snapshot_bytes=1)
    assert "snapshot_capacity_exceeded" in capacity["unresolved"]
    entry.write_text("def echo(value):\n    return value\n", encoding="utf-8")
    missing = run_observed("eval.py", [], str(cwd), [prepare_site(str(entry), "echo", 1)], str(tmp_path / "missing.json"))
    assert "site_not_observed" in missing["unresolved"]


def test_profile_closure_threads_and_raising_returns_are_explicit(tmp_path):
    from fact_generation.execution.runtime_observer import prepare_site, run_observed

    cwd, entry = fixture(tmp_path, '''import sys, threading
def measured(value):
    return value
def raises():
    raise ValueError("author exception")
try: raises()
except ValueError: pass
thread = threading.Thread(target=lambda: None)
thread.start()
thread.join()
measured(2)
sys.setprofile(None)
measured(3)
''')
    result = run_observed("eval.py", [], str(cwd), [
        prepare_site(str(entry), "measured", 2), prepare_site(str(entry), "raises", 4),
    ], str(tmp_path / "interruptions.json"))
    assert result["execution"]["status"] == "completed"
    assert "thread_start_observed" in result["unresolved"]
    assert "profile_changed" in result["unresolved"]
    assert any(row["event"] == "unwind" for row in result["events"])
    assert sum(row["event"] == "call" and row["site"] == 0 for row in result["events"]) == 1
    assert result["scope_limits"] and result["intrusion"]


def test_cli_preserves_author_termination_after_saving_report(tmp_path, monkeypatch):
    from fact_generation.execution import runtime_observer

    monkeypatch.setattr("sys.addaudithook", lambda hook: None)
    for name, termination, error in (
        ("success", "pass", None),
        ("author_exit", "raise SystemExit(7)", SystemExit),
        ("author_error", "raise ValueError('original author failure')", ValueError),
        ("startup", "pass", SystemExit),
    ):
        cwd = tmp_path / name
        cwd.mkdir()
        entry = cwd / "eval.py"
        entry.write_text("def measured():\n    return 3\nmeasured()\n" + termination + "\n", encoding="utf-8")
        original = entry.read_bytes()
        site = runtime_observer.prepare_site(str(entry), "measured", 1)
        output = tmp_path / (name + ".json")
        config = tmp_path / (name + "-config.json")
        config.write_text(json.dumps(dict(entry="eval.py", args=[], cwd=str(cwd),
            sites=[{key: value for key, value in site.items() if key != "code_sha256"}], output=str(output))), encoding="utf-8")
        with monkeypatch.context() as local:
            local.setattr("sys.argv", ["observer", str(config)])
            if name == "startup":
                local.setattr("sys.getprofile", lambda: object())
            if error is None:
                runtime_observer.main()
            else:
                with pytest.raises(error) as caught:
                    runtime_observer.main()
                if error is SystemExit:
                    assert caught.value.code == (7 if name == "author_exit" else 1)
                else:
                    assert caught.value.args == ("original author failure",)
                    frames, traceback = [], caught.value.__traceback__
                    while traceback is not None:
                        frames.append((traceback.tb_frame.f_code.co_filename, traceback.tb_lineno))
                        traceback = traceback.tb_next
                    assert (str(entry), 4) in frames
            saved = json.loads(output.read_text())
            assert saved["execution"]["status"] == {
                "success": "completed", "author_exit": "system_exit", "author_error": "raised", "startup": "not_started",
            }[name]
            assert entry.read_bytes() == original
            if name != "startup":
                assert [event["event"] for event in saved["events"]] == ["call", "return"]
                assert saved["source_hashes_after"][str(entry)] == site["sha256"]
            else:
                assert saved["events"] == [] and saved["unresolved"] == ["startup_identity_or_request_invalid"]
