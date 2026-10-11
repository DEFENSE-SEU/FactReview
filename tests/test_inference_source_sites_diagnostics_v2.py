"""One public combined control; all external boundaries remain mocked."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from fact_generation.execution import inference_runtime
from tests.test_inference_binding_diagnostics_v2 import (
    test_public_binding_diagnostics_are_closed_and_preserve_scientific_qualification as public_binding_fixture,
)


def test_source_sites_diagnostics_preserve_gate_short_circuit_and_binding(tmp_path, monkeypatch):
    original_request = inference_runtime.bind_inference_request
    saved = {}

    def capture(request, **kwargs):
        if request.source_science.get("metric_adapter") == "mean_squared_error":
            saved.update(request=request.model_copy(deep=True), kwargs=kwargs)
        return original_request(request, **kwargs)

    monkeypatch.setattr(inference_runtime, "bind_inference_request", capture)
    (tmp_path / "materials").mkdir()
    public_binding_fixture(tmp_path / "materials", monkeypatch)
    request, kwargs = saved["request"], saved["kwargs"]
    kwargs = {**kwargs, "materials": kwargs["materials"].model_copy(update={
        "content_list_path": str(Path(kwargs["materials"].source_pdf).parent / "content.json"),
    })}
    original_inputs = [item.model_dump(mode="json") for item in
                       (request, kwargs["claim"], kwargs["materials"])]
    workspace_files = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in Path(request.workspace).rglob("*") if path.is_file()}
    marker = "PRIVATE_SOURCE_SITE_CONTENT_7281"
    forbidden_calls = []

    def forbidden(*args, **kwargs):
        forbidden_calls.append(True)
        pytest.fail("Real LLM, retrieval, network, Docker and author processes are forbidden")

    monkeypatch.setattr("llm.client.llm_json", forbidden)
    operator = tmp_path / "operator"

    def selected(name, sites, *, run_dir=None):
        directory = run_dir or operator / name
        (directory / "attempt_0").mkdir(parents=True, exist_ok=True)
        return request.model_copy(update={"run_dir": str(directory), "repair_round": 0,
                                          "source_sites": sites})

    failure = {"status": "unresolved", "reason": "inference_request_binding_unavailable",
               "binding_failure": "inference_source_sites_unbound"}
    filename = "inference_source_sites_diagnostics.json"
    healthy = selected("healthy", copy.deepcopy(request.source_sites))
    expected = original_request(healthy, **kwargs)
    assert expected["status"] == "bound"
    assert original_request(healthy, **kwargs, source_sites_diagnostics=True) == expected
    assert not (Path(healthy.run_dir) / "attempt_0" / filename).exists()

    class DictSubclass(dict):
        pass

    variants = (
        (None, "source_sites_not_dict", [False, None, None, None]),
        (DictSubclass(request.source_sites), "source_sites_not_dict", [False, None, None, None]),
        ({"status": marker, "version": marker, "unresolved": [marker]},
         "source_sites_status_unbound", [True, False, None, None]),
        ({"status": "bound", "version": marker, "unresolved": [marker]},
         "source_sites_version_unbound", [True, True, False, None]),
        ({"status": "bound", "version": "python-source-sites-v1", "unresolved": [marker]},
         "source_sites_unresolved", [True, True, True, False]),
        ({"status": "bound", "version": "python-source-sites-v1", "unresolved": ()},
         "source_sites_unresolved", [True, True, True, False]),
    )
    fields = ("is_dict", "status_bound", "version_current", "unresolved_empty")
    for index, (sites, code, checks) in enumerate(variants):
        current = selected(f"variant_{index}", sites)
        path = Path(current.run_dir) / "attempt_0" / filename
        assert original_request(current, **kwargs) == failure and not path.exists()
        assert original_request(current, **kwargs, source_sites_diagnostics=True) == failure
        payload = path.read_bytes()
        assert len(payload) <= 512 and marker.encode() not in payload
        assert json.loads(payload) == {
            "schema": "torch-inference-source-sites-diagnostics-v1",
            "scope": "first_source_sites_binding_rejection", "code": code,
            "checks": dict(zip(fields, checks, strict=True)),
        }
        assert path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
        assert original_request(current, **kwargs, source_sites_diagnostics=True) == failure
        assert path.read_bytes() == payload  # Existing exclusive sidecar cannot be replaced.

    for value in (False, 1, marker):
        current = selected(f"disabled_{value}", None)
        assert original_request(current, **kwargs, source_sites_diagnostics=value) == failure
        assert not (Path(current.run_dir) / "attempt_0" / filename).exists()

    # Each equality/truth conversion and later short-circuit exception has its original order.
    class Truth:
        def __init__(self, trace, name, value):
            self.trace, self.name, self.value = trace, name, value

        def __bool__(self):
            self.trace.append((self.name, "bool"))
            if type(self.value) is list:
                value = self.value.pop(0)
                if isinstance(value, Exception):
                    raise value
                return value
            return self.value

    class Compared:
        def __init__(self, trace, name, value):
            self.trace, self.name, self.value = trace, name, value

        def __eq__(self, other):
            self.trace.append((self.name, "eq"))
            if isinstance(self.value, Exception):
                raise self.value
            return Truth(self.trace, self.name, self.value)

    for stopping in range(3):
        observed = []
        for enabled in (False, True):
            trace = []
            values = [True] * stopping + [False] + [RuntimeError(marker)] * (2 - stopping)
            sites = dict(zip(("status", "version", "unresolved"),
                             (Compared(trace, str(i), value) for i, value in enumerate(values)), strict=True))
            current = selected(f"trace_{stopping}_{enabled}", sites)
            assert original_request(current, **kwargs, source_sites_diagnostics=enabled) == failure
            observed.append(trace)
        expected_trace = [item for i in range(stopping + 1)
                          for item in ((str(i), "eq"), (str(i), "bool"))]
        if stopping < 2:
            expected_trace.append((str(stopping), "bool"))
        assert observed[0] == observed[1] == expected_trace
    for error_type, expected_failure in ((ValueError, "local_requirement_failed"), (RuntimeError, None)):
        observed = []
        for enabled in (False, True):
            trace = []
            current = selected(f"exception_{error_type.__name__}_{enabled}",
                               {"status": Compared(trace, "status", error_type(marker))})
            if expected_failure is None:
                with pytest.raises(RuntimeError, match=marker):
                    original_request(current, **kwargs, source_sites_diagnostics=enabled)
            else:
                assert original_request(current, **kwargs, source_sites_diagnostics=enabled) == {
                    **failure, "binding_failure": expected_failure,
                }
            assert not (Path(current.run_dir) / "attempt_0" / filename).exists()
            observed.append(trace)
        assert observed[0] == observed[1] == [("status", "eq")]

    for final, expected_failure in ((ValueError(marker), "local_requirement_failed"),
                                    (RuntimeError(marker), None), (True, "key_error")):
        observed = []
        for enabled in (False, True):
            trace = []
            current = selected(f"final_truth_{expected_failure}_{enabled}",
                               {"status": Compared(trace, "status", [False, final])})
            if expected_failure is None:
                with pytest.raises(RuntimeError, match=marker):
                    original_request(current, **kwargs, source_sites_diagnostics=enabled)
            else:
                assert original_request(current, **kwargs, source_sites_diagnostics=enabled) == {
                    **failure, "binding_failure": expected_failure,
                }
            assert not (Path(current.run_dir) / "attempt_0" / filename).exists()
            observed.append(trace)
        assert observed[0] == observed[1] == [("status", "eq"), ("status", "bool"), ("status", "bool")]

    # Earlier and later scientific guards retain their original results and publish no sites sidecar.
    for label, updates, code in (
        ("earlier", {"source_science_files": {}}, "inference_source_closure_unavailable_or_changed"),
        ("later", {"source_sites": {**request.source_sites, "sites": []}}, "inference_site_index_unbound"),
    ):
        current = selected(label, copy.deepcopy(request.source_sites)).model_copy(update=updates)
        assert original_request(current, **kwargs) == original_request(
            current, **kwargs, source_sites_diagnostics=True) == {**failure, "binding_failure": code}
        assert not (Path(current.run_dir) / "attempt_0" / filename).exists()

    from fact_generation.execution import inference_source_sites_diagnostics as diagnostic

    for label, code, checks in (
        ("foreign_code", marker, dict(zip(fields, [False, None, None, None], strict=True))),
        ("foreign_fields", "source_sites_not_dict", {marker: False}),
        ("foreign_value", "source_sites_not_dict", dict(zip(fields, [marker, None, None, None], strict=True))),
        ("numeric_bool", "source_sites_not_dict", dict(zip(fields, [0, None, None, None], strict=True))),
        ("contradictory", "source_sites_not_dict", dict.fromkeys(fields, True)),
    ):
        current = selected(label, None)
        diagnostic.SourceSitesDiagnostic(current, kwargs["materials"])._record(code, checks)
        assert not (Path(current.run_dir) / "attempt_0" / filename).exists()

    for label, directory in (
        ("workspace", Path(request.workspace) / "diagnostic_probe"),
        ("author", Path(kwargs["materials"].repository.root) / "diagnostic_probe"),
        ("materials", Path(kwargs["materials"].source_pdf).parent / "diagnostic_probe"),
    ):
        current = selected(label, None, run_dir=directory)
        assert original_request(current, **kwargs, source_sites_diagnostics=True) == failure
        assert not (directory / "attempt_0" / filename).exists()
    current = selected("round", None).model_copy(update={"repair_round": True})
    assert original_request(current, **kwargs, source_sites_diagnostics=True) == failure
    assert not (Path(current.run_dir) / "attempt_True" / filename).exists()

    current = selected("io_failure", None)
    path = Path(current.run_dir) / "attempt_0" / filename
    original_open = Path.open
    io_attempts = []

    def broken_open(self, *args, **kwargs):
        if self == path:
            io_attempts.append(args[0])
            raise OSError(marker)
        return original_open(self, *args, **kwargs)

    with monkeypatch.context() as local:
        local.setattr(Path, "open", broken_open)
        assert original_request(current, **kwargs, source_sites_diagnostics=True) == failure
        collector = diagnostic.SourceSitesDiagnostic(current, kwargs["materials"])
        for _ in range(2):
            with pytest.raises(ValueError, match=r"^inference_source_sites_unbound$"):
                collector.require_bound(None)
    assert io_attempts == ["x", "x"] and not path.exists()  # One attempt per collector, including failed IO.

    current = selected("linked_path_rejection", None)
    original_plain = diagnostic.plain_path

    def blocked_plain(value):
        if Path(value) == Path(current.run_dir):
            raise ValueError(marker)
        return original_plain(value)

    with monkeypatch.context() as local:
        local.setattr(diagnostic, "plain_path", blocked_plain)
        assert original_request(current, **kwargs, source_sites_diagnostics=True) == failure
    assert not (Path(current.run_dir) / "attempt_0" / filename).exists()
    assert [item.model_dump(mode="json") for item in
            (request, kwargs["claim"], kwargs["materials"])] == original_inputs
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in workspace_files.items())
    assert forbidden_calls == []

    # Public binder observers cannot replace the original source-sites guard.
    import builtins

    observer_calls = []

    def noop(*args, **kwargs):
        observer_calls.append(True)

    class Foreign:
        require_bound = staticmethod(noop)

    class Subclass(diagnostic.SourceSitesDiagnostic):
        require_bound = staticmethod(noop)

    def bind(current, observer=None):
        return inference_runtime.bind_inference_proposal(current.source_science, plan=current.plan,
            claim=kwargs["claim"], materials=kwargs["materials"], workspace=current.workspace,
            source_files=current.source_science_files, limits=current.config.docker_options["torch_inference"],
            source_sites=current.source_sites, source_sites_diagnostic=observer)

    rejected = selected("observer_rejection", {"sites": copy.deepcopy(request.source_sites["sites"])})
    original_import = builtins.__import__
    helper_imports = []

    def traced_import(name, *args, **kwargs):
        if name.endswith("inference_source_sites_diagnostics"):
            helper_imports.append(name)
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as local:
        local.setattr(builtins, "__import__", traced_import)
        assert bind(healthy) == expected["binding"]
        with pytest.raises(ValueError, match=r"^inference_source_sites_unbound$"):
            bind(rejected)
    assert helper_imports == []  # None default imports no new helper, even on a rejected guard.

    for observer in (Foreign(), Subclass(rejected, kwargs["materials"])):
        with pytest.raises(ValueError, match=r"^inference_source_sites_unbound$"):
            bind(rejected, observer)
        assert bind(healthy, observer) == expected["binding"]
        assert not (Path(rejected.run_dir) / "attempt_0" / filename).exists()
    assert observer_calls == []

    overridden = diagnostic.SourceSitesDiagnostic(rejected, kwargs["materials"])
    overridden.require_bound = noop
    with pytest.raises(ValueError, match=r"^inference_source_sites_unbound$"):
        bind(rejected, overridden)
    assert json.loads((Path(rejected.run_dir) / "attempt_0" / filename).read_bytes()) == {
        "schema": "torch-inference-source-sites-diagnostics-v1",
        "scope": "first_source_sites_binding_rejection", "code": "source_sites_status_unbound",
        "checks": dict(zip(fields, [True, False, None, None], strict=True)),
    }
    legitimate = diagnostic.SourceSitesDiagnostic(healthy, kwargs["materials"])
    legitimate.require_bound = noop
    assert bind(healthy, legitimate) == expected["binding"]
    assert observer_calls == [] and forbidden_calls == []
    assert [item.model_dump(mode="json") for item in
            (request, kwargs["claim"], kwargs["materials"])] == original_inputs
