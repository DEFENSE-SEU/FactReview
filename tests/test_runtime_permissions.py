"""Permission provisioning must fail closed before any author transport."""

import json

from fact_generation.execution import runner as v2
from fact_generation.execution import runtime_permissions
from tests.test_execution_workspace_readonly import _ordinary_inputs


def test_mount_permission_failure_stops_author_launch(tmp_path, monkeypatch):
    plan, claim, materials = _ordinary_inputs(tmp_path)
    monkeypatch.setattr(v2, "docker_ensure_paper_image", lambda *a, **kw: (True, "fixture-image"))

    def denied(**kwargs):
        raise PermissionError("Configured sandbox UID cannot own the runtime files")

    def forbidden(*args, **kwargs):
        raise AssertionError("Author transport must not start after permission setup fails")

    monkeypatch.setattr(runtime_permissions, "prepare_runtime_permissions", denied)
    monkeypatch.setattr(v2, "run_command", forbidden)
    result = v2.execute_plans(
        [plan], [claim], materials, tmp_path / "execution",
        config=v2.ExecutionConfig(refine_with_llm=False, max_attempts=0),
    )
    assert not result.claims[0].evidence
    assert not result.claims[0].questions
    assert len(result.delivery_checks) == 1
    assert result.delivery_checks[0].component == "execution.environment"
    assert result.delivery_checks[0].responsibility == "system"
    assert "Runtime mount permissions unavailable" in result.ledger[0]["reason"]
    audits = list((tmp_path / "execution").rglob("runtime_permissions.json"))
    assert len(audits) == 1
    assert json.loads(audits[0].read_text("utf-8"))["status"] == "failed"
