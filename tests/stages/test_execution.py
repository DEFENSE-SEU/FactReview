"""Execution stage tests.

Code execution requires a Docker daemon, so this stage's only test is gated
behind ``@pytest.mark.requires_docker`` and skipped by default. The real
execution path is covered by manual runs of ``demos/`` papers (see the plan
doc's verification section).
"""

from __future__ import annotations

import shutil

import pytest


@pytest.mark.requires_docker
def test_docker_daemon_is_available_for_execution_stage() -> None:
    # Smoke check that an environment claiming to be Docker-capable actually
    # has the CLI available — protects against running the gated test on a
    # host where the orchestrator would crash in a confusing way.
    if shutil.which("docker") is None:
        pytest.skip("docker CLI not available")
