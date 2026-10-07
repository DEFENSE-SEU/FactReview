"""Shared fixtures for the FactReview test suite.

The new suite is organised by pipeline stage, not by source file. Fixtures
here build the smallest valid input for each stage so individual stage tests
can stay short. Anything that ends up specific to one stage lives in that
stage's test file instead.
"""

from __future__ import annotations

import socket
import sys
import threading
import uuid
from pathlib import Path
from unittest.mock import Mock

import pytest

from schemas.paper import Paper, PaperMetadata, Section, Table


def pytest_configure(config):
    # Production switches long Windows venv paths to a hashed fallback. Keep
    # fixture paths short so the existing run-local assertions exercise their
    # intended branch consistently across operating systems.
    if sys.platform == "win32" and config.option.basetemp is None:
        parent = Path(__file__).resolve().parents[1] / "runs" / "pytest"
        parent.mkdir(parents=True, exist_ok=True)
        target = (parent / uuid.uuid4().hex[:8]).resolve()
        if not target.is_relative_to(parent.resolve()) or target.exists():
            raise ValueError("pytest needs a fresh temporary directory inside runs/pytest")
        config.option.basetemp = str(target)


@pytest.fixture(autouse=True)
def isolated_external_boundaries(monkeypatch, request):
    if any(
        request.node.get_closest_marker(marker)
        for marker in ("requires_docker", "requires_llm", "requires_mineru")
    ):
        return
    monkeypatch.setattr("fact_generation.execution.tools.docker._docker_info_field", Mock(return_value=""))
    monkeypatch.setattr(
        socket, "create_connection", Mock(side_effect=OSError("Unit-test network access disabled"))
    )
    original_pair, original_connect = socket.socketpair, socket.socket.connect
    internal = threading.local()

    def socket_pair(*args, **kwargs):
        # Windows uses a loopback connection for asyncio's internal wakeup pair.
        internal.socket_pair = True
        try:
            return original_pair(*args, **kwargs)
        finally:
            internal.socket_pair = False

    def connect(sock, address):
        if getattr(internal, "socket_pair", False) and address[0] in {"127.0.0.1", "::1"}:
            return original_connect(sock, address)
        raise OSError("Unit-test network access disabled")

    monkeypatch.setattr(socket, "socketpair", socket_pair)
    monkeypatch.setattr(socket.socket, "connect", connect)


@pytest.fixture
def tiny_paper() -> Paper:
    """Smallest Paper that exercises every claim-type heuristic + one table."""
    return Paper(
        metadata=PaperMetadata(
            paper_key="tiny",
            title="Tiny Method",
            authors=["A. Author"],
            year=2024,
        ),
        pdf_path=Path("tiny.pdf"),
        sections=[
            Section(
                id="sec_1",
                title="Introduction",
                text=(
                    "We propose TinyMethod, a novel framework. "
                    "Our method outperforms baselines on FB15k-237 and WN18RR with MRR of 0.355. "
                    "We prove that TinyMethod generalizes prior work (Proposition 4.1). "
                    "Source code is available at https://github.com/example/tiny."
                ),
                char_start=0,
            ),
        ],
        tables=[
            Table(
                id="table_1",
                caption="Link prediction results.",
                rows=[
                    ["Method", "MRR"],
                    ["Baseline", "0.30"],
                    ["TinyMethod", "0.355"],
                ],
            ),
        ],
    )


@pytest.fixture
def review_md_with_claims_table() -> str:
    """Minimal review markdown that the claim-audit batched LLM expects.

    Includes the headers (Section 3 Claims, Section 4 Summary, Section 5
    Experiment with Ablation Result) the audit code keys off of so individual
    audit tests can focus on the LLM verdict instead of restating the
    structure each time.
    """
    return (
        "## 2. Technical Positioning\n"
        "(skipped for the audit-focused fixture)\n\n"
        "## 3. Claims\n"
        "(legend)\n\n"
        "| Claim | Evidence | Assessment | Status | Location |\n"
        "|---|---|---|---|---|\n"
        "| TinyMethod is leading on SWE-Bench. | Table 1 reports 59.0 +/- 1.9 vs. 57.7. | "
        'ok | <span style="color: green;">✓ Supported</span> | Table 1 |\n'
        "## 4. Summary\n"
        "Summary text.\n\n"
        "**Strengths:**\n- s1\n\n"
        "**Weaknesses:**\n- w1\n\n"
        "## 5. Experiment\n"
        "### Ablation Result\n"
        "| Dim | Cfg | Full | Paper | Delta |\n"
        "|---|---|---|---|---|\n"
        "| A | no | 1.0 | 0.5 | -0.5 |\n"
    )
