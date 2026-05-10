"""Shared fixtures for the FactReview test suite.

The new suite is organised by pipeline stage, not by source file. Fixtures
here build the smallest valid input for each stage so individual stage tests
can stay short. Anything that ends up specific to one stage lives in that
stage's test file instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from schemas.paper import Paper, PaperMetadata, Section, Table


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
