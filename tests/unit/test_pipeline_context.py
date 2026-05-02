from __future__ import annotations

import json
from pathlib import Path


def test_reuse_job_id_finds_previous_run_runtime_job(tmp_path: Path) -> None:
    from common.pipeline_context import bootstrap_bridge_state

    repo_root = tmp_path / "repo"
    job_id = "11111111-1111-4111-8111-111111111111"
    paper_pdf = tmp_path / "paper.pdf"
    paper_pdf.write_bytes(b"%PDF-1.4\n")

    previous_job_dir = repo_root / "runs" / "compgcn_2026-04-25_120000" / "runtime" / "jobs" / job_id
    previous_job_dir.mkdir(parents=True)
    (previous_job_dir / "job.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "artifacts": {"source_pdf_path": str(paper_pdf)},
                "usage": {},
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )

    new_run_dir = repo_root / "runs" / "compgcn_2026-04-25_130000"
    state = bootstrap_bridge_state(
        repo_root=repo_root,
        run_dir=new_run_dir,
        paper_pdf=paper_pdf,
        paper_key="compgcn",
        reuse_job_id=job_id,
    )

    assert state.job_dir == previous_job_dir.resolve()
    assert state.job_json_path == (previous_job_dir / "job.json").resolve()
