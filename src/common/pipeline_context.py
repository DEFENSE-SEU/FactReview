"""Shared runtime state for the pipeline.

Holds the cross-stage primitives that every stage reads/writes:

- ``init_full_pipeline_context`` / ``ensure_full_pipeline_context`` mark a run
  directory as managed by ``pipeline_full`` (or by a standalone stage CLI) and
  validate the marker before a stage executes.
- ``RuntimeBridgeState`` and the ``*_bridge_state`` helpers carry the
  parse-stage outputs (PDF path, job id, agent payload) into downstream stages
  via a small JSON file under ``stages/preprocessing/parse/``.
- ``load_job_state_snapshot`` / ``load_stage_assets_snapshot`` and
  ``materialize_stage_inputs_snapshot`` keep run-local copies of the agent
  runner's outputs so downstream stages do not depend on
  ``runtime/jobs/<job_id>/`` continuing to exist.
- ``read_json_file`` / ``write_json_file`` and ``resolve_artifact_path`` are the
  small filesystem utilities every stage needs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_BRIDGE_FILE = "_runtime_bridge.json"
_PIPELINE_CONTEXT_FILE = "_full_pipeline_context.json"
_JOB_STATE_SNAPSHOT_FILE = "_job_state_snapshot.json"
_STAGE_ASSETS_SNAPSHOT_FILE = "_stage_assets_snapshot.json"

# Directory under run_dir/stages/ that holds the parse stage's bridge file and
# snapshot artifacts. Kept here (not in parse/) because the snapshot loaders
# are called from every downstream stage and we want a single source of truth.
_PARSE_STAGE_PATH: tuple[str, ...] = ("preprocessing", "parse")
_CLAIM_EXTRACT_STAGE_PATH: tuple[str, ...] = ("preprocessing", "claim_extract")
_REFCHECK_STAGE_PATH: tuple[str, ...] = ("fact_generation", "refcheck")
_POSITIONING_STAGE_PATH: tuple[str, ...] = ("fact_generation", "positioning")
_EXECUTION_STAGE_PATH: tuple[str, ...] = ("fact_generation", "execution")
_REPORT_STAGE_PATH: tuple[str, ...] = ("review", "report")
_TEASER_STAGE_PATH: tuple[str, ...] = ("review", "teaser")


@dataclass(frozen=True)
class RuntimeBridgeState:
    paper_pdf: Path
    paper_key: str
    job_id: str
    job_dir: Path
    job_json_path: Path
    own_payload: dict[str, Any]


# ── JSON / path utilities ────────────────────────────────────────────────────


def read_json_file(path: Path) -> dict[str, Any]:
    """Lenient JSON reader. Returns ``{}`` for missing or malformed files;
    any other I/O error (permission denied, disk error, etc.) propagates so
    the caller is not silently fed an empty dict on a real failure."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_artifact_path(repo_root: Path, raw: Any) -> Path | None:
    token = str(raw or "").strip()
    if not token:
        return None
    p = Path(token)
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return p


# ── Pipeline context marker ──────────────────────────────────────────────────


def _init_pipeline_context(*, run_dir: Path, runner: str, stage: str = "") -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"runner": runner, "stage": stage, "version": 1}
    path = run_dir / _PIPELINE_CONTEXT_FILE
    write_json_file(path, payload)
    return path


def init_full_pipeline_context(*, run_dir: Path) -> Path:
    """Mark a run directory as managed by src.pipeline_full."""
    return _init_pipeline_context(run_dir=run_dir, runner="full_pipeline")


def init_standalone_stage_context(*, run_dir: Path, stage: str = "") -> Path:
    """Mark a run directory as managed by a standalone stage command."""
    return _init_pipeline_context(run_dir=run_dir, runner="standalone_stage", stage=stage)


def ensure_full_pipeline_context(*, run_dir: Path, allow_standalone: bool = False, stage: str = "") -> None:
    """Validate stage context; optionally bootstrap standalone mode."""
    marker = run_dir / _PIPELINE_CONTEXT_FILE
    payload = read_json_file(marker)
    if not payload:
        if allow_standalone:
            init_standalone_stage_context(run_dir=run_dir, stage=stage)
            return
        raise RuntimeError(
            "Stage modules are internal-only and must be run via full_pipeline. "
            f"Missing pipeline context marker: {marker}"
        )
    runner = str(payload.get("runner") or "").strip()
    if runner == "full_pipeline":
        return
    if allow_standalone and runner == "standalone_stage":
        return
    if allow_standalone:
        raise RuntimeError(
            "Invalid pipeline context marker. Expected full_pipeline or standalone_stage, "
            f"got runner={runner!r}. Marker: {marker}"
        )
    if runner != "full_pipeline":
        raise RuntimeError(
            "Invalid pipeline context marker. Please run through scripts/execute_review_pipeline.py."
        )


# ── Stage directory helpers ──────────────────────────────────────────────────
#
# One helper per stage so that every stage_runner and pipeline_full.py asks for
# its output directory the same way. If the on-disk layout ever changes,
# updating the ``_*_STAGE_PATH`` tuples above is the only edit needed.


def parse_stage_dir(run_dir: Path) -> Path:
    """Return ``run_dir/stages/preprocessing/parse``."""
    return run_dir.joinpath("stages", *_PARSE_STAGE_PATH)


def claim_extract_stage_dir(run_dir: Path) -> Path:
    """Return ``run_dir/stages/preprocessing/claim_extract``."""
    return run_dir.joinpath("stages", *_CLAIM_EXTRACT_STAGE_PATH)


def refcheck_stage_dir(run_dir: Path) -> Path:
    """Return ``run_dir/stages/fact_generation/refcheck``."""
    return run_dir.joinpath("stages", *_REFCHECK_STAGE_PATH)


def positioning_stage_dir(run_dir: Path) -> Path:
    """Return ``run_dir/stages/fact_generation/positioning``."""
    return run_dir.joinpath("stages", *_POSITIONING_STAGE_PATH)


def execution_stage_dir(run_dir: Path) -> Path:
    """Return ``run_dir/stages/fact_generation/execution``."""
    return run_dir.joinpath("stages", *_EXECUTION_STAGE_PATH)


def report_stage_dir(run_dir: Path) -> Path:
    """Return ``run_dir/stages/review/report``."""
    return run_dir.joinpath("stages", *_REPORT_STAGE_PATH)


def teaser_stage_dir(run_dir: Path) -> Path:
    """Return ``run_dir/stages/review/teaser``."""
    return run_dir.joinpath("stages", *_TEASER_STAGE_PATH)


# ── Snapshot paths ───────────────────────────────────────────────────────────


def _job_state_snapshot_path(run_dir: Path) -> Path:
    return parse_stage_dir(run_dir) / _JOB_STATE_SNAPSHOT_FILE


def _stage_assets_snapshot_path(run_dir: Path) -> Path:
    return parse_stage_dir(run_dir) / _STAGE_ASSETS_SNAPSHOT_FILE


def _bridge_path(run_dir: Path) -> Path:
    return parse_stage_dir(run_dir) / _BRIDGE_FILE


def load_job_state_snapshot(run_dir: Path) -> dict[str, Any]:
    return read_json_file(_job_state_snapshot_path(run_dir))


def load_stage_assets_snapshot(run_dir: Path) -> dict[str, Any]:
    return read_json_file(_stage_assets_snapshot_path(run_dir))


# ── Snapshot materialization ─────────────────────────────────────────────────


def _snapshot_file(*, source: Path | None, destination: Path) -> str:
    if source is None or (not source.exists()) or (not source.is_file()):
        return ""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination.resolve())


def materialize_stage_inputs_snapshot(
    *,
    repo_root: Path,
    run_dir: Path,
    state: RuntimeBridgeState,
    own_payload: dict[str, Any],
) -> None:
    """Persist run-local snapshots so downstream stages do not depend on
    ``runtime/jobs/<job_id>/job.json`` continuing to exist."""
    stage_dir = parse_stage_dir(run_dir)
    snapshot_root = stage_dir / "snapshot_artifacts"
    snapshot_root.mkdir(parents=True, exist_ok=True)

    job_state = read_json_file(state.job_json_path)
    if not job_state:
        own_artifacts = own_payload.get("artifacts")
        artifacts: dict[str, Any] = own_artifacts if isinstance(own_artifacts, dict) else {}
        job_state = {
            "status": own_payload.get("status"),
            "message": own_payload.get("message"),
            "error": own_payload.get("error"),
            "artifacts": artifacts,
            "usage": own_payload.get("usage") or {},
            "metadata": own_payload.get("metadata") or {},
            "annotation_count": int(own_payload.get("annotation_count") or 0),
            "final_report_ready": bool(own_payload.get("final_report_ready")),
            "pdf_ready": bool(own_payload.get("pdf_ready")),
        }
    write_json_file(_job_state_snapshot_path(run_dir), job_state)

    job_artifacts = job_state.get("artifacts")
    artifacts = job_artifacts if isinstance(job_artifacts, dict) else {}
    annotations_src = resolve_artifact_path(repo_root, artifacts.get("annotations_path"))
    final_md_src = resolve_artifact_path(repo_root, artifacts.get("final_markdown_path"))
    final_pdf_src = resolve_artifact_path(repo_root, artifacts.get("report_pdf_path"))
    semantic_src = (state.job_dir / "semantic_scholar_candidates.json").resolve()

    snapshot_payload = {
        "job_json_path": str(state.job_json_path),
        "job_dir": str(state.job_dir),
        "job_state_snapshot_path": str(_job_state_snapshot_path(run_dir)),
        "annotations_snapshot_path": _snapshot_file(
            source=annotations_src,
            destination=snapshot_root / "annotations.json",
        ),
        "final_markdown_snapshot_path": _snapshot_file(
            source=final_md_src,
            destination=snapshot_root / "final_review.md",
        ),
        "report_pdf_snapshot_path": _snapshot_file(
            source=final_pdf_src,
            destination=snapshot_root / "final_review.pdf",
        ),
        "semantic_scholar_candidates_snapshot_path": _snapshot_file(
            source=semantic_src if semantic_src.exists() else None,
            destination=snapshot_root / "semantic_scholar_candidates.json",
        ),
    }
    write_json_file(_stage_assets_snapshot_path(run_dir), snapshot_payload)


# ── Bridge state ─────────────────────────────────────────────────────────────


def _reuse_job_candidates(*, repo_root: Path, run_dir: Path, job_ref: str) -> list[Path]:
    token = str(job_ref or "").strip()
    if not token:
        return []

    candidates: list[Path] = []
    raw_path = Path(token).expanduser()
    looks_like_path = raw_path.is_absolute() or any(sep in token for sep in ("/", "\\", os.sep))
    if looks_like_path:
        candidate = raw_path.resolve()
        candidates.append(candidate.parent if candidate.name == "job.json" else candidate)

    candidates.append((run_dir / "runtime" / "jobs" / token).resolve())

    runs_root = repo_root / "runs"
    if runs_root.exists():
        try:
            candidates.extend(sorted(p.resolve() for p in runs_root.glob(f"**/runtime/jobs/{token}")))
        except Exception:
            pass

    candidates.append((repo_root / "data" / "jobs" / token).resolve())

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _pick_python_executable(repo_root: Path) -> Path:
    candidate = repo_root / ".venv" / "bin" / "python"
    if candidate.exists():
        try:
            chk = subprocess.run(
                [str(candidate), "-c", "import agents"],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=15,
            )
            if chk.returncode == 0:
                return candidate
        except (subprocess.TimeoutExpired, OSError):
            pass
        return candidate
    return Path("python3")


def _run_review_runtime(*, repo_root: Path, run_dir: Path, paper_pdf: Path, title: str) -> dict[str, Any]:
    py_exec = _pick_python_executable(repo_root)
    script = repo_root / "scripts" / "execute_review_runtime_job.py"

    env = os.environ.copy()
    env["DATA_DIR"] = str((run_dir / "runtime").resolve())

    # Hard ceiling: a single agent runtime job for one paper should never need
    # more than a few hours. Without this, a hung subprocess would pin the
    # entire pipeline indefinitely.
    runtime_timeout_seconds = 3 * 60 * 60
    try:
        proc = subprocess.run(
            [str(py_exec), str(script), "--paper-pdf", str(paper_pdf), "--title", title],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=runtime_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"agent runtime pipeline timed out after {runtime_timeout_seconds}s") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"agent runtime pipeline failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}\n")

    text = (proc.stdout or "").strip()
    payload: dict[str, Any] | None = None
    if text:
        try:
            parsed = json.loads(text)
            payload = parsed if isinstance(parsed, dict) else None
        except Exception:
            start = text.rfind("{")
            if start >= 0:
                try:
                    parsed = json.loads(text[start:])
                    payload = parsed if isinstance(parsed, dict) else None
                except Exception:
                    payload = None
    if payload is None:
        raise RuntimeError(f"cannot parse agent runtime output: {text}")
    return payload


def load_bridge_state(run_dir: Path) -> RuntimeBridgeState | None:
    payload = read_json_file(_bridge_path(run_dir))
    if not payload:
        return None

    paper_pdf = Path(str(payload.get("paper_pdf") or "")).resolve()
    paper_key = str(payload.get("paper_key") or "").strip() or "paper"
    job_id = str(payload.get("job_id") or "").strip()
    job_dir = Path(str(payload.get("job_dir") or "")).resolve()
    job_json_path = Path(str(payload.get("job_json_path") or "")).resolve()
    raw_own_payload = payload.get("own_payload")
    own_payload: dict[str, Any] = raw_own_payload if isinstance(raw_own_payload, dict) else {}
    if not job_json_path.exists() and job_dir.exists():
        job_json_path = job_dir / "job.json"
    if not (paper_pdf.exists() and job_id):
        return None

    return RuntimeBridgeState(
        paper_pdf=paper_pdf,
        paper_key=paper_key,
        job_id=job_id,
        job_dir=job_dir,
        job_json_path=job_json_path,
        own_payload=own_payload,
    )


def save_bridge_state(
    *,
    run_dir: Path,
    paper_pdf: Path,
    paper_key: str,
    own_payload: dict[str, Any],
) -> RuntimeBridgeState:
    job_id = str(own_payload.get("job_id") or "").strip()
    job_dir = Path(str(own_payload.get("job_dir") or "")).resolve()
    job_json_path = Path(str(own_payload.get("job_json_path") or "")).resolve()
    if not job_json_path.exists() and job_dir.exists():
        job_json_path = job_dir / "job.json"

    bridge_payload = {
        "paper_pdf": str(paper_pdf.resolve()),
        "paper_key": paper_key,
        "job_id": job_id,
        "job_dir": str(job_dir),
        "job_json_path": str(job_json_path),
        "own_payload": own_payload,
    }
    write_json_file(_bridge_path(run_dir), bridge_payload)

    return RuntimeBridgeState(
        paper_pdf=paper_pdf.resolve(),
        paper_key=paper_key,
        job_id=job_id,
        job_dir=job_dir,
        job_json_path=job_json_path,
        own_payload=own_payload,
    )


def require_bridge_state(*, run_dir: Path) -> RuntimeBridgeState:
    existing = load_bridge_state(run_dir)
    if existing is not None:
        return existing
    raise FileNotFoundError(
        f"Bridge state missing at {_bridge_path(run_dir)}. "
        "Ensure parse stage has been completed by full_pipeline."
    )


def bootstrap_bridge_state(
    *,
    repo_root: Path,
    run_dir: Path,
    paper_pdf: Path | None = None,
    paper_key: str,
    reuse_job_id: str = "",
) -> RuntimeBridgeState:
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = load_bridge_state(run_dir)
    if existing is not None:
        return existing

    job_id = str(reuse_job_id or "").strip()
    if job_id:
        candidates = _reuse_job_candidates(repo_root=repo_root, run_dir=run_dir, job_ref=job_id)
        job_dir = next((p for p in candidates if (p / "job.json").exists()), candidates[0])
        job_json_path = job_dir / "job.json"
        if not job_json_path.exists():
            searched = ", ".join(str(p / "job.json") for p in candidates)
            raise FileNotFoundError(f"reused job.json not found; searched: {searched}")
        job_state = read_json_file(job_json_path)
        if not job_state:
            raise RuntimeError(f"reused job state is empty/invalid: {job_json_path}")

        reuse_artifacts = job_state.get("artifacts")
        artifacts = reuse_artifacts if isinstance(reuse_artifacts, dict) else {}
        source_pdf = resolve_artifact_path(repo_root, artifacts.get("source_pdf_path"))
        fallback_pdf = paper_pdf.resolve() if paper_pdf is not None else None
        resolved_pdf = source_pdf if (source_pdf is not None and source_pdf.exists()) else fallback_pdf
        if resolved_pdf is None or (not resolved_pdf.exists()):
            raise FileNotFoundError(
                "cannot resolve source_pdf_path from reused job state; "
                "please provide --paper-pdf when using --reuse-job-id."
            )
        key = str(paper_key or "").strip() or resolved_pdf.parent.name or "paper"

        own_payload = {
            "job_id": job_id,
            "status": job_state.get("status"),
            "message": job_state.get("message"),
            "error": job_state.get("error"),
            "artifacts": artifacts,
            "usage": job_state.get("usage") or {},
            "metadata": job_state.get("metadata") or {},
            "annotation_count": int(job_state.get("annotation_count") or 0),
            "final_report_ready": bool(job_state.get("final_report_ready")),
            "pdf_ready": bool(job_state.get("pdf_ready")),
            "job_json_path": str(job_json_path),
            "job_dir": str(job_dir),
            "latest_output_md": str(
                resolve_artifact_path(repo_root, artifacts.get("latest_output_md_path"))
                or (job_dir / "latest_extraction.md").resolve()
            ),
            "latest_output_pdf": str(
                resolve_artifact_path(repo_root, artifacts.get("latest_output_pdf_path"))
                or (job_dir / "latest_extraction.pdf").resolve()
            ),
        }
        return save_bridge_state(
            run_dir=run_dir,
            paper_pdf=resolved_pdf,
            paper_key=key,
            own_payload=own_payload,
        )

    if paper_pdf is None:
        raise FileNotFoundError(
            f"Bridge state missing at {_bridge_path(run_dir)}. "
            "Provide paper_pdf to bootstrap a standalone stage or run parse first."
        )
    resolved_pdf = paper_pdf.resolve()
    if not resolved_pdf.exists():
        raise FileNotFoundError(f"paper pdf not found: {resolved_pdf}")

    key = str(paper_key or "").strip() or resolved_pdf.parent.name or "paper"
    own_payload = _run_review_runtime(repo_root=repo_root, run_dir=run_dir, paper_pdf=resolved_pdf, title=key)
    return save_bridge_state(
        run_dir=run_dir,
        paper_pdf=resolved_pdf,
        paper_key=key,
        own_payload=own_payload,
    )
