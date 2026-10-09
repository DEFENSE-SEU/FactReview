"""Offline installed-wheel smoke test; reuse installed third-party dependencies.

Application modules must come from the wheel's new installation directory.
LLM, retrieval and Docker are never called. This verifies distribution contents
and local report rendering, not dependency resolution or scientific accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import traceback
import uuid
from pathlib import Path
from unittest.mock import patch


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def installed_smoke(target: Path, output: Path) -> int:
    # -I disables PYTHONPATH; also remove editable checkout paths added by .pth.
    roots = {
        "agent_runtime",
        "assessment",
        "common",
        "fact_generation",
        "llm",
        "preprocessing",
        "review",
        "schemas",
        "screening",
        "verification",
        "util",
        "refcopilot",
    }
    sys.path = [str(target)] + [
        p
        for p in sys.path
        if p
        and (
            Path(p).name in {"site-packages", "dist-packages"}
            or not any((Path(p) / name / "__init__.py").is_file() for name in roots)
        )
    ]
    attempts = []

    def external(*args, **kwargs):
        attempts.append("blocked external boundary")
        raise RuntimeError("Distribution smoke forbids external calls")

    class NoExternalProcess(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            external()

    outcome = {
        "passed": False,
        "external_attempts": attempts,
        "boundary": "installed application; existing third-party dependencies; mocked reference result",
    }
    try:
        with (
            patch.object(socket.socket, "connect", external),
            patch.object(socket, "create_connection", external),
            patch.object(subprocess, "Popen", NoExternalProcess),
        ):
            import refcopilot
            from refcopilot.llm.client import _load_factreview_llm

            import pipeline_full
            import pipeline_v2
            from fact_generation.refcheck.refcheck import check_references_with_records
            from review.report.v2 import write_review
            from schemas.claim import Claim, Condition
            from schemas.review import FinalReview

            # Verify lazy FactReview integration without sending a model request.
            _load_factreview_llm()
            with patch.object(refcopilot.RefCopilotPipeline, "run", return_value=refcopilot.Report()) as run:
                bundle = check_references_with_records("Distribution smoke bibliography", api_key="")
                assert run.call_count == 1
                assert bundle.records is not None
                assert bundle.payload["total_refs"] == 0

            with patch.object(sys, "argv", ["factreview", "paper.pdf", "--anonymity-policy", "required"]):
                args = pipeline_full.parse_args()
                assert args.anonymity_policy == "required"
            assert callable(pipeline_v2.run_v2_pipeline)
            review = FinalReview(
                paper_key="distribution-smoke",
                run_id="offline",
                claims=[
                    Claim(
                        id="c1",
                        text="The method is stable.",
                        loc={"page": 1},
                        conditions=[Condition(id="s", description="stability")],
                        needs=[],
                    )
                ],
                execution_requested=False,
            )
            files = write_review(review, output / "report", render_pdf=True)
            assert "pdf" in files and "pdf_error" not in files
            from pypdf import PdfReader

            pdf_text = " ".join(page.extract_text() for page in PdfReader(files["pdf"]).pages)
            assert "distribution-smoke" in pdf_text and "The method is stable." in pdf_text
            sources = {}
            for name, module in list(sys.modules.items()):
                if name.split(".")[0] not in roots | {"pipeline_full", "pipeline_v2"}:
                    continue
                path = getattr(module, "__file__", None)
                if path:
                    path = Path(path).resolve()
                    if not path.is_relative_to(target):
                        raise ValueError(f"Application module outside installed wheel: {name}")
                    sources[name] = str(path.relative_to(target))
            assert not attempts
            outcome.update(
                passed=True,
                imported_application_modules=sources,
                report=files,
                third_party_versions={
                    name: importlib.metadata.version(name)
                    for name in ("pydantic", "pypdf", "reportlab", "httpx", "rapidfuzz")
                },
            )
    except Exception as exc:
        # Store a fixed message and exception type, never provider configuration.
        outcome.update(
            error_type=type(exc).__name__,
            error="Installed-wheel smoke failed",
            frames=[
                {"file": Path(frame.filename).name, "line": frame.lineno}
                for frame in traceback.extract_tb(exc.__traceback__)
            ],
        )
        if isinstance(exc, ModuleNotFoundError):
            outcome["missing_module"] = exc.name
    save(output / "smoke.json", outcome)
    return 0 if outcome["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--installed-target", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    output = (args.output or Path("runs/distribution") / uuid.uuid4().hex[:12]).resolve()
    if args.installed_target:
        return installed_smoke(args.installed_target.resolve(strict=True), output)
    output.mkdir(parents=True, exist_ok=False)
    target = output / "installed"
    driver = output / "smoke_driver.py"
    driver.write_bytes(Path(__file__).read_bytes())
    save(
        output / "input.json",
        {
            "wheel": str(wheel),
            "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "driver_sha256": hashlib.sha256(driver.read_bytes()).hexdigest(),
        },
    )
    with (output / "install.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--disable-pip-version-check",
                "--target",
                str(target),
                str(wheel),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        print(f"Wheel installation failed: {output}")
        return result.returncode
    # No credentials or editable-source hints reach the smoke process.
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper()
        in {
            "SYSTEMROOT",
            "WINDIR",
            "PATH",
            "PATHEXT",
            "TEMP",
            "TMP",
            "HOME",
            "USERPROFILE",
            "APPDATA",
            "LOCALAPPDATA",
            "LANG",
            "LC_ALL",
            "LD_LIBRARY_PATH",
            "VIRTUAL_ENV",
        }
    }
    with (output / "smoke.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                str(driver),
                str(wheel),
                "--output",
                str(output),
                "--installed-target",
                str(target),
            ],
            cwd=output,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(json.dumps({"output": str(output), "passed": result.returncode == 0}))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
