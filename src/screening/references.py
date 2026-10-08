"""Entry-level RefCopilot check over the shared bibliography."""

import json
from pathlib import Path

from common import run_stats
from fact_generation.refcheck.refcheck import check_references
from schemas.claim import Finding
from schemas.materials import SharedMaterials
from screening.checks import paper_finding


def check_bibliography(
    materials: SharedMaterials, output_dir: Path, *, checker=None
) -> tuple[list[Finding], list[str]]:
    if not materials.bibliography:
        run_stats.record_module_status("reference_check", "skipped")
        return [], ["Reference check unavailable: parser supplied no bibliography entries."]
    with run_stats.timed_module("reference_check"):
        try:
            findings, issues, status = _check_bibliography(materials, output_dir, checker=checker)
        except Exception as exc:
            run_stats.record_module_status("reference_check", "failed", warning=str(exc))
            raise
        run_stats.record_module_status("reference_check", status, warning="; ".join(issues))
        return findings, issues


def _check_bibliography(materials, output_dir, *, checker=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "bibliography.txt"
    path.write_text("\n\n".join(b.text for b in materials.bibliography), encoding="utf-8")
    result = (checker or check_references)(paper=str(path))
    result_path = output_dir / "reference_check.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not result.get("ok"):
        return [], [f"Reference check failed: {result.get('error_message', 'unknown error')}; {result_path}"], "failed"
    if not isinstance(result.get("total_refs"), int) or result["total_refs"] <= 0:
        return [], [f"Reference check incomplete: no processed bibliography entries; {result_path}"], "failed"
    if not isinstance(result.get("issues"), list):
        return [], [f"Reference check returned no valid issues list; {result_path}"], "failed"
    findings, issues = [], []
    if result["total_refs"] != len(materials.bibliography):
        issues.append(
            f"Reference coverage needs review: checker processed {result['total_refs']} entries "
            f"from {len(materials.bibliography)} parser bibliography blocks; {result_path}"
        )
    for index, row in enumerate(result.get("issues", [])):
        title = str(row.get("reference_title") or row.get("reference") or "").strip()
        matches = [b for b in materials.bibliography if title and title.casefold() in b.text.casefold()]
        if len(matches) != 1 or matches[0].loc is None:
            issues.append(
                f"Reference finding lacks a unique paper location: {title}; {result_path}#issues.{index}"
            )
            continue
        block = matches[0]
        text = str(
            row.get("details") or row.get("message") or row.get("code") or row.get("type") or "Entry check"
        )
        text += f" [RefCopilot: {result_path}#issues.{index}]"
        findings.append(
            paper_finding(
                materials,
                block,
                quote=block.text,
                text=text,
                kind="reference",
                level=str(row.get("severity") or "unverified"),
            )
        )
    return findings, issues, "ok"
