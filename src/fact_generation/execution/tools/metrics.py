from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _as_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def compute_check(run_artifacts_dir: str, check: dict[str, Any]) -> dict[str, Any]:
    """
    Deterministic check runner. Supported check types:
    - file_exists: expects { "type":"file_exists", "path":"relative/to/artifacts" }
    - json_value:  expects { "type":"json_value", "path":"...", "json_path":["a","b",0], "expected": 1.23, "tolerance": 0.0 }
    - csv_agg:     expects { "type":"csv_agg", "path":"...", "expr": {"groupby":[...], "agg": {"col":"mean"}}, "expected": {...}, "tolerance": 0.0 }
      NOTE: csv_agg is intentionally limited; extend as needed later.
    """
    ctype = str(check.get("type") or "").strip()
    rel = str(check.get("path") or "").strip()
    artifact_path = Path(run_artifacts_dir) / rel

    if ctype == "file_exists":
        ok = artifact_path.exists()
        return {"type": ctype, "path": rel, "passed": ok, "observed": ok, "expected": True}

    if ctype == "json_value":
        if not artifact_path.exists():
            return {"type": ctype, "path": rel, "passed": False, "error": "missing_file"}
        obj = json.loads(artifact_path.read_text(encoding="utf-8", errors="ignore") or "{}")
        jp = check.get("json_path") or []
        cur: Any = obj
        try:
            for part in jp:
                if isinstance(part, int) and isinstance(cur, list):
                    cur = cur[part]
                elif isinstance(part, str) and isinstance(cur, dict):
                    cur = cur.get(part)
                else:
                    cur = None
            expected = check.get("expected")
            tol = float(check.get("tolerance") or 0.0)
            obs_f = _as_float(cur)
            exp_f = _as_float(expected)
            if obs_f is None or exp_f is None:
                passed = cur == expected
            else:
                passed = abs(obs_f - exp_f) <= tol
            return {
                "type": ctype,
                "path": rel,
                "passed": passed,
                "observed": cur,
                "expected": expected,
                "tolerance": tol,
            }
        except Exception as e:
            return {"type": ctype, "path": rel, "passed": False, "error": str(e)}

    if ctype == "csv_agg":
        if not artifact_path.exists():
            return {"type": ctype, "path": rel, "passed": False, "error": "missing_file"}
        try:
            import pandas as pd  # type: ignore
        except Exception as e:
            # Avoid making the whole framework unusable due to pandas/numpy issues when csv_agg isn't needed.
            return {
                "type": ctype,
                "path": rel,
                "passed": False,
                "error": f"pandas_unavailable: {type(e).__name__}: {e}",
            }
        df = pd.read_csv(artifact_path)
        expr = check.get("expr") or {}
        groupby = expr.get("groupby") or []
        agg = expr.get("agg") or {}
        expected = check.get("expected")
        tol = float(check.get("tolerance") or 0.0)
        try:
            g = df.groupby(groupby).agg(agg).reset_index()
            # compare by converting to json-like list
            observed = g.to_dict(orient="records")
            # baseline expected can be provided as list-of-records; compare len and numeric within tol
            passed = True
            if isinstance(expected, list):
                if len(expected) != len(observed):
                    passed = False
                else:
                    for exp_row, obs_row in zip(expected, observed, strict=False):
                        for k, exp_v in exp_row.items():
                            obs_v = obs_row.get(k)
                            exp_f = _as_float(exp_v)
                            obs_f = _as_float(obs_v)
                            if exp_f is not None and obs_f is not None:
                                if abs(exp_f - obs_f) > tol:
                                    passed = False
                            else:
                                if exp_v != obs_v:
                                    passed = False
            else:
                # if no expected given, treat as "computed successfully"
                passed = True
            return {
                "type": ctype,
                "path": rel,
                "passed": passed,
                "observed": observed,
                "expected": expected,
                "tolerance": tol,
            }
        except Exception as e:
            return {"type": ctype, "path": rel, "passed": False, "error": str(e)}

    return {"type": ctype or "unknown", "passed": False, "error": "unsupported_check_type"}
