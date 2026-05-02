from __future__ import annotations

import os
from typing import Any

from langgraph.graph import END, START, StateGraph

from .nodes.finalize import finalize_node
from .nodes.fix import fix_node
from .nodes.judge import judge_node
from .nodes.plan import plan_node
from .nodes.prepare import prepare_node
from .nodes.run import run_node

State = dict[str, Any]


def _is_inconclusive_no_baseline(state: State) -> bool:
    results = (state.get("judge", {}) or {}).get("results") or []
    if not isinstance(results, list):
        return False
    return any(isinstance(r, dict) and r.get("type") == "inconclusive_no_baseline" for r in results)


def _compute_exit_status(state: State) -> str:
    """
    Three-value exit status for clear reviewer semantics:
    - "success"      : passed baseline checks deterministically  (exit code 0)
    - "inconclusive" : run succeeded but no/insufficient baseline (exit code 2)
    - "failed"       : execution failed or checks failed          (exit code 1)
    """
    if state.get("status") == "failed":
        return "failed"
    judge = state.get("judge", {}) or {}
    if judge.get("passed") is True:
        return "success"
    if _is_inconclusive_no_baseline(state) and bool((state.get("run_result", {}) or {}).get("success")):
        return "inconclusive"
    return "failed"


def _route_after_prepare(state: State) -> str:
    if state.get("status") == "failed":
        # Some prepare failures are recoverable via fix loop.
        last_err = ""
        try:
            hist = state.get("history") or []
            if isinstance(hist, list):
                for h in reversed(hist):
                    if isinstance(h, dict) and h.get("kind") == "prepare_error":
                        data = h.get("data") or {}
                        if isinstance(data, dict):
                            last_err = str(data.get("error") or "")
                        break
        except Exception:
            last_err = ""

        recoverable = {
            "docker_env_ensure_failed",
            "docker_image_build_failed",
        }
        if last_err in recoverable:
            return "fix"
        return "finalize"
    return "plan"


def _route_after_plan(state: State) -> str:
    if state.get("status") == "failed":
        return "finalize"
    return "run"


def _route_after_run(state: State) -> str:
    if state.get("status") == "failed":
        return "fix"
    return "judge"


def _route_after_judge(state: State) -> str:
    if state.get("judge", {}).get("passed") is True:
        return "finalize"
    # If judge is inconclusive, do not enter fix loop.
    results = (state.get("judge", {}) or {}).get("results") or []
    if isinstance(results, list) and any(
        isinstance(r, dict) and r.get("type") == "inconclusive_no_baseline" for r in results
    ):
        return "finalize"
    return "fix"


def _route_after_fix(state: State) -> str:
    if state.get("status") == "failed":
        return "finalize"
    # try again
    return "run"


class ExecutionOrchestrator:
    def __init__(
        self,
        run_root: str,
        max_attempts: int = 5,
        no_llm: bool = False,
        llm_provider: str = "",
        llm_model: str = "",
        llm_base_url: str = "",
        llm_judge_mode: str = "",
        dry_run: bool = False,
        auto_tasks: bool = False,
        auto_tasks_mode: str = "smoke",
        auto_tasks_force: bool = False,
        enable_refcheck: bool = False,
        enable_bibtex: bool = False,
        paper_extracted_dir: str = "",
        run_dir: str = "",
    ) -> None:
        self.run_root = run_root
        self.max_attempts = max_attempts
        self.no_llm = no_llm
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm_judge_mode = (llm_judge_mode or os.getenv("EXECUTION_LLM_JUDGE_MODE", "assist")).strip()
        self.dry_run = dry_run
        self.auto_tasks = auto_tasks
        self.auto_tasks_mode = auto_tasks_mode
        self.auto_tasks_force = auto_tasks_force
        self.enable_refcheck = enable_refcheck
        self.enable_bibtex = enable_bibtex
        self.paper_extracted_dir = paper_extracted_dir
        self.run_dir = run_dir

        self._workflow = self._build_workflow()
        self._app = self._workflow.compile()

    def _build_workflow(self) -> StateGraph:
        g = StateGraph(State)
        g.add_node("prepare", prepare_node)
        g.add_node("plan", plan_node)
        g.add_node("run", run_node)
        g.add_node("judge", judge_node)
        g.add_node("fix", fix_node)
        g.add_node("finalize", finalize_node)

        g.add_edge(START, "prepare")
        g.add_conditional_edges("prepare", _route_after_prepare)
        g.add_conditional_edges("plan", _route_after_plan)
        g.add_conditional_edges("run", _route_after_run)
        g.add_conditional_edges("judge", _route_after_judge)
        g.add_conditional_edges("fix", _route_after_fix)
        g.add_edge("finalize", END)
        return g

    async def run(
        self,
        paper_root: str,
        paper_pdf: str,
        paper_key: str,
        tasks_path: str,
        baseline_path: str,
        local_source_path: str = "",
        no_pdf_extract: bool = False,
    ) -> dict[str, Any]:
        initial: State = {
            "status": "running",
            "attempt": 0,
            "max_attempts": self.max_attempts,
            "config": {
                "paper_root": paper_root,
                "paper_pdf": paper_pdf,
                "paper_key": paper_key,
                "tasks_path": tasks_path,
                "baseline_path": baseline_path,
                "local_source_path": local_source_path,
                "run_root": self.run_root,
                "no_llm": self.no_llm,
                "no_pdf_extract": no_pdf_extract,
                "llm_provider": self.llm_provider,
                "llm_model": self.llm_model,
                "llm_base_url": self.llm_base_url,
                "llm_judge_mode": self.llm_judge_mode,
                "dry_run": self.dry_run,
                "auto_tasks": self.auto_tasks,
                "auto_tasks_mode": self.auto_tasks_mode,
                "auto_tasks_force": self.auto_tasks_force,
                "enable_refcheck": self.enable_refcheck,
                "enable_bibtex": self.enable_bibtex,
                "paper_extracted_dir": self.paper_extracted_dir,
                "run_dir": self.run_dir,
            },
            "history": [],
        }
        final_state: State = await self._app.ainvoke(
            initial, config={"configurable": {"thread_id": "execution"}}
        )
        exit_status = _compute_exit_status(final_state)
        return {
            "success": exit_status == "success",
            "exit_status": exit_status,  # "success" | "inconclusive" | "failed"
            "state": final_state,
        }
