# FactReview v2 Refactor — Agent Prompt

Paste everything inside `<prompt>` into Claude Code (or another coding agent) at the repository root.
The structure follows a seven-layer long-task harness (context, procedure, tools, permissions,
review, effort, completion) adapted to this refactor.

<prompt>

<goal>
Refactor FactReview into the v2 architecture defined in `docs/method_v2_spec.md`. Leave a working
pipeline, passing tests, a verification report (`checks.md`), and enough saved state
(`progress.md`) for another session to continue. Completion is defined by the acceptance checks
below. Stop when they all pass, or when a blocker or the run limit is reached, and report exactly
what remains.
</goal>

<role>
Act as the builder and maintainer of this codebase. Read `docs/method_v2_spec.md`, `README.md`, and
the modules listed in the code map before editing. Make routine decisions within scope; record every
decision that changes behaviour or outputs in `progress.md` with its reason.
</role>

<principles>
- Establish a baseline first: run the existing test suite and the CompGCN demo (no execution) and
  save their outputs before changing anything.
- The spec is the fixed evaluator. If the code and the spec disagree, follow the spec; if the spec is
  ambiguous or contradicts how experiments were run, stop and ask (see open items).
- Change one stage at a time, run its tests, keep what works, record failed attempts.
- Prefer the simpler implementation when results are comparable. Reuse existing modules; do not
  rewrite working infrastructure (MinerU parsing, RefCopilot, Docker execution, PDF rendering).
- Never weaken a check or a test to make it pass.
</principles>

<inputs>
Task: implement the v2 method spec — upfront claim extraction, L1 checks, four L2 branches,
claim-linked execution plans, L3 alignment/approval/bounded repair, rule-based claim assessment with
four statuses, and the new report layout.
Deliverable and output path: code changes on branch `refactor/method-v2`; `checks.md` and
`progress.md` at repo root; updated tests under `tests/`.
Acceptance checks: see <layer_7_completion>.
Source material: `docs/method_v2_spec.md` (authoritative); current code (see code map);
`demos/Graph/compgcn` for end-to-end checks.
Allowed changes: `src/`, `scripts/`, `tests/`, `docs/`, `README.md`, `pyproject.toml`.
Do not modify `RefCopilot/` internals (call it as a library), `demos/*/` reference outputs, or `LICENSE`.
Run limit: stop after each phase in <layer_2_procedure> for review; at most 3 fix attempts per
failing check before reporting it as a blocker.
Fill any missing field before starting. If a missing detail changes the outcome, ask one focused question.
</inputs>

<code_map>
Current pipeline (`src/pipeline_full.py`): parse → refcheck → positioning → execution → report → teaser.
Claims are currently extracted at the END by the report agent (`src/agent_runtime/agent_prompt.py`,
"Contribution extraction constraints": C1–C3 only, rule (d) merges all performance claims), and
statuses are capped afterwards by `src/review/report/claim_audit.py`.

| v2 component | Reuse / change | Current location |
|---|---|---|
| Shared materials: text | reuse | `src/preprocessing/parse/` (MinerU) |
| Shared materials: page images, figure crops, caption + referencing sentences | NEW | add under `src/preprocessing/` |
| Shared materials: repository index | NEW (read-only) | add under `src/preprocessing/` |
| L1 writing check | NEW | add `src/screening/writing.py` |
| L1 figure check (VLM) | NEW | add `src/screening/figures.py` |
| L1 reference check | reuse, entry-level only | `src/fact_generation/refcheck/` (RefCopilot) |
| L1 claim extraction | MOVE + REWRITE | out of `agent_prompt.py` into a new upfront stage, e.g. `src/screening/claims.py`; replace C1–C3 limit and rule (d) with the v2 splitting rule |
| L2 Literature | reuse + extend | `src/fact_generation/positioning/` (+ citation-support check; cutoff and self-exclusion rules) |
| L2 Theory | NEW | add `src/verification/theory.py` |
| L2 Code | NEW | add `src/verification/code.py` (uses repository index) |
| L2 Experiments | NEW, absorbs logic | add `src/verification/experiments.py`; move Δ-vs-σ and missing-ablation logic out of `claim_audit.py`; emits plans |
| L3 execution | reuse graph, change plan input | `src/fact_generation/execution/` (`nodes/plan.py` consumes L2 plans; `task_infer.py` becomes a candidate-command source) |
| L3 alignment | generalise | `execution/tools/alignment.py` is CompGCN-specific (score_func/opn); implement `Aligned(dataset, metric, setting)` generically |
| L3 bounded repair | enforce | `execution/nodes/fix.py` allow-flags; default `max_attempts` 5 → see open item |
| Assessment | NEW, replaces status capping | add `src/assessment/` with rule-based aggregation; retire label logic in `claim_audit.py` |
| Schemas | CHANGE | `src/schemas/claim.py`: new `ClaimStatus`, `Claim.conditions`, `Claim.needs`, `Evidence`, `ExecutionPlan`, `Finding`, `AuthorQuestion`; drop `ClaimType` from routing |
| Report | CHANGE | `src/review/report/` consumes claim records instead of extracting claims; v2 layout |
| Teaser | CHANGE | `src/review/teaser/` uses the four v2 statuses |
| Cutoff | CHANGE | `src/util/cutoff_date.py`: submission-deadline cutoff + concurrent window; never derive from the paper's own arXiv id by default |
</code_map>

<layer_1_context>
Read `docs/method_v2_spec.md`, `CLAUDE.md` if present, and `progress.md` if it exists. Inspect the
files referenced in `progress.md` before trusting its status. Load only the source needed for the
current phase.
- Keep durable project facts in `docs/`; keep this task's temporary notes in `progress.md`.
- If the spec and an instruction here conflict, the spec wins; record the conflict.
</layer_1_context>

<layer_2_procedure>
Work in phases. Each phase ends with its tests passing and a `progress.md` update.

Phase 0 — Baseline. Run `pytest`; run `python scripts/execute_review_pipeline.py demos/Graph/compgcn/paper.pdf`
without execution if credentials exist (otherwise record that it was skipped). Save results to
`progress.md`.

Phase 1 — Schemas (spec §2, §6.1). Add `ClaimStatus {supported, flawed, questioned, unverified}`,
`Claim(text, loc, conditions, needs, importance, questions)`, `Evidence(source, pointer, covered,
direction, sufficient, note)`, `ExecutionPlan(claim_id, task, run_mode, y_paper, feasibility,
blocker, priority)`, `Finding(kind, loc, evidence, level)`. Provide a v1→v2 label adapter for old
run artifacts only (in_conflict→flawed is NOT automatic; map to questioned unless evidence is
re-assessed). Tests: schema round-trips.

Phase 2 — Shared materials (spec §3). Page rendering, figure crops from parser boxes, caption and
referencing-sentence linking by citation anchor, bibliography, read-only repository index. Tests on
fixtures: every figure gets crop + caption + ≥0 referencing sentences; repo index lists entry scripts
and configs.

Phase 3 — L1 (spec §5.1). Writing check, figure check (three categories only; printed-size
legibility), reference check wrapper (entry-level), upfront claim extraction with the v2 splitting
rule and multi-label `needs`. Remove claim extraction duties from `agent_prompt.py`. Tests: splitting
rule on crafted sentences (independent conclusions split; multi-setting claim not split).

Phase 4 — L2 (spec §5.2). Dispatcher sends each claim only to branches in `needs`. Implement
Literature (citation support + novelty retrieval with cutoff, concurrent window, self-exclusion, no
author search), Theory (main text first, appendix proofs), Code (repo index), Experiments (five
aspects + plan emission). Every evidence item must carry a verifiable `pointer`. Tests: dispatch
matrix; self-exclusion drops a near-duplicate title; post-cutoff works never become flaw evidence;
plans are claim-linked and carry feasibility/priority.

Phase 5 — L3 (spec §5.3). Plan intake from L2; approval gate (configurable: interactive / rule-based
auto-approve; record which was used); ordering policy; training only for high priority within budget;
generic `Aligned`; repair allowlist enforced with diffs recorded; judge-to-evidence mapping incl. the
author-artifact rule for `flawed`; blocked plans become evidence with reason; ledger. Tests: alignment
truth table; forbidden edit is rejected; mismatch → questioned; mismatch from shipped data/logs → flawed.

Phase 6 — Assessment (spec §6). Pure, deterministic rule engine over evidence items implementing the
five ordered rules, partial-coverage rule, and notes for non-decisive concerns. Tests: table-driven,
one case per rule plus conflict, partial coverage, paper-internal-only support.

Phase 7 — Report and teaser (spec §7). Four-part report; claim list ordered by status; evidence
source types visible; questions for authors; no accept/reject wording. Teaser uses v2 statuses.
Tests: report sections exist and order is correct; no recommendation language.

Phase 8 — End to end. Run the CompGCN demo (with execution if Docker and credentials are available).
Compare against the Phase 0 baseline and record differences in `checks.md`.
</layer_2_procedure>

<layer_3_tools>
Use local files and already configured connections (Codex/LLM client in `src/llm/`, MinerU, Semantic
Scholar/OpenAlex, Docker). Confirm each external source is reachable before a long run.
- Identify the exact file, page, or record needed; keep tool outputs focused on the next decision.
- Record retrieval or connection failures and the missing access in `progress.md`.
- Retry only after changing something relevant.
- Unit tests must not call external services; mock LLM, retrieval, and Docker.
If a required tool is unavailable, name the blocker and finish everything that does not depend on it.
</layer_3_tools>

<layer_4_permissions>
Keep edits inside the allowed paths. Work on branch `refactor/method-v2`; never push to `main`.
- Ask before pushing, opening a PR, or running paid/long jobs (full execution on GPU).
- Before re-running anything that writes outside `runs/`, check whether the first attempt succeeded.
- Commit at the end of each phase so every phase is recoverable.
- The repair allowlist and retrieval rules (cutoff, self-exclusion, no author search, no review
  pages) are enforced in code, not only in prompts.
</layer_4_permissions>

<layer_5_review>
After each phase, perform a distinct review pass (use a separate reviewer agent if available; say
which method was used). Save `checks.md` with one row per requirement:

requirement | verdict | evidence | correction

Use pass, fail, or unresolved. Evidence = test names, file paths, or actual command output. Correct
failures, then rerun affected checks. Leave missing evidence visible.
</layer_5_review>

<layer_6_effort>
Use the configured model and effort for routine edits; reserve deeper review for the assessment rule
engine, alignment, and claim extraction prompts, where errors silently change statuses. Report the
settings actually used; do not claim the prompt changes runtime effort.
</layer_6_effort>

<layer_7_completion>
Before starting, copy this checklist into `checks.md`. Each item must point to observable evidence.
1. `pytest` passes; no test was weakened or skipped to pass.
2. Pipeline order is parse/materials → L1 → L2 → L3 (optional) → assessment → report → teaser.
3. Claims are extracted upfront; the report agent no longer extracts claims; C1–C3 cap and the
   "merge all performance claims" rule are gone; the v2 splitting rule is tested.
4. Every claim has `loc`, `conditions`, `needs`; every evidence item has a verifiable `pointer`.
5. Dispatch sends a claim only to branches in its `needs`.
6. Figure check input = crop + caption + referencing sentences; only the three categories are reported.
7. Literature retrieval enforces cutoff, concurrent window, self-exclusion, no author search, no review pages.
8. Only Experiments emits execution plans; plans are claim-linked with feasibility and priority.
9. L3 counts a run as evidence only when `Aligned = 1`; forbidden repairs are rejected; the max round
   count equals the confirmed value.
10. Assessment implements spec §6.3 exactly; every rule has a test.
11. Statuses are only `supported/flawed/questioned/unverified` across schemas, report, and teaser.
12. Report follows spec §7; contains no accept/reject recommendation.
13. CompGCN end-to-end run completes (or the blocker is reported); differences from baseline recorded.
14. `progress.md` matches the files in the workspace.
Finish with output paths and verification results. If a limit or blocker prevents completion, return
a partial status with the exact remaining work.
</layer_7_completion>

<open_items>
Ask the maintainer before implementing these; do not guess:
1. Repair budget: spec and paper say ≤ 3 rounds; code default `max_attempts = 5`.
2. Approval mode and training budget actually used in experiments.
3. Cutoff source: venue submission deadline per run (how is it supplied?) vs current arXiv-id derivation.
4. Per-metric tolerance defaults (`execution/nodes/plan.py::_default_tolerance`, `tools/alignment.py`): keep or revise.
5. Whether v1 run artifacts must remain readable by the new report stage.
</open_items>

<execution_loop>
1. Inspect saved state and choose one bounded step.
2. Build or revise code within scope.
3. Run the tests relevant to that change.
4. Keep useful work; undo only your failed changes.
5. Record the outcome and checkpoint progress.
6. Continue until completion, a blocker, or the run limit.
Do not weaken acceptance checks to obtain a pass. Avoid repeating a failed approach without a new reason.
</execution_loop>

<handoff>
After each phase, update `progress.md`:
Task: the requested outcome and active constraints.
Outputs: exact paths to the current saved files.
Completed: finished phases and observed check results.
Decisions: choices made and their supporting evidence.
Open issues: failures, uncertainties, blockers, open items still unanswered.
Next action: one concrete step to resume the work.
On resumption, read this note, inspect its referenced files, and continue from the recorded next action.
</handoff>

<delivery>
Return the branch name, `checks.md`, and `progress.md`. State what was checked, what passed, and what
remains unresolved. Ask before pushing or opening the pull request. Report measured usage only when
available.
</delivery>

</prompt>
