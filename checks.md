# FactReview v2 acceptance checks

Checkpoint: Phase 0 complete, 2026-10-08 (Asia/Shanghai); awaiting phase review.
Specification: `docs/method_v2_spec.md`; procedure: `docs/refactor_v2_prompt.md`.
Verdicts describe this checkpoint; passing rows must be revalidated after implementation.
Phases 1–8 have not started. The live baseline is blocked by missing MinerU credentials.

| requirement | verdict | evidence | correction |
|---|---|---|---|
| 1. `pytest` passes; no test was weakened or skipped to pass. | pass | `runs/v2_baseline/pytest-short-temp.log`: 256 passed, 3 deselected in 21.51s, exit 0. Existing marker selection preserved; no source/test/config changes. Explicit local harness mocks indirect Docker and loopback probes. | First run: 252 passed, 4 Windows temp-path failures (`pytest-baseline.log`). One environment correction: shorter `--basetemp`; assertions unchanged. Use recorded command in `progress.md`; standard Windows temp paths remain a known limitation. |
| 2. Pipeline order is parse/materials → L1 → L2 → L3 (optional) → assessment → report → teaser. | unresolved | `src/pipeline_full.py` awaits v2 integration. | Phases 2–8. |
| 3. Claims are extracted upfront; the report agent no longer extracts claims; C1–C3 cap and the merge-all-performance-claims rule are gone; the v2 splitting rule is tested. | unresolved | `src/agent_runtime/agent_prompt.py`; extraction tests pending. | Phase 3. |
| 4. Every claim has `loc`, `conditions`, `needs`; every evidence item has a verifiable `pointer`. | unresolved | `src/schemas/claim.py`; v2 schemas pending. | Phases 1–4. |
| 5. Dispatch sends a claim only to branches in its `needs`. | unresolved | L2 dispatcher and dispatch-matrix tests pending. | Phase 4. |
| 6. Figure check input = crop + caption + referencing sentences; only the three categories are reported. | unresolved | Shared figure materials and L1 figure check pending. | Phases 2–3. |
| 7. Literature retrieval enforces cutoff, concurrent window, self-exclusion, no author search, no review pages. | unresolved | `src/util/cutoff_date.py`; v2 retrieval enforcement pending. Explicit submission deadline and 3-month concurrent window confirmed. | Implement in Phase 4. |
| 8. Only Experiments emits execution plans; plans are claim-linked with feasibility and priority. | unresolved | Experiments branch and plan contracts pending. | Phases 1, 4–5. |
| 9. L3 counts a run as evidence only when Aligned = 1; forbidden repairs are rejected; max round count equals the confirmed value. | unresolved | Generic alignment and repair allowlist pending. Repair default/cap 3, recorded approval modes, and configurable training budget default 0 runs confirmed. | Implement in Phase 5. |
| 10. Assessment implements spec §6.3 exactly; every rule has a test. | unresolved | Deterministic assessment engine and ordered-rule tests pending. | Phase 6. |
| 11. Statuses are only supported/flawed/questioned/unverified across schemas, report, and teaser. | unresolved | v2 status migration and v1 artifact adapter pending. | Phases 1, 6–7. |
| 12. Report follows spec §7; contains no accept/reject recommendation. | unresolved | Four-part report and output tests pending. | Phase 7. |
| 13. CompGCN end-to-end run completes (or blocker is reported); differences from baseline recorded. | unresolved | `runs/v2_baseline/preflight.json`: MinerU token absent, repository `.env` absent. `compgcn-baseline.txt`: authorized credential skip, no live request or parser fallback. `baseline-manifest.json`: hashes for 7 unchanged reference artifacts. | Configure MinerU locally for a fresh live baseline; Phase 8 run/comparison remains outstanding. Existing demo outputs do not establish a fresh baseline. |
| 14. `progress.md` matches the files in the workspace. | pass | Separate reviewer inspected the checklist, logs, isolation harness, credential preflight, and 7 reference hashes. `git diff -- src scripts tests pyproject.toml RefCopilot demos LICENSE` is empty. | Revalidate at every phase checkpoint. |

Phase 0 review method: separate reviewer agent (`baseline_review`), inherited model/effort, no override.
The original default suite deselects 2 `e2e` tests and 1 `requires_docker` test. No new skip, marker change,
or weakened assertion was introduced. Reviewed selected paths use existing LLM/retrieval mocks;
`baseline_isolation.py` supplies the missing indirect Docker/loopback mocks for this local baseline.
This checkpoint records 2 pass / 12 unresolved; it does not authorize publishing the refactor.
