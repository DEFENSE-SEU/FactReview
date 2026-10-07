# FactReview v2 acceptance checks

Checkpoint: Phases 0–3 complete, 2026-10-08 (Asia/Shanghai); independent automatic reviews. Phase 4 underway.
Specification: `docs/method_v2_spec.md`; procedure: `docs/refactor_v2_prompt.md`.
Verdicts describe this checkpoint; passing rows must be revalidated after implementation.
Phases 4–8 remain outstanding. The live baseline is blocked by missing MinerU credentials.

| requirement | verdict | evidence | correction |
|---|---|---|---|
| 1. `pytest` passes; no test was weakened or skipped to pass. | pass | `runs/v2_baseline/pytest-short-temp.log`: 256 passed, 3 deselected in 21.51s, exit 0. Existing marker selection preserved; no source/test/config changes. Explicit local harness mocks indirect Docker and loopback probes. | First run: 252 passed, 4 Windows temp-path failures (`pytest-baseline.log`). One environment correction: shorter `--basetemp`; assertions unchanged. Use recorded command in `progress.md`; standard Windows temp paths remain a known limitation. |
| 2. Pipeline order is parse/materials → L1 → L2 → L3 (optional) → assessment → report → teaser. | unresolved | `src/pipeline_full.py` awaits v2 integration. | Phases 2–8. |
| 3. Claims are extracted upfront; the report agent no longer extracts claims; C1–C3 cap and the merge-all-performance-claims rule are gone; the v2 splitting rule is tested. | pass | `screening/claims.py`, `screening/stage.py`; 24 claim tests cover independent conclusions, multiple settings, >3 claims, exact grounding and report-prompt duties. Phase 3 combined log: 45 passed. | Main-pipeline integration remains check 2. Semantic LLM accuracy remains a live-run limitation. |
| 4. Every claim has `loc`, `conditions`, `needs`; every evidence item has a verifiable `pointer`. | unresolved | Phase 1 schema suite: 62 passed (`phase1-schema.log`); strict fields, coverage, source-specific pointer structure, exact finite plan targets. | Verify actual source existence in producer branches during Phases 2–4. |
| 5. Dispatch sends a claim only to branches in its `needs`. | unresolved | L2 dispatcher and dispatch-matrix tests pending. | Phase 4. |
| 6. Figure check input = crop + caption + referencing sentences; only the three categories are reported. | pass | Material tests cover crops, printed dimensions, linking; screening tests enforce categories, full context and actual printed pixels; three provider transport tests inspect encoded bytes (`runs/v2_screening/pytest.log`). | Missing images or failed checks remain explicit issues. |
| 7. Literature retrieval enforces cutoff, concurrent window, self-exclusion, no author search, no review pages. | unresolved | `src/util/cutoff_date.py`; v2 retrieval enforcement pending. Explicit submission deadline and 3-month concurrent window confirmed. | Implement in Phase 4. |
| 8. Only Experiments emits execution plans; plans are claim-linked with feasibility and priority. | unresolved | Experiments branch and plan contracts pending. | Phases 1, 4–5. |
| 9. L3 counts a run as evidence only when Aligned = 1; forbidden repairs are rejected; max round count equals the confirmed value. | unresolved | Generic alignment and repair allowlist pending. Repair default/cap 3, recorded approval modes, and configurable training budget default 0 runs confirmed. | Implement in Phase 5. |
| 10. Assessment implements spec §6.3 exactly; every rule has a test. | unresolved | Deterministic assessment engine and ordered-rule tests pending. | Phase 6. |
| 11. Statuses are only supported/flawed/questioned/unverified across schemas, report, and teaser. | unresolved | Canonical schemas expose only four v2 statuses. Explicit `legacy_claim.py`/`legacy_review.py` retain historical contracts; adapter tests verify `in_conflict` → `questioned`. | Complete report, teaser and assessment migration in Phases 6–7. |
| 12. Report follows spec §7; contains no accept/reject recommendation. | unresolved | Four-part report and output tests pending. | Phase 7. |
| 13. CompGCN end-to-end run completes (or blocker is reported); differences from baseline recorded. | unresolved | `runs/v2_baseline/preflight.json`: MinerU token absent, repository `.env` absent. `compgcn-baseline.txt`: authorized credential skip, no live request or parser fallback. `baseline-manifest.json`: hashes for 7 unchanged reference artifacts. | Configure MinerU locally for a fresh live baseline; Phase 8 run/comparison remains outstanding. Existing demo outputs do not establish a fresh baseline. |
| 14. `progress.md` matches the files in the workspace. | pass | Independent Phase 0/1 reviews; Phase 1 schema log (62 passed) and stage-import/tail log (2 passed). Phase 1 changes and later-phase preparation recorded in `progress.md`. Protected demo/RefCopilot/LICENSE files unchanged. | Revalidate at every phase checkpoint. |

Phase 0 review method: separate reviewer agent (`baseline_review`), inherited model/effort, no override.
The original default suite deselects 2 `e2e` tests and 1 `requires_docker` test. No new skip, marker change,
or weakened assertion was introduced. Reviewed selected paths use existing LLM/retrieval mocks;
`baseline_isolation.py` supplies the missing indirect Docker/loopback mocks for this local baseline.
This checkpoint records 4 pass / 10 unresolved; it does not authorize publishing the refactor.

Phase 1 review: `phase_reviewer` identified three schema/adapter defects, corrected with regression
tests. `runs/v2_baseline/phase1-schema.log`: 62 passed; `phase1-import-tail.log`: 2 passed.
The legacy tests retain all original assertions. Full-suite baseline remains recorded above;
the final combined suite will be rerun after integration.

Phase 2 review: `phase_reviewer` found appendix/Oxford-list anchor and list-item loss defects;
both corrected with regression tests. `runs/v2_materials/pytest.log`: 25 passed. No demo,
RefCopilot or LICENSE edits. The v2 entry point is ready for later main-pipeline integration.

Phase 3 review: `phase_reviewer` closed zero-reference-processing and LLM-failure masking findings.
`runs/v2_screening/pytest.log`: 45 passed. Report-agent claim extraction duties removed.
