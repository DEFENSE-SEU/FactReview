# FactReview v2 progress

## Task

Open the documentation-only PR, then implement the authoritative v2 method specification through Phases 0–8 on `refactor/method-v2`. Use an independent agent to review each phase, address findings, verify and commit, then continue automatically. The maintainer removed the phase-pause requirement on 2026-10-08 and requested continuous work until completion or an explicit stop. Keep all 14 acceptance checks observable. Never weaken or add skips to tests. Unit tests must mock LLM, retrieval, and Docker. Do not modify RefCopilot internals, demo reference outputs, or LICENSE. Keep refactor commits local until all checks pass or the user explicitly requests a push; the adopted prompt also requires approval before publishing the refactor.

## Outputs

- Documentation PR: https://github.com/DEFENSE-SEU/FactReview/pull/12
- Documentation branch: `docs/method-v2-spec`; initial patch commit `ab983b9`, automatic-review amendment `281e0f5` (also applied to the refactor branch as `84c2aeb`).
- Current branch: `refactor/method-v2`, created directly from the documentation branch.
- `docs/method_v2_spec.md`: authoritative specification, unchanged from the patch.
- `docs/refactor_v2_prompt.md`: adopted procedure, amended to use automatic phase reviews and continuous execution per the maintainer's request.
- `checks.md`: all 14 acceptance requirements.
- `progress.md`: this handoff.
- `runs/v2_baseline/`: local baseline logs (ignored by Git).
- `runs/v2_baseline/dependency-install.log`, `dependencies.txt`: installation output and exact installed versions.
- `runs/v2_baseline/pytest-baseline.log`, `pytest-baseline.exitcode`: first baseline (exit 1).
- `runs/v2_baseline/pytest-short-temp.log`, `pytest-short-temp.exitcode`: passing baseline retry (exit 0).
- `runs/v2_baseline/baseline_isolation.py`: explicit local pytest plugin for indirect Docker and loopback probes.
- `runs/v2_baseline/preflight.json`, `compgcn-baseline.txt`: credential readiness and live-demo skip reason.
- `runs/v2_baseline/baseline-manifest.json`: source commit identities and checksums of 7 unchanged demo reference artifacts.
- Phase 1: `src/schemas/claim.py`, `review.py`, `__init__.py`, `legacy_claim.py`, `legacy_review.py`, `v1_adapter.py`; `tests/test_schemas_v2.py` and import-only legacy adjustments in `tests/test_schemas.py`.
- Phase 1 evidence: `runs/v2_baseline/phase1-schema.log` (62 passed), `phase1-schema.exitcode` (0), `phase1-import-tail.log` (2 mocked legacy integration tests passed).
- Phase 2: `src/preprocessing/materials.py`, `src/schemas/materials.py`, `tests/test_materials_v2.py`; `runs/v2_materials/pytest.log` (25 passed).

## Completed

- Fetched origin and fast-forwarded main from `87dad71` to `5dc71d9` before creating the documentation branch.
- Applied the attached patch with `git am`; no fallback copying needed.
- Pushed only the documentation branch and opened PR #12 with the requested title and attached `PR_DESCRIPTION.md` as the body. The PR changes exactly two documentation files.
- Phase 0 complete: installed existing extras into `.venv` using Python 3.12.10. No application code, test assertions, or project configuration changed.
- Initial baseline: 252 passed, 4 failed, 3 deselected in 29.41s (exit 1). Failures: `test_no_docker_host_venv_install_follows_nested_verify_missing_module`, `test_no_docker_host_venv_install_repairs_dgl_graphbolt_torch_pin`, `test_no_docker_host_venv_installs_pyg_native_package_from_torch_wheel_index`, `test_install_run_venv_jupyter_kernel_uses_run_local_prefix`.
- Diagnosis: Windows default pytest temporary paths exceed 100 characters; `src/fact_generation/execution/nodes/fix.py::_host_venv_dir` then selects `C:\\frv-venvs/<hash>`, while these four tests expect run-local `.venv`. Preserved the initial failure output.
- One environment correction: selected a new shorter pytest `--basetemp` under `runs/v2_baseline/t1`. Reran the same 256 selected tests with unchanged assertions: **256 passed, 3 deselected in 21.51s**, exit 0. The 3 default deselections are 2 `e2e` tests and 1 `requires_docker` test.
- CompGCN live no-execution baseline: skipped under the prompt's explicit credential exception. `MINERU_API_TOKEN` and repository `.env` are absent; strict parsing requires the token (`src/preprocessing/parse/mineru_adapter.py:49–57`). No live pipeline request or parser fallback used. Codex cached credentials are present; validity/reachability remains untested. No external source reachability claim is made.
- Independently reviewed the local harness, selected external-call paths, failure cause, checkpoint files, and all 7 unchanged artifact hashes with reviewer agent `baseline_review`.
- Phase 1 complete: strict v2 claim/evidence/plan/finding/question schemas and v2 review container; historical schemas isolated in explicit legacy modules. Schema checks: 62 passed. Legacy stage-import/report-tail integration: 2 passed with mocked services. Existing schema assertions retained; only their legacy imports changed.
- Phase 1 automatic review (`phase_reviewer`) found ambiguous execution target keys/non-finite targets and two legacy Markdown parsing defects. Corrections enforce exact condition-id targets with finite values, preserve escaped pipes, flag malformed rows, and normalize emphasized status labels. Regression tests cover every finding. Ruff passes on Phase 1 files.
- Real CompGCN v1 `report.md` adapter smoke: 3 historical display records (`questioned`, `supported`, `supported`), zero issues. Historical labels remain explicitly unassessed; no new evidence or missing coordinates fabricated.
- Phase 1 committed as `f2888b0`.
- Phase 2 complete: direct MinerU adapter entry, original Markdown/content list, located blocks, bibliography, 200 dpi pages/crops, 96 dpi printed crops and a read-only repository index. 25 fixture tests passed; Ruff check/format passed.
- Phase 2 automatic review (`phase_reviewer`) found missing appendix/Oxford-list figure anchors and dropped MinerU list items. Both fixed with regression coverage; unrecognized parser content leaves explicit issues. Python 3.11 compatibility for junction checks verified. Printed crop dimensions use PDF physical dimensions; the initial clip-rounding failure was fixed without changing assertions.
- Refactor branch remains local. Phases 3–8 are outstanding; independent preparation of later-phase files is kept outside earlier phase commits.

### Baseline command

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = "$PWD\src;$PWD\runs\v2_baseline"
.\.venv\Scripts\python.exe -m pytest -p baseline_isolation --basetemp runs/v2_baseline/t1
```

The first run used the same command without `--basetemp`. For a repeat, choose a **new** short directory under `runs/v2_baseline/` and save a new log; pytest clears an existing explicit base-temp directory. Verify the resolved path stays within that baseline directory. Inspect previous results before repeating operations that could write outside `runs/`. The local ignored logs/plugin survive in this workspace; this tracked note preserves the command, outcomes, limitations, and next action for other checkouts.

## Decisions

- Use per-command Git committer identity `ChaoqianO <224349230+ChaoqianO@users.noreply.github.com>`, derived from the authenticated GitHub account. Git had no configured identity; global Git configuration was left unchanged. Patch author/message retained by `git am`.
- Use `.venv` for repository dependencies; preserve global Python packages. Install the existing runtime/dev/refcheck/positioning extras needed to collect and exercise the current suite.
- Preserve the existing default pytest marker selection; record deselections explicitly. Phase 0 left tests unchanged. Phase 1 moved old schema imports to the explicit compatibility modules while preserving every assertion.
- Apply the user's required unit-test mocks via the explicit local baseline plugin. A separate review found existing Docker argument-construction tests indirectly call `docker info` through `_docker_info_field`; the stale-loopback test calls `socket.create_connection`. The plugin mocks these two probes, preserving test-specific overrides. Existing selected LLM/retrieval paths were reviewed for their own mocks. The plugin is scoped baseline tooling and provides no universal network-access guarantee. Incorporate durable unit-test isolation in the affected implementation phase.
- Use shorter test temporary paths for this Windows baseline. No production path handling or assertion was changed. The long-default-path failures remain documented as test portability work.
- The directly requested root files `checks.md` and `progress.md`, plus explicitly requested baseline outputs in `runs/`, are authorized outputs alongside the prompt's source-path list.
- Phase review uses a separate reviewer agent with inherited configured model/effort and no override. Exact runtime model/effort identifiers are unavailable to this task; no claim is made that prompt wording changes them.
- The maintainer removed phase pauses on 2026-10-08. Documentation PR #12 was updated with automatic reviews; implementation continues immediately after each phase's review, checks and commit. Prepare independent components in parallel, keeping phase commits and integration validation in dependency order.
- Canonical v2 schemas forbid unknown fields; evidence coverage and execution targets use stable condition IDs. Each target value is keyed by condition ID, so two datasets using the same metric remain distinct. New records require an actual page, section or character span; missing legacy coordinates stay missing in a separate historical envelope.
- The v1 adapter migrates display labels conservatively and retains original records. Reading a historical artifact performs no v2 reassessment; the adapter never creates sufficient evidence from old prose. Unknown labels and malformed rows remain visible as issues.
- Shared materials use the original MinerU adapter directly because the legacy parse stage also runs retrieval and report generation internally. Its old runtime remains available for compatibility; v2 integration will use the new material entry point.
- Bounding-box units are explicit (`normalized_1000` by default, or `pdf_points`), with 1-based PDF page locations. Missing/invalid boxes and unmatched text remain visible issues. Figure checks receive the physically downscaled 96 dpi image. Repository indexing reads and hashes files without executing or modifying them; L3 creates a separate execution workspace.
- **Maintainer-confirmed configuration (2026-10-08):** configurable repair limit defaults to 3, capped at 3 per spec; rule-based automatic approval by default with optional interactive approval and actual mode recorded; training budget defaults to **0 runs**, configurable, with training eligible only for high-priority plans within budget; explicit `--submission-deadline YYYY-MM-DD` with a 3-month concurrent window and no default arXiv-derived cutoff; existing per-metric tolerances centralized in one table; one-way v1 artifact adapter maps `in_conflict` to `questioned` unless re-assessed. The user confirmed the bundled choices in Chinese. These choices are authorized for later phases; no v2 behavior is implemented yet.

## Open issues

- All five configuration choices and a default training budget of 0 runs are confirmed. Continue user-facing communication in Chinese.
- No actual experimental approval mode or training budget has been supplied. Do not describe proposed defaults as historical experiment settings.
- Live baseline unavailable until MinerU is configured locally. Never store credentials in tracked notes or ask for secret values in chat.
- Four existing unit tests depend on short Windows temporary paths; the recorded baseline command passes with a short base-temp directory. Ordinary pytest using long default Windows paths still has the documented failures.
- Baseline Docker/socket isolation lives in ignored local tooling. Make unit tests self-contained during subsequent relevant work while retaining their assertions.
- Phases 3–8 remain outstanding; main-pipeline integration and remote repository material snapshots are still pending.

## Next action

Complete Phase 3 screening and upfront claim extraction, including actual VLM image transport and removal of report-agent extraction duties. Independently review, test and commit, then continue to the four L2 branches.
