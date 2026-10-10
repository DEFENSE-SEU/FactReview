# Execution ablations

Switches for the L3 execution ablations (§5.3 of `method_v2_spec.md`). Each one is a CLI flag of
`src/pipeline_full.py` and, where noted, a field of `ExecutionConfig` (`src/fact_generation/execution/v2_config.py`)
that can also be set through `--execution-config`. A CLI flag given explicitly overrides the JSON file.

| Ablation | Switch | Default (main experiment) | Ablation setting |
|---|---|---|---|
| No execution | `--run-execution` | off: L3 is skipped and the execution stage is recorded as disabled | run without `--run-execution` |
| No alignment rule | `--alignment` / `alignment` | `strict`: dataset, metric and settings must all equal the claim condition | `relaxed` |
| Repair rounds 0–3 | `--max-attempts` / `max_attempts` | `3` | `0`, `1`, `2`, `3` |
| Repair proposer | `--repairer` / `repairer` | `default`: only adds a missing dependency after `ModuleNotFoundError` | `llm` |
| Approval mode | `--approval-mode` / `approval_mode` | `auto` | — |
| Training budget | `--training-budget` / `training_budget` | `0` (no training runs) | — |

The main experiment runs with `--run-execution`, `--approval-mode auto` and `--training-budget 0`.

## Alignment

With `relaxed`, an observation whose dataset equals the claim condition's dataset is compared with
the paper number and becomes execution evidence; metric and settings are not checked. When several
observations share that dataset, the one with the claimed metric is chosen; if that still leaves
more than one, or none, the condition stays unaligned. Unit checks, released-resource checks and the
tolerance rules are unchanged. The evidence note and each `alignment` ledger entry record
`alignment_mode`.

`relaxed` applies on every path that compares an observation with the claim condition: candidate
selection in `execute_plans` (`src/fact_generation/execution/v2.py`) and, for Docker runs with a
source-science proposal, the actual-metadata check in `build_consumption_context`
(`src/fact_generation/execution/runtime_science.py`). There it also skips the requirement that the
condition settings be observable model/split/sample_count, and the semantic review receives only
the `condition/dataset` and `condition/description` obligations; the paper/source role obligations
stay. Two checks are not relaxed because they verify resource use and not condition matching: the
released-predictions measurement (`prediction_measurement.py`) still requires the author output to
equal the released config it read, and the actual partition must be the one the proposal selected.

## Repair

The default repairer only proposes dependencies, so path, launch-argument and wrapper repairs never
occur and runs with `--max-attempts 1..3` can be identical. Use `--repairer llm` for the repair-round
ablation. The LLM repairer (`llm_repair` in `src/fact_generation/execution/v2.py`) first applies the
default dependency fix; otherwise it sends the command, dependencies and the last 4000 characters of
stdout/stderr to `llm_json` and asks for one repair or `null`. Its proposal passes through the same
`Repair` schema and `_apply_repair` whitelist as any other repairer, and a rejected proposal ends the
repair loop with an `execution.repair` failure in the ledger.
