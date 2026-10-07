# FactReview v2 — Method Specification

Status: agreed design for the ARR resubmission. This document is the single source of truth for
the v2 refactor and for the Method section of the paper. Anything not covered here is out of scope
until it is added here first.

## 0. Goal and principles

**Goal.** Make LLM-assisted peer review trustworthy by working at the level of individual claims:
every judgement is attached to a specific claim in the paper and backed by evidence a reviewer can
check. FactReview assists human reviewers; it never issues an accept/reject recommendation.

**Trustworthy** means three testable properties:

| Property | Meaning | Implemented by | Measured by |
|---|---|---|---|
| Traceable | Every judgement points to a location in the paper and to concrete evidence | `loc` on claims, `pointer` on evidence | Per-item feedback quality (evidence validity) |
| Evidence-bounded | No conclusion is stronger than its evidence | Four statuses with explicit rules; execution evidence only after alignment | False-support rate, false-accusation rate |
| Checkable | A reviewer can re-check a judgement in reasonable time | Claim-level evidence report | Reviewer-assistance study, incl. adoption of wrong items |

## 1. Deliverables (what the system returns)

1. **Claim records** — one per extracted claim: text, location, conditions, evidence needs,
   evidence items (with source type and pointer), status, and questions for the authors.
2. **Findings** — issues not tied to a single claim: writing, figures, references, missing related
   work or baselines. Each has a location and evidence.
3. **Questions for authors** — reviewer-facing items: what needs clarification, what could not be
   verified and why.

## 2. Core data structures (paper notation = code names)

```text
claim_i     = < text_i, loc_i, conditions_i, needs_i >
              loc_i        = (page, section, char span)
              conditions_i = datasets, metrics, settings under which the claim is asserted
              needs_i      ⊆ {Literature, Theory, Code, Experiments}   (multi-label)

ev          = < source, pointer, covered, direction, sufficient >
              source    ∈ {paper_internal, literature, theory, code, execution}
              pointer   = verifiable location (quote+page, DOI/arXiv id+passage, file:line, log path+key)
              covered   ⊆ conditions_i     (which part of the claim this item speaks to)
              direction ∈ {support, flaw}
              sufficient: bool             (see §6.2)

plan_j      = < claim_i, task_j, y_paper_j, run_mode, feasibility, priority >

Aligned(run_k, claim_i) = 1[ dataset(run_k) = dataset(claim_i)
                             ∧ metric(run_k)  = metric(claim_i)
                             ∧ setting(run_k) ∈ conditions_i ]

edits_k ⊆ A_allowed = {dependencies, paths, launch arguments, wrappers},  k ≤ 3
```

Status set: `Supported`, `Flawed`, `Questioned`, `Unverified`.

## 3. Shared materials (prepared once, before any check)

| Material | Preparation | Consumers |
|---|---|---|
| Paper text | MinerU, section-aware, keeps tables, equations, captions, citation anchors, page locations | Claim extraction, writing check, all L2 branches |
| Page images + figure crops | Render each page; crop each figure using parser bounding boxes; link caption and every body sentence that references the figure | Figure check (VLM) |
| Bibliography | Parsed reference list | Reference check, Literature branch |
| Repository index | Read-only file index of the released repo (docs, configs, entry scripts, source) | Code branch, Experiments plan generation |
| Execution workspace | Fresh checkout created only when L3 runs | L3 |

## 4. Layering rationale

Checks go progressively deeper, and each layer works on objects produced by the previous one:

- **L1 reads the paper as written** — is it clear, are references real, what does it claim.
- **L2 cross-checks claims against existing material** — literature, derivations, code, experiment design.
- **L3 produces new evidence** — by running released artifacts.

L1 → L2 handoff: tagged claim records. L2 → L3 handoff: claim-linked execution plans.
Findings that need no execution (L1 checks, Literature/Theory/Code branches) go straight to assessment.

## 5. Layers

Every module is specified as: problem → why it matters → how it is checked → output.

### 5.1 L1 — Screening and claim extraction

**Writing check.** Typos, grammatical errors, and unclear sentences. Each item quotes the original
sentence and gives its location. Two levels: *definite error* (typo, grammar) and *clarity issue*
(reviewer's judgement). Reported as a separate list after claim results. Rationale for reviewers:
lets them substantiate "needs proofreading / unclear writing" with concrete locations.

**Figure check.** Input per figure = cropped image + caption + all body sentences referencing it.
Checks three things only:
- *Self-containedness*: legend, axis labels and units, panel labels present;
- *Legibility*: text readable at printed size (downscale crop to its printed size before judging);
- *Text–figure consistency*: caption and body references match what the figure shows (e.g. body
  cites panel (c) that does not exist; caption mentions a dashed line that is absent).

Excluded: colour, style, aesthetics. Tables are checked from parsed text (headers, units), not by VLM.

**Reference check.** Entry-level only: does the reference exist; are authors, year, venue correct;
is it retracted/withdrawn. (Existing RefCopilot.) Whether a cited work supports the citing sentence
is *not* checked here — that belongs to the Literature branch.

**Claim extraction.** Extract review-relevant claims (atomic, localizable, consequential, checkable).
Splitting rule:
- Split when a statement contains independent conclusions that could receive different outcomes
  (e.g. "best accuracy and faster inference").
- Do **not** split when one conclusion is asserted over several settings (e.g. "outperforms
  baselines on five datasets"); keep one claim whose `conditions` list the settings.

Each claim gets `loc`, `conditions`, `needs` (multi-label), and an importance level (core / secondary).

**Handoff L1 → L2.** Each claim is dispatched to exactly the branches in `needs_i`.

### 5.2 L2 — Typed verification

All four branches are equal peers and run in parallel per claim. Each returns evidence items.

**Literature.**
1. *Citation support*: for citations attached to claim sentences, does the cited work support the sentence?
2. *Novelty and positioning*: actively retrieve uncited related work; compare the paper with the
   nearest prior work on mechanism, target setting, and evaluation protocol. Also report missing
   related work / baselines as findings even when no explicit novelty claim exists.

Retrieval constraints: cutoff = submission deadline of the target venue; works within 3 months
before the cutoff are labelled *concurrent* (reported, never used against novelty); the paper's own
versions (preprints, venue page) are excluded by title/abstract similarity; author identity is never
searched; review pages of the submission are never accessed.

Novelty status mapping: same mechanism and setting → `Flawed`; partial overlap → `Questioned`;
adequate search finds nothing close → `Supported` **with the search scope listed**; inadequate search → `Unverified`.

**Theory.** Main text first; consult appendix proofs for theorems stated in the main text. Checks
derivation steps, unstated assumptions, edge cases, notation consistency. No proof anywhere → `Unverified`.

**Code.** Compare paper descriptions with configs and source via the repository index
(architecture, loss, optimizer, hyper-parameters, data processing, evaluation protocol).

**Experiments.** From the paper alone, check five aspects:
- correspondence (each experimental claim has an experiment),
- fairness (same data, budget, tuning for compared methods),
- isolation (ablation for each credited component),
- stability (variance / seeds / significance where gaps are small),
- consistency (numbers in abstract/text vs tables).

Paper-internal evidence may support `Supported`, but its source type stays `paper_internal` and the
report shows it, so reviewers can see it was not independently reproduced.

**Handoff L2 → L3 (execution plans).** For each experimental claim whose numbers could be
re-obtained from released artifacts, emit one plan entry:

| Field | Content |
|---|---|
| claim | the target claim (plans are claim-linked, not experiment-linked) |
| task | candidate entry script + config found in the repository index (may be empty) |
| run_mode | `evaluation` (released weights) / `analysis` (recompute from released data) / `training` |
| y_paper | reported target value(s) taken from the paper |
| feasibility | `ready` / `blocked` + blocker reason (missing code, data, weights, budget) |
| priority | `high` / `medium` / `low`, by link to the paper's core contribution |

Unresolved details (exact arguments, metric output location) are left to L3.

### 5.3 L3 — Execution verification

1. **Approval.** Show the plan with estimated cost; the operator confirms. Training is allowed only
   for high-priority plans within the training budget. The approval mode used in experiments must be
   reported (TODO: actual mode + budget).
2. **Order.** High priority first; within a priority, `ready` before `blocked`; evaluation/analysis before training.
3. **Refine plan.** Fix commands, arguments, metric output location from the repo.
4. **Sandbox run.** Docker, logged.
5. **Align.** A result is evidence only if `Aligned(run, claim) = 1`.
6. **Bounded fix.** Allowed edits: dependencies, paths, launch arguments, wrappers. Forbidden:
   model/architecture, losses, data, evaluation logic, baselines. At most **3** rounds.
7. **Judge → evidence.**
   - aligned and consistent with `y_paper` → `support` evidence;
   - aligned but clearly inconsistent → concern → claim usually `Questioned` (list gap and run conditions);
   - inconsistency derived **from the authors' own released data/logs** (no environment explanation possible) → `Flawed`;
   - not alignable or failed → no evidence; if nothing else supports the claim → `Unverified` with reason.
   Consistency is judged by metric type, reported variance, and gap size; per-metric default
   tolerances are listed in the appendix.
8. **Blocked plans** are kept; the blocker becomes evidence of *why* the claim is unverified.
9. **Execution ledger.** Commands, return codes, environments, configs, logs, metrics, alignment
   decisions, accepted repairs (diffs), runtime, tokens.

## 6. Claim assessment (3.5)

### 6.1 Status definitions

- **Supported** — sufficient support evidence covers the whole claim (all `conditions`) and there is
  no concern that bears on whether the claim holds.
- **Flawed** — sufficient flaw evidence that no reasonable author explanation could overturn.
- **Questioned** — a concrete concern that an author response could resolve; or support and flaw
  evidence conflict on the same part of the claim.
- **Unverified** — evidence is insufficient to decide.

Flawed vs Questioned test: *could any reasonable author explanation change the conclusion?*
No → Flawed. Yes → Questioned.

### 6.2 Sufficiency (per branch, one line each; full rubric in appendix)

- Literature: retrieved work directly describes the same method/result, not merely the same topic.
- Theory: a specific step, missing assumption, or notation contradiction can be pointed to.
- Code: a specific config entry or source line can be pointed to.
- Experiments: execution aligned and consistent/inconsistent; or, paper-only, a specific missing
  control/ablation/statistic or a text–table contradiction can be pointed to.
- All branches: evidence without a verifiable `pointer` is never sufficient; speculation or
  common-sense judgement is at most an insufficient cue (can yield `Questioned`, never `Flawed`).

### 6.3 Aggregation rules (applied in order)

1. Support and flaw evidence both sufficient on the same part → `Questioned` (show both sides).
2. Sufficient flaw evidence not overturnable by explanation → `Flawed`.
3. A concern that bears on whether the claim holds → `Questioned`.
4. Support evidence covers all `conditions`, no such concern → `Supported`.
5. Otherwise → `Unverified`.

Partial coverage never yields `Supported`: uncovered parts with contrary cues → `Questioned`, else `Unverified`.
Concerns that do not affect whether the claim holds (e.g. large gap but no variance) are attached as
notes and do not change the status.

## 7. Report layout

1. Overview — counts per status; the most important Flawed/Questioned items.
2. Claim list — ordered Flawed, Questioned, Unverified, Supported. Each: location, status, evidence
   with source type (paper-internal / literature / theory / code / execution), questions for authors.
3. Other findings — writing, figures, references, missing related work/baselines.
4. Execution ledger — per-run details for reviewers who want them.

No accept/reject recommendation anywhere (report, teaser).

## 8. Open items (must be confirmed against implementation/experiments)

- Repair budget: paper says ≤ 3 rounds; code default `max_attempts = 5` (`execution/stage_runner.py`).
- Approval mode and training budget used in experiments.
- Cutoff source: code derives cutoff from the paper's arXiv id (`util/cutoff_date.py`); v2 uses the
  venue submission deadline.
- Appendix rubrics: per-branch sufficiency; per-metric default tolerances (`execution/nodes/plan.py::_default_tolerance`).
- Outside the Method section: cite ReviewEval and ReviewRL in Related Work; revisit title; main
  figure label `scope` → `conditions`; benchmark labels migrate from the v1 statuses.
