# Qayd search campaign, 22–23 September 2026

Status: complete. The local release candidate passes the declared primary correctness/quality gates, separate tuning checks and the ordinary-CLI sample. Retain the three changes within the scope described below; no push or publication was performed.

Local evidence directory: [`../results/search-campaign-20260922/`](../results/search-campaign-20260922/report.md). The raw runs, scripts, frozen binaries and input manifests remain in that retained, Git-ignored directory.

## Scope and provenance

The original `dev` checkout remains at `9884454f67f1ba10d7f00117f4d3591f4ef4abda`. All implementation is isolated on `feat/search-campaign-20260922` in the sibling `qayd-search-20260922` worktree. Nothing was pushed or published. The first candidate source was `22c981443aa9b1b010479590e2b42dc1af943b93`; its completed seed-0 measurements are retained as preliminary evidence. Corrected source `1da6051070d8d7198c3728c505a05f9805b8374c` passed shared qualification and is frozen separately as candidate v2. Exact binaries, patches and hashes are retained beside this report.

The fixed validation sample contains the same 15 instances as the earlier comparison. Four separate instances were selected for diagnosis; two Examination instances form the quality tuning set. Compressed input hashes, objective directions and paths are in `sample.json`, `tuning.json` and `examination-tuning.json`. No instance selection changed after seeing confirmation results.

Apple M4 Pro, 12 logical CPUs, 24 GiB RAM, macOS 27.2 arm64. Each measured run uses one solver thread, 6 GiB memory, a 180-second internal wall budget, an external 180-second CPU cap and 270-second wall cap. Parsing is outside the internal solver timer. Three runs execute concurrently. The independent checker has a 1 GiB heap. The primary baseline and final runs use the same full-sample order. The smaller ablation subsets keep the same per-run limits and maximum concurrency, but change the mix of concurrent instances; their RSS differences are not by themselves causal evidence of a source-level memory change. There are no concurrent builds. Seeds are 0, 1 and 2. This shared desktop is a local comparison environment, not a publication timing platform.

ACE 2.6 is used only for its pinned XCSP SolutionChecker. Neither the ACE nor Choco solver is executed. Official historical competition numbers are not treated as machine-matched baselines.

## Diagnosis

- **Automatic objective search:** the additive-penalty path found a verified CP scout and immediately returned to exact search. The strong available LS configuration was not given an improvement phase. The paired 30-second tuning diagnostics reproduce the gap: automatic 1246/1562, explicit LS 559/655 on the two Examination instances. More improvements published by exact search did not imply better final quality.
- **Historical Examination3 regression:** 1733 to 1777 is reproduced at the current baseline (seed 0, 180 seconds), but the historical comparison combines several earlier changes. Its first solution improved from 2562 to 2080 while the old trajectory reached 1741 almost immediately. The old reports do not isolate the effects of the CP scout from literal-element lowering. No single-change causal claim is made for that historical 2.54% regression.
- **GolombRuler and Telecom:** missing default incumbents and rejected LS compilation are separate issues. Duplicate expression definitions made functional normalization fail, which was incorrectly reported as memory exhaustion. Removing the optional functional optimization while retaining each equality as a real constraint lets LS execute, but the 30-second held-out runs still have no incumbent. Golomb's wide all-different difference model and Telecom B's large disjunction/intension model remain difficult. Telecom A reaches its optimum quickly, so the whole family does not share one failure mode.
- **Seed-sensitive value ordering:** NonDominatingQueens starts at 666 and ends at 831 on baseline seed 0, but seed 1 advances through every integer from 0 to 45. Frozen source permits an objective-direction reversal for one seed class even with a single worker. The wiring is established; its exact causal contribution versus other seeded decisions needs a controlled ablation. See `nondominating-seed-diagnosis.md`.
- **Crossword:** the original dense support representation really requires about 74.1 GB of support payload on the four shared table bodies. The rejection was not just a spurious estimate. Hybrid supports reduce the estimated total substantially, but the conservative estimate remains above the fixed 6 GiB cap. A separately labelled 12 GiB probe compiles the 200 tables in 4.93 seconds, searches, and reaches a tracked peak near 4.1 GiB. It finds no incumbent. That single observation does not justify weakening the 6 GiB preflight.

## Separable changes and hypotheses

1. `28a9356`: retain and publish the verified scout immediately, then seed LS from it for at most one third of the remaining budget, capped at 60 seconds. Recompute functionals and check feasibility; replay any improvement through the canonical model. Leave at least two thirds for exact search and preserve the scout on interruption. Hypothesis: improve additive-penalty quality without losing early feasibility.
2. `8a30bc6`: store rare table supports sparsely, promote dense ones when smaller, and omit negative-table-only scratch for positive tables. Update the conservative estimator with the representation. Hypothesis: lower table memory without changing semantics or materially regressing table search.
3. `ca1d7ee`: demote ambiguous expression producers into enforced constraints, retaining the restrictions on unsupported cycles and variable-element ambiguity. Correct the unsupported-normalization diagnostic. Hypothesis: restore a valid LS capability without inventing feasibility gains.

`cfa3422` adds explicit incumbent snapshots for independent replay. `67d1aec` and `22c9814` complete the CLI test inventory and Clippy cleanup. No strategy depends on an instance name or path. Native scheduling heuristics were not changed.

Qualification found one R1 runtime defect: a verified scout already at its root objective bound could still consume the optional refinement slice before CP proved optimality. `1da6051` uses the existing sound root-domain bound to skip that fruitless phase, leaving proof authority with CP. A new regression fails on the old source and passes after correction for zero and positive bounds on seeds 0, 1 and 2. A separate above-bound control still requires refinement. This is a qualification correction to R1, not another tuned strategy. The original binary and results are preserved; the final matrix uses a new immutable v2 artifact with unchanged inputs, budgets and acceptance rules. See `qualification-amendment.json`.

## Validation and measurement rules

The main paired matrix uses `--all-solutions`. Every published assignment, including intermediate incumbents, is replayed with the pinned independent checker. The option requests full snapshots and can select a monolithic plan for disconnected homogeneous integer components. Both sides of every instrumented pair use that policy. A separate unchanged-CLI sample checks the usual default command; its final assignments are independently checked, while its legacy intermediate objective-only events are not independently replayable. These two measurement arms are not merged.

The macOS runner records wall duration and peak RSS; its per-run `cpu_seconds` field is unavailable. The external CPU cap is enforced, but this report makes no measured CPU-efficiency claim. Under a wall-time deadline, even a fixed seed can stop at a different point when throughput varies.

RSS is sampled over the process tree every 250 ms. The solver also reports its allocator peak, in whole MiB despite the CLI label `MB`; these are different measurements. R1 showed a seed-1 Eternity RSS increase (240.6 to 350.6 MiB), although Eternity does not execute the seeded LS path, its complete incumbent sequence is identical, and its allocation peak stays at 267 MiB. The cause of that RSS difference is unresolved. R2 lowers the same allocation peak to 164 MiB. R3 normalization is inactive on this instance. `resource-costs.csv` preserves both metrics separately.

TTFF uses the first raw `o` event cross-checked against the harness receive timestamp. Objective distributions retain failed and no-incumbent runs. An independent assignment check establishes validity, not an independently checked optimality proof; optimality counts require the solver's proof claim and a validated incumbent.

Acceptance was declared before confirmation: no invalid solution, no lost feasibility, improved Examination tuning and held-out medians, and no repeatable material regression above 5% median elsewhere. Three seeds are the minimum, with more evidence required if variability makes the decision uncertain. Full details are in `protocol.json`, `hypotheses.md` and `confirmation-plan-v2.json`.

## Correctness qualification

The corrected candidate passes 1405 Rust workspace tests (14 existing ignored), 793 Python tests, 18 LP tests, 11 golden cases, 24 golden-harness unit tests and 28 parallel-harness unit tests. Clippy passes with warnings denied for the workspace and all targets in standard, Python and LP configurations. Formatting and diff checks pass. New checks cover seeded assignment mapping, stale derived values, cancellation/budget boundaries, positive and negative tables, wildcards, repeated scopes, backtracking, explanations and ambiguous functional definitions.

All failed qualification attempts are retained: a missing CLI-help inventory entry, a Clippy boolean simplification and missing `maturin` in the isolated Python environment. These were corrected without changing solver acceptance rules or golden expectations. See `qualification-summary-v2.json` and the raw logs.

The first snapshot replay wrapper also omitted the checker protocol's required `s` status line. Those version-1 invocation failures are preserved. Version 2 adds the required envelope, still checks the exact assignment and objective, and is the only audit version used for acceptance. This was a benchmark wrapper defect, not evidence of an invalid solver assignment.

## Results, decision and remaining work

The primary matrix has **45/45 resource-matched, audited pairs**. Each arm produces 12 validated solutions and two optimality claims per seed, or 36 feasible observations and six optimalities over 45 runs. There are seven objective improvements, 29 ties and nine pairs without an incumbent on either side. No primary objective regression, lost feasibility, invalid incumbent or metadata-integrity failure occurs. All **1471 assignment snapshot checks** pass, including repeated/final snapshots; they cover every intermediate objective event and all 72 feasible-run TTFF observations.

| Validation instance | Baseline objectives, seeds 0/1/2 | Final objectives, seeds 0/1/2 | Median change |
|---|---|---|---|
| Examination3 D1-2-17, min | 1777 / 1725 / 1739 | 755 / 743 / 766 | 1739 to 755, -56.58% |
| Eternity 18x18, min | 4521141 / 4206550 / 4522435 | 4420386 / 4206549 / 4521467 | 4521141 to 4420386, -2.23% |
| Examination1 D2-1-18, min | 1322 / 1350 / 1336 | 1322 / 1350 / 1336 | unchanged |
| Feedback P0100-0200, min | 240 / 241 / 240 | 240 / 241 / 240 | unchanged |
| NonDominatingQueens 45, max | 831 / 45 / 698 | 831 / 47 / 698 | unchanged |

The other feasible instances retain all three objectives. Crossword has three memory rejections per arm; GolombRuler and Telecom B each have three UNKNOWN outcomes per arm. The complete fifteen-instance distributions and individual runtime/RSS/TTFF observations are in `primary-final180.md`, its four CSV files and `primary-final180.json`. Two feasible instance medians improve; the cross-instance median improvement is zero. This is not a uniform gain across every family.

Median TTFF falls from 16.978 to 13.330 seconds on Eternity and from 3.528 to 2.583 seconds on NonDominatingQueens. Examination3 preserves early feasibility, 0.212 to 0.214 seconds. Corresponding median RSS values are 240.59 to 149.17 MiB on Eternity (-38.00%), 552.98 to 469.45 MiB on NonDominatingQueens (-15.11%), and 171.56 to 108.64 MiB on Examination3 (-36.68%). No final median RSS increase exceeds 5%; the largest is 2.50% on MaximumAgreementForest. Most runs still use the full time allowance, so these changes do not establish a broad runtime reduction.

BACPj is a timing caveat: median TTFF rises from 0.342 to 0.441 seconds and process wall time from 0.396 to 0.496 seconds. The difference is concentrated in seed 0; seeds 1 and 2 essentially tie. A separately declared timing diagnostic repeated seed 0 three times per arm, with one job and alternating solver order after the main campaign. Paired TTFF differences were -14.8, +2.0 and +4.6 ms; all six runs retained the same validated optimum. The original +0.267 s seed-0 difference was not reproduced in this serialized check. The single-job repetitions do not replace or explain away the original jobs=3 observation. Commands, all results and audits are in `bacp-timing-protocol.json`, `bacp-timing-executions.json` and `bacp-timing-summary.json`.

Interactions: final v2 exactly retains the three R1 Examination3 objectives and all R2 table-subset objectives. R1's isolated Examination1 seed-0 change, 1322 to 1325, is absent from the combined candidate. R1/2's higher RSS on Clique is also absent from the final full-sample comparison. The source-level cause of every RSS difference is not isolated. `ablation-findings.md` reports these costs, including the intermediate regressions, without suppressing them.

The gain on Examination3 is large and consistent across these three seeds. The Eternity objective gain is modest relative to its variation between seeds, although all three paired outcomes improve. Claims remain limited to this small local sample and the separately selected tuning cases. The fixed-sample primary conditions pass. The separate tuning set also confirms the benefit. The ordinary-CLI seed-0 comparison passes final-assignment validation with 12 solutions, two optimalities, four improved objectives, eight ties and three pairs without a solution. Its improvements are Eternity 4521464 to 4420386, Examination1 1325 to 1319, Examination3 1777 to 755, and NonDominatingQueens 766 to 831. This single-seed arm is kept separate from the instrumented three-seed evidence; its intermediate objective-only events cannot be independently replayed.

On the separate tuning instances, three-seed objectives are:

| Tuning instance | Baseline auto | Final auto | Baseline explicit LS | Auto median improvement |
|---|---|---|---|---|
| Examination1 D1-1-16 | 1215 / 1169 / 1208 | 534 / 553 / 539 | 483 / 457 / 489 | 1208 to 539, -55.38% |
| Examination3 D1-2-16 | 1556 / 1544 / 1507 | 631 / 614 / 619 | 632 / 600 / 619 | 1544 to 619, -59.91% |

The full-budget explicit LS median remains better on the first tuning case (483 versus 539), and ties on the second (619). The automatic policy recovers much of the available quality while retaining early verified feasibility and an exact-search tail. It is not the strongest configuration on every case. R1's broad Examination ambition is only partly fulfilled: the admitted held-out Examination3 improves, whereas the directional-penalty Examination1 D2 model remains outside its admission and unchanged.

The evidence supports retaining the seeded refinement within its existing structural admission, the table representation change, and the normalization/diagnostic correction. Keep full-budget LS optional. Keep `--all-solutions` explicit, because its full-assignment requirement can alter plan selection. Do not lower the Crossword memory guard or change the global default to LS. The normalization correction is retained for sound constraint handling and accurate diagnostics, not for a measured 180-second feasibility gain on the hard families.

The next experiment with the clearest measured opportunity is a general, conservative admission rule for finite directional penalties `max(0, affine-expression)`. Test overflow/domain bounds and a tiny independent oracle, then use distinct D2 tuning and validation instances under the same three-seed protocol. A separate subsequent experiment should isolate the single-worker seed-class objective-direction reversal identified on NonDominatingQueens. No fourth improvement strategy was added tonight.

Phase 13 scheduling qualification remains open. This campaign does not establish its 30/30 JSSP and RCPSP gates, same-budget CP-SAT/Hexaly comparisons, or larger PSPLIB/MMLIB coverage. Passing scheduling correctness tests is not a substitute for those performance gates.

## Reproduction

The exact per-stage argument vectors, binaries and input hashes are retained in this directory. `campaign.py` delegates execution and validation to the repository FAST COP runner. `confirm_campaign_v2.py` follows the frozen matrix and stops on a failed independent snapshot audit. `check_snapshots.py` retains every checker result. `compare_campaign.py --include-ablations --include-tuning --include-preliminary --require-complete` regenerates JSON, CSV and Markdown comparisons without running a solver.

Use `final-v2-artifacts.json` for the candidate binary hash, source revision, compiler and Amthal revision, `final-v2-solvers.json` for all commands, and `confirmation-executions-v2.json` for stage completion evidence. Raw `.jsonl`, solver logs, snapshot files and checker logs remain the authoritative measurements.


The packaged Crossword sizing script was executed after all timed runs and exactly reproduced the stored table counts and arithmetic (`crossword-sizing-replay.json`). The final resource CSV contains 220 retained solver-run records. Version-2 snapshot audits across the campaign contain 3050 accepted checks, including diagnostic and preliminary runs and repeated/final assignments; the primary comparison accounts for 1471 of these. Legacy intermediate objective-only events remain explicitly unaudited.

`REPRODUCTION.md` gives the local build, replay and analysis commands. `release-candidate.patch` contains the source/test diff from the baseline. `planning-update.patch` records the changes to the locally ignored PLAN/LOG files without importing their unrelated historical backlog into Git. `evidence-index.json` fingerprints the retained artifacts. The original `dev` checkout is preserved; the release candidate remains on its local feature branch.

Weekly allowance checks moved from 71% used initially to 79% at final analysis: eight observed percentage points consumed, below the twenty-point limit, with 21% remaining. No reset credit was used. Further solver experiments stop here: the strongest gains are confirmed, the unresolved hard cases have no promising short-run feasibility signal, and the next hypotheses need a separate controlled campaign.
