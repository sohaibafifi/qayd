# XCSP25 FAST COP harness

This directory contains the reproducible benchmark used to compare Qayd with
ACE 2.5, ACE-rr, and the exact Choco XCSP25 release. It is intentionally
separate from `bench/cop`, whose CSV runner is useful for quick smoke tests but
does not stream or validate competition output.

The full protocol runs one solver at a time by default, with one search worker,
180 seconds of solver CPU time, 270 seconds of wall time, seed 0, and 64 GiB of
memory. Each final feasible solution is checked independently with the XCSP3
`SolutionChecker` shipped in ACE 2.5. A timeout first sends `SIGTERM`, waits one
second for the solver to publish its final incumbent, then sends `SIGKILL` if
needed. The checker heap is controlled separately and defaults to the smaller
of 4 GiB and the solver memory limit.

## Pinned inputs and solvers

`manifest.v1.json` describes all 250 XCSP25 FAST COP instances. Every entry has
an instance id, model family and family group, objective sense, compressed
size, repository-relative path, and SHA-256. Objective detection streams the
whole compressed XML document, so objectives located after very large tables
are not missed.

`solvers.v1.json` contains structured argument vectors, not shell command
strings. It pins:

- ACE 2.5 from the PyCSP3 2.5 wheel, SHA-256
  `cb0d0741d7f626c5166371efa2ce8cbe5dc62c49f10180e3ebf4be008106dcc7`;
- Choco Parser 5.0.0-beta.1, the XCSP25 version, SHA-256
  `6dec5d54b335a18170314539e6c5ecc779778600154db28b59cffacdca898600`;
- the current release build of Qayd. Its changing executable SHA-256 is stored
  in every result record and participates in the resume key.

Fetch or verify the reference binaries with:

```bash
python3 bench/fastcop/fetch_solvers.py
python3 bench/fastcop/fetch_solvers.py --verify-only
```

The fetcher extracts ACE from the pinned wheel instead of rebuilding its source
with a different JDK. Downloads are installed only after both the container and
artifact hashes match.

Regenerate and verify the committed corpus manifest with:

```bash
python3 bench/fastcop/manifest.py
python3 bench/fastcop/manifest.py --check
```

## Running campaigns

The complete campaign is:

```bash
./bench/fastcop/pipeline.sh
```

For a staged study, first use short budgets and a limit:

```bash
./bench/fastcop/pipeline.sh 10 0 bench/fastcop/results/stage-10s 8192 1
./bench/fastcop/pipeline.sh 30 0 bench/fastcop/results/stage-30s 65536
./bench/fastcop/pipeline.sh 180 0 bench/fastcop/results/final-180s 65536
```

If `data/XCSP25/COP25` is present, the pipeline reuses it. Otherwise it calls
the existing XCSP fetcher with `--limit 0`. Runs are sequential when the
default `jobs=1` is retained. A stable run key makes an interrupted JSONL
campaign resumable without repeating completed cases.
The optional fifth pipeline argument selects the first `N` instances of every
model variant. Prefer it to an alphabetic `--limit` for short representative
studies. The sixth argument is the number of concurrent runs and defaults to
one. The seventh argument is the checker Java heap in MiB.

For 25 instances at 180 seconds with two concurrent runs and an 8 GiB limit per
run:

```bash
./bench/fastcop/pipeline.sh \
  180 \
  25 \
  bench/fastcop/results/sample-25-180s-j2 \
  8192 \
  0 \
  2 \
  4096
```

`--jobs` parallelizes complete solver runs, not the search inside a solver.
Every solver still receives one search worker. The memory limit applies to each
run, so the configured aggregate is `jobs * memory-mb`, in addition to the
harness and operating system. Checker processes may also overlap, with an
aggregate Java heap of `jobs * checker-memory-mb`. Result records remain
serialized by the parent in deterministic plan order, and the execution
identity includes the job count. A campaign can therefore resume safely only
with the same selection, solver limits, concurrency setting and solver
artifacts. An incompatible reuse of an output file or log directory is rejected
before a solver starts; choose a new output directory when changing those
parameters.
CPU accounting based on process-global `RUSAGE_CHILDREN` is deliberately
disabled for concurrent runs; wall time, peak RSS, solver limits and all
anytime events remain recorded.

Parallel measurements share CPU caches, memory bandwidth and thermal limits.
Use them for faster diagnostic campaigns. Keep `jobs=1`, or use isolated
machines, for the final comparison intended for publication.

For a focused Qayd run:

```bash
python3 bench/fastcop/run.py \
  --solver qayd \
  --family Fortress1 \
  --per-family 2 \
  --cpu-limit 30 \
  --wall-limit 45 \
  --memory-mb 65536 \
  --checker-memory-mb 4096 \
  --output bench/fastcop/results/fortress-qayd.jsonl
```

## Root LP ablation

The LP contribution is measured with three variants of the same release binary:
`qayd-native` disables the optional root relaxation,
`qayd-amthal-bound-only` computes the certified dual bound without primal phase
guidance, and `qayd-amthal` enables both with a 50 ms root budget. This separates
bound quality and overhead from the search-phase effect. Build the instrumented
binary once, then run a paired campaign with solver-order rotation:

```bash
cargo build --release --features lp-relaxation --bin qayd
python3 bench/fastcop/run.py \
  --solvers bench/fastcop/solvers.lp-ablation.v1.json \
  --solver qayd-native \
  --solver qayd-amthal-bound-only \
  --solver qayd-amthal \
  --interleave-solvers \
  --per-family 1 \
  --cpu-limit 30 \
  --wall-limit 45 \
  --memory-mb 8192 \
  --jobs 1 \
  --output bench/fastcop/results/lp-ablation-30s/results.jsonl
python3 bench/fastcop/lp_ablation.py \
  bench/fastcop/results/lp-ablation-30s/results.jsonl \
  --output bench/fastcop/results/lp-ablation-30s/report.md \
  --json bench/fastcop/results/lp-ablation-30s/summary.json
```

The report compares only checker-accepted incumbents. It also extracts root
bound coverage, certification rate, LP time, proof counts, incumbent timing,
searched nodes and family-level wins from the verbose logs. Use `--jobs 1` for
publishable timing. Parallel jobs remain useful for a quick diagnostic run.

`run.py` writes every `o` event and its arrival time, first and best incumbent,
status and proof claim, return code, timeout path, peak RSS when available,
full structured command, solver/checker/instance hashes, seed, host, Git
revision, and paths to complete stdout, stderr, and checker logs. A feasible
objective is excluded from scoring unless its final `v` block passes the
checker. `--no-check` exists only for harness diagnosis and should not be used
for competitive numbers.

Solver execution and solution validation have separate identities. The
`execution_key` depends only on the solver artifact and command, resolved
solver and limit-launcher hashes, instance, seed, solver limits, concurrency,
and the execution portion of the harness.
Changing the checker binary, checker heap, checker timeout, or validation code
does not make a completed solver execution pending. `run_key` remains as a
compatibility alias on new records. Old records without a versioned
`execution_key` can be rechecked, but are deliberately not resumed as current
executions. Use `--recheck-only` when the validation settings change or when a
prior checker attempt was inconclusive.

## Rechecking stored solutions

Use `--recheck-only` to validate existing stdout logs without launching any
solver. The runner rematerializes each compressed instance, invokes only the
checker, and atomically replaces the JSONL file after all selected checks
finish. It does not append duplicate runs. The usual `--solver`, `--family`,
`--instance`, `--limit`, `--per-family`, and `--jobs` selectors apply.

For example, the large `TankAllocation2-0800` checker needs more than the old
1 GiB derived heap:

```bash
python3 bench/fastcop/run.py \
  --recheck-only \
  --output bench/fastcop/results/holdout-25-180s-j4/results.jsonl \
  --instance '^TankAllocation2-0800$' \
  --checker-memory-mb 4096 \
  --checker-timeout 300 \
  --jobs 4
```

Add `--dry-run` first to inspect the exact records that will be checked. An
accepted assignment restores its stored solver status and incumbent. Checker
OOM, timeout, forced kill, and other checker failures are classified as
`checker-oom`, `checker-timeout`, `checker-killed`, and `checker-error`. They
leave the candidate unverified with status `UNKNOWN`; they do not mark the
solver or its model family invalid. Only an explicit checker rejection or an
objective mismatch is a false answer.

## Scoring

The scorer implements the dynamic FAST COP classes:

- `Opt` and `Unsat`: 1 point for a proof;
- `BB1`: 1 point for the best unproved objective when nobody proves it;
- `BB2`: 0.5 point for the same objective when another solver proves it;
- a weaker objective: 0 points.

It reports both one common pool and every pairwise pool because BB1 and BB2 are
relative to the participating solvers:

```bash
python3 bench/fastcop/score.py bench/fastcop/results/final-180s/results.jsonl \
  --manifest bench/fastcop/results/final-180s/manifest.v1.json \
  --mode both \
  --invalidation family \
  --output bench/fastcop/results/final-180s/scores.json
```

By default, one rejected solution invalidates that solver's complete model
family. Use `--invalidation none` for diagnostic scoring, or
`--family-field family_group` to apply the penalty to grouped model variants.
The checker validates assignments and objective values, but it cannot certify
an UNSAT or optimality claim. Such proof claims remain the responsibility of
the solver and should be cross-checked when results conflict.

With `--manifest`, scoring also requires the complete solver by instance grid,
canonical family labels and instance hashes, one seed and resource profile,
one solver configuration, and one execution harness. Only checker-accepted
incumbents can receive BB points, and a proof is credited only after a clean
exit without timeout or forced termination.

To compare a new Qayd-only run with a frozen prior pool without assembling a
JSONL file by hand, pass both files and opt in to replacing Qayd:

```bash
python3 bench/fastcop/score.py \
  bench/fastcop/results/holdout/results.jsonl \
  bench/fastcop/results/holdout-qayd-new/results.jsonl \
  --replace-solver qayd \
  --manifest bench/fastcop/results/holdout/manifest.v1.json \
  --allow-mixed-execution-harness \
  --mode both
```

The mixed-harness flag is intentionally explicit and marks the report as a
diagnostic comparison. Omit it for a controlled campaign in which every solver
was run by the same harness revision.

These local measurements reproduce the protocol, not the competition hardware.
Hardware, JVM, operating system, and executable hashes must accompany any
published comparison.

The scorer consumes fresh local JSONL records only. It does not silently mix
the XCSP website's historical traces with locally measured runs. A future
official-trace adapter can emit the same result schema, with explicit source
and trace hashes, without changing the runner or scoring rules.

## Anytime analysis

`anytime.py` reconstructs the best objective known at configurable elapsed-time
checkpoints from the same JSONL files. It sorts events by their recorded arrival
time and recomputes the minimum or maximum envelope, so it does not rely on the
solver emitting monotone objective values. An event at exactly a checkpoint is
included. An optimality or UNSAT proof becomes visible only at or after its
`proof_elapsed_seconds`; a proof without that timestamp is never backdated.

For example:

```bash
python3 bench/fastcop/anytime.py \
  bench/fastcop/results/final-180s/results.jsonl \
  --checkpoints 1,5,10,30,60,180 \
  --mode both \
  --invalidation family \
  --output bench/fastcop/results/final-180s/anytime.json
```

Checkpoint values can also be space-separated. The JSON report contains every
run's first incumbent, best incumbent, last strict improvement, time to each,
and final plateau duration. It also contains incumbent and proof counts plus
the exact FAST COP pool and pairwise scores at every checkpoint. These scores
reuse `score.py`, including its BB1, BB2 and family-invalidation rules. Use
`--mode none` when only trajectory metrics are wanted.

The run-level plateau is measured from the last strict improvement to the
recorded wall-time horizon, extended to the latest event or proof timestamp if
needed. A checkpoint plateau is measured from the current best's first arrival
to that checkpoint. JSON is written to stdout unless `--output` is supplied;
the compact text summary goes to stderr in that case, or to stdout alongside a
file output. `--quiet` suppresses only the text summary.
