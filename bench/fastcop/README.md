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
needed.

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
one.

For 25 instances at 180 seconds with two concurrent runs and an 8 GiB limit per
run:

```bash
./bench/fastcop/pipeline.sh \
  180 \
  25 \
  bench/fastcop/results/sample-25-180s-j2 \
  8192 \
  0 \
  2
```

`--jobs` parallelizes complete solver runs, not the search inside a solver.
Every solver still receives one search worker. The memory limit applies to each
run, so the configured aggregate is `jobs * memory-mb`, in addition to the
harness and operating system. Result records remain serialized by the parent
in deterministic plan order, and the run identity includes the job count. A
campaign can therefore resume safely only with the same selection, limits,
concurrency setting and solver artifacts. An incompatible reuse of an output
file or log directory is rejected before a solver starts; choose a new output
directory when changing those parameters.
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
  --output bench/fastcop/results/fortress-qayd.jsonl
```

`run.py` writes every `o` event and its arrival time, first and best incumbent,
status and proof claim, return code, timeout path, peak RSS when available,
full structured command, solver/checker/instance hashes, seed, host, Git
revision, and paths to complete stdout, stderr, and checker logs. A feasible
objective is excluded from scoring unless its final `v` block passes the
checker. `--no-check` exists only for harness diagnosis and should not be used
for competitive numbers.

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
