# Parallel CP regression harness

This directory is the dedicated Phase 10.5 corpus for clause sharing, split
cubes, objective probes, and incumbent-driven CP LNS. It uses only the Python
standard library.

The harness deliberately separates two questions:

1. `check` runs each complete-search request once, replays its mathematical
   result, and verifies that the requested parallel mechanism emitted evidence.
   It never applies a wall-time threshold.
2. `measure` runs the same complete-search requests repeatedly and stores the
   wall-time samples plus their median. `compare` reads two stored reports and
   applies a configurable tolerance, 10 percent by default. It never launches a
   solver.

Every correctness record and timing case retains the instance path and SHA-256,
the canonical request, and the parsed status, objective, solution, and parallel
metrics. Campaign metadata retains the manifest hash, source revision, and
solver binary hash. A comparison refuses reports whose corpus, instance hash,
request, or result signature differs.

## Fast tests

Comparator and parser tests use synthetic samples. They make no timing claims
and do not launch `qayd`, so they are suitable for normal CI:

```sh
python3 -m unittest discover -s tests/python -p 'test_parallel_regression.py'
```

Build the release binary, then run correctness separately:

```sh
cargo build --release --bin qayd
python3 bench/parallel/harness.py check \
  --binary target/release/qayd \
  --label current-correctness \
  --revision "$(git rev-parse HEAD)+working-tree" \
  --out bench/parallel/results/current-correctness.jsonl
```

The correctness output is JSONL. A failed oracle or missing mechanism metric
returns a nonzero exit status after writing all completed records.

## Honest pre-refactor baseline

No timing numbers are checked in. The manifest pins the pre-refactor tree to
`6c09b9fddc2738f584e53d464d7a76b98cab4d6c`, where `src/parallel.rs` still
contains the implementation being protected. Capture that exact tree in a
detached worktree and record both its revision and binary hash:

```sh
parallel_baseline_rev="6c09b9fddc2738f584e53d464d7a76b98cab4d6c"
git show "${parallel_baseline_rev}:src/parallel.rs" >/dev/null
parallel_baseline_dir="$(mktemp -d /tmp/qayd-parallel-baseline.XXXXXX)"
git worktree add --detach "$parallel_baseline_dir" "$parallel_baseline_rev"
cargo build \
  --manifest-path "$parallel_baseline_dir/Cargo.toml" \
  --release --bin qayd \
  --target-dir "$parallel_baseline_dir/target"

python3 bench/parallel/harness.py check \
  --binary "$parallel_baseline_dir/target/release/qayd" \
  --label pre-refactor-correctness \
  --revision "$parallel_baseline_rev" \
  --out bench/parallel/results/pre-refactor-correctness.jsonl
```

This process avoids benchmarking a dirty refactor binary and labelling it as a
baseline. Keep the detached worktree until the candidate campaign finishes so
both binaries can be measured as adjacent pairs on the same machine.

## Paired candidate capture and comparison

Build the candidate in the main worktree, then capture each baseline and
candidate sample next to its peer. The first binary alternates by repetition
and scenario so sustained machine drift does not consistently favor one side:

```sh
cargo build --release --bin qayd
parallel_candidate_rev="$(git rev-parse HEAD)+working-tree"
python3 bench/parallel/harness.py measure-pair \
  --baseline-binary "$parallel_baseline_dir/target/release/qayd" \
  --candidate-binary target/release/qayd \
  --baseline-label pre-refactor \
  --candidate-label phase-10.5-candidate \
  --baseline-revision "$parallel_baseline_rev" \
  --candidate-revision "$parallel_candidate_rev" \
  --repetitions 6 --warmups 1 \
  --baseline-out bench/parallel/results/pre-refactor-timing.json \
  --candidate-out bench/parallel/results/candidate-timing.json

python3 bench/parallel/harness.py compare \
  bench/parallel/results/pre-refactor-timing.json \
  bench/parallel/results/candidate-timing.json \
  --tolerance 0.10 \
  --json bench/parallel/results/comparison.json \
  --markdown bench/parallel/results/comparison.md
```

Run baseline and candidate captures on the same otherwise idle host, with the
same release profile and CPU power policy. Six repeats and one warmup are the
defaults. Paired captures require an even number of at least four repeats so
each binary runs first equally often. More repeats are useful when a case is
close to the 10 percent gate.
The paired command records the alternating execution order in both reports and
the comparator rejects reports from different paired captures. The scenario
order also rotates between repetitions.

Use repeated `--case` flags for a focused diagnosis, for example
`--case probes --case lns`. Baseline and candidate reports must contain the
same selected scenarios before they can be compared.

The generated `results/` directory is ignored because timing artifacts are
machine-specific. Preserve publication artifacts in the project-approved
results location only after recording the host and build configuration.
