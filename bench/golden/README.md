# Architecture golden harness

This directory is the blocking differential firewall for Phase 10.5 lot A0.
It runs a small, complete-search corpus through Rust native, Python, XCSP,
FlatZinc, SAT, PB, list, routing, and scheduling surfaces. Each case records its
request, instance SHA-256, status, objective senses and vector, canonical
solution, validity, bound, canonical gap, and proof claim.

Run the complete suite from the repository root:

```bash
uv run python bench/golden/run.py
```

The default preparation step builds the workspace binaries and rebuilds the
Python extension from the current source. For an already prepared workspace:

```bash
uv run python bench/golden/run.py --no-prepare
```

Useful focused runs:

```bash
uv run python bench/golden/run.py --list
uv run python bench/golden/run.py --no-prepare --surface sat
uv run python bench/golden/run.py --no-prepare --case xcsp-minimize
```

## Blocking semantics

The v1 manifest is strict and deterministic:

- Status is exact. SAT against UNSAT, UNSAT against SAT, and a missing exact
  conclusion all fail. A time-limited `SATISFIABLE` result in place of
  `OPTIMAL` belongs in a separate performance campaign, not this corpus.
- Objective vectors are compared exactly in declared lexicographic order.
  Every feasible result is independently replayed, and exhaustive tiny-model
  enumeration checks the optimum.
- Canonical solutions are compared exactly after the surface parser converts
  protocol output to assignments, ordered lists, routes, or task start maps.
  The current fixtures deliberately have unique canonical optima.
- Validity never comes from the solver's status. The harness validates the
  published candidate with an independent finite-domain, CNF, route, or
  schedule oracle.
- A minimization bound may not exceed its incumbent. A maximization bound may
  not be below its incumbent. An optimal terminal bound must close every
  represented objective prefix.
- `gap` is derived from the primary primal, bound, and objective sense. Its
  absolute value is `primal - bound` for minimization and `bound - primal` for
  maximization. Its relative value uses
  `absolute / max(abs(primal), abs(bound), 1)`. It is `null` when one of those
  three inputs is unavailable.
- Cases in the same `equivalence_group` are compared after execution across
  surfaces. Status, senses, objectives, canonical solution, validity, bound
  values, and gap must match. Protocol-specific bound sources and proof kinds
  are intentionally excluded from that comparison. The `integer-minimize`
  group exercises the same tiny complete-search request through Rust native,
  Python, XCSP, and FlatZinc.
- `OPTIMAL` and `UNSATISFIABLE` require a proof claim that the independent
  oracle verifies. The SAT proof case additionally hashes the emitted DRAT
  artifact and checks every step as RUP through the empty clause. The RUP
  checker is intentionally small and is not a general RAT checker.

The process execution timeout is only a stuck-test guard. It is not a solver
budget and any timeout fails the blocking suite.

## Golden update policy

There is no update, bless, snapshot, or regeneration option. Test commands only
read the manifest and fixtures. A changed fixture fails its SHA-256 check before
the solver runs. The harness also hashes the manifest, schema, and every fixture
before preparation, then checks them again after preparation and every case.

An intentional semantic change is reviewed manually:

1. Explain why the model or public result is expected to change.
2. Edit the fixture and expected semantic record by hand.
3. Recompute the instance hash with `shasum -a 256 <fixture>`.
4. Review the manifest diff as an API or solver semantic change.
5. Run the harness and its unit tests.

Performance baselines, elapsed time, RSS, and heuristic tolerances do not live
in this manifest. Keeping those campaigns separate prevents machine variance
from weakening the correctness gate.

## Files

- `manifest.v1.json` is the reviewed corpus and expected semantic record.
- `schema.v1.json` documents the versioned manifest shape.
- `harness.py` runs commands, normalizes protocols, replays solutions, and
  performs semantic comparisons.
- `python_probe.py` exposes structured records for Python modeling surfaces.
- `fixtures/` contains all hashed tiny instances.
- `tests/architecture_golden_native.rs` is the native Rust surface probe.

Run the harness unit tests with:

```bash
python3 -m unittest discover -s bench/golden -p 'test_*.py'
```
