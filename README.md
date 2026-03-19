<p align="center">
  <img src="qayd-logo.svg" width="120" alt="qayd logo">
</p>

<h1 align="center">qayd</h1>

<p align="center">Yet another constraint-programming solver, in Rust.</p>

---

`qayd` solves CSP and COP instances over **integer variables with finite domains**,
using a propagation engine and backtracking search. The target constraint catalogue
is [XCSP3-core](https://xcsp.org).

Priorities, in order: **correct → simple → fast.**

## Features

- Sparse-set trailed domains, event-driven propagation to fixpoint.
- DFS with `dom/wdeg` branching, geometric restarts, branch-and-bound for optimization.
- Constraints: `intension`, `extension`, `regular`, `mdd`, `allDifferent`, `allEqual`,
  `ordered`, `lex`, `precedence`, `sum`, `count`, `nValues`, `cardinality`, `minimum`,
  `maximum`, `element`, `channel`, `slide`, `noOverlap`, `cumulative`, `binPacking`,
  `knapsack`, `circuit`, `instantiation`.
- XCSP3-core reader (`.xml`, `.lzma`, `.xz`), with time limit and Ctrl+C handling.

## Build

```bash
cargo build --release
cargo test            # 86 tests: unit, integration, brute-force oracle
```

## Solve an instance

```bash
qayd [-v] [-t SECONDS] <instance.xml[.lzma|.xz]>
```

- `-v` — emit `c` comment lines (model size, bounds, search stats, wall time).
- `-t SECONDS` — stop after a time budget, reporting the best solution so far.

## As a library

```rust
use qayd::constraints::primitives::not_equal_offset;
use qayd::{count_solutions, Solver, VarId};

// 4-Queens: count the solutions.
let n = 4;
let mut solver = Solver::new();
let q: Vec<VarId> = (0..n).map(|_| solver.new_var_range(0, n - 1)).collect();
for i in 0..n as usize {
    for j in (i + 1)..n as usize {
        let (di, dj) = (i as i32, j as i32);
        not_equal_offset(&mut solver, q[i], q[j], 0);
        not_equal_offset(&mut solver, q[i], q[j], di - dj);
        not_equal_offset(&mut solver, q[i], q[j], dj - di);
    }
}
assert_eq!(count_solutions(&mut solver, &q), 2);
```

## License

MIT
