<p align="center">
  <img src="qayd-logo.svg" width="120" alt="qayd logo">
</p>

<h1 align="center">qayd</h1>

<p align="center">Yet another constraint-programming solver, in Rust.</p>

---

`qayd` solves CSP and COP instances over **integer variables with finite domains**.
It pairs a finite-domain propagation engine with a CDCL search that learns from
conflicts. The target constraint catalogue is [XCSP3-core](https://xcsp.org).

Priorities, in order: **correct, then simple, then fast.**

About 6,600 lines of code (LCG / CDCL engine: 1,617; constraint catalogue: 1,927; XCSP3 bridge: 1,635; core kernel: 1,386).

## Build

```bash
cargo build --release
cargo test
```

## Solve an instance

```bash
qayd [-v] [-t SECONDS] <instance.xml[.lzma|.xz]>
```

- `-v` emits `c` comment lines (model size, search stats, wall time).
- `-t SECONDS` stops after a time budget, reporting the best solution so far.

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
