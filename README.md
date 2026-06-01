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

## Features

- **Finite-domain kernel.** Trailed integer domains use sparse sets for compact
  domains, explicit support storage for sparse domains, and reversible bounds
  with trailed holes for wide contiguous ranges.
- **XCSP3-core front-end.** Reads CSP and mono-objective COP instances from XML,
  `.lzma`, and `.xz` files and emits XCSP3 competition output. Supported
  families include `intension`, `extension`, `regular`, `mdd`, `allDifferent`,
  `allEqual`, `ordered`, `lex`, `precedence`, `sum`, `count`, `nValues`,
  `cardinality`, `minimum`, `maximum`, `element`, `channel`, `noOverlap`,
  `cumulative`, `binPacking`, `knapsack`, `instantiation`, `circuit`, and
  `slide`. See the [XCSP3 format paper](https://arxiv.org/abs/1611.03398).
- **Global filtering.** Positive extension tables use
  [Compact-Table](https://arxiv.org/abs/1604.06641)-style bitsets and residues.
  `regular` follows [Pesant's layered automaton
  propagator](https://doi.org/10.1007/978-3-540-30201-8_36).
  `allDifferent` uses [Régin's matching
  filter](https://aaai.org/Library/AAAI/1994/aaai94-055.php). Fixed-resource
  `cumulative` includes [edge
  finding](https://vilim.eu/petr/cp2009.pdf).
- **Learning search.** [Lazy Clause
  Generation](https://people.eng.unimelb.edu.au/pstuckey/papers/lazy.pdf)
  combines finite-domain propagation with 1-UIP learning, backjumping, sparse
  watched clauses, [LBD](https://www.ijcai.org/Proceedings/09/Papers/074.pdf)
  reduction, [dom/wdeg](https://dl.acm.org/doi/10.5555/3000001.3000033),
  [VSIDS](https://doi.org/10.1145/378239.379017), phase saving, and restarts.
  Sparse domains allocate atoms by support; wide domains allocate shared atoms
  on demand.
- **COP and parallel search.** Branch-and-bound keeps wide linear and expression
  objectives symbolic. Opt-in portfolio workers share incumbents; workers with
  materialized objectives also share short low-LBD clauses. `--split` enables
  proof-job stealing inspired by [Buffered Work
  Stealing](https://doi.org/10.1007/978-3-031-95973-8_13), and `--probe`
  dedicates materialized-objective workers to optimistic probes. `--lns`
  dedicates workers to bounded incumbent-driven [Large Neighborhood
  Search](https://doi.org/10.1007/978-3-319-91086-4_4).

## Build

```bash
cargo build --release
cargo test
```

## Solve an instance

```bash
qayd [-h] [-v] [-t SECONDS] [--seed SEED] [-p THREADS] [--split] [--probe N] [--lns N] <instance.xml[.lzma|.xz]>
```

- `-h`, `--help` prints the usage.
- `-v`, `--verbose` emits `c` comment lines (model size, search stats, wall time, incumbent source).
- `-t SECONDS`, `--time SECONDS` stops after a time budget, reporting the best solution so far.
- `--seed SEED` sets the reproducible search seed. It defaults to `RANDOMSEED`, then `0`.
- `-p THREADS`, `--threads THREADS` sets the COP portfolio worker count. It defaults to `NBCORE`, then `1`.
- `--split` divides COP proof search into disjoint jobs for worker stealing.
- `--probe N` dedicates up to `N` COP workers to optimistic objective probes.
- `--lns N` dedicates up to `N` COP workers to bounded Large Neighborhood Search.

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
