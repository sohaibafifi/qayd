<p align="center">
  <img src="qayd-logo.svg" width="120" alt="qayd logo">
</p>

<h1 align="center">qayd</h1>

<p align="center">Yet another constraint-programming solver, in Rust.</p>

## What It Is

`qayd` solves constraint satisfaction and optimization problems over finite
integer domains. The core solver has trailed domains, propagators, search, Lazy
Clause Generation, and optional parallel search.

The project also has an experimental list-domain engine for list-style models:
routes, bins, ordered assignments, and scheduling prototypes. That engine is
used by the Python examples and is being shaped toward list, lambda, and
partition modeling.

Priority order: correct, simple, fast.

## Features

- **Finite-domain kernel.** Trailed integer domains use sparse sets for compact
  domains, explicit support storage for sparse domains, and reversible bounds
  with trailed holes for wide contiguous ranges.
- **Support for lists, Intervals and lamdas** 
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
  A lone sequential search periodically *rephases*: every few restarts it
  ignores saved phases and dives with the inverted polarity to escape regions
  the saved phase keeps pulling it back to (this finds first solutions on
  feasibility-hard COP instances that otherwise time out). Sparse domains
  allocate atoms by support; wide domains allocate shared atoms on demand. Short
  learned clauses are strengthened by budgeted CP-aware vivification through the
  global propagators.
- **Parallel search.** CSP find-one/UNSAT runs a portfolio of CDCL workers that
  differ by seed and restart cadence and cross-pollinate short low-LBD learned
  clauses; the first to find a solution or prove unsatisfiability wins. COP
  branch-and-bound keeps wide linear and expression
  objectives symbolic. Opt-in portfolio workers share incumbents; workers with
  materialized objectives also share short low-LBD clauses. `--split` enables
  proof-job stealing inspired by [Buffered Work
  Stealing](https://doi.org/10.1007/978-3-031-95973-8_13), and `--probe`
  dedicates materialized-objective workers to optimistic probes. `--lns`
  dedicates workers to bounded incumbent-driven [Large Neighborhood
  Search](https://doi.org/10.1007/978-3-319-91086-4_4).
- **Fast COP mode.** `--fast-cop` is an incumbent-only mode for the Fast COP style of use:
  it searches for feasible solutions and objective improvements, without trying to prove optimality.
  It uses local scoring for common constraints, constructive starts for guarded table/element patterns,
  and  focused repair for simple Boolean exact-cover rows.
- **A beta support for MiniZinc.** The `qayd-fzn` driver speaks the MiniZinc solver protocol,
  so it can be used as a backend for the MiniZinc CLI and IDE. The `--mzn` flag enables some
  MiniZinc-specific behaviour. (see [for details](frontends/flatzinc/minizinc/README.md)).

## Entry Points

- `qayd`: CLI for XCSP3 instances, including `.xml`, `.xml.lzma`, and `.xz`.
- `qayd-fzn`: beta FlatZinc driver for MiniZinc.
- Rust crate: direct finite-domain modeling and solver internals.
- Python module: modeling experiments, especially list, lambda, and routing
  examples.

## Build And Test

```bash
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
```

Python extension checks:

```bash
cargo clippy --all-targets --features python -- -D warnings
```

## CLI

```bash
cargo run --bin qayd -- [options] <instance.xml[.lzma|.xz]>
```

Common options:

- `-t`, `--time SECONDS`: stop after a time limit.
- `-v`, `--verbose`: print progress and search statistics.
- `--seed SEED`: set a reproducible search seed.
- `-p`, `--threads N`: set worker count.
- `--fast-cop`: search for good COP incumbents without proving optimality.
- `--turbo`: use the faster incumbent-focused COP path when applicable.
- `--split`, `--probe N`, `--lns N`: optional parallel COP strategies.

## Python Examples

The Python examples are the easiest way to inspect the list-domain modeling API:

```bash
uv run examples/python/vrp.py
uv run examples/python/cvrptw.py
uv run examples/python/bin_packing.py
```

Typical list model shape:

```python
import qayd as cp

model = cp.Model()
routes = model.list_vars(customers, count=k)

for route in routes:
    model.add(cp.sum(route, lambda i: demand[i]) <= capacity)

model.minimize(
    cp.sum(cp.sum_edges(r, lambda i, j: dist[i][j], start=0, end=0) for r in routes)
)
solution = model.solve(time_limit=30)
```

## License

MIT
