<p align="center">
  <img src="qayd-logo.svg" width="120" alt="qayd logo">
</p>

<h1 align="center">qayd</h1>

<p align="center">Yet another constraint-programming solver, in Rust.</p>

## What It Is

`qayd` solves constraint satisfaction and optimization problems over finite
integer domains. The core solver has trailed domains, propagators, search, Lazy
Clause Generation, and optional parallel search.

The project also has an experimental collection engine for list-style models:
routes, bins, ordered assignments, and scheduling prototypes. That engine is
used by the Python examples and is being shaped toward list, lambda, and
partition modeling.

Priority order: correct, simple, fast.

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

The Python examples are the easiest way to inspect the collection modeling API:

```bash
uv run examples/python/vrp.py
uv run examples/python/cvrptw.py
uv run examples/python/bin_packing.py
```

Typical list model shape:

```python
import qayd as cp

model = cp.Model()
routes = model.list_vars(k, customers)

for route in routes:
    model.add(cp.sum(route, lambda i: demand[i]) <= capacity)

model.minimize(
    cp.sum(cp.sum_edges(r, lambda i, j: dist[i][j], start=0, end=0) for r in routes)
)
solution = model.solve(time_limit=30)
```

## Docs

- `docs/`: small static docs site.
- `LISTS.md`: current plan for list, lambda, and partition support.
- `TURBO.md`: notes for the incumbent-focused COP path.
- `frontends/flatzinc/minizinc/README.md`: MiniZinc setup notes.

## License

MIT
