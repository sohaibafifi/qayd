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
- **Lists, intervals, and lambdas.**
- **Scheduling primitives.** Native Python intervals support optional
  alternatives, precedence, unary and cumulative resources, asymmetric setup
  sequences, piecewise-constant capacity calendars, state functions, and
  makespan minimization. Alternative masters compose directly with the other
  scheduling primitives.
- **XCSP3-core front-end.** Reads CSP and mono-objective COP instances from XML,
  `.lzma`, and `.xz` files and emits XCSP3 competition output. Supported
  families include `intension`, `extension`, `regular`, `mdd`, `allDifferent`,
  `allEqual`, `ordered`, `lex`, `precedence`, `sum`, `count`, `nValues`,
  `cardinality`, `minimum`, `maximum`, `element`, `channel`, `noOverlap`,
  `cumulative`, `binPacking`, `knapsack`, `instantiation`, `circuit`, and
  `slide`. See the [XCSP3 format paper](https://arxiv.org/abs/1611.03398).
  `--core` computes an exact minimal unsatisfiable subset of XCSP source
  constraints after an UNSAT result.
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
- **Parallel search.**
- **Optional certified LP relaxation.**
- **Local-search COP engine.**
- **Adaptive ordered-list search.**
- **Beta MiniZinc support.** See
  [the MiniZinc integration notes](frontends/flatzinc/minizinc/README.md).
- **Beta SAT and Pseudo-Boolean frontends.**

## Entry Points

- `qayd`: CLI for XCSP3 instances, including `.xml`, `.xml.lzma`, and `.xz`.
- `qayd-fzn`: beta FlatZinc driver for MiniZinc.
- `qayd-sat`: beta DIMACS CNF frontend.
- `qayd-pb`: beta OPB frontend.
- Rust crate: canonical semantic modeling through `Model`, `ModelPackage`, and
  `SolveRequest`, plus lower-level solver internals.
- Python module: modeling experiments, especially list, lambda, and routing
  examples.

Every frontend builds the same semantic package. The orchestrator validates it,
compiles an executable backend plan, owns budgets and cancellation, then replays
the final assignment before publication. A minimal Rust solve is:

```rust
use qayd::model::{Constraint, Model, ModelPackage, Relation};
use qayd::orchestrator::{IgnoreEvents, SolveRequest};

let mut model = Model::new();
let value = model.int_range(0, 9);
model.add_constraint(Constraint::Linear {
    terms: vec![(1, value)],
    relation: Relation::Ge,
    rhs: 4,
});

let result = qayd::solve(
    &ModelPackage::new(model),
    &SolveRequest::default(),
    &mut IgnoreEvents,
)?;
# Ok::<(), qayd::orchestrator::SolveError>(())
```

## Build And Test

```bash
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
```

Optional LP backend checks:

```bash
cargo test --features lp-relaxation --test lp_relaxation
cargo clippy --all-targets --features lp-relaxation -- -D warnings
```

Python extension checks:

```bash
cargo clippy --all-targets --features python -- -D warnings
```

To enable the optional certified LP bounds in the Python extension, build both
features explicitly:

```bash
uv run maturin develop --profile pyext --features python,lp-relaxation
uv run examples/python/routing/api/vrp.py \
  data/vrplib/CVRP/X-n101-k25.vrp \
  --time-limit 120 --threads 8 --linear-backend amthal
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
- `--ls`: search for good COP incumbents without proving optimality.
- `--split`, `--probe N`, `--lns N`: optional parallel COP strategies.
- `--core`: after an UNSAT result, print an exact source-constraint MUS.
- `--linear-backend auto|native|amthal`: select the optional LP backend.
- `--lp-root-ms N`: cap the root relaxation wall-clock time.
- `--lp-node-ms N`: cap each persistent in-search LP reoptimization; zero, the default, disables node LP while retaining the root bound.
- `--lp-node-depth N`: set the minimum depth interval between in-search LP solves.
- `--lp-max-vars`, `--lp-max-rows`, `--lp-max-nonzeros`: cap active columns and retained matrix size. Rows touching the objective are retained first.
- `--lp-min-coverage`, `--lp-phase-max-vars`: control eligibility and phase guidance; the phase guard counts physical CP variables, and zero disables LP phase guidance.

The Amthal backend is linked from the private (not yet released) sibling crate `../amthal` only by
`--features lp-relaxation`. Node relaxations retain one private simplex session
per search worker, reuse a still-feasible primal, and turn a bound into pruning
only after exact rational recertification. Without that feature, `auto` preserves the native
path and an explicit `amthal` request returns a clear configuration error.
Verbose output reports both the physical variable count and compact LP column
count, nonlinear auxiliary columns, interval fallbacks, candidate and retained
rows, nonzeros, objective coverage, and an explicit construction status such as
`ready`, `no-rows`, or `invalid-objective`.

## Python Examples

The Python examples live under `examples/python/`, grouped by domain: `routing/`,
`scheduling/`, `packing/`, `optimization/`, `csplib/`, and `mus/`
(infeasibility analysis). Routing and scheduling examples have native and API
versions. The CSPLib collection has a catalog and a common runner:

```bash
uv run python -m examples.python.csplib list
uv run python -m examples.python.csplib prob007 --size 12
```


```python
from qayd.datasets import load_instance, read_cvrplib, read_jsplib, read_psplib, read_solomon

instance = load_instance("data/X-n101-k25.vrp")  # marker-based detection
distance = instance.edge_weights
```

Supported families are CVRPLIB/TSPLIB CVRP, Solomon and Gehring-Homberger
VRPTW, JSPLIB job shop, and PSPLIB RCPSP/MRCPSP.

Typical routing API shape:

```python
import qayd as cp

model = cp.Model()
customers = model.customers(range(1, n + 1))
for customer in customers:
    customer.demand = demand[customer.id]

routes = model.routes(customers, vehicles=k, depot=0, travel=dist)
for route in routes:
    model.add(route.sum(lambda customer: customer.demand) <= capacity)

model.minimize(routes.sum(lambda route: route.distance()))
solution = model.solve(time_limit=30)
print(solution.objective, solution.dual_bound, solution.relative_gap)
```

Typical scheduling API shape:

```python
model = cp.Model()
tasks = model.tasks(range(n))
for task in tasks:
    task.duration = duration[task.id]
    task.demand = demand[task.id]

schedule = model.schedule(tasks, horizon=horizon)
for before, after in precedences:
    model.add(schedule[before].end <= schedule[after].start)
for resource in resources:
    model.add(schedule.resource(lambda task: task.demand[resource]) <= capacity[resource])

model.minimize(schedule.makespan())
solution = model.solve(time_limit=30)
```

## License

MIT
