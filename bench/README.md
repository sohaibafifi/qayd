# qayd benchmarking - SAT / PB / CSP / COP

Fetch competition instances, run the matching `qayd` frontend against a
literature baseline solver, and compare. One directory per problem class; the
runner and comparator are shared in `common/`.

| Class | dir    | qayd frontend | baseline (literature)                 | instances | source |
|-------|--------|---------------|---------------------------------------|-----------|--------|
| SAT   | `sat/` | `qayd-sat`    | **CaDiCaL**                           | DIMACS CNF | SAT Competition (GBD) |
| PB    | `pb/`  | `qayd-pb`     | **Sat4j** PB (`LanceurPseudo2007`)    | linear OPB | PB Competition (CRIL) |
| CSP   | `csp/` | `qayd`        | **Choco** (`ChocoXCSP`)               | XCSP3, decision    | XCSP Competition (CRIL) |
| COP   | `cop/` | `qayd`        | **Choco** (`ChocoXCSP`)               | XCSP3, optimization | XCSP Competition (CRIL) |

Baselines are all reference/medal solvers and are swappable via the `--cmd`
template in `common/run.py` (e.g. Kissat for SAT, RoundingSat for PB, ACE for
XCSP).


## Quick start

```sh
# build the qayd frontends
cargo build --release -p qayd-sat -p qayd-pb          # SAT + PB
cargo build --release --bin qayd                       # XCSP (CSP + COP)

# get the baseline jars (cadical must be on PATH separately)
bash solvers/fetch_solvers.sh

# fetch a bounded slice per class
python3 sat/fetch.py --track main_2024 --limit 30 --max-mb 2
python3 pb/fetch.py  --archive selected-PB24 --kind opb --max-files 60
python3 csp/fetch.py --year 25 --limit 30
python3 cop/fetch.py --year 25 --limit 30

# run + compare (10s per instance)
./sat/pipeline.sh 10 ; ./pb/pipeline.sh 10 ; ./csp/pipeline.sh 10 ; ./cop/pipeline.sh 10
```

For CPU-tuned local benchmark binaries, prefix the build command with
`RUSTFLAGS="-Ctarget-cpu=native"`. The repository does not force that flag so
release and CI builds stay portable.

`pipeline.sh [TIMEOUT_S] [LIMIT]` - per-instance wall-clock timeout and an
optional instance cap (0 = all fetched).

## Session probe

The Python session probe measures rolling-horizon re-solves under the current
architecture: `SolveSession.solve()` keeps the shared learned-clause pool across
epochs, while each epoch still solves from a cloned base solver. It compares
that path with cold `Model.solve()` calls using the same assumptions, hints, and
branch order.

```sh
uv run --with maturin maturin develop --features python
uv run python bench/session_probe.py --vars 32 --conflicts 50 --epochs 6 --epoch-time-limit 3
```

Use `--format jsonl` or `--format csv` for scripts. The reported
`cold_over_session_time` is a rough ratio for the synthetic instance, not a
claim about a true live push/pop continuation. Epoch solves are capped at one
second by default; pass `--epoch-time-limit 0` for an unbounded exact run.

## Dataset parsers and native launchers

The pure-Python `qayd.datasets` package supplies the normalized input layer for
the routing and scheduling campaign:

| Parser | Benchmark families | Normalization contract |
|--------|--------------------|------------------------|
| `read_cvrplib` | CVRPLIB X and TSPLIB-style CVRP | zero-based nodes, TSPLIB integer distances |
| `read_solomon` | Solomon and Gehring-Homberger VRPTW | zero-based nodes, explicit windows and service times |
| `read_jsplib` | ABZ, FT, LA, ORB, SWV, Taillard and YN job shop | zero-based machine ids |
| `read_psplib` | PSPLIB RCPSP/MRCPSP `.sm` and `.mm` | typed modes, successors and resource kinds |
| `read_vrp_solution` | CVRPLIB/Solomon route solutions | routes plus numeric BKS cost |

`load_instance(path)` detects these formats from structural markers. SAT,
linear OPB and XCSP3 continue to use their existing qayd frontends rather than
duplicating parsers in Python.

For the DIMACS VRPTW convention, use
`solomon.distance_matrix(scale=10, rounding="truncate")`. This stores distances
truncated to one decimal place as integers and makes objective replay stable
across solvers.

The existing native examples double as qayd launchers for CVRPLIB,
Solomon/Homberger, JSPLIB and PSPLIB. With no positional argument they retain
their generated demonstration; with a file argument they parse, solve and
independently verify that instance. `--json` produces a campaign-ready record.
Small format-correct smoke instances live under `examples/instances/`.

## Data sources

- **SAT** - Global Benchmark Database, <https://benchmark-database.de>, which
  serves each SAT-Competition CNF as an individually addressable xz file.
  Tracks `main_2020`…`main_2024` (400 each), plus `parallel_*`, `cloud_*`,
  `incremental_*`. (Zenodo has the official monolithic tarballs, e.g.
  `10.5281/zenodo.15095752`, but they can't be size-sliced.)
- **PB** - CRIL, <https://www.cril.univ-artois.fr/PB24/benchs/> :
  `selected-PB<YY>.tar` (competition-used) / `normalized-PB<YY>.tar` (all
  submissions), years 06/07/09/10/11/12/16/24, plus `normalized-WBO.tar`.
- **CSP/COP** - XCSP, `instancesXCSP<YY>.zip` from
  <https://www.cril.univ-artois.fr/~lecoutre/compets/> ; members are
  pre-split into `CSP<YY>/`, `COP<YY>/`, `MiniCSP<YY>/`, `MiniCOP<YY>/`. The
  fetchers reuse an already-extracted copy under `data/XCSP<YY>/` when present.

### Fetching "everything"

Full multi-year pulls are tens of GB (SAT) / ~7 GB (PB), and most competition
instances are unsolvable under a short timeout - hence the default caps.

```sh
python3 sat/fetch.py --track main_2024 --limit 0 --max-mb 0      # all 400 of a track
for y in 2020 2021 2022 2023 2024; do python3 sat/fetch.py --track main_$y --limit 0 --max-mb 0; done
for y in 06 07 09 10 11 12 16 24; do python3 pb/fetch.py --archive normalized-PB$y; done
python3 csp/fetch.py --year 25 --limit 0 ; python3 cop/fetch.py --year 25 --limit 0
```

## Metrics (`common/compare.py`)

- **solved** - SAT / UNSAT / OPTIMUM count per solver.
- **PAR2** - SAT-competition ranking score: Σ runtimes, each unsolved instance
  charged `2 × timeout`. Lower is better.
- **VBS** - virtual best solver: solved by *either*.
- **speed** - among solved-by-both, who was faster.
- **objective** (PB/COP) - min/max-aware optimum agreement / qayd better / worse.
- **CONTRADICTIONS** - one solver says SAT/OPTIMUM, the other proves UNSAT: a
  **soundness bug** in one of them, printed loudly. Primary correctness check.
