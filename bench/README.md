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
