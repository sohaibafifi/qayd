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

## Collection backend policy

The Python collection frontend uses a size-aware `engine="auto"` policy. Exact
enumeration is limited to 10 items for ordered lists, and exact assignment or
packing to 24 items and 192 item-list cells. Exact scheduling is limited to 48
intervals and 96 modes. Integer routing lowering retains its 32-node cap.
Larger models go directly to the matching local-search backend, without first
building the exact model mirror. Explicit `engine="exact"` has a larger
capability envelope, with classification work capped at 1,000,000 estimated
units and exact construction at 100,000 item-list cells.

`time_limit` is one wall-clock budget shared by validation, classification,
construction, exact fallbacks, portfolio workers, and search. This matters for
campaign results: reported solve time is not a search-only measurement.

## Certified bounds and gaps

The API and native VRP, VRPTW, JSSP, and RCPSP launchers include `dual_bound`,
`absolute_gap`, `relative_gap`, and `bound_method` in every feasible JSON
record. The dual is a certified lower bound for the minimization objective.
Routing selects the strongest valid result among assignment, Held-Karp 1-tree,
and stabilized route-column relaxations; compact CVRPs up to 16 customers use
exact subset pricing. Fleet minimization also uses exact bin-packing bounds.
VRPTW adds route incompatibilities, conflict-aware packing, travel-aware
interval energy, and an exact elementary route-cover dual up to 16 customers,
following the certified preprocessing and pricing ideas in `vrptw_lb`.
Scheduling combines critical paths with no-overlap and cumulative-energy
bounds. Unsupported shapes report `null`, never a heuristic number disguised
as a certificate. API and native launcher pairs accept the same explicit CLI
arguments and do not read `QAYD_*` configuration variables.

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

## Competitive routing and scheduling campaign

The runner uses one normalized JSONL contract for qayd API, qayd native,
Hexaly, HGS-CVRP, LKH-3 and OR-Tools CP-SAT. Every feasible adapter result is
replayed before it is accepted. Missing certified bounds remain `null`.

CVRPLIB `-kN` names provide a minimum feasible fleet, not a maximum fleet.
The competitive CVRP models therefore allow up to one route per customer and
minimize distance, matching the official BKS convention. Passing `--vehicles N`
to a qayd launcher deliberately changes the convention to a fleet limit and is
recorded as such. HGS is likewise run without `-veh` in competitive campaigns.

```sh
# Optional public datasets and open baseline binaries
python3 bench/fetch_collections.py --collection all
python3 bench/solvers/fetch_competitive.py --solver hgs
python3 bench/solvers/fetch_competitive.py --solver lkh --accept-lkh-academic-license

# Fast end-to-end validation with the repository's five tiny instances
uv run bench/campaign.py --suite smoke \
  --solver qayd-api --solver qayd-native --solver ortools-cp-sat \
  --budget 1 --seed 0,1,2,3,4 --threads 1 \
  --out bench/results/smoke.jsonl

# Aggregate and enforce the five-seed publication gate
uv run bench/report.py bench/results/smoke.jsonl \
  --markdown bench/results/smoke.md --json bench/results/smoke-summary.json \
  --require-claim-ready
```

For the full campaign, use `--suite competitive`, budgets such as
`--budget 1,10,60,600`, and pass external installations explicitly with
`--hexaly-home`, `--hgs-binary`, and `--lkh-binary`. The runner is resumable by
default. Use `--restart` only when intentionally replacing an output file.

Qayd LS profiling is explicit and enabled by the campaign runner. Use
`--no-profile-qayd` for quality-only measurements or `--max-iterations N` for a
deterministic work budget. The same controls are available as
`Model.solve(profile=True, max_iterations=N)` and as `--profile` plus
`--max-iterations` on both API/native routing launchers. These controls apply
to the ordered-list LS backend. Scheduling publishes no artificial candidate
counter.

Exact ablations are explicit too. The campaign and paired routing launchers
accept `--[no-]routing-two-way`, `--[no-]routing-nearest-neighbor`, and
`--[no-]routing-warm-start`; flexible job-shop launchers accept `--schedule-cdcl`. The
Python equivalents are keyword arguments to `Model.solve`. The XCSP binary
uses `--force-scope-reasons` and `--shared-pool-cap N` for its LCG ablations.
There is no process-wide solver configuration through environment variables.

## Clean-room behavior probes

`behavior_probe.py` studies solver behavior through generated CVRPLIB inputs and
documented solver API outputs only. Vendor-distributed runtime files are loaded
only to call that API, and the campaign may record an opaque executable hash for
provenance. It does not disassemble, decompile, inspect executable code, read
non-public files, or call hidden APIs. Four paired transformations are generated
at each requested size:

| Factor | Controlled change | Measurement |
|---|---|---|
| `distance-scale` | multiply every edge cost by a constant | numerical scale invariance after objective normalization |
| `index-permutation` | apply an isomorphic customer permutation | sensitivity to indices and tie-breaking |
| `single-edge` | change one symmetric customer edge | locality of the search response |
| `capacity-threshold` | tighten capacity without changing demands | response near a feasibility threshold |

The runner reuses `campaign.py`, so every solution is replayed and the same
wall time, process-tree RSS, certified dual, throughput, version, seed, and
thread fields are retained. Multiple budgets are independent checkpoints. They
provide an anytime quality curve without claiming access to an internal
incumbent event stream.

```sh
# Fast local qayd fingerprint, about 40 one-second or three-second solves
uv run python bench/behavior_probe.py \
  --workspace bench/results/behavior-qayd \
  --solver qayd-api --customers 40 --budget 1,3 \
  --seed 0,1,2,3,4 --threads 1

# Paired public-interface comparison when a licensed Hexaly install is present
uv run python bench/behavior_probe.py \
  --workspace bench/results/behavior-qayd-hexaly \
  --solver qayd-api --solver hexaly --hexaly-home /opt/hexaly_14_5 \
  --customers 40,80 --budget 1,10,60 \
  --seed 0,1,2,3,4 --threads 1
```

The workspace contains the exact generated instances, `manifest.json`, the
campaign JSONL and provenance sidecar, plus `report.json` and `report.md`.
Runs resume by default. Use `--restart` to replace the campaign, `--generate-only`
to inspect inputs before solving, or `--analyze-only` to rebuild reports from
existing results. All configuration is passed as command-line arguments.

## Linear backend probe

`linear_backend_probe.py` compares Amthal and HiGHS on the same MIPLIB easy
instances. Runs are sequential and single-threaded. The order alternates by
instance to reduce systematic thermal bias. Every record contains solver and
instance hashes, external wall time, process-tree RSS, primal, dual, gap, node
count, and validation against the official `miplib.solu` reference bundled by
the Amthal fetcher.

```sh
cargo build --manifest-path ../amthal/Cargo.toml --release
uv run python bench/linear_backend_probe.py \
  --amthal-binary ../amthal/target/release/amthal \
  --highs-binary /opt/homebrew/bin/highs \
  --time-limit 10 --restart \
  --out bench/results/linear-backends/results.jsonl
```

The report includes solved counts, reference disagreements, PAR2, paired speed
wins, memory, and the median Amthal/HiGHS time ratio. `--limit N` provides a
smoke subset, and runs resume by stable run id when `--restart` is omitted.

The `positioning` suite is a fixed, stratified first-study sample containing
five CVRPLIB X instances up to 1000 customers, six Solomon VRPTW instances,
six JSPLIB job shops, and six PSPLIB j30 projects. It is intended for five-seed
local studies before committing to the full `competitive` suite.

After the four family campaigns, build a per-instance study with actual
lexicographic medians, paired wins, BKS gaps, failure counts, and memory:

```sh
uv run python bench/positioning_report.py \
  bench/results/positioning-cvrp.jsonl \
  bench/results/positioning-vrptw.jsonl \
  bench/results/positioning-jssp.jsonl \
  bench/results/positioning-rcpsp.jsonl \
  --markdown bench/results/positioning-study.md \
  --json bench/results/positioning-study.json
```

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
