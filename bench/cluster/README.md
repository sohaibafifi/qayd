# Cluster benchmarks

Run these commands from the Qayd repository root. This folder is self-contained:
it does not require the other local `bench/` workflows.

```text
config/       campaign settings and official corpus references
fixtures/     small test instances, kept in Git
lib/          execution, validation and download helpers
scripts/      OAR build and worker scripts
status.py     live monitoring
collect.py    CSV/JSONL result collection
runs/         generated tools, plans, logs and results; ignored by Git
```

Downloaded datasets live under `data/`, including `data/benchmarks/` for native
collections and PB. Retired workflows and their results are kept locally in
`drafts/bench/`.

## 1. Prepare

For LP, place an authorized Amthal checkout at `../amthal`. The `--lp` option
selects `integrations/amthal/Cargo.toml`, where LP is enabled by default. It
does not add an LP feature to the public workspace.

```bash
python3.12 bench/cluster prepare tools --lp
source bench/cluster/runs/env.sh
# Ritchie example: exclude entire GPU hosts.
export CPU_ONLY_PROPERTY="network_address NOT IN (SELECT network_address FROM resources WHERE gpumodel IS NOT NULL)"
python bench/cluster prepare build --lp --cores 4 --walltime 02:00:00 \
  --property "$CPU_ONLY_PROPERTY" --scratch-dir "$QAYD_SCRATCH_DIR" --execute
```

The Ritchie example excludes every host with a GPU resource, using OAR's
`gpumodel` property. Adapt `CPU_ONLY_PROPERTY` for other clusters, and set
`QAYD_SCRATCH_DIR` to your writable scratch directory on a compute node.
Queue, resources, paths and walltime remain launch options. Without `--execute`,
the build command only displays its submission. Binaries use release mode and
the Python extension uses the optimized `pyext` profile. Omit `--lp` for a public
build without Amthal.

To include LKH-3 in the routing comparisons, add
`--accept-lkh-academic-license` to `prepare tools`, then submit the build job
above. Downloads happen on the frontend; compilation happens in OAR. If the
official LKH server is unavailable, preparation uses a commit-pinned archive
of version 3.0.13 with a checked SHA-256. The source URL and hash are recorded in
`runs/tools/solvers/LKH-3.0.13/qayd-source.json`. This pin identifies the mirror
archive; no publisher checksum is available.

After the build job succeeds, download and verify the datasets:

```bash
python bench/cluster prepare corpus
```

SAT datasets are deferred. For shared storage configuration, use
`python bench/cluster prepare storage --help`.

## 2. Run or resume

```bash
export CAMPAIGN=bench/cluster/runs/campaign
python bench/cluster run --out "$CAMPAIGN" --hexaly --jobs 2 --execute --detach
```

Each category has its own profile: `xcsp-csp`, `xcsp-cop`, `xcsp-fastcop`,
`minizinc`, `pb`, `cvrp`, `cvrptw`, `jssp` and `rcpsp`. With `--hexaly`, the
published-table comparisons use `hexaly-cvrp`, `hexaly-jssp` and `hexaly-rcpsp`.
Each profile advances through `smoke`, `gate` (qualification) and `full`.
A failed qualification blocks only that profile. Full runs start automatically
after its qualification succeeds; there is no combined `main/full` in a new
campaign.

The controller shares at most two OAR jobs by default across all profiles, with
GNU Parallel inside each job, and requires CPU-only resources plus `idempotent`
and `besteffort`. Repeating the command resumes compatible work. Source,
binaries, inputs and configuration are frozen. After code or binary changes,
repeat the build step and choose a new `--out` directory for a clean campaign.
The controller verifies an LP-enabled build receipt for the direct comparisons.

You can also run several single-node OAR jobs, with GNU Parallel inside each job.

```bash
python bench/cluster run --out "$CAMPAIGN" --hexaly \
  --jobs 4 --resource "/nodes=1" --workers auto \
  --property "$CPU_ONLY_PROPERTY" --execute --detach
```

To run or resume only PB, select its profile. Repeat `--profile` to select
several categories. The selection is remembered; use `--profile all` to return
to all profiles. A selected profile still needs a clean qualification.

```bash
python bench/cluster run --out "$CAMPAIGN" --profile pb --execute --detach
```

Run one controller per campaign. To change a running controller's selection,
stop its process and restart it with the new options; submitted OAR jobs remain
part of the same bounded pool.
## 3. Monitor

```bash
python bench/cluster status --out "$CAMPAIGN" --details --watch 60
```

`Ctrl+C` closes the monitor. Remaining budgets exclude queueing, preemptions
and checking. Solver details include queued smoke and qualification runs.

## 4. Collect results

```bash
python bench/cluster collect --out "$CAMPAIGN"
```

Exports are written to `$CAMPAIGN/exports/latest/`: `results.csv`,
`results.jsonl`, `summary.json` and `summary.md`. Saved partial arms and errors
are retained. Collection also accepts locally accessible copies of old results;
it does not rerun solvers or rewrite their plans and qualifications.
Use `--stage full`, `--profile NAME`, `--dest DIR` or `--require-ready` as needed.

Use `--help` after any command for options.
