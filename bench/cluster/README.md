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
export CPU_ONLY_PROPERTY="gpudevices='-1'"
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

The controller advances through smoke, qualification and full runs. It uses at
most two OAR jobs by default, with GNU Parallel inside each job, and requires
CPU-only resources plus `idempotent` and `besteffort`. Repeating the command
resumes compatible work. Source, binaries, inputs and configuration are frozen.
The controller verifies an LP-enabled build receipt for the main comparison.

You can also run several single-node OAR jobs, with GNU Parallel inside each job.

```
python bench/cluster run --out "$CAMPAIGN" --hexaly \
  --jobs 4 --resource "/nodes=1" --workers auto \
  --property "$CPU_ONLY_PROPERTY" --execute --detach
  ```
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

Use `--help` after any command for options. Existing campaigns on Ritchie keep
their original checkout and frozen controller; this reorganization is for new
builds. Detailed internal notes live in `../../.agents/benchmarking`.
