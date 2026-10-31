#!/usr/bin/env bash
#OAR -t idempotent
#OAR -t besteffort
# One OAR allocation, a bounded GNU Parallel pool, no nested submissions.
set -euo pipefail
PLAN=$1
TASKS=$2
WORKERS=$3
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO"
command -v parallel >/dev/null || { echo 'GNU Parallel is required' >&2; exit 2; }
export QAYD_CAMPAIGN_WORKERS="$WORKERS"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Use the exact Python interpreter frozen in the plan, including its environment.
PYTHON=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime"]["python"])' "$PLAN")
"$PYTHON" -c 'import json,os,sys; p=json.load(open(sys.argv[1])); n=int(sys.argv[2])*p["threads"]; available=len(os.sched_getaffinity(0)); assert n <= available, (n, available)' "$PLAN" "$WORKERS"
parallel --quote --jobs "$WORKERS" --halt soon,fail=1 --line-buffer \
  "$PYTHON" "$REPO/bench/cluster/lib/campaign.py" run --plan "$PLAN" --task '{}' :::: "$TASKS"
