#!/usr/bin/env bash
# COP track (optimization): candidate qayd vs a baseline qayd binary.
# Compares proved-optimum counts AND incumbent objective quality (min/max aware).
# Usage: ./pipeline.sh [TIMEOUT_S] [LIMIT] [BASELINE_BINARY] [CANDIDATE_BINARY]
# Fetch instances first with fetch.py. Binary paths are ordinary arguments so
# campaigns never depend on machine-specific environment variables or folders.
set -euo pipefail
cd "$(dirname "$0")"
TIMEOUT_S="${1:-10}"
LIMIT="${2:-0}"
BASELINE_BINARY="${3:-qayd-old}"
CANDIDATE_BINARY="${4:-../../target/release/qayd}"
COMMON=../common

mkdir -p results
python3 "$COMMON/run.py" --dir instances --timeout "$TIMEOUT_S" --limit "$LIMIT" \
    --cmd "$CANDIDATE_BINARY -t {t} {f}" --out results/qayd.csv

python3 "$COMMON/run.py" --dir instances --timeout "$TIMEOUT_S" --limit "$LIMIT" \
    --cmd "$BASELINE_BINARY -t {t} {f}" --out results/qayd-old.csv

echo "------------------- COP: qayd vs qayd-old -------------------"
python3 "$COMMON/compare.py" results/qayd.csv results/qayd-old.csv \
    --timeout "$TIMEOUT_S" --name-a qayd --name-b qayd-old
