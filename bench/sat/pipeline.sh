#!/usr/bin/env bash
# SAT track: qayd-sat vs CaDiCaL over DIMACS CNF competition instances.
# Usage: ./pipeline.sh [TIMEOUT_S] [LIMIT]   (fetch first with fetch.py)
set -euo pipefail
cd "$(dirname "$0")"
T="${1:-10}"; LIMIT="${2:-0}"
C=../common; QSAT=../../target/release/qayd-sat
mkdir -p results
python3 "$C/run.py" --dir instances --timeout "$T" --limit "$LIMIT" \
    --cmd "$QSAT -t {t} {f}" --out results/qayd.csv
python3 "$C/run.py" --dir instances --timeout "$T" --limit "$LIMIT" \
    --cmd "cadical {f}"      --out results/cadical.csv
echo "------------------- SAT: qayd-sat vs cadical -------------------"
python3 "$C/compare.py" results/qayd.csv results/cadical.csv \
    --timeout "$T" --name-a qayd-sat --name-b cadical
