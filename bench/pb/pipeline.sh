#!/usr/bin/env bash
# PB track: qayd-pb vs Sat4j over linear OPB competition instances.
# Usage: ./pipeline.sh [TIMEOUT_S] [LIMIT]   (fetch first with fetch.py)
set -euo pipefail
cd "$(dirname "$0")"
T="${1:-10}"; LIMIT="${2:-0}"
C=../common; QPB=../../target/release/qayd-pb
S4J="java -cp ../solvers/s4j-pb.jar:../solvers/s4j-core.jar org.sat4j.pb.LanceurPseudo2007"
mkdir -p results
python3 "$C/run.py" --dir instances --timeout "$T" --limit "$LIMIT" \
    --cmd "$QPB -t {t} {f}" --out results/qayd.csv
python3 "$C/run.py" --dir instances --timeout "$T" --limit "$LIMIT" \
    --cmd "$S4J {f}"        --out results/sat4j.csv
echo "------------------- PB: qayd-pb vs sat4j -------------------"
python3 "$C/compare.py" results/qayd.csv results/sat4j.csv \
    --timeout "$T" --name-a qayd-pb --name-b sat4j
