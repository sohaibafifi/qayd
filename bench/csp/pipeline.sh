#!/usr/bin/env bash
# CSP track (decision): qayd vs Choco over XCSP3 instances.
# Usage: ./pipeline.sh [TIMEOUT_S] [LIMIT]   (fetch first with fetch.py)
set -euo pipefail
cd "$(dirname "$0")"
T="${1:-10}"; LIMIT="${2:-0}"
C=../common; QAYD=../../target/release/qayd
CHOCO="java -cp ../solvers/choco.jar org.chocosolver.parser.xcsp.ChocoXCSP"
mkdir -p results
python3 "$C/run.py" --dir instances --timeout "$T" --limit "$LIMIT" \
    --cmd "$QAYD -t {t} {f}"           --out results/qayd.csv
python3 "$C/run.py" --dir instances --timeout "$T" --limit "$LIMIT" \
    --cmd "$CHOCO -limit=${T}s {f}"    --out results/choco.csv
echo "------------------- CSP: qayd vs choco -------------------"
python3 "$C/compare.py" results/qayd.csv results/choco.csv \
    --timeout "$T" --name-a qayd --name-b choco
