#!/usr/bin/env bash
# SAT track: qayd-sat vs CaDiCaL over DIMACS CNF competition instances.
# Usage: ./pipeline.sh [TIMEOUT_S] [LIMIT] [JOBS] [MEMORY_MB] [OUTPUT_DIR]
set -euo pipefail
cd "$(dirname "$0")"
T="${1:-10}"; LIMIT="${2:-0}"; JOBS="${3:-1}"; MEM="${4:-0}"
OUT="${5:-results}"
C=../common; QSAT=../../target/release/qayd-sat
CADICAL="$(command -v cadical)"
mkdir -p "$OUT"
python3 "$C/run.py" --dir instances --timeout "$T" --limit "$LIMIT" --jobs "$JOBS" --mem-mb "$MEM" \
    --finalization-seconds 1 \
    --verify-kind sat --solver qayd-sat --artifact "$QSAT" \
    --cmd "$QSAT -t {t} {f}" --out "$OUT/qayd.csv" \
    --log-dir "$OUT/logs/qayd" --provenance-out "$OUT/qayd.provenance.json"
python3 "$C/run.py" --dir instances --timeout "$T" --limit "$LIMIT" --jobs "$JOBS" --mem-mb "$MEM" \
    --finalization-seconds 1 \
    --verify-kind sat --solver cadical --artifact "$CADICAL" \
    --cmd "$CADICAL -t {t} {f}" --out "$OUT/cadical.csv" \
    --log-dir "$OUT/logs/cadical" --provenance-out "$OUT/cadical.provenance.json"
echo "------------------- SAT: qayd-sat vs cadical -------------------"
python3 "$C/compare.py" "$OUT/qayd.csv" "$OUT/cadical.csv" \
    --timeout "$T" --finalization-seconds 1 \
    --name-a qayd-sat --name-b cadical --details 10
