#!/usr/bin/env bash
# PB track: qayd-pb vs Sat4j over linear OPB competition instances.
# Usage: ./pipeline.sh [TIMEOUT_S] [LIMIT] [JOBS] [MEMORY_MB] [OUTPUT_DIR]
set -euo pipefail
cd "$(dirname "$0")"
T="${1:-10}"; LIMIT="${2:-0}"; JOBS="${3:-1}"; MEM="${4:-0}"
OUT="${5:-results}"
C=../common; QPB=../../target/release/qayd-pb
S4J="java -cp ../solvers/s4j-pb.jar:../solvers/s4j-core.jar org.sat4j.pb.LanceurPseudo2007"
mkdir -p "$OUT"
python3 "$C/run.py" --dir instances --timeout "$T" --limit "$LIMIT" --jobs "$JOBS" --mem-mb "$MEM" \
    --finalization-seconds 1 \
    --verify-kind pb --solver qayd-pb --artifact "$QPB" \
    --cmd "$QPB -t {t} {f}" --out "$OUT/qayd.csv" \
    --log-dir "$OUT/logs/qayd" --provenance-out "$OUT/qayd.provenance.json"
python3 "$C/run.py" --dir instances --timeout "$T" --limit "$LIMIT" --jobs "$JOBS" --mem-mb "$MEM" \
    --finalization-seconds 1 \
    --verify-kind pb --solver sat4j --artifact ../solvers/s4j-pb.jar --artifact ../solvers/s4j-core.jar \
    --cmd "$S4J Default {t} {f}" --out "$OUT/sat4j.csv" \
    --log-dir "$OUT/logs/sat4j" --provenance-out "$OUT/sat4j.provenance.json"
echo "------------------- PB: qayd-pb vs sat4j -------------------"
python3 "$C/compare.py" "$OUT/qayd.csv" "$OUT/sat4j.csv" \
    --timeout "$T" --finalization-seconds 1 \
    --name-a qayd-pb --name-b sat4j --details 10
