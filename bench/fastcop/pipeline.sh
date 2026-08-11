#!/usr/bin/env bash
# Reproducible XCSP25 FAST COP campaign: Qayd, ACE, ACE-rr, and Choco.
# Usage: pipeline.sh [CPU_SECONDS] [LIMIT] [OUTPUT_DIR] [MEMORY_MB] [PER_FAMILY] [JOBS] [CHECKER_MEMORY_MB]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
CPU_SECONDS="${1:-180}"
LIMIT="${2:-0}"
OUTPUT_DIR="${3:-$HERE/results/$(date -u +%Y%m%dT%H%M%SZ)}"
MEMORY_MB="${4:-65536}"
PER_FAMILY="${5:-0}"
JOBS="${6:-1}"
if [[ "$MEMORY_MB" -lt 4096 ]]; then
    DEFAULT_CHECKER_MEMORY_MB="$MEMORY_MB"
else
    DEFAULT_CHECKER_MEMORY_MB=4096
fi
CHECKER_MEMORY_MB="${7:-$DEFAULT_CHECKER_MEMORY_MB}"
DATA="$ROOT/data/XCSP25/COP25"

if ! [[ "$CPU_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "CPU_SECONDS must be a positive integer" >&2
    exit 2
fi
if ! [[ "$LIMIT" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "LIMIT must be a non-negative integer" >&2
    exit 2
fi
if ! [[ "$MEMORY_MB" =~ ^[1-9][0-9]*$ ]]; then
    echo "MEMORY_MB must be a positive integer" >&2
    exit 2
fi
if ! [[ "$PER_FAMILY" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "PER_FAMILY must be a non-negative integer" >&2
    exit 2
fi
if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "JOBS must be a positive integer" >&2
    exit 2
fi
if ! [[ "$CHECKER_MEMORY_MB" =~ ^[1-9][0-9]*$ ]]; then
    echo "CHECKER_MEMORY_MB must be a positive integer" >&2
    exit 2
fi
if [[ "$LIMIT" -gt 0 && "$PER_FAMILY" -gt 0 ]]; then
    echo "LIMIT and PER_FAMILY are mutually exclusive" >&2
    exit 2
fi

WALL_SECONDS="$(( (CPU_SECONDS * 3 + 1) / 2 ))"
SELECTION_ARGS=()
if [[ "$PER_FAMILY" -gt 0 ]]; then
    SELECTION_ARGS=(--per-family "$PER_FAMILY")
else
    SELECTION_ARGS=(--limit "$LIMIT")
fi

if [[ ! -d "$DATA" ]]; then
    python3 "$ROOT/bench/cop/fetch.py" --year 25 --limit 0 --out "$HERE/instances"
    DATA="$HERE/instances/COP25"
fi

mkdir -p "$OUTPUT_DIR"
python3 "$HERE/manifest.py" \
    --instances "$DATA" \
    --output "$OUTPUT_DIR/manifest.v1.json" \
    --expect-count 250
python3 "$HERE/fetch_solvers.py"
cargo build --manifest-path "$ROOT/Cargo.toml" --release

python3 "$HERE/run.py" \
    --manifest "$OUTPUT_DIR/manifest.v1.json" \
    --solver qayd \
    --solver ace \
    --solver ace-rr \
    --solver choco \
    --cpu-limit "$CPU_SECONDS" \
    --wall-limit "$WALL_SECONDS" \
    --memory-mb "$MEMORY_MB" \
    --checker-memory-mb "$CHECKER_MEMORY_MB" \
    --jobs "$JOBS" \
    --seed 0 \
    "${SELECTION_ARGS[@]}" \
    --output "$OUTPUT_DIR/results.jsonl" \
    --log-dir "$OUTPUT_DIR/logs"

python3 "$HERE/score.py" "$OUTPUT_DIR/results.jsonl" \
    --manifest "$OUTPUT_DIR/manifest.v1.json" \
    --mode both \
    --invalidation family \
    --output "$OUTPUT_DIR/scores.json"

echo "FAST COP results: $OUTPUT_DIR"
