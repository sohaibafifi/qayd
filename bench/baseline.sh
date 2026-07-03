#!/usr/bin/env bash
# Self-baseline: run the release `qayd` over the XCSP CSP + COP tracks of one
# competition year and freeze the per-instance CSVs as a reference for
# before/after solver comparisons (bench/common/compare.py).
# Usage: ./baseline.sh [TIMEOUT_S] [YEAR_DIR] [TAG]
#   TIMEOUT_S  per-instance wall clock (default 120)
#   YEAR_DIR   extracted competition dir (default data/XCSP24); accepts both
#              CSP/COP and CSP<YY>/COP<YY> subdir layouts
#   TAG        suffix for output files (default <year>-<timeout>s)
set -euo pipefail
cd "$(dirname "$0")/.."

T="${1:-120}"
YEAR_DIR="${2:-data/XCSP24}"
TAG="${3:-$(basename "$YEAR_DIR")-${T}s}"
QAYD="$PWD/target/release/qayd"
OUT="bench/baselines"

[ -x "$QAYD" ] || { echo "build first: cargo build --release --bin qayd" >&2; exit 1; }
csp_dir=$(compgen -G "$YEAR_DIR/CSP*" | head -1) || { echo "no CSP dir under $YEAR_DIR" >&2; exit 1; }
cop_dir=$(compgen -G "$YEAR_DIR/COP*" | head -1) || { echo "no COP dir under $YEAR_DIR" >&2; exit 1; }
mkdir -p "$OUT"

{
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host: $(hostname)"
    echo "commit: $(git rev-parse --short HEAD)$(git diff --quiet || echo ' +dirty')"
    echo "rustflags: ${RUSTFLAGS:-<default>}"
    echo "cmd: qayd -t $T {f}  (single thread, default seed)"
    echo "sets: $csp_dir  $cop_dir"
} > "$OUT/PROVENANCE-$TAG.txt"

python3 bench/common/run.py --dir "$csp_dir" --timeout "$T" \
    --cmd "$QAYD -t {t} {f}" --out "$OUT/csp-$TAG.csv"
python3 bench/common/run.py --dir "$cop_dir" --timeout "$T" \
    --cmd "$QAYD -t {t} {f}" --out "$OUT/cop-$TAG.csv"
echo "baseline complete: $OUT/csp-$TAG.csv $OUT/cop-$TAG.csv"
