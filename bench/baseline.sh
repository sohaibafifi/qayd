#!/usr/bin/env bash
# Self-baseline: run the release `qayd` over the XCSP CSP + COP tracks of one
# competition year and freeze the per-instance CSVs as a reference for
# before/after solver comparisons (bench/common/compare.py).
# Usage: ./baseline.sh [TIMEOUT_S] [YEAR_DIR] [TAG] [JOBS] [MEM_MB]
#   TIMEOUT_S  per-instance wall clock (default 120)
#   YEAR_DIR   extracted competition dir (default data/XCSP24); accepts both
#              CSP/COP and CSP<YY>/COP<YY> subdir layouts
#   TAG        suffix for output files (default <year>-<timeout>s)
#   JOBS       instances run concurrently (default: half the logical cores).
#              Co-running instances add timing noise: compare only runs made
#              with the SAME value (it is recorded in the provenance file).
#   MEM_MB     harness-level address-space cap PER INSTANCE (default: 80% of
#              RAM divided by JOBS). Solver-agnostic (works on binaries without
#              --mem-limit): the OS kills an over-consuming instance instead of
#              the machine dying under JOBS concurrent runs. Enforced on Linux.
set -euo pipefail
cd "$(dirname "$0")/.."

T="${1:-120}"
YEAR_DIR="${2:-data/XCSP24}"
TAG="${3:-$(basename "$YEAR_DIR")-${T}s}"
CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu)
JOBS="${4:-$(( CORES / 2 > 0 ? CORES / 2 : 1 ))}"
RAM_MB=$(( $(free -m 2>/dev/null | awk '/^Mem:/{print $2}' || sysctl -n hw.memsize | awk '{print int($1/1048576)}') ))
MEM_MB="${5:-$(( RAM_MB * 4 / 5 / JOBS ))}"
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
    echo "jobs: $JOBS  (same value required on both sides of a comparison)"
    echo "mem_mb: $MEM_MB per instance (harness rlimit; same value on both sides)"
    echo "sets: $csp_dir  $cop_dir"
} > "$OUT/PROVENANCE-$TAG.txt"

python3 bench/common/run.py --dir "$csp_dir" --timeout "$T" --jobs "$JOBS" --mem-mb "$MEM_MB" \
    --cmd "$QAYD -t {t} {f}" --out "$OUT/csp-$TAG.csv"
python3 bench/common/run.py --dir "$cop_dir" --timeout "$T" --jobs "$JOBS" --mem-mb "$MEM_MB" \
    --cmd "$QAYD -t {t} {f}" --out "$OUT/cop-$TAG.csv"
echo "baseline complete: $OUT/csp-$TAG.csv $OUT/cop-$TAG.csv"
