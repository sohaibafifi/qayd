#!/usr/bin/env bash
# Register this qayd MiniZinc bundle with the local MiniZinc installation.
#
# Standalone: works from wherever the bundle was unpacked, no Rust toolchain
# needed. Writes ~/.minizinc/solvers/qayd.msc with absolute paths into this
# directory, so the bundle must stay where it is after installing (re-run
# this script if you move it).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
SOLVERS_DIR="$HOME/.minizinc/solvers"

[ -x "$HERE/qayd-fzn" ] || { echo "error: qayd-fzn not found next to install.sh" >&2; exit 1; }
[ -d "$HERE/mznlib" ] || { echo "error: mznlib/ not found next to install.sh" >&2; exit 1; }

# The bundled qayd.msc uses relative paths; rewrite them as absolute so the
# installed copy works from any working directory.
mkdir -p "$SOLVERS_DIR"
sed -e "s|\"\./qayd-fzn\"|\"$HERE/qayd-fzn\"|" \
    -e "s|\"\./mznlib\"|\"$HERE/mznlib\"|" \
    "$HERE/qayd.msc" > "$SOLVERS_DIR/qayd.msc"

echo "installed: $SOLVERS_DIR/qayd.msc"
echo "try: minizinc --solver qayd <model.mzn>"
