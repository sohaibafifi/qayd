#!/usr/bin/env bash
# Register qayd as a MiniZinc solver (CLI + IDE).
#
# Builds the release binary and writes a solver configuration into
# ~/.minizinc/solvers/ pointing at the wrapper, which runs the binary
# straight from target/release (rebuilds are picked up automatically).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
WRAPPER="$REPO/frontends/flatzinc/minizinc/qayd-fzn-mzn.sh"
SOLVERS_DIR="$HOME/.minizinc/solvers"

cargo build --release -p qayd-flatzinc --manifest-path "$REPO/Cargo.toml"
chmod +x "$WRAPPER"

mkdir -p "$SOLVERS_DIR"
cat > "$SOLVERS_DIR/qayd.msc" <<EOF
{
  "id": "org.qayd.qayd",
  "name": "qayd",
  "description": "qayd constraint-programming solver",
  "version": "0.1.0",
  "executable": "$WRAPPER",
  "tags": ["cp", "int"],
  "stdFlags": ["-t"],
  "supportsMzn": false,
  "supportsFzn": true,
  "needsSolns2Out": true,
  "inputType": "FZN"
}
EOF

echo "installed: $SOLVERS_DIR/qayd.msc"
echo "try: minizinc --solver qayd $REPO/data/fzn/all_diff.mzn"
