#!/usr/bin/env bash
# Register qayd as a MiniZinc solver (CLI + IDE).
#
# Builds the release binary and writes a solver configuration into
# ~/.minizinc/solvers/ pointing straight at target/release/qayd-fzn -
# the binary speaks the MiniZinc protocol natively, and rebuilds are
# picked up automatically. The config also points at the bundled mznlib,
# which keeps supported globals (all_different, table, regular, ...) as
# native FlatZinc predicates instead of decompositions.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
BIN="$REPO/target/release/qayd-fzn"
MZNLIB="$REPO/frontends/flatzinc/minizinc/mznlib"
SOLVERS_DIR="$HOME/.minizinc/solvers"
VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' "$REPO/Cargo.toml" | head -1)

cargo build --release -p qayd-flatzinc --manifest-path "$REPO/Cargo.toml"

mkdir -p "$SOLVERS_DIR"
cat > "$SOLVERS_DIR/qayd.msc" <<EOF
{
  "id": "org.qayd.qayd",
  "name": "qayd",
  "description": "qayd constraint-programming solver",
  "version": "$VERSION",
  "executable": "$BIN",
  "mznlib": "$MZNLIB",
  "tags": ["cp", "int"],
  "stdFlags": ["-t"],
  "supportsMzn": false,
  "supportsFzn": true,
  "needsSolns2Out": true,
  "inputType": "FZN"
}
EOF

echo "installed: $SOLVERS_DIR/qayd.msc (version $VERSION)"
echo "try: minizinc --solver qayd $REPO/data/fzn/all_diff.mzn"
