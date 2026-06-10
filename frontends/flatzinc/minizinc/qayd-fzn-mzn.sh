#!/usr/bin/env bash
# MiniZinc driver entry point for qayd: force the MiniZinc output protocol,
# pass every driver flag (e.g. `-t <ms>`) straight through.
# Runs the binary from the workspace target dir, so `cargo build --release`
# is always picked up — no staged copy to go stale.
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
exec "$REPO/target/release/qayd-fzn" --mzn "$@"
