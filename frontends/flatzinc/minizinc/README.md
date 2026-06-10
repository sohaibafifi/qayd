# MiniZinc integration

Registers `qayd` as a solver for the MiniZinc CLI and IDE.

Two distribution paths:

- **From a release**: download the `qayd-minizinc-<tag>-<target>` bundle —
  binary, relocatable `qayd.msc`, `mznlib/`, and a standalone `install.sh`
  that needs no Rust toolchain. The bundle templates live in [`dist/`](dist).
- **From the repo** (development): the install below points the solver config
  at `target/release/`, so rebuilds are picked up automatically.

## Install

```bash
bash frontends/flatzinc/minizinc/install.sh
```

This builds `qayd-fzn` and writes `~/.minizinc/solvers/qayd.msc` pointing
straight at `target/release/qayd-fzn`. Every `cargo build --release` is
picked up automatically, no reinstall needed. MiniZinc (CLI and IDE) finds
the config on its own.

## Use

CLI:

```bash
minizinc --solver qayd model.mzn data.dzn
```

IDE: restart the IDE after installing, then pick **qayd** in the solver
configuration dropdown and press Run. The IDE time limit works (passed as the
standard `-t <ms>` flag).
