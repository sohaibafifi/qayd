# MiniZinc integration

Registers `qayd` as a solver for the MiniZinc CLI and IDE.

## Install

```bash
bash frontends/flatzinc/minizinc/install.sh
```

This builds `qayd-fzn` and writes `~/.minizinc/solvers/qayd.msc`. The config
points at `qayd-fzn-mzn.sh`, which runs the binary straight from
`target/release/` — every `cargo build --release` is picked up automatically,
no reinstall needed. MiniZinc (CLI and IDE) finds the config on its own.

## Use

CLI:

```bash
minizinc --solver qayd model.mzn data.dzn
```

IDE: restart the IDE after installing, then pick **qayd** in the solver
configuration dropdown and press Run. The IDE time limit works (passed as the
standard `-t <ms>` flag).

## How it works

- The config uses no solver MiniZinc library (`mznlib`), so MiniZinc decomposes
  every global into FlatZinc builtins — exactly the dialect exercised by
  `data/fzn/challenge-std`. A qayd-specific `mznlib` keeping `all_different`,
  `table`, `regular`, `cumulative`, ... as native globals is the natural next
  step for propagation strength.
- `qayd-fzn-mzn.sh` invokes `qayd-fzn --mzn`, which speaks the MiniZinc solver
  protocol: `name = value;` solution items honouring `output_var` /
  `output_array` annotations, `----------` after each solution, `==========`
  after an optimality/exhaustiveness proof, `=====UNSATISFIABLE=====` /
  `=====UNKNOWN=====` otherwise. Under `--mzn`, `-t` is in milliseconds (the
  driver's convention); the native CLI keeps seconds.
- `needsSolns2Out: true` lets the driver format solutions through the model's
  `output` statement.
