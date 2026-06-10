# qayd — MiniZinc solver bundle

Everything needed to use qayd from the MiniZinc CLI and IDE:

```
qayd-fzn(.exe)  the FlatZinc solver binary (speaks the MiniZinc protocol)
qayd.msc        solver configuration (relative paths — works in place)
mznlib/         solver library: globals qayd propagates natively
install.sh      registers the bundle (Linux/macOS bundles)
install.ps1     registers the bundle (Windows bundles)
```

## Quick start (no install)

`qayd.msc` uses paths relative to itself, so it works straight from the
unpacked directory:

```bash
minizinc --solver /path/to/this/dir/qayd.msc model.mzn data.dzn
```

## Install (CLI + IDE discovery)

Linux / macOS:

```bash
./install.sh
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```

Either writes `qayd.msc` into your user MiniZinc solver directory
(`~/.minizinc/solvers/` or `%USERPROFILE%\.minizinc\solvers\`) with absolute
paths into this directory — after that `minizinc --solver qayd ...` works and
the IDE lists **qayd** in its solver dropdown (restart the IDE once). Keep
the bundle where it is, or re-run the installer after moving it.

## Notes

- The IDE/driver time limit is passed as the standard `-t <ms>` flag.
- `mznlib/` keeps `all_different`, `table`, `regular`, `cumulative`,
  `disjunctive`, `circuit`, `global_cardinality*`, `count_eq`, `lex_*`,
  `value_precede`, `increasing`, `all_equal`, `member`, `bin_packing_load`
  as native FlatZinc predicates so qayd's dedicated propagators see them
  instead of decompositions.
