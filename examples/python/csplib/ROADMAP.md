# CSPLib internal implementation record

## Scope

The collection follows the 97 identifiers in the current numerical catalog:
`prob001` through `prob091`, followed by `prob110`, `prob115`, `prob116`,
`prob131`, `prob132`, and `prob133`.

Every identifier is registered in `catalog.py` and has a runnable Python module
under `problems/`. Each module provides a Qayd model, a decoder, an independent
domain validator, and a command-line entry point. Published instance formats are
parsed directly where practical. Other modules accept documented JSON data or
parameters suitable for generating an instance.

## Verification gates

- The catalog contains exactly the 97 expected identifiers.
- Every registered command-line entry point solves and validates its default
  instance.
- `tests/python/test_csplib.py` checks known results and runs all registered
  entry points.
- Python sources pass Ruff formatting, Ruff checks, bytecode compilation, and
  `git diff --check`.
- The public `README.md` contains usage and API documentation only.

Performance studies remain separate from correctness checks. A successful
default instance demonstrates the encoding and validator path, not competitive
performance on every published benchmark instance.
