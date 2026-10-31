# Tests

Run commands from the repository root.

```text
integration/  public Rust integration tests, grouped by subsystem
internal/     crate-private Rust tests, grouped by subsystem
internal.rs   independent harness for the crate-private suites
common/       shared Rust oracle and model helpers
fixtures/     small checked-in test inputs
python/       Python suites grouped by API, domain and tooling
```

Rust integration targets are registered in `Cargo.toml`. Their names and
separate executables are preserved, including tests with custom allocators.
The private Amthal workspace selects the same shared test sources.

```bash
cargo test --workspace
cargo test --test xcsp
cargo test --test internal schedule_state::
cargo test --manifest-path integrations/amthal/Cargo.toml --test lp_relaxation
```

The LP command requires the authorized sibling Amthal checkout. Python API
tests require an environment with the Qayd extension installed.

```bash
uv run --no-sync python -m pytest tests/python
uv run --no-sync python -m pytest tests/python/cluster
uv run --no-sync python -m pytest tests/python/scheduling
```

Keep new tests in the matching subsystem directory. Register a new Rust
integration target in `Cargo.toml`; register crate-private suites in
`internal.rs`. Keep fixtures under `fixtures/` and generated results outside
the test sources.
