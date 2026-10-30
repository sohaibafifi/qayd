# Private Amthal integration

This standalone package builds Qayd with the optional private Amthal LP
backend. It shares Qayd's source files and is excluded from the public Cargo
workspace. Public builds and `--all-features` do not select this package or
require Amthal.

## Local setup

Use an authorized local Amthal checkout that provides the `LpSession` API.
The integration manifest expects it next to the Qayd checkout:

```text
parent/
  qayd/
    integrations/amthal/Cargo.toml
  amthal/
    Cargo.toml
```

Amthal remains private and is not downloaded by Qayd's public CI or release
workflows. No private source is bundled in this integration.

## Build and validate

Run these commands from the Qayd repository root:

```bash
cargo build --manifest-path integrations/amthal/Cargo.toml --release
cargo test --manifest-path integrations/amthal/Cargo.toml --test lp_relaxation
cargo clippy --manifest-path integrations/amthal/Cargo.toml --all-targets -- -D warnings
```

The `lp-relaxation` feature is enabled by default for this package. It builds
the `qayd`, `qayd-fzn`, `qayd-sat`, and `qayd-pb` binaries, with outputs under
`integrations/amthal/target/release/` unless Cargo's target directory is
overridden. Its separate lockfile and target directory keep private dependency
resolution and build artifacts out of the public workspace.

## Python development install

To install the shared Python package with the private LP backend for local
development, run from the Qayd repository root:

```bash
uv run maturin develop --manifest-path integrations/amthal/Cargo.toml --profile pyext --features python
```

This editable installation uses the Python sources in `frontends/python` and
builds the `qayd._core` extension with LP enabled.
