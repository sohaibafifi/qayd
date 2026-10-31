"""Run the real bootstrap entry point with simulated compute-node tools."""
import os
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).resolve().parents[3] / "bench/cluster/scripts/bootstrap.oar"
DRIVER = r'''
command() {
  if [[ "$1" == -v ]]; then
    printf '%s\n' "$2" >> "$BOOTSTRAP_TEST_TRACE"
    case " $BOOTSTRAP_TEST_MISSING " in
      *" $2 "*) return 1 ;;
      *) return 0 ;;
    esac
  fi
  builtin command "$@"
}
uname() { printf 'Linux x86_64\n'; }
export -f command uname
exec /bin/bash "$@"
'''


def invoke(tmp_path, *, prepare=False, missing="curl"):
    repo = tmp_path / "qayd"
    repo.mkdir()
    (repo / "Cargo.toml").write_text("[package]\n")
    python = repo / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nprintf 'Reached shared Python\\n'\nexit 71\n")
    python.chmod(0o755)
    trace = tmp_path / "checked-tools"
    env = {**os.environ, "OAR_JOB_ID": "test-only", "BOOTSTRAP_TEST_TRACE": str(trace),
           "BOOTSTRAP_TEST_MISSING": missing}
    args = ["/bin/bash", "-c", DRIVER, "test-driver", str(SCRIPT), "--repo", str(repo)]
    if prepare:
        args.append("--prepare")
    result = subprocess.run(args, text=True, capture_output=True, env=env, check=False)
    return result, trace.read_text().splitlines()


def test_build_node_needs_no_download_tools_and_uses_shared_python(tmp_path):
    result, checked = invoke(tmp_path, missing="curl uv python3.12 sha256sum")
    assert result.returncode == 71, result.stderr
    assert "Reached shared Python" in result.stdout
    assert not {"curl", "uv", "python3.12", "sha256sum"}.intersection(checked)


def test_frontend_prepare_still_requires_curl(tmp_path):
    result, checked = invoke(tmp_path, prepare=True)
    assert result.returncode == 2
    assert "Missing prepare prerequisites: curl" in result.stderr
    assert "curl" in checked


def test_build_reports_all_missing_compilers_in_one_pass(tmp_path):
    result, checked = invoke(tmp_path, missing="cc c++ cmake")
    assert result.returncode == 2
    assert "Missing build prerequisites: cc c++ cmake" in result.stderr
    assert {"cc", "c++", "cmake", "parallel"}.issubset(checked)


def test_build_uses_scratch_for_every_cargo_phase_and_cleans_only_its_temporary_directory(tmp_path):
    repo = tmp_path / "qayd source"
    repo.mkdir()
    (repo/"Cargo.toml").write_text("[package]\n")
    (repo/"integrations/amthal").mkdir(parents=True)
    (repo/"integrations/amthal/Cargo.toml").write_text("[package]\n")
    (tmp_path/"amthal").mkdir()
    (tmp_path/"amthal/Cargo.toml").write_text("[package]\n")
    python = repo/".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\ncat >/dev/null\n")
    python.chmod(0o755)
    (python.parent/"activate").write_text(r'''
git() {
  if [[ "${3:-}" == rev-parse ]]; then
    case "$2" in
      *cadical*) printf 'c60730422e758ef1cebe7aeddf2dda31c996bf04\n' ;;
      *) printf 'b062d5fccb1f64864dd65a94dac35929059ab089\n' ;;
    esac
  fi
  return 0
}
java() { return 0; }
rustc() { return 0; }
minizinc() { return 0; }
python() { printf '%s\n' "$*" >> "$BOOTSTRAP_TEST_PYTHON_TRACE"; }
cargo() {
  [[ "$RUST_TEST_THREADS" == "$CARGO_BUILD_JOBS" ]] || return 91
  printf '%s|%s|%s|%s|%s\n' "$CARGO_TARGET_DIR" "$CARGO_INCREMENTAL" \
    "$CARGO_PROFILE_DEV_DEBUG" "$CARGO_PROFILE_TEST_DEBUG" "$*" >> "$BOOTSTRAP_TEST_BUILD_TRACE"
}
''')
    shared_logs = tmp_path/"shared logs"
    shared_logs.mkdir()
    (repo/"bench/cluster").mkdir(parents=True)
    (repo/"bench/cluster/runs").symlink_to(shared_logs, target_is_directory=True)
    for name in ("cargo/bin/rustup", "cadical-3.0.1/build/cadical"):
        path = repo/"bench/cluster/runs/tools"/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    cached = repo/"target/debug/keep"
    cached.parent.mkdir(parents=True)
    cached.write_text("preexisting cache")
    scratch = tmp_path/"node scratch"
    cargo_trace, python_trace = tmp_path/"cargo-trace", tmp_path/"python-trace"
    env = {**os.environ, "OAR_JOB_ID": "test-only", "BOOTSTRAP_TEST_MISSING": "curl uv python3.12",
           "BOOTSTRAP_TEST_TRACE": str(tmp_path/"tool-trace"),
           "BOOTSTRAP_TEST_BUILD_TRACE": str(cargo_trace),
           "BOOTSTRAP_TEST_PYTHON_TRACE": str(python_trace)}
    result = subprocess.run(["/bin/bash", "-c", DRIVER, "test-driver", str(SCRIPT),
                             "--repo", str(repo), "--scratch-dir", str(scratch), "--lp"],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout+result.stderr
    records = [line.split("|") for line in cargo_trace.read_text().splitlines()]
    assert len(records) == 3  # Rust tests and both Clippy feature sets.
    assert all(Path(record[0]).is_relative_to(scratch) for record in records)
    assert all(record[1:4] == ["0", "0", "0"] for record in records)
    assert "--manifest-path integrations/amthal/Cargo.toml" in records[0][4]
    build = next(line for line in python_trace.read_text().splitlines() if "campaign.py build" in line)
    assert f"--target-dir {records[0][0]}" in build and "--lp" in build
    installer_calls = [line for line in python_trace.read_text().splitlines() if "install_cosoco.py" in line]
    assert installer_calls == ["bench/cluster/lib/install_cosoco.py --verify-only"]
    assert not list(scratch.iterdir())
    assert cached.read_text() == "preexisting cache"


def test_reactivating_a_venv_does_not_remove_the_benchmark_tools(tmp_path):
    repo=tmp_path/'qayd'
    activate=repo/'.venv/bin/activate'
    activate.parent.mkdir(parents=True)
    activate.write_text('export PATH=/usr/bin:/bin\n')
    tool=repo/'bench/cluster/runs/tools/MiniZincIDE-2.9.5-bundle-linux-x86_64/bin/minizinc'
    tool.parent.mkdir(parents=True)
    tool.write_text('#!/bin/sh\n'); tool.chmod(0o755)
    body=SCRIPT.read_text().split("<<'ENV'\n",1)[1].split('\nENV\n',1)[0]
    env_file=repo/'env.sh'
    env_file.write_text('QAYD_CAMPAIGN_ROOT='+str(repo)+'\n'+body+'\n')
    result=subprocess.run(['/bin/bash','-c','source "$1"; command -v minizinc; source "$1"; command -v minizinc',
                           'activation-test',str(env_file)],capture_output=True,text=True,check=True)
    assert result.stdout.splitlines()==[str(tool),str(tool)]
