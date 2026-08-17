import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "bench" / "common" / "run.py"


def test_runner_replays_assignment_and_writes_complete_provenance(tmp_path):
    instances = tmp_path / "instances"
    nested = instances / "nested"
    nested.mkdir(parents=True)
    instance = nested / "tiny.cnf"
    instance.write_text("p cnf 2 2\n1 2 0\n-1 2 0\n", encoding="utf-8")
    tools = tmp_path / "tools with spaces"
    tools.mkdir()
    solver = tools / "solver.py"
    solver.write_text(
        "print('s SATISFIABLE')\nprint('v -1 2 0')\n",
        encoding="utf-8",
    )
    output = tmp_path / "results.csv"
    provenance = tmp_path / "provenance.json"
    logs = tmp_path / "logs"

    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--dir",
            str(instances),
            "--cmd",
            f'"{sys.executable}" "{solver}" {{f}}',
            "--timeout",
            "2",
            "--verify-kind",
            "sat",
            "--solver",
            "fake-sat",
            "--artifact",
            str(solver),
            "--out",
            str(output),
            "--provenance-out",
            str(provenance),
            "--log-dir",
            str(logs),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["instance"] == "nested/tiny.cnf"
    assert rows[0]["status"] == "SAT"
    assert rows[0]["valid"] == "1"
    assert rows[0]["validation_reason"] == "accepted"
    assert len(rows[0]["instance_sha256"]) == 64
    assert Path(rows[0]["stdout_log"]).read_text(encoding="utf-8") == (
        "s SATISFIABLE\nv -1 2 0\n"
    )
    assert Path(rows[0]["stderr_log"]).read_text(encoding="utf-8") == ""

    sidecar = json.loads(provenance.read_text(encoding="utf-8"))
    assert sidecar["schema"] == "qayd.satpb.campaign/v1"
    assert sidecar["complete"] is True
    assert sidecar["configuration"]["solver"] == "fake-sat"
    assert sidecar["configuration"]["instances"] == [
        {"path": "nested/tiny.cnf", "sha256": rows[0]["instance_sha256"]}
    ]
    assert sidecar["results"]["records"] == 1
    assert sidecar["results"]["invalid"] == 0
    assert sidecar["results"]["inconclusive_validations"] == 0
    assert len(sidecar["configuration_sha256"]) == 64
    assert len(sidecar["results"]["sha256"]) == 64


def test_runner_reserves_external_time_for_final_output(tmp_path):
    instances = tmp_path / "instances"
    instances.mkdir()
    (instances / "tiny.cnf").write_text("p cnf 1 1\n1 0\n", encoding="utf-8")
    solver = tmp_path / "solver.py"
    solver.write_text(
        "import time\n"
        "time.sleep(1.2)\n"
        "print('s SATISFIABLE')\n"
        "print('v 1 0')\n",
        encoding="utf-8",
    )
    output = tmp_path / "results.csv"
    logs = tmp_path / "logs"

    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--dir",
            str(instances),
            "--cmd",
            f'"{sys.executable}" "{solver}" {{t}} {{f}}',
            "--timeout",
            "1",
            "--finalization-seconds",
            "1",
            "--verify-kind",
            "sat",
            "--log-dir",
            str(logs),
            "--out",
            str(output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    with output.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["timedout"] == "0"
    assert row["valid"] == "1"
    assert 1.0 < float(row["time"]) < 2.0


def test_runner_records_invalid_solver_assignment_without_accepting_it(tmp_path):
    instances = tmp_path / "instances"
    instances.mkdir()
    (instances / "tiny.cnf").write_text("p cnf 1 1\n1 0\n", encoding="utf-8")
    solver = tmp_path / "bad_solver.py"
    solver.write_text(
        "print('s SATISFIABLE')\nprint('v -1 0')\n",
        encoding="utf-8",
    )
    output = tmp_path / "results.csv"
    provenance = tmp_path / "provenance.json"

    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--dir",
            str(instances),
            "--cmd",
            f'"{sys.executable}" "{solver}" {{f}}',
            "--timeout",
            "2",
            "--verify-kind",
            "sat",
            "--artifact",
            str(solver),
            "--out",
            str(output),
            "--provenance-out",
            str(provenance),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    with output.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["status"] == "SAT"
    assert row["valid"] == "0"
    assert row["validation_reason"] == "unsatisfied-clause"
    sidecar = json.loads(provenance.read_text(encoding="utf-8"))
    assert sidecar["results"]["invalid"] == 1


def test_external_wall_limit_does_not_grant_an_extra_five_seconds(tmp_path):
    instances = tmp_path / "instances"
    instances.mkdir()
    (instances / "tiny.cnf").write_text("p cnf 1 1\n1 0\n", encoding="utf-8")
    solver = tmp_path / "slow_solver.py"
    solver.write_text("import time\ntime.sleep(10)\n", encoding="utf-8")
    output = tmp_path / "results.csv"

    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--dir",
            str(instances),
            "--cmd",
            f'"{sys.executable}" "{solver}" {{f}}',
            "--timeout",
            "1",
            "--out",
            str(output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    with output.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["timedout"] == "1"
    assert float(row["time"]) < 2.5


def test_sigterm_grace_accepts_a_final_verified_incumbent(tmp_path):
    instances = tmp_path / "instances"
    instances.mkdir()
    (instances / "tiny.cnf").write_text("p cnf 1 1\n1 0\n", encoding="utf-8")
    solver = tmp_path / "graceful_solver.py"
    solver.write_text(
        "import signal, time\n"
        "def stop(_signal, _frame):\n"
        "    print('s SATISFIABLE', flush=True)\n"
        "    print('v 1 0', flush=True)\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "while True:\n"
        "    time.sleep(0.1)\n",
        encoding="utf-8",
    )
    output = tmp_path / "results.csv"

    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--dir",
            str(instances),
            "--cmd",
            f'"{sys.executable}" "{solver}" {{f}}',
            "--timeout",
            "1",
            "--grace-seconds",
            "1",
            "--verify-kind",
            "sat",
            "--out",
            str(output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    with output.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["timedout"] == "1"
    assert row["status"] == "SAT"
    assert row["valid"] == "1"
