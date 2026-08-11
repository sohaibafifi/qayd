import argparse
import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.fastcop import manifest as fastcop_manifest
from bench.fastcop import run as fastcop_run
from bench.fastcop import score as fastcop_score


def write_script(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    return path


def test_objective_detection_reads_past_old_eight_megabyte_cutoff(tmp_path):
    instance = tmp_path / "large.xml"
    padding = " " * ((8 << 20) + 1024)
    instance.write_text(
        "<instance><variables>" + padding + "</variables>"
        "<objectives><maximize>x</maximize></objectives></instance>",
        encoding="utf-8",
    )

    assert fastcop_manifest.detect_objective_sense(instance) == "max"


def test_streaming_records_all_incumbents_and_best_for_sense(tmp_path):
    solver = write_script(
        tmp_path / "solver.py",
        """import time
print('o 11', flush=True)
time.sleep(0.04)
print('o 7 extra-data', flush=True)
time.sleep(0.04)
print('o 9', flush=True)
print('s SATISFIABLE', flush=True)
print('v <instantiation/>', flush=True)
""",
    )
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"

    result = fastcop_run.execute_streaming(
        [sys.executable, "-u", str(solver)],
        "min",
        wall_seconds=2,
        cpu_seconds=2,
        memory_mb=512,
        grace_seconds=0.2,
        stdout_path=stdout,
        stderr_path=stderr,
    )

    assert [event["value"] for event in result["incumbents"]] == [11, 7, 9]
    assert result["first_incumbent"]["value"] == 11
    assert result["best_incumbent"]["value"] == 7
    assert result["incumbents"][0]["elapsed_seconds"] < result["incumbents"][1]["elapsed_seconds"]
    assert result["claimed_status"] == "SAT"
    assert result["has_solution"] is True


def test_timeout_sends_term_before_kill(tmp_path):
    solver = write_script(
        tmp_path / "term_solver.py",
        """import signal
import sys
import time

def stop(_signal, _frame):
    print('TERM_SEEN', file=sys.stderr, flush=True)
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
print('o 5', flush=True)
while True:
    time.sleep(0.05)
""",
    )
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"

    result = fastcop_run.execute_streaming(
        [sys.executable, "-u", str(solver)],
        "max",
        wall_seconds=0.2,
        cpu_seconds=10,
        memory_mb=512,
        grace_seconds=0.5,
        stdout_path=stdout,
        stderr_path=stderr,
    )

    assert result["timed_out"] is True
    assert result["termination_signal"] == "SIGTERM"
    assert result["killed"] is False
    assert "TERM_SEEN" in stderr.read_text(encoding="utf-8")


def test_solution_checker_acceptance_is_parsed(tmp_path):
    checker = write_script(
        tmp_path / "checker.py",
        """import sys
text = sys.stdin.read()
if 'v <instantiation' in text:
    print('OK\\t7')
else:
    print('ERROR: no instantiation found')
""",
    )
    accepted = tmp_path / "accepted.out"
    accepted.write_text("s SATISFIABLE\no 7\nv <instantiation/>\n", encoding="utf-8")
    rejected = tmp_path / "rejected.out"
    rejected.write_text("s SATISFIABLE\no 7\n", encoding="utf-8")

    valid = fastcop_run.validate_solution(
        [sys.executable, str(checker)],
        accepted,
        tmp_path / "valid.log",
        2,
        expected_objective=7,
    )
    invalid = fastcop_run.validate_solution(
        [sys.executable, str(checker)],
        rejected,
        tmp_path / "invalid.log",
        2,
        expected_objective=7,
    )
    mismatch = fastcop_run.validate_solution(
        [sys.executable, str(checker)],
        accepted,
        tmp_path / "mismatch.log",
        2,
        expected_objective=8,
    )
    bare_checker = write_script(
        tmp_path / "bare_checker.py",
        "print('OK')\n",
    )
    bare = fastcop_run.validate_solution(
        [sys.executable, str(bare_checker)],
        accepted,
        tmp_path / "bare.log",
        2,
        expected_objective=123,
    )
    bare_without_objective = fastcop_run.validate_solution(
        [sys.executable, str(bare_checker)],
        accepted,
        tmp_path / "bare-csp.log",
        2,
        expected_objective=None,
    )

    assert valid["valid"] is True
    assert valid["reason"] == "accepted"
    assert invalid["valid"] is False
    assert invalid["reason"] == "checker-rejected"
    assert mismatch["valid"] is False
    assert mismatch["reason"] == "objective-mismatch"
    assert mismatch["reported_objective"] == 7
    assert bare["valid"] is None
    assert bare["reason"] == "checker-missing-objective"
    assert bare["reported_objective"] is None
    assert bare_without_objective["valid"] is True
    assert bare_without_objective["reason"] == "accepted"


def test_solution_checker_infrastructure_failures_do_not_invalidate_a_family(tmp_path):
    out_of_memory = write_script(
        tmp_path / "oom_checker.py",
        "import sys\nprint('java.lang.OutOfMemoryError: Java heap space')\nraise SystemExit(1)\n",
    )
    slow = write_script(
        tmp_path / "slow_checker.py",
        "import time\ntime.sleep(10)\nprint('OK')\n",
    )
    killed = write_script(
        tmp_path / "killed_checker.py",
        "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n",
    )
    broken = write_script(
        tmp_path / "broken_checker.py",
        "raise SystemExit(2)\n",
    )
    solver_output = tmp_path / "solver.out"
    solver_output.write_text("s SATISFIABLE\no 1\nv <instantiation/>\n", encoding="utf-8")

    crashed = fastcop_run.validate_solution(
        [sys.executable, str(out_of_memory)],
        solver_output,
        tmp_path / "oom.log",
        timeout_seconds=2,
        expected_objective=1,
    )
    timed_out = fastcop_run.validate_solution(
        [sys.executable, str(slow)],
        solver_output,
        tmp_path / "timeout.log",
        timeout_seconds=0.05,
        expected_objective=1,
    )
    killed_result = fastcop_run.validate_solution(
        [sys.executable, str(killed)],
        solver_output,
        tmp_path / "killed.log",
        timeout_seconds=2,
        expected_objective=1,
    )
    broken_result = fastcop_run.validate_solution(
        [sys.executable, str(broken)],
        solver_output,
        tmp_path / "broken.log",
        timeout_seconds=2,
        expected_objective=1,
    )

    assert crashed["valid"] is None
    assert crashed["reason"] == "checker-oom"
    assert timed_out["valid"] is None
    assert timed_out["reason"] == "checker-timeout"
    assert killed_result["valid"] is None
    assert killed_result["reason"] == "checker-killed"
    assert broken_result["valid"] is None
    assert broken_result["reason"] == "checker-error"


def test_solution_checker_is_cancelled_with_the_campaign(tmp_path):
    checker = write_script(
        tmp_path / "slow_checker.py",
        "import time\ntime.sleep(10)\nprint('OK')\n",
    )
    solver_output = tmp_path / "solver.out"
    solver_output.write_text("s SATISFIABLE\no 1\nv <instantiation/>\n", encoding="utf-8")
    stop_event = threading.Event()
    timer = threading.Timer(0.1, stop_event.set)
    timer.start()
    try:
        with pytest.raises(fastcop_run.RunCancelled):
            fastcop_run.validate_solution(
                [sys.executable, str(checker)],
                solver_output,
                tmp_path / "checker.log",
                timeout_seconds=20,
                expected_objective=1,
                stop_event=stop_event,
            )
    finally:
        timer.cancel()


def test_checker_infrastructure_failure_is_unknown_not_invalid():
    record = {
        "raw_best_incumbent": {"value": 7},
        "execution_error": None,
        "returncode": 0,
    }
    validation = {
        "attempted": True,
        "valid": None,
        "reason": "checker-oom",
    }
    fastcop_run.apply_validation_projection(
        record,
        validation,
        check_solution=True,
        solver_summary={
            "claimed_status": "SAT",
            "claimed_proof": False,
            "has_solution": True,
            "best_incumbent": {"value": 7},
        },
    )

    assert record["status"] == "UNKNOWN"
    assert record["invalid"] is False
    assert record["best_incumbent"] is None


@pytest.mark.parametrize("timed_out,killed,returncode", [(False, False, 1), (True, True, -15)])
def test_incomplete_solver_execution_cannot_keep_an_optimality_proof(
    timed_out, killed, returncode
):
    record = {
        "raw_best_incumbent": {"value": 7},
        "execution_error": None,
        "returncode": returncode,
        "timed_out": timed_out,
        "killed": killed,
    }
    fastcop_run.apply_validation_projection(
        record,
        {"attempted": True, "valid": True, "reason": "accepted"},
        check_solution=True,
        solver_summary={
            "claimed_status": "OPTIMUM",
            "claimed_proof": True,
            "has_solution": True,
            "best_incumbent": {"value": 7},
        },
    )

    assert record["status"] == "SAT"
    assert record["proof"] is False


def test_incomplete_unsat_execution_is_not_a_proof():
    record = {
        "raw_best_incumbent": None,
        "execution_error": None,
        "returncode": 1,
        "timed_out": False,
        "killed": False,
    }
    fastcop_run.apply_validation_projection(
        record,
        {"attempted": False, "valid": None, "reason": "unsat-proof-not-checkable"},
        check_solution=True,
        solver_summary={
            "claimed_status": "UNSAT",
            "claimed_proof": True,
            "has_solution": False,
            "best_incumbent": None,
        },
    )

    assert record["status"] == "ERROR"
    assert record["proof"] is False


def test_clean_solver_execution_keeps_a_validated_optimality_proof():
    record = {
        "raw_best_incumbent": {"value": 7},
        "execution_error": None,
        "returncode": 0,
        "timed_out": False,
        "killed": False,
    }
    fastcop_run.apply_validation_projection(
        record,
        {"attempted": True, "valid": True, "reason": "accepted"},
        check_solution=True,
        solver_summary={
            "claimed_status": "OPTIMUM",
            "claimed_proof": True,
            "has_solution": True,
            "best_incumbent": {"value": 7},
        },
    )

    assert record["status"] == "OPTIMUM"
    assert record["proof"] is True


def result(
    solver,
    instance,
    sense,
    objective=None,
    status="SAT",
    proof=False,
    family="F",
    invalid=False,
):
    return {
        "schema": fastcop_score.RESULT_SCHEMA,
        "run_key": f"{solver}-{instance}",
        "solver": solver,
        "instance": instance,
        "family": family,
        "family_group": family,
        "objective_sense": sense,
        "status": "INVALID" if invalid else status,
        "proof": proof,
        "best_incumbent": None if objective is None else {"value": objective},
        "validation": {
            "valid": False if invalid else (True if objective is not None else None),
            "expected_objective": objective,
            "reported_objective": objective,
        },
        "invalid": invalid,
        "returncode": 0,
        "execution_error": None,
        "timed_out": False,
        "killed": False,
    }


def test_exact_opt_unsat_bb1_bb2_scoring_for_min_and_max():
    records = [
        result("a", "min-proof", "min", 10, "OPTIMUM", True),
        result("b", "min-proof", "min", 10),
        result("c", "min-proof", "min", 11),
        result("a", "max-open", "max", 7),
        result("b", "max-open", "max", 7),
        result("c", "max-open", "max", 5),
        result("a", "unsat", "min", None, "UNSAT", True),
        result("b", "unsat", "min", None, "UNKNOWN", False),
        result("c", "unsat", "min", None, "UNKNOWN", False),
    ]

    report = fastcop_score.score_records(records, mode="both", invalidation="none")
    rows = {row["instance"]: row for row in report["pool"]["instance_scores"]}

    assert rows["min-proof"]["scores"]["a"] == {
        "score": 1.0,
        "class": "Opt",
        "objective": 10,
        "status": "OPTIMUM",
    }
    assert rows["min-proof"]["scores"]["b"]["class"] == "BB2"
    assert rows["min-proof"]["scores"]["b"]["score"] == 0.5
    assert rows["max-open"]["scores"]["a"]["class"] == "BB1"
    assert rows["max-open"]["scores"]["b"]["class"] == "BB1"
    assert rows["max-open"]["scores"]["c"]["score"] == 0.0
    assert rows["unsat"]["scores"]["a"]["class"] == "Unsat"
    assert rows["unsat"]["scores"]["a"]["score"] == 1.0
    assert "a__vs__b" in report["pairwise"]


def test_false_answer_can_invalidate_only_its_family():
    records = [
        result("a", "f-bad", "min", None, family="F", invalid=True),
        result("b", "f-bad", "min", 4, family="F"),
        result("a", "f-good", "min", 3, "OPTIMUM", True, family="F"),
        result("b", "f-good", "min", 3, family="F"),
        result("a", "g-good", "max", 8, "OPTIMUM", True, family="G"),
        result("b", "g-good", "max", 8, family="G"),
    ]

    official = fastcop_score.score_records(records, mode="pool", invalidation="family")
    without_family_penalty = fastcop_score.score_records(
        records, mode="pool", invalidation="none"
    )
    rows = {
        row["instance"]: row for row in official["pool"]["instance_scores"]
    }

    assert rows["f-good"]["scores"]["a"]["class"] == "Invalidated"
    assert rows["g-good"]["scores"]["a"]["class"] == "Opt"
    assert official["pool"]["solvers"]["a"]["score"] == 1.0
    assert official["pool"]["solvers"]["a"]["invalidated_families"] == ["F"]
    assert without_family_penalty["pool"]["solvers"]["a"]["score"] == 2.0


def test_scorer_rejects_unverified_incumbents_and_incomplete_proofs():
    error = result("error", "open", "min", 5, status="ERROR")
    error["validation"] = {"valid": None}
    error["execution_error"] = "launch failed"
    valid = result("valid", "open", "min", 10)

    killed_optimum = result(
        "killed", "proof", "min", 7, status="OPTIMUM", proof=True
    )
    killed_optimum["returncode"] = -15
    killed_optimum["timed_out"] = True
    killed_optimum["killed"] = True
    competitor = result("valid", "proof", "min", 8)

    broken_unsat = result(
        "broken", "unsat", "min", None, status="UNSAT", proof=True
    )
    broken_unsat["returncode"] = 1
    unknown = result("valid", "unsat", "min", None, status="UNKNOWN")

    report = fastcop_score.score_records(
        [error, valid, killed_optimum, competitor, broken_unsat, unknown],
        mode="pool",
        invalidation="none",
    )
    rows = {row["instance"]: row for row in report["pool"]["instance_scores"]}

    assert rows["open"]["scores"]["error"]["class"] == "None"
    assert rows["open"]["scores"]["valid"]["class"] == "BB1"
    assert rows["proof"]["scores"]["killed"]["class"] == "BB1"
    assert rows["proof"]["scores"]["killed"]["class"] != "Opt"
    assert rows["unsat"]["scores"]["broken"]["class"] == "None"


def test_score_loader_rejects_missing_and_duplicate_run_keys(tmp_path):
    missing = result("a", "i", "min", 1)
    missing.pop("run_key")
    missing_path = tmp_path / "missing.jsonl"
    missing_path.write_text(json.dumps(missing) + "\n", encoding="utf-8")
    with pytest.raises(fastcop_score.ScoreError, match="missing run_key"):
        fastcop_score.load_records([missing_path])

    duplicate = result("a", "i", "min", 1)
    duplicate_path = tmp_path / "duplicate.jsonl"
    duplicate_path.write_text(
        json.dumps(duplicate) + "\n" + json.dumps(duplicate) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(fastcop_score.ScoreError, match="duplicate run_key"):
        fastcop_score.load_records([duplicate_path])


def test_score_loader_requires_explicit_cross_file_solver_replacement(tmp_path):
    old = result("qayd", "i", "min", 10)
    new = result("qayd", "i", "min", 8)
    new["run_key"] = "qayd-i-new"
    old_path = tmp_path / "old.jsonl"
    new_path = tmp_path / "new.jsonl"
    old_path.write_text(json.dumps(old) + "\n", encoding="utf-8")
    new_path.write_text(json.dumps(new) + "\n", encoding="utf-8")

    with pytest.raises(fastcop_score.ScoreError, match="multiple runs"):
        fastcop_score.load_records([old_path, new_path])
    audit = {}
    merged = fastcop_score.load_records(
        [old_path, new_path], replace_solvers=["qayd"], audit=audit
    )
    assert len(merged) == 1
    assert merged[0]["best_incumbent"]["value"] == 8
    assert audit["replace_solvers"] == ["qayd"]
    assert audit["replacements"] == [
        {
            "solver": "qayd",
            "instance": "i",
            "old_run_key": "qayd-i",
            "new_run_key": "qayd-i-new",
            "old_input": str(old_path),
            "new_input": str(new_path),
        }
    ]
    assert all(len(item["sha256"]) == 64 for item in audit["inputs"])


def test_manifest_gate_rejects_incomplete_or_mislabeled_campaign():
    manifest_hash = "a" * 64
    manifest = {
        "instances": [
            {
                "id": "i",
                "family": "F",
                "family_group": "FG",
                "objective_sense": "min",
                "sha256": "b" * 64,
            }
        ]
    }

    def campaign_record(solver):
        record = result(solver, "i", "min", 7)
        record.update(
            {
                "family_group": "FG",
                "instance_sha256": "b" * 64,
                "seed": 0,
                "limits": {
                    "cpu_seconds": 180.0,
                    "wall_seconds": 270.0,
                    "memory_mb": 4096,
                    "grace_seconds": 1.0,
                    "parallel_jobs": 4,
                },
                "provenance": {
                    "manifest_sha256": manifest_hash,
                    "solver_config_sha256": "c" * 64,
                    "execution_harness_sha256": "d" * 64,
                },
            }
        )
        return record

    records = [campaign_record("a"), campaign_record("b")]
    report = fastcop_score.score_records(
        records,
        mode="pool",
        manifest=manifest,
        manifest_hash=manifest_hash,
    )
    assert report["campaign"]["instance_count"] == 1

    mislabeled = [dict(records[0]), dict(records[1])]
    mislabeled[0]["family"] = "typo"
    with pytest.raises(fastcop_score.ScoreError, match="family mismatch"):
        fastcop_score.score_records(
            mislabeled,
            mode="pool",
            manifest=manifest,
            manifest_hash=manifest_hash,
        )

    with pytest.raises(fastcop_score.ScoreError, match="campaign is missing"):
        fastcop_score.score_records(
            records[:1],
            mode="pool",
            solvers=["a", "b"],
            manifest=manifest,
            manifest_hash=manifest_hash,
        )


def test_manifest_serialization_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(fastcop_manifest, "REPO_ROOT", tmp_path)
    instances = tmp_path / "instances"
    instances.mkdir()
    (instances / "Coprime-02.xml").write_text(
        "<instance><objectives><minimize>x</minimize></objectives></instance>",
        encoding="utf-8",
    )

    first = fastcop_manifest.canonical_json(fastcop_manifest.build_manifest(instances))
    second = fastcop_manifest.canonical_json(fastcop_manifest.build_manifest(instances))

    assert first == second
    assert json.loads(first)["instances"][0]["objective_sense"] == "min"


def test_per_family_selection_is_deterministic_and_representative():
    instances = [
        {"id": "A-1", "family": "A"},
        {"id": "A-2", "family": "A"},
        {"id": "A-3", "family": "A"},
        {"id": "B-1", "family": "B"},
        {"id": "B-2", "family": "B"},
    ]
    benchmark = {"instances": instances}

    selected = fastcop_run.select_instances(
        benchmark, families=[], pattern=None, limit=0, per_family=2
    )

    assert [item["id"] for item in selected] == ["A-1", "A-2", "B-1", "B-2"]


def test_execution_and_validation_identities_are_independent():
    execution_arguments = dict(
        solver_name="qayd",
        artifact_hash="solver-artifact",
        instance_id="instance-id",
        instance_hash="instance",
        objective_sense="min",
        cpu_seconds=5.0,
        wall_seconds=8.0,
        memory_mb=1024,
        seed=0,
        grace_seconds=1.0,
        parallel_jobs=1,
        command=["{artifact}", "-t", "5", "{instance}"],
        launcher={"path": "/workspace/qayd", "sha256": "launcher-hash"},
        limit_launcher={"python": {"sha256": "python-hash"}},
        execution_harness_hash="execution-harness",
    )
    baseline = fastcop_run.make_execution_identity(**execution_arguments)

    for field, value in [
        ("grace_seconds", 2.0),
        ("parallel_jobs", 2),
        (
            "launcher",
            {"path": "/workspace/qayd", "sha256": "different-launcher"},
        ),
        ("limit_launcher", {"python": {"sha256": "different-python"}}),
        ("execution_harness_hash", "different-execution-harness"),
    ]:
        changed = dict(execution_arguments)
        changed[field] = value
        assert fastcop_run.make_execution_identity(**changed) != baseline

    validation_arguments = dict(
        execution_key=fastcop_run.identity_key(baseline),
        checker_hash="checker-artifact",
        checker_memory_mb=1024,
        checker_timeout=120.0,
        checker_command=["java", "-Xmx1024m", "{instance}"],
        checker_launcher={"path": "/usr/bin/java", "sha256": "java-hash"},
        limit_launcher={"python": {"sha256": "python-hash"}},
        solver_stdout_hash="stdout",
        expected_objective=7,
        validation_harness_hash="validation-harness",
    )
    validation = fastcop_run.make_validation_identity(**validation_arguments)
    for field, value in [
        ("checker_hash", "different-checker"),
        ("checker_memory_mb", 2048),
        ("checker_timeout", 60.0),
        (
            "checker_launcher",
            {"path": "/usr/bin/java", "sha256": "different-java"},
        ),
        ("limit_launcher", {"python": {"sha256": "different-python"}}),
        ("validation_harness_hash", "different-validation-harness"),
    ]:
        changed = dict(validation_arguments)
        changed[field] = value
        assert fastcop_run.make_validation_identity(**changed) != validation
        assert fastcop_run.make_execution_identity(**execution_arguments) == baseline


def test_canonical_command_handles_an_embedded_instance_placeholder(tmp_path):
    planned = fastcop_run.canonicalize_command(
        [f"--input={fastcop_run.INSTANCE_SENTINEL}"]
    )
    scratch = tmp_path / "qayd-fastcop-instance-test"
    scratch.mkdir()
    materialized = scratch / "instance.xml"
    materialized.touch()
    executed = fastcop_run.canonicalize_command(
        [f"--input={materialized}"]
    )

    assert planned == ["--input={instance}"]
    assert executed == planned


def test_jobs_must_be_positive():
    assert fastcop_run.positive_int("2") == 2
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        fastcop_run.positive_int("0")
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        fastcop_run.positive_int("-1")
    assert fastcop_run.nonnegative_float("0") == 0
    for value in ("0", "nan", "inf", "-inf"):
        with pytest.raises(argparse.ArgumentTypeError, match="finite and positive"):
            fastcop_run.positive_float(value)


def test_parallel_runs_overlap_and_write_jsonl_in_plan_order(tmp_path):
    tasks = [
        fastcop_run.RunTask(
            position=position,
            solver_name="solver",
            solver={},
            instance_item={"id": f"instance-{position}"},
            run_key=f"key-{position}",
        )
        for position in range(1, 4)
    ]
    barrier = threading.Barrier(2, timeout=5)
    release_second = threading.Event()
    state_lock = threading.Lock()
    active = 0
    peak = 0
    called = []
    finished = []

    def worker(task):
        nonlocal active, peak
        with state_lock:
            called.append(task.position)
            active += 1
            peak = max(peak, active)
        barrier.wait()
        if task.position == 3:
            with state_lock:
                finished.append(3)
            release_second.set()
        else:
            assert release_second.wait(timeout=5)
            with state_lock:
                finished.append(2)
        with state_lock:
            active -= 1
        return {"position": task.position}

    stop_event = threading.Event()
    output = tmp_path / "results.jsonl"
    results = []
    for task, record in fastcop_run.ordered_run_results(
        tasks,
        completed_keys={"key-1"},
        jobs=2,
        worker=worker,
        stop_event=stop_event,
    ):
        results.append((task, record))
        if record is not None:
            fastcop_run.append_record(output, record)

    assert [task.position for task, _record in results] == [1, 2, 3]
    assert results[0][1] is None
    assert [record["position"] for _task, record in results[1:]] == [2, 3]
    assert sorted(called) == [2, 3]
    assert finished == [3, 2]
    assert peak == 2
    assert stop_event.is_set() is False
    written = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["position"] for record in written] == [2, 3]


def test_campaign_lock_rejects_a_second_writer(tmp_path):
    output = tmp_path / "results.jsonl"
    first = fastcop_run.acquire_campaign_lock(output)
    try:
        with pytest.raises(fastcop_run.HarnessError, match="another FAST COP campaign"):
            fastcop_run.acquire_campaign_lock(output)
    finally:
        fastcop_run.release_campaign_lock(first)

    log_directory = tmp_path / "logs"
    first_log = fastcop_run.acquire_log_lock(log_directory)
    try:
        with pytest.raises(fastcop_run.HarnessError, match="another FAST COP campaign"):
            fastcop_run.acquire_log_lock(log_directory)
    finally:
        fastcop_run.release_campaign_lock(first_log)


def test_existing_output_must_match_the_current_plan(tmp_path):
    task = fastcop_run.RunTask(
        position=1,
        solver_name="qayd",
        solver={},
        instance_item={"id": "instance-1"},
        run_key="expected-key",
    )
    matching = {
        "solver": "qayd",
        "instance": "instance-1",
        "run_key": "expected-key",
    }
    fastcop_run.ensure_output_matches_plan(
        tmp_path / "results.jsonl", [task], [matching]
    )

    incompatible = {**matching, "run_key": "different-key"}
    with pytest.raises(fastcop_run.HarnessError, match="incompatible run"):
        fastcop_run.ensure_output_matches_plan(
            tmp_path / "results.jsonl", [task], [incompatible]
        )

    outside = {**matching, "instance": "instance-2"}
    with pytest.raises(fastcop_run.HarnessError, match="outside the current selection"):
        fastcop_run.ensure_output_matches_plan(
            tmp_path / "results.jsonl", [task], [outside]
        )

    second_task = fastcop_run.RunTask(
        position=2,
        solver_name="qayd",
        solver={},
        instance_item={"id": "instance-2"},
        run_key="second-key",
    )
    second_record = {
        "solver": "qayd",
        "instance": "instance-2",
        "run_key": "second-key",
    }
    with pytest.raises(fastcop_run.HarnessError, match="canonical prefix"):
        fastcop_run.ensure_output_matches_plan(
            tmp_path / "results.jsonl", [task, second_task], [second_record]
        )


def test_legacy_run_key_is_quarantined_from_current_execution_plan():
    execution_identity = fastcop_run.make_execution_identity(
        solver_name="qayd",
        artifact_hash="artifact-hash",
        instance_id="instance-1",
        instance_hash="instance-hash",
        objective_sense="min",
        cpu_seconds=5.0,
        wall_seconds=8.0,
        memory_mb=1024,
        seed=0,
        grace_seconds=1.0,
        parallel_jobs=1,
        command=["{artifact}", "-t", "5", "{instance}"],
        launcher={"path": "/workspace/qayd", "sha256": "launcher-hash"},
        limit_launcher={"python": {"sha256": "python-hash"}},
        execution_harness_hash="execution-harness",
    )
    execution_key = fastcop_run.identity_key(execution_identity)
    task = fastcop_run.RunTask(
        position=1,
        solver_name="qayd",
        solver={},
        instance_item={"id": "instance-1"},
        run_key=execution_key,
        execution_identity=execution_identity,
    )
    legacy = {
        "solver": "qayd",
        "instance": "instance-1",
        "run_key": "legacy-key-containing-checker-settings",
        "instance_sha256": "instance-hash",
        "objective_sense": "min",
        "seed": 0,
        "limits": {
            "cpu_seconds": 5.0,
            "wall_seconds": 8.0,
            "memory_mb": 1024,
            "grace_seconds": 1.0,
            "parallel_jobs": 1,
            "checker_memory_mb": 256,
        },
        "command": [
            "/workspace/qayd",
            "-t",
            "5",
            "/tmp/qayd-fastcop-instance-old/instance.xml",
        ],
        "provenance": {
            "artifact": "/workspace/qayd",
            "artifact_sha256": "artifact-hash",
            "checker_sha256": "old-checker",
            "harness_sha256": "old-validation-harness",
        },
    }

    with pytest.raises(fastcop_run.HarnessError, match="cannot be resumed safely"):
        fastcop_run.ensure_output_matches_plan(
            Path("results.jsonl"), [task], [legacy]
        )

    legacy_key = fastcop_run.ensure_record_execution_key(
        legacy, allow_legacy=True
    )
    assert legacy_key != execution_key
    assert legacy["execution_key"] == legacy_key
    assert fastcop_run.ensure_record_execution_key(legacy, allow_legacy=True) == legacy_key
    assert legacy["run_key"] == "legacy-key-containing-checker-settings"


def test_resume_rejects_an_execution_key_that_does_not_match_its_identity():
    identity = fastcop_run.make_execution_identity(
        solver_name="qayd",
        artifact_hash="artifact-hash",
        instance_id="instance-1",
        instance_hash="instance-hash",
        objective_sense="min",
        cpu_seconds=5.0,
        wall_seconds=8.0,
        memory_mb=1024,
        seed=0,
        grace_seconds=1.0,
        parallel_jobs=1,
        command=["{artifact}", "{instance}"],
        launcher={"path": "/workspace/qayd", "sha256": "launcher-hash"},
        limit_launcher={"python": {"sha256": "python-hash"}},
        execution_harness_hash="execution-harness",
    )
    task = fastcop_run.RunTask(
        position=1,
        solver_name="qayd",
        solver={},
        instance_item={"id": "instance-1"},
        run_key=fastcop_run.identity_key(identity),
        execution_identity=identity,
    )
    record = {
        "solver": "qayd",
        "instance": "instance-1",
        "run_key": task.run_key,
        "execution_key": task.run_key,
        "execution_identity": {**identity, "seed": 99},
    }

    with pytest.raises(fastcop_run.HarnessError, match="does not match its key"):
        fastcop_run.ensure_output_matches_plan(
            Path("results.jsonl"), [task], [record]
        )


def test_recheck_only_reuses_stdout_and_atomically_replaces_validation(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(fastcop_run, "REPO_ROOT", tmp_path)
    instance = tmp_path / "instance.xml"
    instance.write_text("<instance/>", encoding="utf-8")
    stdout = tmp_path / "stored.stdout.log"
    stdout.write_text(
        "s SATISFIABLE\no 7\nv <instantiation/>\n", encoding="utf-8"
    )
    stderr = tmp_path / "stored.stderr.log"
    stderr.write_text("", encoding="utf-8")
    marker = tmp_path / "solver-ran"
    solver = write_script(
        tmp_path / "solver.py",
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
    )
    checker = write_script(
        tmp_path / "checker.py",
        "import sys\nsys.stdin.read()\nprint('OK\\t7')\n",
    )
    solver_config = tmp_path / "solvers.json"
    solver_config.write_text(
        json.dumps(
            {
                "schema": fastcop_run.SOLVER_SCHEMA,
                "checker": {
                    "artifact": checker.name,
                    "expected_sha256": None,
                    "argv": [
                        sys.executable,
                        "{ace_artifact}",
                        "{instance}",
                    ],
                },
                "solvers": {
                    "fake": {
                        "artifact": solver.name,
                        "expected_sha256": None,
                        "argv": [
                            sys.executable,
                            "{artifact}",
                            "{instance}",
                        ],
                        "version": "test",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "results.jsonl"
    record = {
        "schema": fastcop_run.RESULT_SCHEMA,
        "run_key": "legacy-validation-coupled-key",
        "solver": "fake",
        "instance": "i",
        "family": "F",
        "family_group": "F",
        "objective_sense": "min",
        "instance_path": instance.name,
        "instance_sha256": fastcop_manifest.sha256_file(instance),
        "seed": 0,
        "limits": {
            "cpu_seconds": 5.0,
            "wall_seconds": 8.0,
            "memory_mb": 512,
            "grace_seconds": 1.0,
            "parallel_jobs": 1,
        },
        "command": [
            sys.executable,
            str(solver),
            "/tmp/qayd-fastcop-instance-old/instance.xml",
        ],
        "status": "INVALID",
        "proof": False,
        "best_incumbent": None,
        "raw_best_incumbent": {"value": 7, "elapsed_seconds": 0.1},
        "validation": {
            "attempted": True,
            "valid": False,
            "reason": "checker-rejected",
        },
        "invalid": True,
        "returncode": 0,
        "execution_error": None,
        "logs": {"stdout": str(stdout), "stderr": str(stderr), "checker": None},
        "provenance": {
            "artifact": str(solver),
            "artifact_sha256": fastcop_manifest.sha256_file(solver),
        },
    }
    fastcop_run.append_record(output, record)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--recheck-only",
            "--solver",
            "fake",
            "--solvers",
            str(solver_config),
            "--output",
            str(output),
            "--checker-memory-mb",
            "256",
            "--checker-timeout",
            "2",
        ],
    )

    assert fastcop_run.main() == 0

    records = fastcop_run.load_result_records(output)
    assert len(records) == 1
    updated = records[0]
    assert marker.exists() is False
    assert updated["run_key"] == "legacy-validation-coupled-key"
    assert isinstance(updated["execution_key"], str)
    assert isinstance(updated["validation_key"], str)
    assert updated["validation"]["reason"] == "accepted"
    assert updated["validation"]["valid"] is True
    assert updated["status"] == "SAT"
    assert updated["invalid"] is False
    assert updated["best_incumbent"]["value"] == 7


def test_parallel_resume_appends_only_the_missing_suffix(tmp_path):
    tasks = [
        fastcop_run.RunTask(
            position=position,
            solver_name="qayd",
            solver={},
            instance_item={"id": f"instance-{position}"},
            run_key=f"key-{position}",
        )
        for position in range(1, 4)
    ]

    def record(task):
        return {
            "schema": fastcop_run.RESULT_SCHEMA,
            "run_key": task.run_key,
            "solver": task.solver_name,
            "instance": task.instance_item["id"],
        }

    output = tmp_path / "results.jsonl"
    fastcop_run.append_record(output, record(tasks[0]))
    existing = fastcop_run.load_result_records(output)
    fastcop_run.ensure_output_matches_plan(output, tasks, existing)
    completed = {item["run_key"] for item in existing}
    called = []

    def worker(task):
        called.append(task.position)
        return record(task)

    for _task, result_record in fastcop_run.ordered_run_results(
        tasks,
        completed_keys=completed,
        jobs=2,
        worker=worker,
        stop_event=threading.Event(),
    ):
        if result_record is not None:
            fastcop_run.append_record(output, result_record)

    final = fastcop_run.load_result_records(output)
    assert sorted(called) == [2, 3]
    assert [item["run_key"] for item in final] == ["key-1", "key-2", "key-3"]


def test_parallel_worker_failure_cancels_other_runs():
    tasks = [
        fastcop_run.RunTask(
            position=position,
            solver_name="qayd",
            solver={},
            instance_item={"id": f"instance-{position}"},
            run_key=f"key-{position}",
        )
        for position in (1, 2)
    ]
    second_started = threading.Event()
    stop_event = threading.Event()

    def worker(task):
        if task.position == 1:
            assert second_started.wait(timeout=5)
            raise fastcop_run.HarnessError("injected failure")
        second_started.set()
        assert stop_event.wait(timeout=5)
        raise fastcop_run.RunCancelled("cancelled by test")

    with pytest.raises(fastcop_run.HarnessError, match="injected failure"):
        list(
            fastcop_run.ordered_run_results(
                tasks,
                completed_keys=set(),
                jobs=2,
                worker=worker,
                stop_event=stop_event,
            )
        )
    assert stop_event.is_set()


@pytest.mark.parametrize(
    "content",
    [
        "{bad json\n",
        "[]\n",
        '{"schema":"qayd.fastcop.result/v1","solver":"qayd","instance":"i"}\n',
    ],
)
def test_resume_rejects_corrupt_jsonl(tmp_path, content):
    output = tmp_path / "results.jsonl"
    output.write_text(content, encoding="utf-8")
    with pytest.raises(fastcop_run.HarnessError):
        fastcop_run.load_result_records(output)
