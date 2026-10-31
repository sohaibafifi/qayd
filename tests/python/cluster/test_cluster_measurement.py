"""Streaming, cancellation and independent witness checks retained for the cluster harness."""
import json
from pathlib import Path
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "bench"))
from cluster.lib.adapters import measurement as fastcop_run
from cluster.lib.adapters import xcsp as fastcop_manifest


def write_script(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    return path



@pytest.mark.parametrize("disappearing_pid", (100, 101))
def test_rss_sampling_tolerates_a_process_exiting_during_proc_read(monkeypatch, disappearing_pid):
    monkeypatch.setattr(Path, "is_file", lambda path: path == Path("/proc/100/status"))

    def read_proc(path, *args, **kwargs):
        if str(path) == f"/proc/{disappearing_pid}/status":
            raise ProcessLookupError(3, "No such process")
        if path == Path("/proc/100/status"):
            return "VmRSS:\t10 kB\n"
        if path == Path("/proc/100/task/100/children"):
            return "101"
        raise AssertionError(f"unexpected proc read: {path}")

    monkeypatch.setattr(Path, "read_text", read_proc)
    assert fastcop_run.process_tree_rss_kb(100) == (0 if disappearing_pid == 100 else 10)



def test_rss_sampling_tolerates_process_exit_during_status_lookup(monkeypatch):
    def exited(_path):
        raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr(Path, "is_file", exited)
    assert fastcop_run.process_tree_rss_kb(100) == 0



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



