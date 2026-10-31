"""Cluster helpers retained from bench/fastcop/run.py; experimental semantics preserved."""
from __future__ import annotations

import os
import queue
import re
import resource
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]


LIMIT_LAUNCHER = Path(__file__).with_name("exec_limit.py").resolve()


STATUS_RE = re.compile(r"^\s*s\s+(.+?)\s*$", re.IGNORECASE)


OBJECTIVE_RE = re.compile(r"^\s*o\s+(-?\d+)(?:\s|$)", re.IGNORECASE)


CHECK_OK_RE = re.compile(r"^OK(?:[ \t]+(-?\d+))?[ \t]*$", re.MULTILINE)


CHECKER_OOM_RE = re.compile(
    r"(?:OutOfMemoryError|Java heap space|GC overhead limit exceeded|"
    r"Cannot reserve enough space|unable to create native thread|"
    r"insufficient memory for the Java Runtime Environment|"
    r"Native memory allocation .* failed|Cannot allocate memory)",
    re.IGNORECASE,
)


class HarnessError(RuntimeError):
    """A reproducibility or execution invariant was violated."""


class RunCancelled(RuntimeError):
    """An in-flight run was stopped because the campaign is terminating."""


def display_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else format(value, ".6g")


def normalize_status(raw: str) -> str:
    upper = raw.upper()
    if "OPTIMUM" in upper or "OPTIMAL" in upper:
        return "OPTIMUM"
    if "UNSAT" in upper:
        return "UNSAT"
    if "SATIS" in upper:
        return "SAT"
    if "UNKNOWN" in upper:
        return "UNKNOWN"
    if "UNSUPPORTED" in upper:
        return "UNSUPPORTED"
    return "UNKNOWN"


class OutputObserver:
    """Parse competition output while preserving every incumbent timestamp."""

    def __init__(self, objective_sense: str) -> None:
        if objective_sense not in ("min", "max"):
            raise HarnessError(f"unsupported objective sense: {objective_sense}")
        self.objective_sense = objective_sense
        self.incumbents: list[dict[str, Any]] = []
        self.status_events: list[dict[str, Any]] = []
        self.has_solution = False

    def observe(self, stream: str, line: str, elapsed: float) -> None:
        if stream != "stdout":
            return
        objective = OBJECTIVE_RE.match(line)
        if objective:
            self.incumbents.append(
                {
                    "value": int(objective.group(1)),
                    "elapsed_seconds": round(elapsed, 6),
                    "line": line.rstrip("\r\n"),
                }
            )
        status = STATUS_RE.match(line)
        if status:
            self.status_events.append(
                {
                    "status": normalize_status(status.group(1)),
                    "elapsed_seconds": round(elapsed, 6),
                    "line": line.rstrip("\r\n"),
                }
            )
        if line.lstrip().startswith("v "):
            self.has_solution = True

    def summary(self) -> dict[str, Any]:
        first = self.incumbents[0] if self.incumbents else None
        if self.objective_sense == "min":
            best = min(self.incumbents, key=lambda item: item["value"], default=None)
        else:
            best = max(self.incumbents, key=lambda item: item["value"], default=None)
        status = self.status_events[-1]["status"] if self.status_events else "UNKNOWN"
        if status == "UNKNOWN" and best is not None:
            status = "SAT"
        proof_event = next(
            (
                event for event in reversed(self.status_events)
                if event["status"] in ("OPTIMUM", "UNSAT")
            ),
            None,
        )
        return {
            "incumbents": self.incumbents,
            "first_incumbent": first,
            "best_incumbent": best,
            "status_events": self.status_events,
            "claimed_status": status,
            "claimed_proof": status in ("OPTIMUM", "UNSAT"),
            "proof_elapsed_seconds": (
                proof_event["elapsed_seconds"] if proof_event is not None else None
            ),
            "has_solution": self.has_solution,
        }


def limited_argv(
    argv: Sequence[str], cpu_seconds: float, memory_mb: int
) -> list[str]:
    return [
        sys.executable,
        str(LIMIT_LAUNCHER),
        "--cpu-seconds",
        display_number(cpu_seconds),
        "--memory-mb",
        str(memory_mb),
        "--",
        *argv,
    ]


def process_tree_rss_kb(root_pid: int) -> Optional[int]:
    """Best-effort resident memory for a process and all descendants."""
    proc_root = Path("/proc")
    try:
        has_status = (proc_root / str(root_pid) / "status").is_file()
    except ProcessLookupError:
        return 0
    if has_status:
        pending = [root_pid]
        descendants = set()
        total = 0
        while pending:
            pid = pending.pop()
            if pid in descendants:
                continue
            descendants.add(pid)
            process_dir = proc_root / str(pid)
            try:
                for line in (process_dir / "status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1])
                        break
                children = process_dir / "task" / str(pid) / "children"
                pending.extend(int(child) for child in children.read_text().split())
            except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
                continue
        return total

    # BSD/macOS has no procfs. This path is slower, but it is only used for
    # local development; competition hosts use the procfs path above.
    try:
        completed = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,rss="],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    children: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            pid, parent, resident = map(int, fields)
        except ValueError:
            continue
        children.setdefault(parent, []).append(pid)
        rss[pid] = resident
    pending = [root_pid]
    descendants = set()
    while pending:
        pid = pending.pop()
        if pid in descendants:
            continue
        descendants.add(pid)
        pending.extend(children.get(pid, ()))
    return sum(rss.get(pid, 0) for pid in descendants)


def terminate_group(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
    try:
        os.killpg(os.getpgid(process.pid), sig)
    except (OSError, ProcessLookupError):
        try:
            process.send_signal(sig)
        except OSError:
            pass


def _pipe_reader(
    stream_name: str,
    pipe,
    log,
    events: "queue.Queue[tuple[str, float, Optional[str]]]",
    start: float,
) -> None:
    try:
        while True:
            raw = pipe.readline()
            if not raw:
                break
            elapsed = time.monotonic() - start
            log.write(raw)
            log.flush()
            events.put((stream_name, elapsed, raw.decode("utf-8", "replace")))
    finally:
        events.put((stream_name, time.monotonic() - start, None))


def execute_streaming(
    argv: Sequence[str],
    objective_sense: str,
    wall_seconds: float,
    cpu_seconds: float,
    memory_mb: int,
    grace_seconds: float,
    stdout_path: Path,
    stderr_path: Path,
    *,
    measure_child_cpu: bool = True,
    stop_event: Optional[threading.Event] = None,
) -> dict[str, Any]:
    """Execute one process, stream both pipes, and enforce graceful timeout."""
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    observer = OutputObserver(objective_sense)
    events: "queue.Queue[tuple[str, float, Optional[str]]]" = queue.Queue()
    usage_before = (
        resource.getrusage(resource.RUSAGE_CHILDREN)
        if measure_child_cpu
        else None
    )
    start = time.monotonic()
    timed_out = False
    cancelled = False
    killed = False
    termination_signal: Optional[str] = None
    peak_rss_kb: Optional[int] = None
    monitor_cpu_seconds = 0.0

    try:
        with stdout_path.open("wb") as stdout_log, stderr_path.open("wb") as stderr_log:
            process = subprocess.Popen(
                limited_argv(argv, cpu_seconds, memory_mb),
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            assert process.stdout is not None and process.stderr is not None
            threads = [
                threading.Thread(
                    target=_pipe_reader,
                    args=("stdout", process.stdout, stdout_log, events, start),
                    daemon=True,
                ),
                threading.Thread(
                    target=_pipe_reader,
                    args=("stderr", process.stderr, stderr_log, events, start),
                    daemon=True,
                ),
            ]
            for thread in threads:
                thread.start()

            closed_streams = 0
            term_deadline: Optional[float] = None
            next_rss_sample = start
            while closed_streams < 2 or process.poll() is None:
                now = time.monotonic()
                if now >= next_rss_sample and process.poll() is None:
                    monitor_before = (
                        resource.getrusage(resource.RUSAGE_CHILDREN)
                        if measure_child_cpu
                        else None
                    )
                    rss = process_tree_rss_kb(process.pid)
                    if monitor_before is not None:
                        monitor_after = resource.getrusage(resource.RUSAGE_CHILDREN)
                        monitor_cpu_seconds += (
                            monitor_after.ru_utime
                            + monitor_after.ru_stime
                            - monitor_before.ru_utime
                            - monitor_before.ru_stime
                        )
                    if rss is not None:
                        peak_rss_kb = max(peak_rss_kb or 0, rss)
                    next_rss_sample = now + 0.25
                if (
                    not cancelled
                    and stop_event is not None
                    and stop_event.is_set()
                    and process.poll() is None
                ):
                    cancelled = True
                    termination_signal = "SIGTERM"
                    terminate_group(process, signal.SIGTERM)
                    term_deadline = now + max(0.0, grace_seconds)
                if not timed_out and wall_seconds > 0 and now - start >= wall_seconds:
                    timed_out = True
                    termination_signal = "SIGTERM"
                    terminate_group(process, signal.SIGTERM)
                    term_deadline = now + max(0.0, grace_seconds)
                if (
                    (timed_out or cancelled)
                    and process.poll() is None
                    and term_deadline is not None
                    and now >= term_deadline
                ):
                    killed = True
                    termination_signal = "SIGKILL"
                    terminate_group(process, signal.SIGKILL)
                    term_deadline = None
                try:
                    stream, elapsed, line = events.get(timeout=0.02)
                except queue.Empty:
                    continue
                if line is None:
                    closed_streams += 1
                else:
                    observer.observe(stream, line, elapsed)

            returncode = process.wait()
            for thread in threads:
                thread.join(timeout=1)
            while True:
                try:
                    stream, elapsed, line = events.get_nowait()
                except queue.Empty:
                    break
                if line is not None:
                    observer.observe(stream, line, elapsed)
    except FileNotFoundError as error:
        elapsed = time.monotonic() - start
        stderr_path.write_text(str(error) + "\n", encoding="utf-8")
        return {
            **observer.summary(),
            "returncode": None,
            "elapsed_wall_seconds": round(elapsed, 6),
            "cpu_seconds": 0.0,
            "peak_rss_kb": None,
            "timed_out": False,
            "cancelled": False,
            "killed": False,
            "termination_signal": None,
            "execution_error": str(error),
        }

    elapsed = time.monotonic() - start
    child_cpu: Optional[float] = None
    if usage_before is not None:
        usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        child_cpu = max(
            0.0,
            (usage_after.ru_utime + usage_after.ru_stime)
            - (usage_before.ru_utime + usage_before.ru_stime)
            - monitor_cpu_seconds,
        )
    return {
        **observer.summary(),
        "returncode": returncode,
        "elapsed_wall_seconds": round(elapsed, 6),
        "cpu_seconds": round(child_cpu, 6) if child_cpu is not None else None,
        "peak_rss_kb": peak_rss_kb,
        "timed_out": timed_out,
        "cancelled": cancelled,
        "killed": killed,
        "termination_signal": termination_signal,
        "execution_error": None,
    }


def validate_solution(
    checker_argv: Sequence[str],
    solver_stdout: Path,
    checker_log: Path,
    timeout_seconds: float,
    expected_objective: Optional[int] = None,
    stop_event: Optional[threading.Event] = None,
) -> dict[str, Any]:
    """Validate the final XCSP instantiation using ACE SolutionChecker."""
    start = time.monotonic()
    try:
        checker_log.parent.mkdir(parents=True, exist_ok=True)
        timed_out = False
        with (
            solver_stdout.open("rb") as source,
            checker_log.open("wb") as checker_output,
        ):
            process = subprocess.Popen(
                list(checker_argv),
                cwd=REPO_ROOT,
                stdin=source,
                stdout=checker_output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = (
                start + timeout_seconds if timeout_seconds > 0 else None
            )
            while process.poll() is None:
                if stop_event is not None and stop_event.is_set():
                    terminate_group(process, signal.SIGTERM)
                    try:
                        process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        terminate_group(process, signal.SIGKILL)
                        process.wait()
                    raise RunCancelled("campaign stopped during solution validation")
                if deadline is not None and time.monotonic() >= deadline:
                    timed_out = True
                    terminate_group(process, signal.SIGTERM)
                    try:
                        process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        terminate_group(process, signal.SIGKILL)
                        process.wait()
                    break
                time.sleep(0.02)
            returncode = process.wait()

        output = checker_log.read_text(encoding="utf-8", errors="replace")
        if timed_out:
            return {
                "attempted": True,
                "valid": None,
                "reason": "checker-timeout",
                "returncode": returncode,
                "elapsed_seconds": round(time.monotonic() - start, 6),
                "log": str(checker_log),
                "output_tail": output[-4096:],
            }
        accepted = CHECK_OK_RE.search(output)
        reported_objective = (
            int(accepted.group(1))
            if accepted is not None and accepted.group(1) is not None
            else None
        )
        checker_accepted = (
            returncode == 0
            and accepted is not None
            and "ERROR:" not in output
        )
        objective_matches = (
            expected_objective is None
            or reported_objective == expected_objective
        )
        if CHECKER_OOM_RE.search(output):
            valid: Optional[bool] = None
            reason = "checker-oom"
        elif returncode == -signal.SIGKILL:
            valid = None
            reason = "checker-killed"
        elif checker_accepted and expected_objective is not None and reported_objective is None:
            valid = None
            reason = "checker-missing-objective"
        elif checker_accepted and objective_matches:
            valid = True
            reason = "accepted"
        elif checker_accepted and not objective_matches:
            valid = False
            reason = "objective-mismatch"
        elif "ERROR:" in output:
            valid = False
            reason = "checker-rejected"
        else:
            # A checker crash, OOM, signal, or malformed empty response says
            # nothing about the candidate. Keep it unverified without
            # invalidating the solver's complete model family.
            valid = None
            reason = "checker-error"
        termination_signal = None
        if returncode is not None and returncode < 0:
            try:
                termination_signal = signal.Signals(-returncode).name
            except ValueError:
                termination_signal = f"SIGNAL_{-returncode}"
        return {
            "attempted": True,
            "valid": valid,
            "reason": reason,
            "returncode": returncode,
            "termination_signal": termination_signal,
            "expected_objective": expected_objective,
            "reported_objective": reported_objective,
            "elapsed_seconds": round(time.monotonic() - start, 6),
            "log": str(checker_log),
            "output_tail": output[-4096:],
        }
    except RunCancelled:
        raise
    except OSError as error:
        try:
            checker_log.parent.mkdir(parents=True, exist_ok=True)
            checker_log.write_text(str(error) + "\n", encoding="utf-8")
        except OSError:
            pass
        return {
            "attempted": True,
            "valid": None,
            "reason": "checker-error",
            "returncode": None,
            "elapsed_seconds": round(time.monotonic() - start, 6),
            "log": str(checker_log),
            "output_tail": str(error),
        }


