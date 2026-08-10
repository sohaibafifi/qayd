#!/usr/bin/env python3
"""Run reproducible, validated XCSP25 FAST COP experiments."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import platform
import queue
import re
import resource
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

try:
    from .manifest import REPO_ROOT, load_manifest, open_instance, sha256_file
except ImportError:
    from manifest import REPO_ROOT, load_manifest, open_instance, sha256_file


SOLVER_SCHEMA = "qayd.fastcop.solvers/v1"
RESULT_SCHEMA = "qayd.fastcop.result/v1"
DEFAULT_MANIFEST = Path(__file__).with_name("manifest.v1.json")
DEFAULT_SOLVERS = Path(__file__).with_name("solvers.v1.json")
DEFAULT_RESULTS = Path(__file__).with_name("results") / "results.jsonl"
LIMIT_LAUNCHER = Path(__file__).with_name("exec_limit.py").resolve()
STATUS_RE = re.compile(r"^\s*s\s+(.+?)\s*$", re.IGNORECASE)
OBJECTIVE_RE = re.compile(r"^\s*o\s+(-?\d+)(?:\s|$)", re.IGNORECASE)
CHECK_OK_RE = re.compile(r"^OK(?:[ \t]+(-?\d+))?[ \t]*$", re.MULTILINE)


class HarnessError(RuntimeError):
    """A reproducibility or execution invariant was violated."""


class RunCancelled(RuntimeError):
    """An in-flight run was stopped because the campaign is terminating."""


@dataclass(frozen=True)
class RunTask:
    position: int
    solver_name: str
    solver: dict[str, Any]
    instance_item: dict[str, Any]
    run_key: str


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def harness_sha256() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__).resolve(), LIMIT_LAUNCHER):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_json(path: Path) -> tuple[dict, str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise HarnessError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise HarnessError(f"expected a JSON object in {path}")
    return value, sha256_bytes(raw)


def resolve_config_path(config_path: Path, relative: str) -> Path:
    path = (config_path.parent / relative).resolve()
    if not path.is_file():
        raise HarnessError(f"solver artifact does not exist: {path}")
    return path


def load_solvers(path: Path) -> tuple[dict, str]:
    config, digest = load_json(path)
    if config.get("schema") != SOLVER_SCHEMA:
        raise HarnessError(f"unsupported solver schema in {path}")
    solvers = config.get("solvers")
    if not isinstance(solvers, dict) or not solvers:
        raise HarnessError("solver configuration has no solvers")
    for name, solver in solvers.items():
        if (
            not isinstance(solver, dict)
            or not isinstance(solver.get("argv"), list)
            or not solver["argv"]
        ):
            raise HarnessError(f"invalid solver entry: {name}")
        if not all(isinstance(token, str) for token in solver["argv"]):
            raise HarnessError(f"non-string argv token for solver {name}")
        artifact = resolve_config_path(path, solver.get("artifact", ""))
        observed = sha256_file(artifact)
        expected = solver.get("expected_sha256")
        if expected is not None and observed != expected:
            raise HarnessError(
                f"artifact hash mismatch for {name}: expected {expected}, got {observed}"
            )
    checker = config.get("checker")
    if (
        not isinstance(checker, dict)
        or not isinstance(checker.get("argv"), list)
        or not checker["argv"]
    ):
        raise HarnessError("solver configuration has no checker")
    checker_artifact = resolve_config_path(path, checker.get("artifact", ""))
    expected = checker.get("expected_sha256")
    observed = sha256_file(checker_artifact)
    if expected is not None and observed != expected:
        raise HarnessError(
            f"checker hash mismatch: expected {expected}, got {observed}"
        )
    return config, digest


def display_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else format(value, ".6g")


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def expand_argv(template: Sequence[str], replacements: dict[str, str]) -> list[str]:
    argv = []
    marker = re.compile(r"\{([a-z_]+)\}")
    for token in template:
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in replacements:
                raise HarnessError(f"unknown command placeholder {{{name}}}")
            return replacements[name]

        expanded = marker.sub(replace, token)
        if "{" in expanded or "}" in expanded:
            raise HarnessError(f"malformed command token: {token}")
        argv.append(expanded)
    return argv


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
    if (proc_root / str(root_pid) / "status").is_file():
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
            except (FileNotFoundError, PermissionError, ValueError):
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
                "valid": False,
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
            reported_objective is None
            or expected_objective is None
            or reported_objective == expected_objective
        )
        valid = checker_accepted and objective_matches
        if valid:
            reason = "accepted"
        elif checker_accepted and not objective_matches:
            reason = "objective-mismatch"
        else:
            reason = "checker-rejected"
        return {
            "attempted": True,
            "valid": valid,
            "reason": reason,
            "returncode": returncode,
            "expected_objective": expected_objective,
            "reported_objective": reported_objective,
            "elapsed_seconds": round(time.monotonic() - start, 6),
            "log": str(checker_log),
            "output_tail": output[-4096:],
        }
    except RunCancelled:
        raise
    except OSError as error:
        checker_log.write_text(str(error) + "\n", encoding="utf-8")
        return {
            "attempted": True,
            "valid": False,
            "reason": "checker-error",
            "returncode": None,
            "elapsed_seconds": round(time.monotonic() - start, 6),
            "log": str(checker_log),
            "output_tail": str(error),
        }


def materialize(
    instance: Path,
    directory: Path,
    stop_event: Optional[threading.Event] = None,
) -> Path:
    destination = directory / "instance.xml"
    with open_instance(instance) as source, destination.open("wb") as target:
        while True:
            if stop_event is not None and stop_event.is_set():
                raise RunCancelled("campaign stopped during instance materialization")
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            target.write(chunk)
    return destination


def executable_provenance(argv0: str) -> dict[str, Any]:
    resolved = shutil.which(argv0)
    if resolved is None:
        candidate = Path(argv0)
        resolved = str(candidate.resolve()) if candidate.exists() else None
    if resolved is None:
        return {"path": None, "sha256": None}
    path = Path(resolved).resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def git_revision() -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def make_replacements(
    artifact: Path,
    ace_artifact: Path,
    instance: Path,
    cpu_seconds: float,
    memory_mb: int,
    seed: int,
) -> dict[str, str]:
    java_reserve_mb = min(4096, max(512, memory_mb // 8))
    java_heap_mb = max(128, memory_mb - java_reserve_mb)
    checker_heap_mb = max(128, min(4096, memory_mb // 4))
    return {
        "artifact": str(artifact),
        "ace_artifact": str(ace_artifact),
        "instance": str(instance),
        "cpu_seconds": display_number(cpu_seconds),
        "memory_mb": str(memory_mb),
        "java_heap_mb": str(java_heap_mb),
        "checker_heap_mb": str(checker_heap_mb),
        "seed": str(seed),
    }


def load_result_records(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise HarnessError(f"invalid JSONL at {path}:{number}: {error}") from error
            if not isinstance(record, dict):
                raise HarnessError(f"result at {path}:{number} is not an object")
            if record.get("schema") != RESULT_SCHEMA:
                raise HarnessError(f"unsupported result schema at {path}:{number}")
            key = record.get("run_key")
            if not isinstance(key, str):
                raise HarnessError(f"missing run_key at {path}:{number}")
            if not isinstance(record.get("solver"), str) or not isinstance(
                record.get("instance"), str
            ):
                raise HarnessError(
                    f"missing solver or instance at {path}:{number}"
                )
            records.append(record)
    return records


def load_completed_keys(path: Path) -> set[str]:
    return {record["run_key"] for record in load_result_records(path)}


def ensure_output_matches_plan(
    output: Path,
    tasks: Sequence[RunTask],
    existing_records: Sequence[dict[str, Any]],
) -> None:
    planned = {
        (task.solver_name, task.instance_item["id"]): task.run_key
        for task in tasks
    }
    observed_order = []
    seen_keys = set()
    for record in existing_records:
        pair = (record["solver"], record["instance"])
        expected = planned.get(pair)
        if expected is None:
            raise HarnessError(
                f"{output} contains {pair[0]} {pair[1]}, which is outside "
                "the current selection; choose a new output"
            )
        if record["run_key"] != expected:
            raise HarnessError(
                f"{output} contains an incompatible run for {pair[0]} "
                f"{pair[1]}; keep the same limits, jobs and artifacts or "
                "choose a new output"
            )
        if record["run_key"] not in seen_keys:
            seen_keys.add(record["run_key"])
            observed_order.append(record["run_key"])

    expected_order = [task.run_key for task in tasks[: len(observed_order)]]
    if observed_order != expected_order:
        raise HarnessError(
            f"{output} is not a canonical prefix of the current run plan; "
            "resume with the original selection or choose a new output"
        )


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as target:
        target.write(canonical_json(record) + "\n")
        target.flush()
        os.fsync(target.fileno())


def acquire_lock(lock_path: Path, target: str):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise HarnessError(
            f"another FAST COP campaign is using {target}"
        ) from error
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    return handle


def acquire_campaign_lock(output: Path):
    return acquire_lock(Path(f"{output}.lock"), f"output {output}")


def acquire_log_lock(log_directory: Path):
    return acquire_lock(
        log_directory / ".fastcop.lock",
        f"log directory {log_directory}",
    )


def release_campaign_lock(handle) -> None:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def make_run_identity(
    solver_name: str,
    solver_config_hash: str,
    artifact_hash: str,
    checker_hash: str,
    instance_id: str,
    instance_hash: str,
    cpu_seconds: float,
    wall_seconds: float,
    memory_mb: int,
    seed: int,
    grace_seconds: float,
    checker_timeout: float,
    check_solution: bool,
    parallel_jobs: int,
    harness_hash: str,
    manifest_hash: str,
) -> dict:
    return {
        "solver": solver_name,
        "solver_config_sha256": solver_config_hash,
        "artifact_sha256": artifact_hash,
        "checker_artifact_sha256": checker_hash,
        "instance_id": instance_id,
        "instance_sha256": instance_hash,
        "cpu_limit_seconds": cpu_seconds,
        "wall_limit_seconds": wall_seconds,
        "memory_limit_mb": memory_mb,
        "seed": seed,
        "grace_seconds": grace_seconds,
        "checker_timeout_seconds": checker_timeout,
        "check_solution": check_solution,
        "parallel_jobs": parallel_jobs,
        "harness_sha256": harness_hash,
        "manifest_sha256": manifest_hash,
    }


def run_one(
    solver_name: str,
    solver: dict,
    solver_config_path: Path,
    solver_config_hash: str,
    checker: dict,
    instance_item: dict,
    manifest_hash: str,
    cpu_seconds: float,
    wall_seconds: float,
    memory_mb: int,
    seed: int,
    grace_seconds: float,
    checker_timeout: float,
    check_solution: bool,
    parallel_jobs: int,
    harness_hash: str,
    log_directory: Path,
    stop_event: Optional[threading.Event] = None,
) -> dict:
    if stop_event is not None and stop_event.is_set():
        raise RunCancelled("campaign stopped before run preparation")
    original_instance = (REPO_ROOT / instance_item["path"]).resolve()
    observed_instance_hash = sha256_file(original_instance)
    if observed_instance_hash != instance_item["sha256"]:
        raise HarnessError(f"instance changed: {instance_item['id']}")
    artifact = resolve_config_path(solver_config_path, solver["artifact"])
    artifact_hash = sha256_file(artifact)
    checker_artifact = resolve_config_path(solver_config_path, checker["artifact"])
    checker_hash = sha256_file(checker_artifact)

    identity = make_run_identity(
        solver_name,
        solver_config_hash,
        artifact_hash,
        checker_hash,
        instance_item["id"],
        observed_instance_hash,
        cpu_seconds,
        wall_seconds,
        memory_mb,
        seed,
        grace_seconds,
        checker_timeout,
        check_solution,
        parallel_jobs,
        harness_hash,
        manifest_hash,
    )
    run_key = sha256_bytes(canonical_json(identity).encode("utf-8"))
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", instance_item["id"])
    base = log_directory / solver_name / f"{safe_id}-{run_key}"
    stdout_path = Path(f"{base}.stdout.log")
    stderr_path = Path(f"{base}.stderr.log")
    checker_log = Path(f"{base}.checker.log")
    try:
        checker_log.unlink()
    except FileNotFoundError:
        pass

    preparation_start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="qayd-fastcop-instance-") as scratch:
        materialized = materialize(original_instance, Path(scratch), stop_event)
        preparation_seconds = time.monotonic() - preparation_start
        replacements = make_replacements(
            artifact, checker_artifact, materialized, cpu_seconds, memory_mb, seed
        )
        argv = expand_argv(solver["argv"], replacements)
        execution = execute_streaming(
            argv,
            instance_item["objective_sense"],
            wall_seconds,
            cpu_seconds,
            memory_mb,
            grace_seconds,
            stdout_path,
            stderr_path,
            measure_child_cpu=parallel_jobs == 1,
            stop_event=stop_event,
        )
        if execution["cancelled"]:
            raise RunCancelled(
                f"campaign stopped while running {solver_name} {instance_item['id']}"
            )
        best = execution["best_incumbent"]
        if not check_solution:
            validation = {
                "attempted": False,
                "valid": None,
                "reason": "disabled",
                "returncode": None,
            }
        elif execution["claimed_status"] == "UNSAT":
            validation = {
                "attempted": False,
                "valid": None,
                "reason": "unsat-proof-not-checkable",
                "returncode": None,
            }
        elif best is None:
            validation = {
                "attempted": False,
                "valid": None,
                "reason": "no-incumbent",
                "returncode": None,
            }
        elif not execution["has_solution"]:
            validation = {
                "attempted": False,
                "valid": None,
                "reason": "missing-final-solution",
                "returncode": None,
            }
        else:
            checker_argv = expand_argv(checker["argv"], replacements)
            validation = validate_solution(
                checker_argv,
                stdout_path,
                checker_log,
                checker_timeout,
                expected_objective=best["value"],
                stop_event=stop_event,
            )

    status = execution["claimed_status"]
    proof = execution["claimed_proof"]
    invalid = validation.get("valid") is False
    feasible_eligible = best is not None and (
        validation.get("valid") is True or not check_solution
    )
    if invalid:
        status = "INVALID"
        proof = False
    elif best is not None and not feasible_eligible:
        status = "UNKNOWN"
        proof = False
    elif execution["execution_error"] is not None:
        status = "ERROR"
        proof = False
    elif execution["returncode"] not in (0, None) and status == "UNKNOWN":
        status = "ERROR"

    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    launcher = executable_provenance(argv[0])
    return {
        "schema": RESULT_SCHEMA,
        "run_key": run_key,
        "solver": solver_name,
        "solver_version": solver.get("version"),
        "instance": instance_item["id"],
        "family": instance_item["family"],
        "family_group": instance_item.get("family_group", instance_item["family"]),
        "objective_sense": instance_item["objective_sense"],
        "instance_path": instance_item["path"],
        "instance_sha256": observed_instance_hash,
        "seed": seed,
        "limits": {
            "cpu_seconds": cpu_seconds,
            "wall_seconds": wall_seconds,
            "memory_mb": memory_mb,
            "grace_seconds": grace_seconds,
            "parallel_jobs": parallel_jobs,
        },
        "command": argv,
        "status": status,
        "proof": proof,
        "proof_elapsed_seconds": execution["proof_elapsed_seconds"],
        "incumbents": execution["incumbents"],
        "first_incumbent": execution["first_incumbent"],
        "best_incumbent": execution["best_incumbent"] if feasible_eligible else None,
        "raw_best_incumbent": execution["best_incumbent"],
        "validation": validation,
        "invalid": invalid,
        "timed_out": execution["timed_out"],
        "killed": execution["killed"],
        "termination_signal": execution["termination_signal"],
        "returncode": execution["returncode"],
        "execution_error": execution["execution_error"],
        "elapsed_wall_seconds": execution["elapsed_wall_seconds"],
        "cpu_seconds": execution["cpu_seconds"],
        "preparation_seconds": round(preparation_seconds, 6),
        "peak_rss_kb": execution["peak_rss_kb"],
        "logs": {
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "checker": str(checker_log) if checker_log.exists() else None,
        },
        "provenance": {
            "recorded_at": started_at,
            "manifest_sha256": manifest_hash,
            "solver_config_sha256": solver_config_hash,
            "artifact": str(artifact),
            "artifact_sha256": artifact_hash,
            "checker_artifact": str(checker_artifact),
            "checker_sha256": checker_hash,
            "launcher": launcher,
            "harness_sha256": harness_hash,
            "cpu_accounting": (
                "rusage-children" if parallel_jobs == 1
                else "unavailable-concurrent-runs"
            ),
            "git_revision": git_revision(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }


def select_instances(
    manifest: dict,
    families: list[str],
    pattern: Optional[str],
    limit: int,
    per_family: int = 0,
) -> list[dict]:
    instances = manifest["instances"]
    if families:
        requested = set(families)
        instances = [
            item for item in instances
            if item["family"] in requested or item.get("family_group") in requested
        ]
    if pattern:
        matcher = re.compile(pattern)
        instances = [item for item in instances if matcher.search(item["id"])]
    if per_family:
        counts: dict[str, int] = {}
        stratified = []
        for item in instances:
            family = item["family"]
            count = counts.get(family, 0)
            if count >= per_family:
                continue
            counts[family] = count + 1
            stratified.append(item)
        instances = stratified
    if limit:
        instances = instances[:limit]
    return instances


def ordered_run_results(
    tasks: Sequence[RunTask],
    completed_keys: set[str],
    jobs: int,
    worker: Callable[[RunTask], dict[str, Any]],
    stop_event: threading.Event,
) -> Iterator[tuple[RunTask, Optional[dict[str, Any]]]]:
    """Run pending tasks concurrently and yield every plan entry in order."""
    if jobs <= 0:
        raise HarnessError("--jobs must be positive")

    if jobs == 1:
        try:
            for task in tasks:
                if task.run_key in completed_keys:
                    yield task, None
                else:
                    yield task, worker(task)
        except BaseException:
            stop_event.set()
            raise
        return

    executor = ThreadPoolExecutor(
        max_workers=jobs,
        thread_name_prefix="qayd-fastcop",
    )
    pending = iter(
        task for task in tasks if task.run_key not in completed_keys
    )
    futures: dict[int, Future[dict[str, Any]]] = {}

    def fill_window() -> None:
        while len(futures) < 2 * jobs:
            try:
                task = next(pending)
            except StopIteration:
                return
            futures[task.position] = executor.submit(worker, task)

    fill_window()
    try:
        for task in tasks:
            if task.run_key in completed_keys:
                yield task, None
                continue
            future = futures.pop(task.position)
            record = future.result()
            fill_window()
            yield task, record
    except BaseException:
        stop_event.set()
        for future in futures.values():
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--solvers", type=Path, default=DEFAULT_SOLVERS)
    parser.add_argument("--solver", action="append", dest="solver_names")
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--cpu-limit", type=positive_float, default=180.0)
    parser.add_argument("--wall-limit", type=positive_float, default=270.0)
    parser.add_argument("--memory-mb", type=positive_int, default=65536)
    parser.add_argument(
        "--jobs",
        type=positive_int,
        default=1,
        help="number of solver runs executed concurrently (memory limit is per run)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grace", type=nonnegative_float, default=1.0)
    parser.add_argument("--checker-timeout", type=positive_float, default=120.0)
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--instance", help="regular expression selecting instance ids")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=nonnegative_int, default=0)
    selection.add_argument(
        "--per-family",
        type=nonnegative_int,
        default=0,
        help="deterministically select the first N instances of every family",
    )
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="do not skip completed run keys")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    lock_handles = []
    try:
        manifest_path = args.manifest.resolve()
        solver_path = args.solvers.resolve()
        manifest = load_manifest(manifest_path)
        manifest_hash = sha256_file(manifest_path)
        solver_config, solver_hash = load_solvers(solver_path)
        configured = solver_config["solvers"]
        solver_names = args.solver_names or list(configured)
        unknown = [name for name in solver_names if name not in configured]
        if unknown:
            raise HarnessError(f"unknown solver(s): {', '.join(unknown)}")
        if len(set(solver_names)) != len(solver_names):
            raise HarnessError("--solver entries must be unique")
        instances = select_instances(
            manifest, args.family, args.instance, args.limit, args.per_family
        )
        if not instances:
            raise HarnessError("instance selection is empty")
        output = args.output.resolve()
        log_directory = (
            args.log_dir.resolve() if args.log_dir
            else output.parent / (output.stem + "-logs")
        )
        if not args.dry_run:
            lock_handles.append(acquire_campaign_lock(output))
            lock_handles.append(acquire_log_lock(log_directory))
        existing_records = load_result_records(output)
        completed = (
            set()
            if args.rerun
            else {record["run_key"] for record in existing_records}
        )
        total = len(solver_names) * len(instances)
        skipped = 0
        checker = solver_config["checker"]
        checker_artifact = resolve_config_path(solver_path, checker["artifact"])
        checker_hash = sha256_file(checker_artifact)
        harness_hash = harness_sha256()
        tasks = []
        for solver_name in solver_names:
            solver = configured[solver_name]
            artifact = resolve_config_path(solver_path, solver["artifact"])
            artifact_hash = sha256_file(artifact)
            for item in instances:
                position = len(tasks) + 1
                identity = make_run_identity(
                    solver_name,
                    solver_hash,
                    artifact_hash,
                    checker_hash,
                    item["id"],
                    item["sha256"],
                    args.cpu_limit,
                    args.wall_limit,
                    args.memory_mb,
                    args.seed,
                    args.grace,
                    args.checker_timeout,
                    not args.no_check,
                    args.jobs,
                    harness_hash,
                    manifest_hash,
                )
                key = sha256_bytes(canonical_json(identity).encode("utf-8"))
                tasks.append(
                    RunTask(
                        position=position,
                        solver_name=solver_name,
                        solver=solver,
                        instance_item=item,
                        run_key=key,
                    )
                )

        run_keys = [task.run_key for task in tasks]
        if len(set(run_keys)) != len(run_keys):
            raise HarnessError("run plan contains duplicate identities")
        ensure_output_matches_plan(output, tasks, existing_records)

        if args.dry_run:
            for task in tasks:
                resumed = task.run_key in completed
                skipped += int(resumed)
                state = "resumed" if resumed else "pending"
                print(
                    f"[{task.position}/{total}] {task.solver_name} "
                    f"{task.instance_item['id']}: {state}",
                    flush=True,
                )
            print(f"results={output} completed={total - skipped} resumed={skipped}")
            return 0

        if args.jobs > 1:
            print(
                f"parallel jobs={args.jobs}; per-run memory={args.memory_mb} MiB; "
                f"aggregate configured memory={args.jobs * args.memory_mb} MiB",
                flush=True,
            )

        stop_event = threading.Event()

        def execute_task(task: RunTask) -> dict[str, Any]:
            try:
                record = run_one(
                    task.solver_name,
                    task.solver,
                    solver_path,
                    solver_hash,
                    checker,
                    task.instance_item,
                    manifest_hash,
                    args.cpu_limit,
                    args.wall_limit,
                    args.memory_mb,
                    args.seed,
                    args.grace,
                    args.checker_timeout,
                    not args.no_check,
                    args.jobs,
                    harness_hash,
                    log_directory,
                    stop_event,
                )
            except RunCancelled:
                raise
            except (HarnessError, OSError) as error:
                raise HarnessError(
                    f"{task.solver_name} {task.instance_item['id']}: {error}"
                ) from error
            if record["run_key"] != task.run_key:
                raise HarnessError(
                    f"run identity changed while executing {task.solver_name} "
                    f"{task.instance_item['id']}"
                )
            return record

        for task, record in ordered_run_results(
            tasks,
            completed,
            args.jobs,
            execute_task,
            stop_event,
        ):
            if record is None:
                skipped += 1
                print(
                    f"[{task.position}/{total}] {task.solver_name} "
                    f"{task.instance_item['id']}: resumed",
                    flush=True,
                )
                continue
            append_record(output, record)
            completed.add(record["run_key"])
            incumbent = record["best_incumbent"]
            objective = f" obj={incumbent['value']}" if incumbent else ""
            validation = record["validation"]["reason"]
            print(
                f"[{task.position}/{total}] {task.solver_name} "
                f"{task.instance_item['id']}: "
                f"{record['status']}{objective} {record['elapsed_wall_seconds']:.2f}s "
                f"validation={validation}",
                flush=True,
            )
        print(f"results={output} completed={total - skipped} resumed={skipped}")
        return 0
    except KeyboardInterrupt:
        print("fastcop run interrupted", file=sys.stderr)
        return 130
    except (HarnessError, OSError, re.error) as error:
        print(f"fastcop run error: {error}", file=sys.stderr)
        return 2
    finally:
        for lock_handle in reversed(lock_handles):
            release_campaign_lock(lock_handle)


if __name__ == "__main__":
    raise SystemExit(main())
