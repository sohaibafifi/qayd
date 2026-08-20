"""Shared primitives for reproducible optimization campaigns.

The older SAT/PB runner intentionally has a compact CSV schema.  Routing and
scheduling need richer records: vector objectives, certified bounds, seeds,
memory, solver provenance, and independent feasibility checks.  This module is
standard-library only so every adapter can emit the same JSON contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import threading
import time
from typing import Any, Iterable, Optional, Sequence


SCHEMA_VERSION = 1
FEASIBLE_STATUSES = {"SAT", "SATISFIABLE", "FEASIBLE", "OPTIMAL", "OPTIMUM"}
_RSS_SAMPLE_INTERVAL_SECONDS = 0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(root: Path, *arguments: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=root, text=True, capture_output=True,
            check=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def git_bytes(root: Path, *arguments: str) -> Optional[bytes]:
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=root, capture_output=True,
            check=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout


def source_tree_sha256(root: Path) -> Optional[str]:
    """Fingerprint HEAD plus every tracked and untracked working-tree edit."""
    commit = git_value(root, "rev-parse", "HEAD")
    diff = git_bytes(root, "diff", "--binary", "--no-ext-diff", "HEAD", "--")
    untracked = git_bytes(root, "ls-files", "--others", "--exclude-standard", "-z")
    if commit is None or diff is None or untracked is None:
        return None
    digest = hashlib.sha256()

    def add(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    add(b"qayd-source-tree/v1")
    add(commit.encode("ascii"))
    add(diff)
    for encoded_path in sorted(filter(None, untracked.split(b"\0"))):
        path = root / os.fsdecode(encoded_path)
        try:
            payload = path.read_bytes()
        except OSError:
            return None
        add(encoded_path)
        add(payload)
    return digest.hexdigest()


def machine_provenance(root: Path) -> dict[str, Any]:
    commit = git_value(root, "rev-parse", "HEAD")
    dirty = bool(git_value(root, "status", "--porcelain"))
    return {
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": commit,
        "dirty": dirty,
        "source_tree_sha256": source_tree_sha256(root),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
    }


def normalize_status(value: object) -> str:
    text = str(value or "UNKNOWN").upper()
    if "OPTIMAL" in text or "OPTIMUM" in text:
        return "OPTIMAL"
    if "INFEASIBLE" in text or "UNSAT" in text:
        return "UNSAT"
    if "FEASIBLE" in text or "SATISFIABLE" in text or text == "SAT":
        return "SATISFIABLE"
    if "ERROR" in text or "INVALID" in text:
        return "ERROR"
    return "UNKNOWN"


def objective_gap(primal: Optional[float], dual: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    if primal is None or dual is None or not math.isfinite(primal) or not math.isfinite(dual):
        return None, None
    absolute = max(0.0, primal - dual)
    relative = absolute / max(1.0, abs(primal), abs(dual))
    return absolute, relative


def complete_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize an adapter result without inventing a certificate."""
    result = dict(record)
    result["schema_version"] = SCHEMA_VERSION
    result["status"] = normalize_status(result.get("status"))
    objectives = result.get("objectives")
    result["objectives"] = list(objectives) if isinstance(objectives, (list, tuple)) else []
    dual = result.get("dual_bound")
    primal = result["objectives"][0] if result["objectives"] else None
    absolute, relative = objective_gap(primal, dual)
    if result.get("absolute_gap") is None:
        result["absolute_gap"] = absolute
    if result.get("relative_gap") is None:
        result["relative_gap"] = relative
    result.setdefault("bound_method", None)
    result.setdefault("verified", False)
    return result


def json_record_from_output(stdout: str) -> dict[str, Any]:
    """Read the last JSON object printed by a solver wrapper."""
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("solver did not emit a JSON object")


def _process_table() -> list[tuple[int, int, int]]:
    """Return pid, parent pid, RSS KiB for all visible processes."""
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,rss="], text=True, capture_output=True,
            check=True, timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            rows.append(tuple(map(int, fields)))
        except ValueError:
            continue
    return rows


def process_tree_rss_kib(root_pid: int) -> int:
    rows = _process_table()
    children: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for pid, parent, value in rows:
        children.setdefault(parent, []).append(pid)
        rss[pid] = value
    stack = [root_pid]
    descendants = set()
    while stack:
        pid = stack.pop()
        if pid in descendants:
            continue
        descendants.add(pid)
        stack.extend(children.get(pid, ()))
    return sum(rss.get(pid, 0) for pid in descendants)


def run_measured(
    argv: Sequence[str], *, timeout: float, cwd: Path, memory_limit_mb: int = 0,
) -> dict[str, Any]:
    """Run one command with a wall limit and sampled process-tree RSS.

    The external timeout includes model parsing and result verification.  The
    solver's internal budget is normally smaller and is passed separately in
    ``argv`` by the campaign driver.  Process-tree RSS is sampled immediately,
    every 500 ms, and once after exit.  Waiting is clipped to the wall deadline
    so the lower-overhead RSS cadence does not reduce timeout precision.
    """
    preexec_fn = None
    if memory_limit_mb and os.name == "posix":
        import resource

        def apply_limit() -> None:
            limit = memory_limit_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            except (OSError, ValueError):
                pass

        preexec_fn = apply_limit

    started = time.perf_counter()
    process = subprocess.Popen(
        list(argv), cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=(os.name == "posix"),
        preexec_fn=preexec_fn,
    )
    captured = {"stdout": "", "stderr": ""}

    def drain(name: str, stream: Any) -> None:
        try:
            captured[name] = stream.read()
        finally:
            stream.close()

    readers = [
        threading.Thread(target=drain, args=("stdout", process.stdout), daemon=True),
        threading.Thread(target=drain, args=("stderr", process.stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()
    peak_rss = process_tree_rss_kib(process.pid)
    timed_out = False
    deadline = started + timeout
    next_rss_sample = time.perf_counter() + _RSS_SAMPLE_INTERVAL_SECONDS
    while process.poll() is None:
        now = time.perf_counter()
        if now >= deadline:
            timed_out = True
            if os.name == "posix":
                import signal
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            break
        wait_seconds = min(deadline, next_rss_sample) - now
        try:
            process.wait(timeout=max(0.0, wait_seconds))
        except subprocess.TimeoutExpired:
            pass
        if process.poll() is not None:
            break
        now = time.perf_counter()
        if now < deadline:
            peak_rss = max(peak_rss, process_tree_rss_kib(process.pid))
            next_rss_sample = now + _RSS_SAMPLE_INTERVAL_SECONDS
    process.wait()
    for reader in readers:
        reader.join()
    peak_rss = max(peak_rss, process_tree_rss_kib(process.pid))
    return {
        "argv": list(argv),
        "wall_seconds": time.perf_counter() - started,
        "peak_memory_mb": peak_rss / 1024.0,
        "return_code": process.returncode,
        "timed_out": timed_out,
        "stdout": captured["stdout"],
        "stderr": captured["stderr"],
    }


def percentile(values: Iterable[float], quantile: float) -> Optional[float]:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def lex_key(objectives: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in objectives)
