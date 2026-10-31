"""Cluster helpers retained from bench/common/competitive.py; experimental semantics preserved."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional


SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


