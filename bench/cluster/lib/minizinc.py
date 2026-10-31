"""MiniZinc source catalog and independent solution replay.

The timed run includes flattening with each solver's own mznlib. Replay uses
the original source and a Gecode completion with all exported decision values
fixed, then checks the recomputed objective. It is outside the solve budget.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
from typing import Any


def messages(output: str) -> list[dict[str, Any]]:
    result = []
    for line in output.splitlines():
        try:
            item = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(item, dict):
            result.append(item)
    return result


def summary(output: str) -> dict[str, Any]:
    events = messages(output)
    solutions = [e for e in events if e.get("type") == "solution"]
    statuses = [e.get("status") for e in events if e.get("type") == "status"]
    errors = [e for e in events if e.get("type") == "error"]
    values = solutions[-1].get("output", {}).get("json", {}) if solutions else {}
    objective = values.get("_objective") if isinstance(values, dict) else None
    raw = statuses[-1] if statuses else "UNKNOWN"
    status = {
        "OPTIMAL_SOLUTION": "OPTIMUM", "ALL_SOLUTIONS": "SAT",
        "SATISFIED": "SAT", "UNSATISFIABLE": "UNSAT",
    }.get(raw, "SAT" if solutions else "UNKNOWN")
    if errors:
        status = "ERROR"
    return {"status": status, "objective": objective, "values": values,
            "has_solution": bool(solutions), "errors": errors}


def interface(executable: str, model: Path, data: list[Path], timeout: int) -> dict:
    completed = subprocess.run(
        [executable, "--model-interface-only", str(model), *map(str, data)],
        text=True, capture_output=True, timeout=timeout, check=False,
    )
    events = messages(completed.stdout)
    interfaces = [event for event in events if event.get("type") == "interface"]
    if completed.returncode or len(interfaces) != 1:
        raise ValueError(f"cannot inspect MiniZinc model: {completed.stderr[-1000:]}")
    return interfaces[0]


def literal(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and math.isfinite(value):
        return repr(value)
    if isinstance(value, dict) and set(value) == {"e"}:
        return "'" + str(value["e"]).replace("'", "\\'") + "'"
    if isinstance(value, dict) and set(value) == {"set"}:
        elements = []
        for element in value["set"]:
            if isinstance(element, list) and len(element) == 2:
                elements.append(f"({literal(element[0])}..{literal(element[1])})")
            else:
                elements.append("{" + literal(element) + "}")
        return " union ".join(elements) or "{}"
    raise ValueError("unsupported MiniZinc JSON value for replay")


def flattened(value: list) -> list:
    return [leaf for element in value for leaf in (flattened(element) if isinstance(element, list) else [element])]


def fixed_constraint(name: str, value) -> str:
    identifier = "'" + name.replace("'", "\\'") + "'"
    if isinstance(value, list):
        return f"constraint array1d({identifier}) = [{','.join(literal(v) for v in flattened(value))}];"
    return f"constraint {identifier} = {literal(value)};"


def replay(executable: str, item: dict, observed: dict, root: Path,
           scratch: Path, timeout: int, measure) -> dict:
    """A timeout/error is inconclusive; only a conclusive rejection is invalid."""
    if not observed["has_solution"]:
        return {"valid": None, "reason": "no-witness"}
    values = observed["values"]
    if not isinstance(values, dict):
        return {"valid": None, "reason": "missing-json-assignment"}
    required = set(item["outputs"])
    if not required:
        return {"valid": None, "reason": "no-exported-decision-witness"}
    if not required.issubset(values):
        return {"valid": None, "reason": "incomplete-decision-output"}
    # Auxiliary _objective is produced by MiniZinc, not an input variable.
    fixed = {key: values[key] for key in required}
    assignment = scratch / "replay.mzn"
    try:
        constraints = [fixed_constraint(key, value) for key, value in fixed.items()]
    except ValueError:
        return {"valid": None, "reason": "unsupported-output-type-for-replay"}
    # Equality constraints also work for variables defined by expressions. Data
    # assignments would incorrectly trigger "multiple assignment" for these.
    assignment.write_text("include " + json.dumps(str(root / item["path"])) + ";\n"
                          + "\n".join(constraints) + "\n", encoding="utf-8")
    argv = [executable, "--solver", "org.gecode.gecode", "--time-limit",
            str(timeout * 1000), "--json-stream", "--output-mode", "json",
            "--output-objective", str(assignment),
            *[str(root / path) for path in item.get("data", [])]]
    checked = measure(argv, timeout, scratch / "checker")
    replayed = summary(checked["stdout"])
    if checked["timed_out"] or checked["returncode"] != 0 or replayed["errors"]:
        return {"valid": None, "reason": "checker-error-or-timeout"}
    if replayed["status"] == "UNSAT":
        return {"valid": False, "reason": "original-model-rejected"}
    if not replayed["has_solution"]:
        return {"valid": None, "reason": "checker-no-witness"}
    # Guard against input values accidentally being ignored by a compiler.
    if any(replayed["values"].get(key) != value for key, value in fixed.items()):
        return {"valid": None, "reason": "checker-did-not-fix-output"}
    if item["sense"] in ("min", "max"):
        if observed["objective"] is None or replayed["objective"] is None:
            return {"valid": None, "reason": "missing-objective"}
        if observed["objective"] != replayed["objective"]:
            return {"valid": False, "reason": "objective-mismatch"}
    return {"valid": True, "reason": "original-model-gecode-replay",
            "objective": replayed["objective"]}
