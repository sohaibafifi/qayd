#!/usr/bin/env python3
"""Compare Amthal and HiGHS on the same MPS corpus and wall-clock budget."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Iterable

from common.competitive import machine_provenance, run_measured, sha256_file


ROOT = Path(__file__).resolve().parents[1]
FLOAT = r"[-+]?(?:inf|nan|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"


def captured(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else None


def number(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def integer(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def normalize_result_status(raw: str | None, primal: float | None) -> str:
    text = (raw or "UNKNOWN").upper().replace(" ", "_")
    if "OPTIMAL" in text:
        return "OPTIMAL"
    if "INFEASIBLE" in text:
        return "UNSAT"
    if "UNBOUNDED" in text:
        return "UNBOUNDED"
    if "TIME" in text or "LIMIT" in text:
        return "SATISFIABLE" if primal is not None else "UNKNOWN"
    if "NUMERICAL" in text or "ERROR" in text:
        return "ERROR"
    return "SATISFIABLE" if primal is not None else "UNKNOWN"


def parse_amthal(stdout: str, stderr: str) -> dict[str, Any]:
    raw_status = captured(r"^status\s*:\s*(.+)$", stdout)
    objective = number(captured(rf"^objective\s*:\s*({FLOAT})", stdout))
    dual = number(captured(rf"^bound\s*:\s*({FLOAT})", stdout))
    gap = number(captured(rf"^gap\s*:\s*({FLOAT})\s*%", stdout))
    return {
        "raw_status": raw_status,
        "status": normalize_result_status(raw_status, objective),
        "objective": objective,
        "dual_bound": dual,
        "relative_gap": gap / 100.0 if gap is not None else None,
        "nodes": integer(captured(r"^nodes\s*:\s*(\d+)", stdout)),
        "solver_seconds": number(captured(rf"^time\s*:\s*({FLOAT})s", stdout)),
        "diagnostic": stderr[-4000:],
    }


def parse_highs(stdout: str, stderr: str) -> dict[str, Any]:
    text = stdout + "\n" + stderr
    raw_status = captured(r"^\s*Status\s+(.+?)\s*$", text)
    objective = number(captured(rf"^\s*Primal bound\s+({FLOAT})", text))
    dual = number(captured(rf"^\s*Dual bound\s+({FLOAT})", text))
    gap = number(captured(rf"^\s*Gap\s+({FLOAT})\s*%", text))
    return {
        "raw_status": raw_status,
        "status": normalize_result_status(raw_status, objective),
        "objective": objective,
        "dual_bound": dual,
        "relative_gap": gap / 100.0 if gap is not None else None,
        "nodes": integer(captured(r"^\s*Nodes\s+(\d+)", text)),
        "solver_seconds": number(captured(rf"^\s*Timing\s+({FLOAT})", text)),
        "diagnostic": text[-4000:],
    }


def process_failed(parsed: dict[str, Any], measured: dict[str, Any]) -> bool:
    """Distinguish a solver-reported limit from a broken process invocation."""
    return bool(
        measured["timed_out"]
        or (
            measured["return_code"] != 0
            and (parsed["raw_status"] is None or parsed["status"] == "ERROR")
        )
    )


def version(binary: Path) -> str:
    try:
        result = subprocess.run(
            [str(binary), "--version"], text=True, capture_output=True,
            timeout=10, check=False,
        )
    except OSError as error:
        return f"missing:{binary}:{error}"
    first = (result.stdout + result.stderr).splitlines()
    label = first[0].strip() if first else binary.name
    return f"{label} [sha256:{sha256_file(binary)[:16]}]"


def references(path: Path) -> dict[str, tuple[str, float | None]]:
    values: dict[str, tuple[str, float | None]] = {}
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = re.match(r"=(opt|best|inf)=\s+(\S+)(?:\s+(\S+))?", line)
            if not match:
                continue
            kind, name, raw = match.groups()
            values[name] = (kind, number(raw))
    return values


def run_id(solver: str, path: Path, digest: str, time_limit: float) -> str:
    value = json.dumps([solver, str(path), digest, time_limit], separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def existing_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    result = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record.get("run_id"), str):
                result.add(record["run_id"])
    return result


def reference_check(record: dict[str, Any]) -> tuple[bool | None, float | None]:
    kind = record.get("reference_status")
    expected = record.get("reference_objective")
    objective = record.get("objective")
    status = record.get("status")
    if kind == "inf":
        return status == "UNSAT" if status in {"UNSAT", "OPTIMAL", "SATISFIABLE"} else None, None
    if not isinstance(expected, (int, float)) or not isinstance(objective, (int, float)):
        return None, None
    error = abs(objective - expected) / max(1.0, abs(expected))
    if status == "OPTIMAL" and kind == "opt":
        return error <= 1e-5, 100.0 * error
    return None, 100.0 * error


def finite_values(values: Iterable[object]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(value)]


def median(values: Iterable[object]) -> float | None:
    numbers = finite_values(values)
    return statistics.median(numbers) if numbers else None


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
    return records


def summarize(records: list[dict[str, Any]], time_limit: float) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_instance: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        groups[record["solver"]].append(record)
        by_instance[record["instance"]][record["solver"]] = record
    rows = []
    for solver, runs in sorted(groups.items()):
        optimal = [record for record in runs if record["status"] == "OPTIMAL"]
        rows.append({
            "solver": solver,
            "runs": len(runs),
            "optimal": len(optimal),
            "feasible_not_proven": sum(record["status"] == "SATISFIABLE" for record in runs),
            "infeasible": sum(record["status"] == "UNSAT" for record in runs),
            "unknown": sum(record["status"] == "UNKNOWN" for record in runs),
            "errors": sum(bool(record.get("process_error", record["status"] == "ERROR")) for record in runs),
            "verified_reference": sum(record.get("reference_correct") is True for record in runs),
            "wrong_reference": sum(record.get("reference_correct") is False for record in runs),
            "median_wall_seconds_optimal": median(record["wall_seconds"] for record in optimal),
            "median_peak_memory_mb": median(record["peak_memory_mb"] for record in runs),
            "median_reference_error_percent": median(record.get("reference_error_percent") for record in runs),
            "par2_seconds": sum(record["wall_seconds"] if record["status"] == "OPTIMAL" else 2 * time_limit for record in runs),
        })
    wins = {"amthal": 0, "highs": 0, "ties": 0, "both_optimal": 0}
    ratios = []
    for solvers in by_instance.values():
        if set(solvers) != {"amthal", "highs"}:
            continue
        amthal, highs = solvers["amthal"], solvers["highs"]
        if amthal["status"] != "OPTIMAL" or highs["status"] != "OPTIMAL":
            continue
        wins["both_optimal"] += 1
        left, right = amthal["wall_seconds"], highs["wall_seconds"]
        ratios.append(left / right if right > 0 else None)
        if left < right * 0.99:
            wins["amthal"] += 1
        elif right < left * 0.99:
            wins["highs"] += 1
        else:
            wins["ties"] += 1
    wins["median_amthal_over_highs_time"] = median(ratios)
    return {"schema_version": 1, "time_limit": time_limit, "rows": rows, "paired": wins}


def shown(value: object, digits: int = 2) -> str:
    return "n/a" if not isinstance(value, (int, float)) else f"{value:.{digits}f}"


def shown_seconds(value: object) -> str:
    return "n/a" if not isinstance(value, (int, float)) else f"{value:.3f}s"


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Amthal versus HiGHS on MIPLIB easy",
        "",
        f"Wall limit: {summary['time_limit']} seconds per solver and instance. Runs are sequential and single-threaded.",
        "",
        "| Solver | Runs | Optimal | Feasible | Infeasible | Unknown | Errors | Reference OK | Reference wrong | PAR2 | Median optimal time | Median RSS | Median reference error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['solver']} | {row['runs']} | {row['optimal']} | {row['feasible_not_proven']} | "
            f"{row['infeasible']} | {row['unknown']} | {row['errors']} | {row['verified_reference']} | "
            f"{row['wrong_reference']} | {shown(row['par2_seconds'], 1)} | "
            f"{shown_seconds(row['median_wall_seconds_optimal'])} | {shown(row['median_peak_memory_mb'], 1)} MB | "
            f"{shown(row['median_reference_error_percent'], 4)}% |"
        )
    paired = summary["paired"]
    lines.extend((
        "",
        "## Paired optimal runs",
        "",
        f"Both optimal: {paired['both_optimal']}. HiGHS wins: {paired['highs']}. "
        f"Amthal wins: {paired['amthal']}. Ties within 1%: {paired['ties']}.",
        f"Median Amthal/HiGHS wall-time ratio: {shown(paired['median_amthal_over_highs_time'])}.",
        "",
    ))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, default=ROOT.parent / "amthal" / "benchmarks" / "easy")
    parser.add_argument("--reference", type=Path, default=ROOT.parent / "amthal" / "benchmarks" / ".cache" / "miplib.solu")
    parser.add_argument("--amthal-binary", type=Path, default=ROOT.parent / "amthal" / "target" / "release" / "amthal")
    parser.add_argument("--highs-binary", type=Path, default=Path("/opt/homebrew/bin/highs"))
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--grace-seconds", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "bench" / "results" / "linear-backends" / "results.jsonl")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    if args.time_limit <= 0 or args.grace_seconds < 0 or args.limit < 0:
        raise SystemExit("time limit must be positive; grace and limit must be non-negative")
    binaries = {"amthal": args.amthal_binary.resolve(), "highs": args.highs_binary.resolve()}
    for solver, binary in binaries.items():
        if not binary.is_file():
            raise SystemExit(f"missing {solver} binary: {binary}")
    instances = sorted(args.instances.resolve().glob("*.mps"))
    if args.limit:
        instances = instances[:args.limit]
    if not instances:
        raise SystemExit(f"no MPS instances under {args.instances}")
    known = references(args.reference.resolve())
    versions = {solver: version(binary) for solver, binary in binaries.items()}
    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sidecar = args.out.with_suffix(args.out.suffix + ".provenance.json")
    provenance = {
        "schema_version": 1,
        "instances": str(args.instances.resolve()),
        "reference": str(args.reference.resolve()),
        "time_limit": args.time_limit,
        "limit": args.limit,
        "solvers": versions,
        "host": machine_provenance(ROOT),
    }
    if args.out.is_file() and not args.restart:
        previous = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
        for field in ("instances", "reference", "time_limit", "limit", "solvers"):
            if previous.get(field) != provenance.get(field):
                raise SystemExit(f"configuration changed at {field}; use a new --out or --restart")
    sidecar.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    completed = set() if args.restart else existing_ids(args.out)
    jobs = []
    for index, instance in enumerate(instances):
        order = ("amthal", "highs") if index % 2 == 0 else ("highs", "amthal")
        digest = sha256_file(instance)
        for solver in order:
            identifier = run_id(solver, instance, digest, args.time_limit)
            if identifier not in completed:
                jobs.append((identifier, solver, instance, digest))
    print(f"linear probe: {len(instances)} instances, {len(jobs)} pending", file=sys.stderr)
    mode = "w" if args.restart else "a"
    with args.out.open(mode, encoding="utf-8") as output:
        for index, (identifier, solver, instance, digest) in enumerate(jobs, 1):
            if solver == "amthal":
                command = [
                    str(binaries[solver]), "--time-limit", str(args.time_limit),
                    "--quiet", str(instance),
                ]
                parser_fn = parse_amthal
            else:
                command = [
                    str(binaries[solver]), "--time_limit", str(args.time_limit),
                    "--parallel", "off", "--random_seed", "0", str(instance),
                ]
                parser_fn = parse_highs
            with tempfile.TemporaryDirectory(prefix="qayd_linear_probe_") as temporary:
                measured = run_measured(
                    command, timeout=args.time_limit + args.grace_seconds,
                    cwd=Path(temporary),
                )
            record = parser_fn(measured["stdout"], measured["stderr"])
            reference_status, reference_objective = known.get(instance.stem, (None, None))
            record.update({
                "schema_version": 1,
                "run_id": identifier,
                "solver": solver,
                "solver_version": versions[solver],
                "instance": instance.name,
                "instance_path": str(instance),
                "instance_sha256": digest,
                "reference_status": reference_status,
                "reference_objective": reference_objective,
                "time_limit": args.time_limit,
                "wall_seconds": measured["wall_seconds"],
                "peak_memory_mb": measured["peak_memory_mb"],
                "return_code": measured["return_code"],
                "timed_out": measured["timed_out"],
                "command": measured["argv"],
            })
            process_error = process_failed(record, measured)
            record["process_error"] = process_error
            if process_error:
                record["status"] = "ERROR"
            correct, error = reference_check(record)
            record["reference_correct"] = correct
            record["reference_error_percent"] = error
            output.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            output.flush()
            print(
                f"[{index}/{len(jobs)}] {solver} {instance.name}: {record['status']} "
                f"obj={record['objective']} wall={record['wall_seconds']:.3f}s",
                file=sys.stderr,
            )
    summary = summarize(load_records(args.out), args.time_limit)
    summary_path = args.out.with_name("summary.json")
    report_path = args.out.with_name("report.md")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(markdown(summary), encoding="utf-8")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
