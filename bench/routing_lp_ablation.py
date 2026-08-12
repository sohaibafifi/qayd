#!/usr/bin/env python3
"""Run and report a controlled CVRPLIB root-LP ablation.

The default arms use the structural routing bound and Amthal root budgets of
250, 1,000, and 3,000 milliseconds. Every arm uses the same installed qayd
extension, instance, search budget, seed, and worker count. Results are
resumable and tied to an exact source-tree and extension fingerprint.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Iterable, Sequence

from common.competitive import (
    json_record_from_output,
    machine_provenance,
    run_measured,
    sha256_file,
    source_tree_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_SCHEMA = "qayd.routing-lp-ablation.result/v1"
SUMMARY_SCHEMA = "qayd.routing-lp-ablation.summary/v1"
DEFAULT_GLOB = "data/vrplib/CVRP/X-*.vrp"
FEASIBLE_STATUSES = {"SAT", "SATISFIABLE", "FEASIBLE", "OPTIMAL", "OPTIMUM"}


class CampaignError(RuntimeError):
    """The campaign cannot be executed or compared safely."""


def integers(values: Iterable[str], *, positive: bool) -> list[int]:
    parsed = []
    for value in values:
        for field in value.split(","):
            if not field.strip():
                continue
            number = int(field)
            if (positive and number <= 0) or (not positive and number < 0):
                raise argparse.ArgumentTypeError(
                    "values must be positive" if positive else "values must be non-negative"
                )
            parsed.append(number)
    return sorted(set(parsed))


def arm_name(root_ms: int) -> str:
    return "structural" if root_ms == 0 else f"lp-{root_ms}ms"


def arm_specs(root_budgets: Sequence[int]) -> list[dict[str, Any]]:
    return [
        {
            "name": arm_name(root_ms),
            "root_ms": root_ms,
            "linear_backend": "native" if root_ms == 0 else "amthal",
        }
        for root_ms in root_budgets
    ]


def discover_instances(explicit: Sequence[str], globs: Sequence[str], limit: int) -> list[dict[str, Any]]:
    paths: set[Path] = set()
    for value in explicit:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if not candidate.is_file():
            raise CampaignError(f"instance not found: {value}")
        paths.add(candidate.resolve())
    patterns = list(globs) if globs else ([] if explicit else [DEFAULT_GLOB])
    for pattern in patterns:
        for candidate in ROOT.glob(pattern):
            if candidate.is_file():
                paths.add(candidate.resolve())
    ordered = sorted(paths)
    if limit:
        ordered = ordered[:limit]
    if not ordered:
        raise CampaignError("no CVRPLIB instances matched")
    return [
        {
            "path": path,
            "relative_path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "sha256": sha256_file(path),
            "known_upper_bound": adjacent_solution_cost(path),
        }
        for path in ordered
    ]


def adjacent_solution_cost(instance: Path) -> int | None:
    solution = instance.with_suffix(".sol")
    if not solution.is_file():
        return None
    for raw in reversed(solution.read_text(encoding="utf-8", errors="replace").splitlines()):
        fields = raw.replace(":", " ").split()
        if len(fields) >= 2 and fields[0].lower() in {"cost", "objective", "value"}:
            try:
                value = float(fields[1])
            except ValueError:
                return None
            return int(value) if value.is_integer() else None
    return None


def prepare_extension() -> None:
    command = [
        "uv",
        "run",
        "maturin",
        "develop",
        "--profile",
        "pyext",
        "--features",
        "python,lp-relaxation",
    ]
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except (OSError, subprocess.SubprocessError) as error:
        raise CampaignError(f"cannot build the LP-enabled Python extension: {error}") from error


def extension_provenance() -> dict[str, str]:
    try:
        completed = subprocess.run(
            [sys.executable, "-c", "import qayd._core; print(qayd._core.__file__)"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        path = Path(completed.stdout.strip()).resolve(strict=True)
    except (OSError, subprocess.SubprocessError) as error:
        raise CampaignError(f"cannot locate the installed qayd extension: {error}") from error
    if not path.is_file():
        raise CampaignError(f"invalid qayd extension path: {path}")
    return {"path": str(path), "sha256": sha256_file(path)}


def artifact_unchanged(artifact: dict[str, str]) -> bool:
    path = Path(artifact["path"])
    return path.is_file() and sha256_file(path) == artifact["sha256"]


def run_id(
    item: dict[str, Any],
    arm: dict[str, Any],
    search_budget: int,
    seed: int,
    args: argparse.Namespace,
) -> str:
    payload = [
        RESULT_SCHEMA,
        item["relative_path"],
        item["sha256"],
        arm,
        search_budget,
        seed,
        args.threads,
        args.launcher,
        args.max_iterations,
        args.lp_max_variables,
        args.lp_max_rows,
        args.lp_max_nonzeros,
        args.lp_route_ng_size,
        args.lp_route_max_labels,
        args.lp_route_dual_stabilization_percent,
    ]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def ordered_jobs(
    instances: Sequence[dict[str, Any]],
    arms: Sequence[dict[str, Any]],
    search_budgets: Sequence[int],
    seeds: Sequence[int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    jobs = []
    block = 0
    for item in instances:
        for search_budget in search_budgets:
            for seed in seeds:
                offset = block % len(arms)
                ordered = [*arms[offset:], *arms[:offset]]
                if (block // len(arms)) % 2:
                    ordered.reverse()
                for position, arm in enumerate(ordered):
                    jobs.append(
                        {
                            "item": item,
                            "arm": arm,
                            "search_budget": search_budget,
                            "seed": seed,
                            "order_position": position,
                            "run_id": run_id(item, arm, search_budget, seed, args),
                        }
                    )
                block += 1
    return jobs


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
            identifier = record.get("run_id")
            if isinstance(identifier, str):
                result.add(identifier)
    return result


def command_for(job: dict[str, Any], args: argparse.Namespace) -> list[str]:
    script = ROOT / "examples" / "python" / "routing" / args.launcher / "vrp.py"
    arm = job["arm"]
    command = [
        sys.executable,
        str(script),
        str(job["item"]["path"]),
        "--time-limit",
        str(job["search_budget"]),
        "--threads",
        str(args.threads),
        "--seed",
        str(job["seed"]),
        "--engine",
        "ls",
        "--linear-backend",
        arm["linear_backend"],
        "--lp-root-ms",
        str(arm["root_ms"]),
        "--lp-max-variables",
        str(args.lp_max_variables),
        "--lp-max-rows",
        str(args.lp_max_rows),
        "--lp-max-nonzeros",
        str(args.lp_max_nonzeros),
        "--lp-route-ng-size",
        str(args.lp_route_ng_size),
        "--lp-route-max-labels",
        str(args.lp_route_max_labels),
        "--lp-route-dual-stabilization-percent",
        str(args.lp_route_dual_stabilization_percent),
        "--json",
    ]
    if args.max_iterations is not None:
        command.extend(("--max-iterations", str(args.max_iterations)))
    return command


def integer_value(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def primal_value(record: dict[str, Any]) -> int | None:
    objectives = record.get("objectives")
    return integer_value(objectives[0]) if isinstance(objectives, list) and objectives else None


def validate_record(record: dict[str, Any], known_upper_bound: int | None) -> list[str]:
    errors = []
    primal = primal_value(record)
    if record.get("status") not in FEASIBLE_STATUSES:
        errors.append("no feasible status")
    if record.get("verified") is not True:
        errors.append("launcher did not verify the incumbent")
    if primal is None:
        errors.append("missing integer primal objective")
    for field in ("dual_bound", "lp_root_bound"):
        bound = integer_value(record.get(field))
        if record.get(field) is not None and bound is None:
            errors.append(f"{field} is not an integer")
        if bound is not None and primal is not None and bound > primal:
            errors.append(f"{field} crosses the verified incumbent")
        if bound is not None and known_upper_bound is not None and bound > known_upper_bound:
            errors.append(f"{field} crosses the known feasible solution")
    completed_ng_size = integer_value(record.get("lp_route_ng_size_completed"))
    requested_ng_size = integer_value(record.get("lp_route_ng_size"))
    if completed_ng_size is None:
        errors.append("missing integer completed ng-route size")
    elif completed_ng_size < 0 or requested_ng_size is None or completed_ng_size > requested_ng_size:
        errors.append("completed ng-route size is outside the requested range")
    return errors


def execute(job: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    command = command_for(job, args)
    measured = run_measured(
        command,
        timeout=job["search_budget"] + args.grace_seconds,
        cwd=ROOT,
        memory_limit_mb=args.memory_limit_mb,
    )
    try:
        result = json_record_from_output(measured["stdout"])
    except ValueError as error:
        result = {"status": "ERROR", "objectives": [], "verified": False, "parse_error": str(error)}
    validation_errors = validate_record(result, job["item"]["known_upper_bound"])
    if measured["return_code"] != 0:
        validation_errors.append(f"launcher returned {measured['return_code']}")
    if measured["timed_out"]:
        validation_errors.append("external timeout")
    return {
        **result,
        "schema": RESULT_SCHEMA,
        "run_id": job["run_id"],
        "arm": job["arm"]["name"],
        "linear_backend": job["arm"]["linear_backend"],
        "lp_root_ms": job["arm"]["root_ms"],
        "instance_path": job["item"]["relative_path"],
        "instance_sha256": job["item"]["sha256"],
        "known_upper_bound": job["item"]["known_upper_bound"],
        "search_budget_seconds": job["search_budget"],
        "seed": job["seed"],
        "threads": args.threads,
        "launcher": args.launcher,
        "lp_route_ng_size": args.lp_route_ng_size,
        "lp_route_max_labels": args.lp_route_max_labels,
        "lp_route_dual_stabilization_percent": args.lp_route_dual_stabilization_percent,
        "order_position": job["order_position"],
        "wall_seconds": measured["wall_seconds"],
        "peak_memory_mb": measured["peak_memory_mb"],
        "return_code": measured["return_code"],
        "timed_out": measured["timed_out"],
        "command": measured["argv"],
        "validation_errors": validation_errors,
        "diagnostic": measured["stderr"][-4000:] if validation_errors else None,
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    seen = set()
    try:
        stream = path.open(encoding="utf-8")
    except OSError as error:
        raise CampaignError(f"cannot read {path}: {error}") from error
    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise CampaignError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if record.get("schema") != RESULT_SCHEMA:
                raise CampaignError(f"unsupported record at {path}:{line_number}")
            identifier = record.get("run_id")
            if not isinstance(identifier, str) or identifier in seen:
                raise CampaignError(f"missing or duplicate run ID at {path}:{line_number}")
            seen.add(identifier)
            records.append(record)
    return records


def finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def median(values: Iterable[object]) -> float | None:
    numbers = [number for value in values if (number := finite(value)) is not None]
    return statistics.median(numbers) if numbers else None


def relative_gap(primal: int | None, dual: int | None) -> float | None:
    if primal is None or dual is None or dual > primal:
        return None
    return (primal - dual) / max(1, abs(primal), abs(dual))


def summarize(records: Sequence[dict[str, Any]], expected_arms: Sequence[str], allow_incomplete: bool) -> dict[str, Any]:
    grouped: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    duplicates = []
    for record in records:
        key = (
            str(record.get("instance_path")),
            int(record.get("search_budget_seconds", -1)),
            int(record.get("seed", -1)),
        )
        arm = str(record.get("arm"))
        if arm in grouped[key]:
            duplicates.append((*key, arm))
        grouped[key][arm] = record
    if duplicates:
        raise CampaignError(f"duplicate comparison slot: {duplicates[0]}")
    expected = set(expected_arms)
    incomplete = [key for key, arms in grouped.items() if set(arms) != expected]
    if incomplete and not allow_incomplete:
        shown = ", ".join(f"{key[0]}/t{key[1]}/s{key[2]}" for key in incomplete[:3])
        raise CampaignError(f"incomplete arm groups: {shown}")

    complete_groups = [(key, arms) for key, arms in grouped.items() if set(arms) == expected]
    structural_name = "structural"
    if structural_name not in expected:
        raise CampaignError("the report requires a structural arm with root budget 0")
    details = []
    for (instance, search_budget, seed), arms in sorted(complete_groups):
        structural = arms[structural_name]
        structural_primal = primal_value(structural)
        structural_dual = integer_value(structural.get("dual_bound"))
        for arm_name_value in expected_arms:
            record = arms[arm_name_value]
            primal = primal_value(record)
            dual = integer_value(record.get("dual_bound"))
            lp_bound = integer_value(record.get("lp_root_bound"))
            details.append(
                {
                    "instance": instance,
                    "search_budget_seconds": search_budget,
                    "seed": seed,
                    "arm": arm_name_value,
                    "verified": record.get("verified") is True and not record.get("validation_errors"),
                    "primal": primal,
                    "dual": dual,
                    "lp_root_bound": lp_bound,
                    "relative_gap": relative_gap(primal, dual),
                    "primal_delta_vs_structural": (
                        primal - structural_primal if primal is not None and structural_primal is not None else None
                    ),
                    "dual_delta_vs_structural": (
                        dual - structural_dual if dual is not None and structural_dual is not None else None
                    ),
                    "dual_gain_vs_structural": (
                        (dual - structural_dual) / max(1, abs(structural_dual))
                        if dual is not None and structural_dual is not None
                        else None
                    ),
                    "wall_seconds": record.get("wall_seconds"),
                    "peak_memory_mb": record.get("peak_memory_mb"),
                    "lp_time_ms": record.get("lp_simplex_time_ms"),
                    "lp_solves": record.get("lp_solves"),
                    "lp_certified": record.get("lp_certified"),
                    "lp_model_status": record.get("lp_model_status"),
                    "bound_method": record.get("bound_method"),
                    "validation_errors": record.get("validation_errors", []),
                }
            )

    arm_rows = []
    for arm_name_value in expected_arms:
        values = [row for row in details if row["arm"] == arm_name_value]
        deltas = [row["primal_delta_vs_structural"] for row in values if row["primal_delta_vs_structural"] is not None]
        arm_rows.append(
            {
                "arm": arm_name_value,
                "runs": len(values),
                "verified": sum(row["verified"] for row in values),
                "invalid": sum(bool(row["validation_errors"]) for row in values),
                "dual_coverage": sum(row["dual"] is not None for row in values),
                "lp_coverage": sum(row["lp_root_bound"] is not None for row in values),
                "primal_wins": sum(delta < 0 for delta in deltas),
                "primal_ties": sum(delta == 0 for delta in deltas),
                "primal_losses": sum(delta > 0 for delta in deltas),
                "median_primal_delta": median(deltas),
                "median_dual_delta": median(row["dual_delta_vs_structural"] for row in values),
                "median_dual_gain": median(row["dual_gain_vs_structural"] for row in values),
                "median_relative_gap": median(row["relative_gap"] for row in values),
                "median_wall_seconds": median(row["wall_seconds"] for row in values),
                "median_peak_memory_mb": median(row["peak_memory_mb"] for row in values),
                "median_lp_time_ms": median(row["lp_time_ms"] for row in values),
                "lp_model_statuses": dict(sorted(Counter(str(row["lp_model_status"]) for row in values).items())),
                "bound_methods": dict(sorted(Counter(str(row["bound_method"]) for row in values).items())),
            }
        )

    instance_rows = []
    instance_groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        instance_groups[(row["instance"], row["search_budget_seconds"], row["arm"])].append(row)
    for (instance, search_budget, arm_name_value), values in sorted(instance_groups.items()):
        deltas = [row["primal_delta_vs_structural"] for row in values if row["primal_delta_vs_structural"] is not None]
        instance_rows.append(
            {
                "instance": instance,
                "search_budget_seconds": search_budget,
                "arm": arm_name_value,
                "runs": len(values),
                "invalid": sum(bool(row["validation_errors"]) for row in values),
                "primal_wins": sum(delta < 0 for delta in deltas),
                "primal_ties": sum(delta == 0 for delta in deltas),
                "primal_losses": sum(delta > 0 for delta in deltas),
                "median_primal_delta": median(deltas),
                "median_dual_delta": median(row["dual_delta_vs_structural"] for row in values),
                "median_dual_gain": median(row["dual_gain_vs_structural"] for row in values),
                "median_relative_gap": median(row["relative_gap"] for row in values),
                "median_lp_root_bound": median(row["lp_root_bound"] for row in values),
                "median_lp_time_ms": median(row["lp_time_ms"] for row in values),
            }
        )

    return {
        "schema": SUMMARY_SCHEMA,
        "records": len(records),
        "comparison_groups": len(grouped),
        "complete_groups": len(complete_groups),
        "incomplete_groups": len(incomplete),
        "invalid_records": sum(bool(record.get("validation_errors")) for record in records),
        "arms": arm_rows,
        "instances": instance_rows,
        "details": details,
    }


def shown(value: object, *, percent: bool = False) -> str:
    number = finite(value)
    if number is None:
        return "n/a"
    return f"{100 * number:.2f}%" if percent else f"{number:.3f}"


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Qayd CVRPLIB root-LP ablation",
        "",
        f"Complete paired groups: {summary['complete_groups']}/{summary['comparison_groups']}. "
        f"Invalid records: {summary['invalid_records']}.",
        "",
        "| Arm | Runs | Verified | LP coverage | Primal W/T/L | Median primal delta | Median dual delta | Median dual gain | Median gap | LP ms | Wall s | RSS MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["arms"]:
        lines.append(
            f"| {row['arm']} | {row['runs']} | {row['verified']} | {row['lp_coverage']} | "
            f"{row['primal_wins']}/{row['primal_ties']}/{row['primal_losses']} | "
            f"{shown(row['median_primal_delta'])} | {shown(row['median_dual_delta'])} | "
            f"{shown(row['median_dual_gain'], percent=True)} | "
            f"{shown(row['median_relative_gap'], percent=True)} | "
            f"{shown(row['median_lp_time_ms'])} | {shown(row['median_wall_seconds'])} | "
            f"{shown(row['median_peak_memory_mb'])} |"
        )
    lines.extend(
        [
            "",
            "Primal deltas are treatment minus structural, so negative values are better. "
            "Dual deltas are treatment minus structural, so positive values are better.",
            "",
            "## LP outcomes",
            "",
        ]
    )
    for row in summary["arms"]:
        lines.append(
            f"- `{row['arm']}`: statuses {row['lp_model_statuses']}; bound methods {row['bound_methods']}."
        )
    lines.extend(
        [
            "",
            "## By instance and search budget",
            "",
            "| Instance | Search s | Arm | Runs | Primal W/T/L | Median primal delta | Median dual delta | Median dual gain | Median gap | Median LP bound | LP ms |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["instances"]:
        lines.append(
            f"| {Path(row['instance']).name} | {row['search_budget_seconds']} | {row['arm']} | {row['runs']} | "
            f"{row['primal_wins']}/{row['primal_ties']}/{row['primal_losses']} | "
            f"{shown(row['median_primal_delta'])} | {shown(row['median_dual_delta'])} | "
            f"{shown(row['median_dual_gain'], percent=True)} | {shown(row['median_relative_gap'], percent=True)} | "
            f"{shown(row['median_lp_root_bound'])} | {shown(row['median_lp_time_ms'])} |"
        )
    return "\n".join(lines) + "\n"


def sidecar_payload(
    args: argparse.Namespace,
    instances: Sequence[dict[str, Any]],
    arms: Sequence[dict[str, Any]],
    search_budgets: Sequence[int],
    seeds: Sequence[int],
    artifact: dict[str, str],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "instances": [
            {
                "path": item["relative_path"],
                "sha256": item["sha256"],
                "known_upper_bound": item["known_upper_bound"],
            }
            for item in instances
        ],
        "arms": list(arms),
        "search_budgets": list(search_budgets),
        "seeds": list(seeds),
        "threads": args.threads,
        "jobs": args.jobs,
        "launcher": args.launcher,
        "max_iterations": args.max_iterations,
        "memory_limit_mb": args.memory_limit_mb,
        "lp_max_variables": args.lp_max_variables,
        "lp_max_rows": args.lp_max_rows,
        "lp_max_nonzeros": args.lp_max_nonzeros,
        "lp_route_ng_size": args.lp_route_ng_size,
        "lp_route_max_labels": args.lp_route_max_labels,
        "lp_route_dual_stabilization_percent": args.lp_route_dual_stabilization_percent,
        "artifact": artifact,
        "host": provenance,
    }


def check_resume(sidecar: Path, payload: dict[str, Any], restart: bool) -> None:
    if restart or not sidecar.exists():
        return
    try:
        previous = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read provenance sidecar {sidecar}: {error}") from error
    comparable = {key: value for key, value in payload.items() if key != "host"}
    previous_comparable = {key: value for key, value in previous.items() if key != "host"}
    if comparable != previous_comparable:
        raise CampaignError("campaign configuration or extension changed; use a new --out or --restart")
    if previous.get("host", {}).get("source_tree_sha256") != payload.get("host", {}).get("source_tree_sha256"):
        raise CampaignError("qayd source tree changed; use a new --out or --restart")


def run_campaign(
    jobs: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    artifact: dict[str, str],
    expected_source: str | None,
) -> None:
    completed = set() if args.restart else existing_ids(args.out)
    pending = [job for job in jobs if job["run_id"] not in completed]
    print(f"campaign: {len(jobs)} runs, {len(pending)} pending, {args.jobs} concurrent jobs", file=sys.stderr)
    mode = "w" if args.restart else "a"
    with args.out.open(mode, encoding="utf-8") as output, ThreadPoolExecutor(max_workers=args.jobs) as pool:
        iterator = iter(pending)
        active: dict[Future[dict[str, Any]], dict[str, Any]] = {}

        def submit_next() -> bool:
            try:
                job = next(iterator)
            except StopIteration:
                return False
            if source_tree_sha256(ROOT) != expected_source:
                raise CampaignError("qayd source tree changed during the campaign")
            if not artifact_unchanged(artifact):
                raise CampaignError("qayd extension changed during the campaign")
            active[pool.submit(execute, job, args)] = job
            return True

        for _ in range(min(args.jobs, len(pending))):
            submit_next()
        finished = 0
        while active:
            ready, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in ready:
                job = active.pop(future)
                record = future.result()
                output.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                output.flush()
                finished += 1
                objective = record.get("objectives") or "-"
                validation = "invalid" if record["validation_errors"] else "valid"
                print(
                    f"[{finished}/{len(pending)}] {record['arm']} {Path(record['instance_path']).name} "
                    f"t={record['search_budget_seconds']} seed={record['seed']}: "
                    f"obj={objective} dual={record.get('dual_bound')} {validation}",
                    file=sys.stderr,
                )
                submit_next()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", action="append", default=[], help="explicit CVRPLIB instance")
    parser.add_argument("--glob", action="append", default=[], help=f"repository glob, default: {DEFAULT_GLOB}")
    parser.add_argument("--limit", type=int, default=0, help="maximum number of sorted instances")
    parser.add_argument("--budget", action="append", default=[], help="search seconds, comma-separated or repeated")
    parser.add_argument("--seed", action="append", default=[], help="seeds, comma-separated or repeated")
    parser.add_argument(
        "--lp-root-ms",
        action="append",
        default=[],
        help="root budgets, including 0 for structural; default: 0,250,1000,3000",
    )
    parser.add_argument("--threads", type=int, default=1, help="threads per solver run")
    parser.add_argument("--jobs", type=int, default=1, help="concurrent solver processes")
    parser.add_argument("--launcher", choices=("api", "native"), default="api")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--lp-max-variables", type=int, default=2000)
    parser.add_argument("--lp-max-rows", type=int, default=1000)
    parser.add_argument("--lp-max-nonzeros", type=int, default=100000)
    parser.add_argument("--lp-route-ng-size", type=int, default=8)
    parser.add_argument("--lp-route-max-labels", type=int, default=2000000)
    parser.add_argument("--lp-route-dual-stabilization-percent", type=int, default=75)
    parser.add_argument("--memory-limit-mb", type=int, default=0)
    parser.add_argument("--grace-seconds", type=float, default=20.0)
    parser.add_argument("--prepare-qayd", action="store_true", help="build the LP-enabled extension first")
    parser.add_argument("--out", type=Path, required=True, help="resumable JSONL result path")
    parser.add_argument("--summary", type=Path, help="summary JSON, defaults beside --out")
    parser.add_argument("--report", type=Path, help="Markdown report, defaults beside --out")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    if (
        args.limit < 0
        or args.threads <= 0
        or args.jobs <= 0
        or args.memory_limit_mb < 0
        or args.grace_seconds < 0
        or args.lp_max_variables <= 0
        or args.lp_max_rows <= 0
        or args.lp_max_nonzeros <= 0
        or not 1 <= args.lp_route_ng_size <= 16
        or args.lp_route_max_labels <= 0
        or not 1 <= args.lp_route_dual_stabilization_percent <= 100
        or (args.max_iterations is not None and args.max_iterations < 0)
    ):
        parser.error("threads, jobs, and LP size limits must be positive; other limits must be non-negative")
    search_budgets = integers(args.budget or ["10"], positive=True)
    seeds = integers(args.seed or ["0,1,2,3,4"], positive=False)
    root_budgets = integers(args.lp_root_ms or ["0,250,1000,3000"], positive=False)
    if 0 not in root_budgets:
        parser.error("--lp-root-ms must include 0 for the structural control")
    args.out = args.out.resolve()
    args.summary = (args.summary or args.out.with_suffix(".summary.json")).resolve()
    args.report = (args.report or args.out.with_suffix(".md")).resolve()
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    sidecar = args.out.with_suffix(args.out.suffix + ".provenance.json")
    if args.analyze_only:
        if sidecar.is_file() and not args.lp_root_ms:
            try:
                stored = json.loads(sidecar.read_text(encoding="utf-8"))
                arms = stored["arms"]
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
                raise CampaignError(f"cannot recover arms from {sidecar}: {error}") from error
        else:
            arms = arm_specs(root_budgets)
    else:
        arms = arm_specs(root_budgets)
        instances = discover_instances(args.instance, args.glob, args.limit)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.prepare_qayd:
            prepare_extension()
        artifact = extension_provenance()
        provenance = machine_provenance(ROOT)
        payload = sidecar_payload(args, instances, arms, search_budgets, seeds, artifact, provenance)
        check_resume(sidecar, payload, args.restart)
        sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        jobs = ordered_jobs(instances, arms, search_budgets, seeds, args)
        run_campaign(jobs, args, artifact, provenance.get("source_tree_sha256"))

    records = load_records(args.out)
    summary = summarize(records, [arm["name"] for arm in arms], args.allow_incomplete)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.write_text(markdown(summary), encoding="utf-8")
    print(f"results={args.out} summary={args.summary} report={args.report}", file=sys.stderr)
    return 1 if summary["invalid_records"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CampaignError as error:
        raise SystemExit(str(error)) from error
