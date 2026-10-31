#!/usr/bin/env python3
"""Audit exact planned coverage and summarize paired category comparisons."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cluster.lib.campaign import load_plan, read, write


def vector(value):
    if isinstance(value, (int, float)) and math.isfinite(value):
        return (value,)
    if isinstance(value, list) and value and all(isinstance(v, (int, float)) and math.isfinite(v) for v in value):
        return tuple(value)
    return None


def audit(plan: dict, directory: Path) -> dict:
    rows, errors, missing, contradictions, comparisons, pairing_warnings = [], [], [], [], [], []
    planned = {task["key"] for task in plan["tasks"]}
    if len(planned) != len(plan["tasks"]):
        errors.append("duplicate planned task keys")
    expected_directories = {f"{i:07d}" for i in range(len(plan["tasks"]))}
    task_directory = directory / "tasks"
    if task_directory.exists():
        for path in task_directory.glob("*/result.json"):
            if path.parent.name not in expected_directories:
                errors.append(f"unplanned result: {path}")
    for index, task in enumerate(plan["tasks"]):
        path = task_directory / f"{index:07d}" / "result.json"
        if not path.is_file():
            missing.append(index)
            continue
        result = read(path)
        if (not result.get("complete") or result.get("plan_sha256") != plan["sha256"]
                or result.get("task_key") != task["key"]):
            errors.append(f"incompatible task {index}")
            continue
        arms = result["arms"]
        hosts = {arm.get("host", {}).get("hostname") for arm in arms}
        if len(hosts) > 1:
            pairing_warnings.append(f"task {index} spans hosts after resume: {sorted(hosts)}")
        if Counter(arm["solver"] for arm in arms) != Counter(task["solvers"]):
            errors.append(f"duplicate or missing solver arms in task {index}")
            continue
        if any(arm.get("plan_sha256") != plan["sha256"] or arm.get("task_key") != task["key"] for arm in arms):
            errors.append(f"incompatible arm provenance in task {index}")
            continue
        for arm in arms:
            if arm["status"] in ("ERROR", "INVALID") or arm["validation"].get("valid") is False:
                errors.append(f"task {index}: {arm['solver']} {arm['status']}")
            rows.append({**arm, "task": index, "track": task["item"]["track"],
                         "family": task["item"]["family"], "instance": task["item"]["id"],
                         "seed": task["seed"], "budget": task["seconds"]})
        feasible = [arm for arm in arms if arm.get("has_incumbent")]
        if feasible and any(arm["status"] == "UNSAT" for arm in arms):
            contradictions.append({"task": index, "reason": "UNSAT-versus-verified-witness"})
        convention = {arm.get("details", {}).get("objective_convention") for arm in feasible}
        if len(convention) > 1:
            errors.append(f"objective conventions differ in task {index}: {convention}")
            continue
        for arm in feasible:
            for other in feasible:
                a, b = vector(arm.get("objective")), vector(other.get("objective"))
                if a is None or b is None:
                    continue
                better = b > a if task["item"]["sense"] == "max" else b < a
                if arm["status"] == "OPTIMUM" and better:
                    contradictions.append({"task": index, "reason": "better-verified-solution-than-claimed-optimum",
                                           "solver": arm["solver"], "other": other["solver"]})
                if arm["solver"].startswith("qayd") and not other["solver"].startswith("qayd"):
                    comparisons.append({"task": index, "track": task["item"]["track"],
                                        "solver": arm["solver"], "baseline": other["solver"],
                                        "outcome": "tie" if a == b else "loss" if better else "win",
                                        "same_fleet": a[0] == b[0] if len(a) == 2 else None})
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["track"], row["family"], row["solver"])].append(row)
    aggregates = []
    for (track, family, solver), group in sorted(grouped.items()):
        per_instance = defaultdict(list)
        for row in group:
            per_instance[row["instance"]].append(row["wall_seconds"])
        aggregates.append({"track": track, "family": family, "solver": solver,
                           "runs": len(group), "instances": len(per_instance),
                           "statuses": dict(Counter(row["status"] for row in group)),
                           "verified_incumbents": sum(row["has_incumbent"] for row in group),
                           "solved_claims": sum(row["solved"] for row in group),
                           "sum_par2": sum(row["par2"] for row in group),
                           "median_instance_wall_seconds": statistics.median(statistics.median(v) for v in per_instance.values()),
                           "maximum_rss_kb": max((row["peak_rss_kb"] for row in group
                                                  if row.get("peak_rss_kb") is not None), default=None)})
    if plan["stage"] in {"smoke", "gate"} and not missing:
        expected_arms = {(task["item"]["track"], solver) for task in plan["tasks"] for solver in task["solvers"]}
        for track, solver in expected_arms:
            supported = [row for row in rows if row["track"] == track and row["solver"] == solver
                         and (row["has_incumbent"] or row["status"] == "UNSAT")]
            if not supported:
                errors.append(f"no conclusive support witness for {track}/{solver}")
    concurrency = {str(row.get("host", {}).get("workers_per_job")) for row in rows}
    if len(concurrency) > 1:
        pairing_warnings.append(f"mixed worker-pool sizes: {sorted(concurrency)}")
    cpu_models = {row.get("host", {}).get("cpu") for row in rows if row.get("host", {}).get("cpu")}
    if len(cpu_models) > 1:
        pairing_warnings.append(f"mixed processor models: {sorted(cpu_models)}")
    return {"plan_sha256": plan["sha256"], "environment": plan["environment"], "stage": plan["stage"],
            "complete": not missing and not errors, "ready": not missing and not errors and not contradictions,
            "missing_tasks": missing, "errors": errors, "contradictions": contradictions,
            "expected_tasks": len(plan["tasks"]), "completed_tasks": len(plan["tasks"]) - len(missing),
            "aggregates": aggregates, "pairwise": comparisons, "records": rows,
            "pairing_warnings": pairing_warnings, "timing_comparable": not pairing_warnings,
            "note": "Feasible witnesses are replayed. UNSAT and optimality are solver claims, not independently checked proofs."}


def markdown(report: dict) -> str:
    lines = ["# Campaign qualification", "",
             f"Stage: {report['stage']}. Completed tasks: {report['completed_tasks']}/{report['expected_tasks']}.",
             f"Ready: {report['ready']}. Errors: {len(report['errors'])}. Contradictions: {len(report['contradictions'])}.",
             "", report["note"], "",
             "| Track | Family | Solver | Instances | Runs | Verified incumbents | Claimed solved | Total PAR2 | Max RSS (MiB) |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["aggregates"]:
        rss = f"{row['maximum_rss_kb']/1024:.1f}" if row["maximum_rss_kb"] is not None else "?"
        lines.append(f"| {row['track']} | {row['family']} | {row['solver']} | {row['instances']} | "
                     f"{row['runs']} | {row['verified_incumbents']} | {row['solved_claims']} | "
                     f"{row['sum_par2']:.1f} | {rss} |")
    lines += ["", "PAR2 measures claimed proofs or verified decision solutions. "
              "Optimization incumbents are counted separately. Tracks and budgets are not pooled.",
              "", "Pairwise outcomes and all runs by instance and seed are retained in report.json."]
    for key in ("errors", "contradictions", "missing_tasks", "pairing_warnings"):
        if report[key]:
            lines += ["", key + ":", "", *["- " + str(item) for item in report[key][:50]]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    plan = load_plan(args.plan)
    report = audit(plan, args.plan.resolve().parent)
    write(args.plan.parent / "report.json", report)
    write(args.plan.parent / "qualification.json", {key: report[key] for key in (
        "ready", "complete", "environment", "plan_sha256", "stage", "expected_tasks", "completed_tasks")})
    (args.plan.parent / "report.md").write_text(markdown(report), encoding="utf-8")
    print(f"{report['completed_tasks']}/{report['expected_tasks']} tasks; ready={report['ready']}")
    if args.require_ready and not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
