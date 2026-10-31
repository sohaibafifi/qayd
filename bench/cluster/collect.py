#!/usr/bin/env python3
"""Export audited cluster results without running solvers or changing their plans."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cluster.status import plan_at
from cluster.lib import report

COLUMNS = ("profile", "stage", "track", "family", "instance", "seed", "solver", "status", "objective",
           "has_incumbent", "validation", "solved", "wall_seconds", "cpu_seconds", "peak_rss_kb", "budget",
           "par2", "timed_out", "task_complete", "plan_sha256", "task_key", "source_sha256", "hostname",
           "oar_job_id", "stdout", "stderr")


def plans_at(out, stage="latest", profiles=None):
    found = {}
    for name in ("full", "gate", "smoke"):
        if stage not in ("latest", "all", name):
            continue
        for path in sorted(out.glob(f"*/{name}/plan.json")):
            profile = path.parent.parent.name
            if profiles and profile not in profiles:
                continue
            if stage == "all" or profile not in found:
                found[profile if stage != "all" else (profile, name)] = path
    missing = set(profiles or []) - {p.parent.parent.name for p in found.values()}
    if missing:
        raise ValueError("no prepared plan for profiles: " + ", ".join(sorted(missing)))
    if not found:
        raise ValueError(f"no {stage} plans found under {out}")
    return sorted(found.values())


def partial_records(plan, directory):
    """Include saved arms of incomplete tasks, with exact planned provenance."""
    rows, errors = [], []
    tasks = directory / "tasks"
    if not tasks.exists():
        return rows, errors
    for entry in sorted(tasks.iterdir()):
        if not entry.is_dir() or not entry.name.isdigit() or (entry / "result.json").exists():
            continue
        index = int(entry.name)
        if not 0 <= index < len(plan["tasks"]):
            errors.append(f"unplanned partial task: {entry}")
            continue
        task = plan["tasks"][index]
        for path in sorted(entry.glob("*.json")):
            arm = json.loads(path.read_text())
            solver = path.stem
            if (solver not in task["solvers"] or arm.get("solver") != solver
                    or arm.get("plan_sha256") != plan["sha256"] or arm.get("task_key") != task["key"]):
                errors.append(f"incompatible partial arm: {path}")
                continue
            if arm.get("status") in {"ERROR", "INVALID"} or arm.get("validation", {}).get("valid") is False:
                errors.append(f"task {index}: {solver} {arm.get('status')}")
            rows.append({**arm, "task": index, "track": task["item"]["track"],
                         "family": task["item"]["family"], "instance": task["item"]["id"],
                         "seed": task["seed"], "budget": task["seconds"], "task_complete": False})
    return rows, errors


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def collect(out, destination, stage="latest", profiles=None):
    out, destination = Path(out).resolve(), Path(destination).resolve()
    paths = plans_at(out, stage, profiles)
    # Exports never belong inside an input stage or its task directory.
    if destination == out or any(path.parent == destination or path.parent in destination.parents for path in paths):
        raise ValueError("export destination must be outside the input stage directories")
    rows, summaries = [], []
    for path in paths:
        plan = plan_at(path)  # Also works on copied results from another checkout.
        if plan.get("stage") != path.parent.name:
            raise ValueError(f"plan stage differs from its directory: {path}")
        audited = report.audit(plan, path.parent)
        partial, errors = partial_records(plan, path.parent)
        records = [{**row, "task_complete": True} for row in audited["records"]] + partial
        profile = path.parent.parent.name
        for row in records:
            row.update(profile=profile, stage=plan["stage"], source_sha256=plan.get("source", {}).get("sha256"),
                       hostname=row.get("host", {}).get("hostname"), oar_job_id=row.get("host", {}).get("oar_job_id"))
        summaries.append({"profile": profile, "stage": plan["stage"], "plan": str(path),
                          "plan_sha256": plan["sha256"], "source_sha256": plan.get("source", {}).get("sha256"),
                          "environment": plan.get("environment"), "ready": audited["ready"] and not errors,
                          "complete": audited["complete"] and not errors,
                          "expected_tasks": len(plan["tasks"]), "completed_tasks": audited["completed_tasks"],
                          "expected_runs": sum(len(t["solvers"]) for t in plan["tasks"]),
                          "saved_runs": len(records), "partial_runs": len(partial),
                          "errors": [*audited["errors"], *errors], "contradictions": audited["contradictions"],
                          "missing_tasks": audited["missing_tasks"], "pairing_warnings": audited["pairing_warnings"],
                          "aggregates": audited["aggregates"]})
        rows.extend(records)
    rows.sort(key=lambda r: (r["profile"], r["stage"], r["task"], r["solver"]))
    jsonl = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, sort_keys=True, allow_nan=False) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})
    summary = {"schema": "qayd.cluster.collection/v1", "created_utc": datetime.now(timezone.utc).isoformat(),
               "campaign": str(out), "selection": stage, "profiles": summaries, "saved_runs": len(rows),
               "ready": all(s["ready"] for s in summaries),
               "note": "Live read: profiles are observed successively. Partial arms remain incomplete; "
                       "witness checks are retained, not rerun. Claims and errors remain explicit. "
                       "Budgets, stages and seeds are never pooled into a performance score.",
               "exports_sha256": {"results.jsonl": hashlib.sha256(jsonl.encode()).hexdigest(),
                                  "results.csv": hashlib.sha256(buffer.getvalue().encode()).hexdigest()}}
    lines = ["# Collected results", "", summary["note"], "",
             "| Profile | Stage | Completed tasks | Saved runs | Partial arms | Errors | Contradictions | Ready |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for item in summaries:
        lines.append(f"| {item['profile']} | {item['stage']} | {item['completed_tasks']}/{item['expected_tasks']} | "
                     f"{item['saved_runs']}/{item['expected_runs']} | {item['partial_runs']} | {len(item['errors'])} | "
                     f"{len(item['contradictions'])} | {item['ready']} |")
    for name, content in (("results.jsonl", jsonl), ("results.csv", buffer.getvalue()),
                          ("summary.json", json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"),
                          ("summary.md", "\n".join(lines) + "\n")):
        atomic_text(destination / name, content)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("bench/cluster/runs/next-v2"), help="campaign root, locally accessible")
    parser.add_argument("--dest", type=Path, help="export directory; default: <out>/exports/latest")
    parser.add_argument("--stage", choices=("latest", "all", "smoke", "gate", "full"), default="latest")
    parser.add_argument("--profile", action="append", help="select a profile; repeat to select several")
    parser.add_argument("--require-ready", action="store_true", help="write exports, then fail if any selected profile is not ready")
    args = parser.parse_args()
    destination = args.dest or args.out / "exports/latest"
    try:
        summary = collect(args.out, destination, args.stage, args.profile)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"Collection failed: {error}") from error
    print(f"{summary['saved_runs']} saved solver runs; {len(summary['profiles'])} profiles; exports: {destination}")
    if args.require_ready and not summary["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
