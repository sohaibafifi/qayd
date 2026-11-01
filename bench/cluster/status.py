#!/usr/bin/env python3
"""Read live campaign results and remaining budgets without rewriting reports."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def plan_at(path):
    plan = read(path)
    expected = plan.pop("sha256")
    actual = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if actual != expected:
        raise ValueError(f"plan hash mismatch: {path}")
    plan["sha256"] = expected
    return plan


class Monitor:
    def __init__(self, out):
        self.out = Path(out)
        self.plans = {}
        self.completed = {}

    def collect(self, path):
        path = Path(path)
        if path not in self.plans:
            self.plans[path] = plan_at(path)
        plan = self.plans[path]
        rows, done, warnings = [], 0, []
        directory = path.parent / "tasks"
        for entry in directory.iterdir() if directory.exists() else ():
            if not entry.name.isdigit() or not 0 <= int(entry.name) < len(plan["tasks"]):
                continue
            index = int(entry.name)
            task = plan["tasks"][index]
            cache_key = (path, plan["sha256"], index)
            if cache_key in self.completed:
                rows.extend(self.completed[cache_key])
                done += 1
                continue
            result_path = entry / "result.json"
            result = read(result_path) if result_path.exists() else None
            arms = result["arms"] if result else [read(entry / (solver + ".json"))
                for solver in task["solvers"] if (entry / (solver + ".json")).is_file()]
            valid = (all(a.get("plan_sha256") == plan["sha256"] and a.get("task_key") == task["key"]
                         for a in arms)
                     and len({a["solver"] for a in arms}) == len(arms)
                     and all(a["solver"] in task["solvers"] for a in arms))
            if result:
                valid = valid and result.get("complete") and result.get("plan_sha256") == plan["sha256"] \
                    and result.get("task_key") == task["key"] \
                    and Counter(a["solver"] for a in arms) == Counter(task["solvers"])
            if not valid:
                warnings.append(f"incompatible provenance or contents: {entry}")
                continue
            summaries = [{"task": index, "solver": a["solver"], "status": a["status"],
                          "verified": bool(a.get("has_incumbent") and a.get("validation", {}).get("valid") is True),
                          "error": a["status"] in ("ERROR", "INVALID") or a.get("validation", {}).get("valid") is False,
                          "wall": a["wall_seconds"], "budget": task["seconds"],
                          "family": task["item"]["family"], "instance": task["item"]["id"]} for a in arms]
            rows.extend(summaries)
            if result:
                done += 1
                self.completed[cache_key] = summaries
        total_runs = sum(len(t["solvers"]) for t in plan["tasks"])
        audited_path = path.parent / "report.json"
        if audited_path.is_file():
            audited = read(audited_path)
            if audited.get("plan_sha256") == plan["sha256"]:
                rejected = {(row["task"], row["solver"]) for row in audited.get("rejected_claims", [])}
                rows = [{**row, "status": "ERROR", "error": True}
                        if (row["task"], row["solver"]) in rejected else row for row in rows]
        remaining = sum(t["seconds"] * len(t["solvers"]) for t in plan["tasks"]) - sum(a["budget"] for a in rows)
        return {"plan": plan, "rows": rows, "done": done, "runs": total_runs,
                "remaining_hours": remaining / 3600, "warnings": warnings}

    def slots(self, state):
        jobs = state.get("active_jobs", [])
        capacity = 0
        for job in jobs:
            if job.get("state") != "Running":
                continue
            job_id = job.get("Job_Id", job.get("job_id", job.get("id")))
            logs = list((self.out / "waves").glob(f"*/*.{job_id}.stdout"))
            match = None
            for path in logs:
                with path.open() as stream:
                    match = re.search(r"OAR \d+: (\d+) workers", stream.read(1024))
                if match:
                    break
            if match is None:
                return None
            capacity += int(match.group(1))
        return capacity


def projection(sample, slots):
    """Project only this profile, using full-run samples in every needed stratum."""
    observed, instances, expected = defaultdict(list), defaultdict(set), Counter()
    for task in sample["plan"]["tasks"]:
        for solver in task["solvers"]:
            expected[(solver, task["item"]["family"], task["seconds"])] += 1
    for row in sample["rows"]:
        key = (row["solver"], row["family"], row["budget"])
        observed[key].append(row["wall"])
        instances[key].add(row["instance"])
    if not slots or sample["warnings"] or any(row["error"] for row in sample["rows"]):
        return None
    seconds = 0
    for key, count in expected.items():
        remaining = count - len(observed[key])
        if remaining:
            if len(observed[key]) < 20 or len(instances[key]) < 10:
                return None
            seconds += remaining * sum(observed[key]) / len(observed[key])
    return seconds / slots / 3600


def display(monitor, details=False, reference_slots=None):
    settings, state = read(monitor.out / "settings.json"), read(monitor.out / "state.json")
    slots = reference_slots if reference_slots else monitor.slots(state)
    print(datetime.now().astimezone().strftime("%d/%m/%Y %H:%M:%S %Z"))
    print(f"Controller: {state['status']} | {state.get('profile', '?')}/{state.get('stage', '?')} | "
          f"OAR jobs: {len(state.get('active_jobs', []))}/{settings['jobs']}")
    print(f"Workers {'(reference)' if reference_slots else 'observed in jobs'}: {slots if slots is not None else '?'}")
    if len(state.get("active_profiles", [])) > 1:
        print("Active profiles: " + ", ".join(state["active_profiles"]))
    if settings.get("selected_profiles") is not None:
        print("Selected profiles: " + ", ".join(settings["selected_profiles"]) + ". Other profiles are not scheduled.")
    print("\nProfile         Stage        Completed tasks  Saved runs  Witnesses Errors  Remaining budget (h)")
    samples, total_hours, unprepared = {}, 0.0, []
    blocked = state.get("profile_errors", {})
    selected = set(settings.get("selected_profiles") or settings["profiles"])
    for profile in sorted(settings["profiles"]):
        directory = monitor.out / profile
        paths = [directory / stage / "plan.json" for stage in ("full", "gate", "smoke")]
        path = next((p for p in paths if p.is_file()), None)
        if path is None:
            queued = profile not in selected
            print(f"{profile:15} {'BLOCKED' if profile in blocked else 'not selected' if queued else 'preparing'}")
            if not queued:
                unprepared.append(profile)
            continue
        sample = monitor.collect(path)
        stage = path.parent.name
        label = "BLOCKED" if profile in blocked else f"{stage}/paused" if profile not in selected else stage
        rows = sample["rows"]
        error_count = sum(a["error"] for a in rows)
        print(f"{profile:15} {label:12} {sample['done']:6}/{len(sample['plan']['tasks']):<7} "
              f"{len(rows):5}/{sample['runs']:<6} {sum(a['verified'] for a in rows):7} {error_count:8} "
              f"{sample['remaining_hours']:19.2f}")
        if stage == "full" and profile not in blocked and profile in selected:
            total_hours += sample["remaining_hours"]
            samples[profile] = sample
        elif stage != "full" and profile in selected:
            unprepared.append(profile)
        for warning in sample["warnings"][:3]:
            print("  WARNING: " + warning)
        if details:
            parent = sample["plan"].get("supplement", {}).get("parent_plans", {}).get("full")
            if parent:
                print(f"  Supplement to {Path(parent).parent.parent.name}; original runs are retained.")
            planned = Counter(s for t in sample["plan"]["tasks"] for s in t["solvers"])
            for solver in sorted(planned):
                subset = [a for a in rows if a["solver"] == solver]
                counts = Counter(a["status"] for a in subset)
                STATUS_ORDER = ("OPTIMUM", "SAT", "UNSAT", "UNKNOWN", "UNSUPPORTED", "INVALID", "ERROR")
                statuses = {name: counts[name] if name in counts else 0 for name in STATUS_ORDER}
                statuses.update((name, counts[name]) for name in sorted(counts.keys() - statuses.keys()))
                progress = ""
                if not subset:
                    active = (profile in state.get("active_profiles", [state.get("profile")]) and stage == state.get("stage")
                              and state.get("active_jobs"))
                    progress = " | running, no saved result yet" if active else " | waiting to run"
                print(f"  {solver:18} {len(subset):5}/{planned[solver]:<5} runs | "
                      f"witnesses={sum(a['verified'] for a in subset):5} | "
                      f"statuses={statuses}{progress}")
    if samples:
        print(f"\nRemaining nominal budget for prepared, unblocked full profiles: {total_hours:,.2f} solver hours.")
        if slots:
            print(f"At {slots} continuously available workers: {total_hours / slots:.1f} hours, or {total_hours / slots / 24:.1f} days.")
    else:
        print("\nFull-campaign budget unavailable: no unblocked full plan has been prepared.")
    if unprepared:
        print("Full plans not yet prepared: " + ", ".join(unprepared) + ". Their budgets are not included.")
    if blocked:
        print("Blocked profiles are excluded from duration estimates.")
    print("This envelope assumes full budget consumption; checking, OAR queueing and preemptions are excluded.")
    current = samples.get(state.get("profile"))
    estimate = projection(current, slots) if current else None
    print("Active profile projection from family/solver averages: " +
          (f"{estimate:.1f} computing hours remaining (indicative)." if estimate is not None else
           "unavailable: insufficient samples, errors or unknown capacity."))
    print("Task = instance x seed with all planned solvers. Saved partial arms are counted separately.")
    print("Witnesses are replayed and accepted solutions. OPTIMUM/UNSAT remain solver claims.")
    for profile, error in blocked.items():
        print(f"BLOCKED {profile}: {error}")
    if state.get("error"):
        print("CONTROLLER ERROR: " + state["error"])
    for pending in settings.get("pending", []):
        print("PENDING: " + pending)
    print(f"Audited reports, updated after each wave: {monitor.out}/<profile>/full/report.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("bench/cluster/runs/next-v2"))
    parser.add_argument("--details", action="store_true", help="per-solver progress, including queued smoke and gate runs")
    parser.add_argument("--watch", type=int, default=0, metavar="SECONDS")
    parser.add_argument("--slots", type=int, help="reference worker capacity for duration scenarios")
    args = parser.parse_args()
    if args.watch < 0 or args.watch and args.watch < 10 or args.slots is not None and args.slots < 1:
        parser.error("--watch must be 0 or >= 10, and --slots must be positive")
    monitor = Monitor(args.out)
    try:
        while True:
            if args.watch and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            display(monitor, args.details, args.slots)
            if not args.watch:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"Monitoring unavailable: {error}") from error


if __name__ == "__main__":
    main()
