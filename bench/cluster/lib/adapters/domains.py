"""Cluster helpers retained from bench/campaign.py; experimental semantics preserved."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[4]


ADAPTERS = Path(__file__).resolve().parent


QAYD_SCRIPTS = {
    ("api", "cvrp"): "examples/python/routing/api/vrp.py",
    ("native", "cvrp"): "examples/python/routing/native/vrp.py",
    ("api", "cvrptw"): "examples/python/routing/api/cvrptw.py",
    ("native", "cvrptw"): "examples/python/routing/native/cvrptw.py",
    ("api", "jssp"): "examples/python/scheduling/api/jssp.py",
    ("native", "jssp"): "examples/python/scheduling/native/jssp.py",
    ("api", "rcpsp"): "examples/python/scheduling/api/rcpsp.py",
    ("native", "rcpsp"): "examples/python/scheduling/native/rcpsp.py",
}


def adjacent_best_known(path: Path) -> float | None:
    solution = path.with_suffix(".sol")
    if solution.is_file():
        for raw in reversed(solution.read_text(encoding="utf-8", errors="replace").splitlines()):
            fields = raw.replace(":", " ").split()
            if len(fields) >= 2 and fields[0].lower() in {"cost", "objective", "value"}:
                try:
                    return float(fields[1])
                except ValueError:
                    return None
    for parent in path.parents:
        metadata = parent / "instances.json"
        if metadata.is_file():
            try:
                entries = json.loads(metadata.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                entries = []
            for entry in entries if isinstance(entries, list) else []:
                if entry.get("name") == path.name and isinstance(entry.get("optimum"), (int, float)):
                    return float(entry["optimum"])
        summary_path = parent / "summary.json"
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                summary = {}
            prefix = str(summary.get("instance_set", ""))
            suffix = path.stem.removeprefix(prefix)
            fields = suffix.split("_")
            if prefix and len(fields) == 2 and all(field.isdigit() for field in fields):
                parameter, instance = map(int, fields)
                for entry in summary.get("solutions", []):
                    if entry.get("parameter") == parameter and entry.get("instance") == instance:
                        value = entry.get("optimal_makespan") if entry.get("is_proven_optimal") else entry.get("upper_bound")
                        return float(value) if isinstance(value, (int, float)) else None
    return None


def command_for(
    solver: str, item: dict[str, Any], budget: int, seed: int,
    args: argparse.Namespace,
) -> list[str]:
    problem = item["problem"]
    instance = str(item["path"])
    threads = effective_threads(solver, item, args)
    common = [instance, "--time-limit", str(budget), "--threads", str(threads), "--seed", str(seed), "--json"]
    if solver.startswith("qayd-"):
        variant = solver.removeprefix("qayd-")
        engine = args.qayd_engine
        if problem == "rcpsp" and item["path"].suffix.lower() == ".mm" and engine == "ls":
            engine = "auto"
        command = [
            sys.executable,
            str(ROOT / QAYD_SCRIPTS[(variant, problem)]),
            *common,
            "--engine",
            engine,
            "--memory-limit-mb",
            str(args.memory_limit_mb),
        ]
        list_controls = problem in {"cvrp", "cvrptw"}
        if args.profile_qayd:
            command.append("--profile")
        scheduling_ls_controls = problem in {"jssp", "rcpsp"} and engine in {"auto", "ls"} and item["path"].suffix.lower() != ".mm"
        if args.max_iterations is not None and (engine == "ls" or scheduling_ls_controls):
            command.extend(("--max-iterations", str(args.max_iterations)))
        if problem in {"cvrp", "cvrptw"}:
            command.append("--routing-two-way" if args.routing_two_way else "--no-routing-two-way")
            command.append("--routing-nearest-neighbor" if args.routing_nearest_neighbor else "--no-routing-nearest-neighbor")
            command.append("--routing-warm-start" if args.routing_warm_start else "--no-routing-warm-start")
        if problem == "jssp":
            command.append("--compact-json")
            command.extend(("--schedule-jssp-search", args.qayd_schedule_jssp_search))
            if args.qayd_schedule_path_relink:
                command.append("--schedule-path-relink")
            if args.qayd_schedule_lns_shadow:
                command.append("--schedule-lns-shadow")
            if args.qayd_schedule_constructor_workers:
                command.extend((
                    "--schedule-constructor-workers",
                    str(args.qayd_schedule_constructor_workers),
                ))
        return command
    if solver == "ortools-cp-sat":
        return [sys.executable, str(ADAPTERS / "ortools_cp_sat.py"), problem, *common]
    if solver == "hexaly":
        command = [
            sys.executable, str(ADAPTERS / "hexaly.py"), problem, *common,
            "--hexaly-license-path", str(Path(args.hexaly_license_path).expanduser().resolve()),
        ]
        if args.hexaly_home:
            command.extend((
                "--hexaly-home", str(Path(args.hexaly_home).expanduser().resolve()),
            ))
        return command
    if solver == "hgs":
        return [
            sys.executable, str(ADAPTERS / "hgs.py"), instance,
            "--time-limit", str(budget), "--seed", str(seed), "--threads", str(threads),
            "--binary", str(Path(args.hgs_binary).resolve()), "--json",
        ]
    if solver == "lkh":
        return [
            sys.executable, str(ADAPTERS / "lkh.py"), problem, instance,
            "--time-limit", str(budget), "--seed", str(seed), "--threads", str(threads),
            "--binary", str(Path(args.lkh_binary).resolve()), "--json",
        ]
    raise AssertionError(solver)


def effective_threads(solver: str, item: dict[str, Any], args: argparse.Namespace) -> int:
    if solver in {"hgs", "lkh"}:
        return 1
    if solver.startswith("qayd-"):
        if args.qayd_engine == "exact":
            return 1
        if item["problem"] == "rcpsp" and item["path"].suffix.lower() == ".mm":
            return 1
    return args.threads


