#!/usr/bin/env python3
"""Verified LKH-3 adapter for CVRPLIB CVRP and Solomon/Homberger VRPTW."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time

from qayd.datasets import read_cvrplib, read_solomon


def write_vrptw(instance: object, path: Path, scale: int) -> None:
    matrix = instance.distance_matrix(scale=scale, rounding="truncate")
    lines = [
        f"NAME : {instance.name}",
        "TYPE : CVRPTW",
        f"DIMENSION : {instance.dimension}",
        f"VEHICLES : {instance.vehicles}",
        f"CAPACITY : {instance.capacity}",
        "EDGE_WEIGHT_TYPE : EXPLICIT",
        "EDGE_WEIGHT_FORMAT : FULL_MATRIX",
        "EDGE_WEIGHT_SECTION",
    ]
    lines.extend(" ".join(map(str, row)) for row in matrix)
    lines.append("DEMAND_SECTION")
    lines.extend(f"{index + 1} {instance.demands[index]}" for index in range(instance.dimension))
    lines.append("TIME_WINDOW_SECTION")
    lines.extend(
        f"{index + 1} {window[0] * scale} {window[1] * scale}"
        for index, window in enumerate(instance.time_windows)
    )
    lines.append("SERVICE_TIME_SECTION")
    lines.extend(f"{index + 1} {instance.service_times[index] * scale}" for index in range(instance.dimension))
    lines.extend(["DEPOT_SECTION", str(instance.depot + 1), "-1", "EOF"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_sintef(path: Path) -> list[list[int]]:
    routes = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line.lower().startswith("route") or ":" not in line:
            continue
        routes.append([int(value) for value in line.split(":", 1)[1].split()])
    if not routes:
        raise ValueError("LKH did not write any SINTEF routes")
    return routes


def solve_cvrp(args: argparse.Namespace, raw_routes: list[list[int]]) -> dict[str, object]:
    instance = read_cvrplib(args.instance)
    routes = []
    for raw in raw_routes:
        normalized = []
        for node_id in raw:
            try:
                node = instance.normalize_node(node_id)
            except KeyError:
                continue
            if node != instance.depot:
                normalized.append(node)
        if normalized:
            routes.append(normalized)
    assert sorted(node for route in routes for node in route) == sorted(instance.customers)
    total = 0
    public = []
    for route in routes:
        load = sum(instance.demands[node] for node in route)
        assert load <= instance.capacity
        sequence = [instance.depot, *route, instance.depot]
        distance = sum(instance.edge_weights[a][b] for a, b in zip(sequence, sequence[1:]))
        total += distance
        public.append({"nodes": [instance.node_ids[node] for node in route], "load": load, "distance": distance})
    return {
        "instance": instance.name,
        "status": "SATISFIABLE",
        "objectives": [total],
        "objective_convention": "cvrplib_unlimited_fleet_distance",
        "routes": public,
        "vehicles_used": len(routes),
        "minimum_vehicles": instance.vehicles,
        "best_known": instance.best_known,
        "verified": True,
    }


def solve_cvrptw(args: argparse.Namespace, raw_routes: list[list[int]]) -> dict[str, object]:
    instance = read_solomon(args.instance)
    scale = args.distance_scale
    matrix = instance.distance_matrix(scale=scale, rounding="truncate")
    routes = []
    for raw in raw_routes:
        route = [node_id - 1 for node_id in raw if 1 <= node_id <= instance.dimension and node_id - 1 != instance.depot]
        if route:
            routes.append(route)
    assert sorted(node for route in routes for node in route) == sorted(instance.customers)
    total = 0
    public = []
    for route in routes:
        load = sum(instance.demands[node] for node in route)
        assert load <= instance.capacity
        clock = instance.time_windows[instance.depot][0] * scale
        previous = instance.depot
        starts = []
        for node in route:
            start = max(instance.time_windows[node][0] * scale, clock + matrix[previous][node])
            assert start <= instance.time_windows[node][1] * scale
            starts.append(start)
            clock = start + instance.service_times[node] * scale
            previous = node
        assert clock + matrix[previous][instance.depot] <= instance.time_windows[instance.depot][1] * scale
        sequence = [instance.depot, *route, instance.depot]
        distance = sum(matrix[a][b] for a, b in zip(sequence, sequence[1:]))
        total += distance
        public.append({
            "nodes": [instance.node_ids[node] for node in route],
            "starts": [start / scale for start in starts],
            "load": load,
            "distance": distance / scale,
        })
    return {
        "instance": instance.name,
        "status": "SATISFIABLE",
        "objectives": [len(routes), total],
        "objective_convention": "fleet_then_dimacs_trunc1_distance",
        "routes": public,
        "distance": total / scale,
        "distance_scale": scale,
        "rounding": "truncate",
        "verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", choices=("cvrp", "cvrptw"))
    parser.add_argument("instance")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1, help="recorded for fairness; LKH-3 is single-threaded")
    parser.add_argument("--distance-scale", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not args.binary.is_file():
        raise SystemExit(f"LKH-3 binary not found: {args.binary}")
    if args.time_limit <= 0 or args.seed < 0 or args.threads <= 0 or args.distance_scale <= 0:
        raise SystemExit("time limit, threads, and scale must be positive; seed must be non-negative")

    with tempfile.TemporaryDirectory(prefix="qayd_lkh_") as directory:
        scratch = Path(directory)
        problem_path = Path(args.instance).resolve()
        if args.problem == "cvrptw":
            problem_path = scratch / "instance.vrp"
            write_vrptw(read_solomon(args.instance), problem_path, args.distance_scale)
        solution_path = scratch / "solution.txt"
        parameter_path = scratch / "run.par"
        parameter_path.write_text("\n".join([
            f"PROBLEM_FILE = {problem_path}",
            f"SINTEF_SOLUTION_FILE = {solution_path}",
            "RUNS = 1",
            f"TIME_LIMIT = {args.time_limit}",
            f"SEED = {args.seed}",
            "TRACE_LEVEL = 0",
        ]) + "\n", encoding="utf-8")
        started = time.perf_counter()
        result = subprocess.run(
            [str(args.binary.resolve()), str(parameter_path)],
            text=True, capture_output=True, check=False,
        )
        elapsed = time.perf_counter() - started
        base = {
            "status": "ERROR" if result.returncode else "UNKNOWN",
            "objectives": [],
            "dual_bound": None,
            "absolute_gap": None,
            "relative_gap": None,
            "bound_method": None,
            "elapsed_seconds": elapsed,
            "solver_engine": "LKH-3",
            "effective_threads": 1,
            "seed_effective": True,
            "verified": False,
        }
        if result.returncode or not solution_path.is_file():
            base["diagnostic"] = (result.stderr or result.stdout)[-2000:]
            record = base
        else:
            try:
                routes = parse_sintef(solution_path)
                solved = solve_cvrp(args, routes) if args.problem == "cvrp" else solve_cvrptw(args, routes)
                record = {**base, **solved}
            except (AssertionError, ValueError, KeyError) as error:
                base["status"] = "ERROR"
                base["diagnostic"] = f"solution replay failed: {error}"
                record = base
    print(json.dumps(record, sort_keys=True) if args.json else record)


if __name__ == "__main__":
    main()
