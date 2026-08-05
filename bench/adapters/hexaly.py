#!/usr/bin/env python3
"""Hexaly adapter with the same normalized models and output as qayd launchers.

The installation is explicit through ``--hexaly-home``.  No license or runtime
path is taken from an environment variable.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

from qayd.datasets import read_cvrplib, read_jsplib, read_psplib, read_solomon


_HEXALY_RUNTIME: tempfile.TemporaryDirectory[str] | None = None


def load_hexaly(home: Path) -> Any:
    global _HEXALY_RUNTIME
    python_dir = home / "bin" / "python"
    library = home / "bin"
    if not python_dir.is_dir() or not library.is_dir():
        raise SystemExit(f"invalid Hexaly installation: {home}")
    native_candidates = [
        *library.glob("libhexaly*.dylib"),
        *library.glob("libhexaly*.so"),
        *library.glob("hexaly*.dll"),
    ]
    if not native_candidates:
        raise SystemExit(f"Hexaly native library not found under {library}")
    # The distributed Python package only probes its own directory and the
    # system loader path.  Build an isolated import view with a link to the
    # explicitly selected native library, without modifying loader env vars.
    _HEXALY_RUNTIME = tempfile.TemporaryDirectory(prefix="qayd_hexaly_")
    runtime = Path(_HEXALY_RUNTIME.name)
    package = runtime / "hexaly"
    shutil.copytree(python_dir / "hexaly", package)
    (package / native_candidates[0].name).symlink_to(native_candidates[0])
    sys.path.insert(0, str(runtime))
    return importlib.import_module("hexaly.optimizer")


def configure(optimizer: Any, args: argparse.Namespace) -> None:
    optimizer.param.time_limit = args.time_limit
    optimizer.param.seed = args.seed
    optimizer.param.nb_threads = args.threads
    optimizer.param.verbosity = 0


def status_and_bounds(optimizer: Any, objective_count: int, module: Any) -> dict[str, Any]:
    status = optimizer.solution.status
    has_primal = status in {module.HxSolutionStatus.FEASIBLE, module.HxSolutionStatus.OPTIMAL}
    status_text = {
        module.HxSolutionStatus.OPTIMAL: "OPTIMAL",
        module.HxSolutionStatus.FEASIBLE: "SATISFIABLE",
        # These are solution statuses. INFEASIBLE means the current solution
        # violates constraints, while INCONSISTENT is the proof of no feasible
        # solution in Hexaly's API terminology.
        module.HxSolutionStatus.INFEASIBLE: "UNKNOWN",
        module.HxSolutionStatus.INCONSISTENT: "UNSAT",
    }[status]
    dual = optimizer.solution.get_objective_bound(0) if objective_count and has_primal else None
    gap = optimizer.solution.get_objective_gap(0) if objective_count and has_primal else None
    return {
        "status": status_text,
        "has_primal": has_primal,
        "dual_bound": dual,
        "relative_gap": gap,
        "bound_method": "Hexaly certified objective bound" if dual is not None else None,
    }


def solve_cvrp(args: argparse.Namespace, hx: Any) -> dict[str, Any]:
    instance = read_cvrplib(args.instance)
    customers = list(instance.customers)
    n = len(customers)
    with hx.HexalyOptimizer() as optimizer:
        model = optimizer.model
        routes = [model.list(n) for _ in range(n)]
        model.constraint(model.partition(routes))
        demands = model.array([instance.demands[node] for node in customers])
        matrix = model.array([[instance.edge_weights[a][b] for b in customers] for a in customers])
        depot_distance = model.array([instance.edge_weights[instance.depot][node] for node in customers])
        route_distances = []
        for route in routes:
            count = model.count(route)
            model.constraint(model.sum(route, model.lambda_function(lambda j: demands[j])) <= instance.capacity)
            between = model.sum(
                model.range(1, count),
                model.lambda_function(lambda i: model.at(matrix, route[i - 1], route[i])),
            )
            route_distances.append(between + model.iif(
                count > 0, depot_distance[route[0]] + depot_distance[route[count - 1]], 0,
            ))
        total_distance = model.sum(route_distances)
        model.minimize(total_distance)
        model.close()
        configure(optimizer, args)
        started = time.perf_counter()
        optimizer.solve()
        elapsed = time.perf_counter() - started
        info = status_and_bounds(optimizer, 1, hx)
        record = {
            "instance": instance.name,
            "status": info["status"],
            "objectives": [],
            "dual_bound": info["dual_bound"],
            "absolute_gap": None,
            "relative_gap": info["relative_gap"],
            "bound_method": info["bound_method"],
            "elapsed_seconds": elapsed,
            "objective_convention": "cvrplib_unlimited_fleet_distance",
            "solver_engine": "Hexaly Optimizer",
            "verified": False,
        }
        if not info["has_primal"]:
            return record
        normalized_routes = [[customers[position] for position in route.value] for route in routes]
        served = sorted(node for route in normalized_routes for node in route)
        assert served == sorted(customers)
        distance = 0
        public_routes = []
        for route in normalized_routes:
            load = sum(instance.demands[node] for node in route)
            assert load <= instance.capacity
            sequence = [instance.depot, *route, instance.depot]
            route_distance = sum(instance.edge_weights[a][b] for a, b in zip(sequence, sequence[1:]))
            distance += route_distance
            public_routes.append({
                "nodes": [instance.node_ids[node] for node in route],
                "load": load,
                "distance": route_distance,
            })
        assert distance == total_distance.value
        record.update({
            "objectives": [distance],
            "absolute_gap": max(0, distance - info["dual_bound"]) if info["dual_bound"] is not None else None,
            "routes": public_routes,
            "vehicles": n,
            "minimum_vehicles": instance.vehicles,
            "vehicles_used": sum(bool(route) for route in normalized_routes),
            "best_known": instance.best_known,
            "verified": True,
        })
        return record


def solve_cvrptw(args: argparse.Namespace, hx: Any) -> dict[str, Any]:
    instance = read_solomon(args.instance)
    customers = list(instance.customers)
    n = len(customers)
    scale = args.distance_scale
    distances = instance.distance_matrix(scale=scale, rounding=args.rounding)
    earliest_all = [window[0] * scale for window in instance.time_windows]
    latest_all = [window[1] * scale for window in instance.time_windows]
    service_all = [duration * scale for duration in instance.service_times]
    with hx.HexalyOptimizer() as optimizer:
        model = optimizer.model
        routes = [model.list(n) for _ in range(instance.vehicles)]
        model.constraint(model.partition(routes))
        demands = model.array([instance.demands[node] for node in customers])
        earliest = model.array([earliest_all[node] for node in customers])
        latest = model.array([latest_all[node] for node in customers])
        service = model.array([service_all[node] for node in customers])
        matrix = model.array([[distances[a][b] for b in customers] for a in customers])
        depot_distance = model.array([distances[instance.depot][node] for node in customers])
        used = [model.count(route) > 0 for route in routes]
        route_distances = []
        latenesses = []
        for route, is_used in zip(routes, used):
            count = model.count(route)
            model.constraint(model.sum(route, model.lambda_function(lambda j: demands[j])) <= instance.capacity)
            between = model.sum(
                model.range(1, count),
                model.lambda_function(lambda i: model.at(matrix, route[i - 1], route[i])),
            )
            route_distances.append(between + model.iif(
                is_used, depot_distance[route[0]] + depot_distance[route[count - 1]], 0,
            ))
            departures = model.array(
                model.range(0, count),
                model.lambda_function(
                    lambda i, previous: model.max(
                        earliest[route[i]],
                        model.iif(
                            i == 0,
                            earliest_all[instance.depot] + depot_distance[route[0]],
                            previous + model.at(matrix, route[i - 1], route[i]),
                        ),
                    ) + service[route[i]]
                ),
                earliest_all[instance.depot],
            )
            customer_late = model.sum(
                model.range(0, count),
                model.lambda_function(lambda i: model.max(0, departures[i] - service[route[i]] - latest[route[i]])),
            )
            home_late = model.iif(
                is_used,
                model.max(0, departures[count - 1] + depot_distance[route[count - 1]] - latest_all[instance.depot]),
                0,
            )
            latenesses.append(customer_late + home_late)
        total_lateness = model.sum(latenesses)
        model.constraint(total_lateness == 0)
        fleet = model.sum(used)
        total_distance = model.sum(route_distances)
        model.minimize(fleet)
        model.minimize(total_distance)
        model.close()
        configure(optimizer, args)
        started = time.perf_counter()
        optimizer.solve()
        elapsed = time.perf_counter() - started
        info = status_and_bounds(optimizer, 2, hx)
        record = {
            "instance": instance.name,
            "status": info["status"],
            "objectives": [],
            "dual_bound": info["dual_bound"],
            "absolute_gap": None,
            "relative_gap": info["relative_gap"],
            "bound_method": info["bound_method"],
            "elapsed_seconds": elapsed,
            "objective_convention": "fleet_then_dimacs_trunc1_distance",
            "distance_scale": scale,
            "rounding": args.rounding,
            "solver_engine": "Hexaly Optimizer",
            "verified": False,
        }
        if not info["has_primal"]:
            return record
        normalized_routes = [[customers[position] for position in route.value] for route in routes]
        assert sorted(node for route in normalized_routes for node in route) == sorted(customers)
        distance = 0
        public_routes = []
        for route in normalized_routes:
            load = sum(instance.demands[node] for node in route)
            assert load <= instance.capacity
            clock = earliest_all[instance.depot]
            previous = instance.depot
            starts = []
            for node in route:
                start = max(earliest_all[node], clock + distances[previous][node])
                assert start <= latest_all[node]
                starts.append(start)
                clock = start + service_all[node]
                previous = node
            assert clock + distances[previous][instance.depot] <= latest_all[instance.depot]
            sequence = [instance.depot, *route, instance.depot]
            route_distance = sum(distances[a][b] for a, b in zip(sequence, sequence[1:]))
            distance += route_distance
            public_routes.append({
                "nodes": [instance.node_ids[node] for node in route],
                "starts": [start / scale for start in starts],
                "load": load,
                "distance": route_distance / scale,
            })
        fleet_value = sum(bool(route) for route in normalized_routes)
        assert fleet_value == fleet.value and distance == total_distance.value
        record.update({
            "objectives": [fleet_value, distance],
            "absolute_gap": max(0, fleet_value - info["dual_bound"]) if info["dual_bound"] is not None else None,
            "routes": public_routes,
            "distance": distance / scale,
            "verified": True,
        })
        return record


def solve_jssp(args: argparse.Namespace, hx: Any) -> dict[str, Any]:
    instance = read_jsplib(args.instance)
    jobs = instance.num_jobs
    machines = instance.num_machines
    for job in instance.jobs:
        assert len(job) == machines and len({operation.machine for operation in job}) == machines
    horizon = max(1, instance.horizon)
    processing = [[0] * machines for _ in range(jobs)]
    order = []
    for job_index, job in enumerate(instance.jobs):
        order.append([operation.machine for operation in job])
        for operation in job:
            processing[job_index][operation.machine] = operation.duration
    with hx.HexalyOptimizer() as optimizer:
        model = optimizer.model
        tasks = [[model.interval(0, horizon) for _ in range(machines)] for _ in range(jobs)]
        for job in range(jobs):
            for machine in range(machines):
                model.constraint(model.length(tasks[job][machine]) == processing[job][machine])
            for index in range(machines - 1):
                model.constraint(tasks[job][order[job][index]] < tasks[job][order[job][index + 1]])
        task_array = model.array(tasks)
        machine_orders = [model.list(jobs) for _ in range(machines)]
        for machine, sequence in enumerate(machine_orders):
            model.constraint(model.count(sequence) == jobs)
            model.constraint(model.and_(
                model.range(0, jobs - 1),
                model.lambda_function(lambda i: model.at(task_array, sequence[i], machine) < model.at(task_array, sequence[i + 1], machine)),
            ))
        makespan = model.max([model.end(tasks[job][order[job][-1]]) for job in range(jobs)])
        model.minimize(makespan)
        model.close()
        configure(optimizer, args)
        started = time.perf_counter()
        optimizer.solve()
        elapsed = time.perf_counter() - started
        info = status_and_bounds(optimizer, 1, hx)
        record = {
            "instance": instance.name,
            "status": info["status"],
            "objectives": [],
            "dual_bound": info["dual_bound"],
            "absolute_gap": None,
            "relative_gap": info["relative_gap"],
            "bound_method": info["bound_method"],
            "elapsed_seconds": elapsed,
            "objective_convention": "makespan",
            "solver_engine": "Hexaly Optimizer",
            "verified": False,
        }
        if not info["has_primal"]:
            return record
        schedule = []
        occupied = [[] for _ in range(machines)]
        makespan_value = 0
        for job_index, job in enumerate(instance.jobs):
            row = []
            previous_end = 0
            for operation in job:
                interval = tasks[job_index][operation.machine].value
                start = interval.start()
                end = interval.end()
                assert end - start == operation.duration and start >= previous_end
                previous_end = end
                makespan_value = max(makespan_value, end)
                occupied[operation.machine].append((start, end))
                row.append({"machine": operation.machine, "start": start, "duration": operation.duration})
            schedule.append(row)
        for intervals in occupied:
            intervals.sort()
            assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
        assert makespan_value == makespan.value
        record.update({
            "objectives": [makespan_value],
            "absolute_gap": max(0, makespan_value - info["dual_bound"]) if info["dual_bound"] is not None else None,
            "schedule": schedule,
            "verified": True,
        })
        return record


def solve_rcpsp(args: argparse.Namespace, hx: Any) -> dict[str, Any]:
    instance = read_psplib(args.instance)
    jobs = list(instance.jobs)
    job_index = {job.job: index for index, job in enumerate(jobs)}
    horizon = max(instance.horizon or 0, sum(max(mode.duration for mode in job.modes) for job in jobs), 1)
    with hx.HexalyOptimizer() as optimizer:
        model = optimizer.model
        options = [[model.optional_interval(0, horizon) for _ in job.modes] for job in jobs]
        presence = [[model.presence(interval) for interval in row] for row in options]
        masters = [model.hull(row) for row in options]
        for index, job in enumerate(jobs):
            model.constraint(model.sum(presence[index]) == 1)
            for mode_index, mode in enumerate(job.modes):
                model.constraint(model.iif(
                    presence[index][mode_index],
                    model.length(options[index][mode_index]) == mode.duration,
                    1,
                ))
            for successor in job.successors:
                model.constraint(masters[index] < masters[job_index[successor]])
        for resource, (kind, capacity) in enumerate(zip(instance.resource_kinds, instance.capacities)):
            if kind in {"renewable", "doubly_constrained"}:
                def capacity_function(resource_index: int, resource_capacity: int) -> Any:
                    def capacity_at(point: Any) -> Any:
                        return model.sum(
                            mode.demands[resource_index] * model.contains(options[index][mode_index], point)
                            for index, job in enumerate(jobs)
                            for mode_index, mode in enumerate(job.modes)
                        ) <= resource_capacity
                    return capacity_at

                model.constraint(model.and_(
                    model.range(horizon),
                    model.lambda_function(capacity_function(resource, capacity)),
                ))
            if kind in {"nonrenewable", "doubly_constrained"}:
                model.constraint(model.sum(
                    mode.demands[resource] * presence[index][mode_index]
                    for index, job in enumerate(jobs)
                    for mode_index, mode in enumerate(job.modes)
                ) <= capacity)
        makespan = model.max([model.end(master) for master in masters])
        model.minimize(makespan)
        model.close()
        configure(optimizer, args)
        started = time.perf_counter()
        optimizer.solve()
        elapsed = time.perf_counter() - started
        info = status_and_bounds(optimizer, 1, hx)
        record = {
            "instance": instance.name,
            "status": info["status"],
            "objectives": [],
            "dual_bound": info["dual_bound"],
            "absolute_gap": None,
            "relative_gap": info["relative_gap"],
            "bound_method": info["bound_method"],
            "elapsed_seconds": elapsed,
            "objective_convention": "makespan",
            "solver_engine": "Hexaly Optimizer",
            "verified": False,
        }
        if not info["has_primal"]:
            return record
        schedule = []
        for index, job in enumerate(jobs):
            selected = next(mode_index for mode_index, value in enumerate(presence[index]) if value.value)
            mode = job.modes[selected]
            interval = options[index][selected].value
            schedule.append({
                "job": job.job,
                "mode": mode.mode,
                "start": interval.start(),
                "duration": mode.duration,
                "demands": list(mode.demands),
            })
        by_job = {job["job"]: job for job in schedule}
        for job in jobs:
            for successor in job.successors:
                assert by_job[job.job]["start"] + by_job[job.job]["duration"] <= by_job[successor]["start"]
        for resource, (kind, capacity) in enumerate(zip(instance.resource_kinds, instance.capacities)):
            if kind in {"renewable", "doubly_constrained"}:
                points = sorted({job["start"] for job in schedule} | {job["start"] + job["duration"] for job in schedule})
                for point in points:
                    assert sum(
                        job["demands"][resource] for job in schedule
                        if job["start"] <= point < job["start"] + job["duration"]
                    ) <= capacity
            if kind in {"nonrenewable", "doubly_constrained"}:
                assert sum(job["demands"][resource] for job in schedule) <= capacity
        makespan_value = max(job["start"] + job["duration"] for job in schedule)
        assert makespan_value == makespan.value
        record.update({
            "objectives": [makespan_value],
            "absolute_gap": max(0, makespan_value - info["dual_bound"]) if info["dual_bound"] is not None else None,
            "schedule": schedule,
            "verified": True,
        })
        return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", choices=("cvrp", "cvrptw", "jssp", "rcpsp"))
    parser.add_argument("instance")
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hexaly-home", type=Path, required=True)
    parser.add_argument("--distance-scale", type=int, default=10)
    parser.add_argument("--rounding", choices=("truncate", "nearest", "ceil"), default="truncate")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.time_limit < 0 or args.threads <= 0 or args.seed < 0 or args.distance_scale <= 0:
        raise SystemExit("time limit and seed must be non-negative; threads and scale must be positive")
    hx = load_hexaly(args.hexaly_home.resolve())
    solve = {
        "cvrp": solve_cvrp,
        "cvrptw": solve_cvrptw,
        "jssp": solve_jssp,
        "rcpsp": solve_rcpsp,
    }[args.problem]
    record = solve(args, hx)
    if args.json:
        print(json.dumps(record, sort_keys=True))
    else:
        print(record)


if __name__ == "__main__":
    main()
