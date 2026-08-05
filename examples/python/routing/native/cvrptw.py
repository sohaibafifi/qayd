"""Capacitated VRPTW with an optional Solomon/Homberger instance.

Without a positional file this keeps the original deterministic generated
example. Pass a Solomon or Gehring-Homberger file to solve that benchmark:

    uv run examples/python/routing/native/cvrptw.py C101.txt --threads 4
"""

import argparse
import contextlib
import json
import math
import sys
import time
from random import Random

import qayd as cp
from qayd.datasets import read_solomon, read_vrp_solution


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", nargs="?")
    parser.add_argument("--customers", type=int, default=15)
    parser.add_argument("--vehicles", type=int)
    parser.add_argument("--time-limit", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--routing-two-way", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--routing-nearest-neighbor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--routing-warm-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--engine", choices=("auto", "exact", "ls"), default="ls")
    parser.add_argument("--distance-scale", type=int, default=10)
    parser.add_argument("--rounding", choices=("truncate", "nearest", "ceil"), default="truncate")
    parser.add_argument("--solution", help="VRP solution file used as an LS warm start")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit one machine-readable result")
    return parser.parse_args()


args = arguments()
if args.threads <= 0 or args.time_limit < 0 or args.seed < 0 or (args.max_iterations is not None and args.max_iterations < 0):
    raise SystemExit("threads must be positive; time limit and seed must be non-negative")

instance = read_solomon(args.instance) if args.instance else None
if instance is None:
    n = args.customers
    rng = Random(args.seed)
    coordinates = [(50, 50)] + [(rng.randint(0, 100), rng.randint(0, 100)) for _ in range(n)]
    demand = [0] + [rng.randint(1, 9) for _ in range(n)]
    service = [0] + [10 for _ in range(n)]
    distance = [
        [round(math.hypot(x1 - x2, y1 - y2)) for x2, y2 in coordinates]
        for x1, y1 in coordinates
    ]
    earliest = [0] + [rng.randint(0, 60) for _ in range(n)]
    latest = [1_000] + [earliest[index] + 80 for index in range(1, n + 1)]
    capacity = 40
    depot = 0
    customers = list(range(1, n + 1))
    vehicles = args.vehicles or (-(-sum(demand) // capacity) + 3)
    scale = 1
    name = f"generated-vrptw-n{n}"
    node_ids = list(range(n + 1))
else:
    if args.distance_scale <= 0:
        raise SystemExit("distance scale must be positive")
    scale = args.distance_scale
    distance = [list(row) for row in instance.distance_matrix(scale=scale, rounding=args.rounding)]
    demand = list(instance.demands)
    service = [value * scale for value in instance.service_times]
    earliest = [window[0] * scale for window in instance.time_windows]
    latest = [window[1] * scale for window in instance.time_windows]
    capacity = instance.capacity
    depot = instance.depot
    customers = list(instance.customers)
    vehicles = args.vehicles or instance.vehicles
    name = instance.name
    node_ids = list(instance.node_ids)

if vehicles <= 0:
    raise SystemExit("vehicle count must be positive")
if args.solution and (instance is None or args.engine != "ls"):
    raise SystemExit("--solution requires a real instance and --engine ls")

D, Q = cp.matrix(distance), cp.array(demand)
E, L, S = cp.array(earliest), cp.array(latest), cp.array(service)
model = cp.Model()
routes = model.list_vars(customers, count=vehicles)
model.minimize(cp.sum(cp.used(route) for route in routes))
model.then_minimize(cp.sum(cp.sum_edges(route, lambda i, j: D[i][j], start=depot, end=depot) for route in routes))
for route in routes:
    model.add(cp.sum(route, lambda customer: Q[customer]) <= capacity)
    lateness = cp.scan_sum(
        route,
        step=lambda current, clock, previous: cp.max(E[current], clock + D[previous][current]) + S[current],
        emit=lambda current, departure, previous: cp.max(0, departure - S[current] - L[current]),
        init=earliest[depot],
        boundary=depot,
        end=depot,
    )
    model.add(lateness <= 0)

hint = None
if args.solution:
    hint = [list(route) for route in read_vrp_solution(args.solution, instance=instance).routes]

solve_options = {
    "engine": args.engine,
    "threads": args.threads,
    "time_limit": args.time_limit,
    "seed": args.seed,
    "verbose": args.verbose,
        "max_iterations": args.max_iterations,
        "profile": args.profile,
        "routing_two_way": args.routing_two_way,
        "routing_nearest_neighbor": args.routing_nearest_neighbor,
        "routing_warm_start": args.routing_warm_start,
    }
if hint is not None:
    solve_options["list_hint"] = hint
started = time.perf_counter()
output = contextlib.redirect_stdout(sys.stderr) if args.json else contextlib.nullcontext()
with output:
    solution = model.solve(**solve_options)
elapsed = time.perf_counter() - started

if solution.lists is None:
    record = {
        "instance": name,
        "status": solution.status,
        "elapsed_seconds": elapsed,
        "objectives": [],
        "objective_convention": "fleet_then_dimacs_trunc1_distance",
        "distance_scale": scale,
        "rounding": args.rounding,
        "dual_bound": solution.dual_bound,
        "absolute_gap": solution.absolute_gap,
        "relative_gap": solution.relative_gap,
        "bound_method": solution.bound_method,
        "alns_iterations": solution.alns_iterations,
        "candidates_evaluated": solution.candidates_evaluated,
        "candidates_per_second": solution.candidates_per_second,
        "full_recompute_percentage": solution.full_recompute_percentage,
        "routing_two_way": args.routing_two_way,
        "routing_nearest_neighbor": args.routing_nearest_neighbor,
        "routing_warm_start": args.routing_warm_start,
    }
    print(json.dumps(record, sort_keys=True) if args.json else f"instance: {name}  status: {solution.status}")
    raise SystemExit(0)

served = sorted(customer for route in solution.lists for customer in route)
assert served == sorted(customers), "every customer served exactly once"
total_distance = 0
route_records = []
for route in solution.lists:
    load = sum(demand[customer] for customer in route)
    assert load <= capacity, "capacity respected"
    clock, previous = earliest[depot], depot
    starts = []
    for customer in route:
        start = max(earliest[customer], clock + distance[previous][customer])
        assert start <= latest[customer], "every service starts within its time window"
        starts.append(start)
        clock = start + service[customer]
        previous = customer
    depot_return = max(earliest[depot], clock + distance[previous][depot])
    assert depot_return <= latest[depot], "route returns within the depot window"
    sequence = [depot, *route, depot]
    route_distance = sum(distance[before][after] for before, after in zip(sequence, sequence[1:]))
    total_distance += route_distance
    route_records.append(
        {
            "nodes": [node_ids[customer] for customer in route],
            "starts": [start / scale for start in starts],
            "load": load,
            "distance": route_distance / scale,
        }
    )

fleet = sum(bool(route) for route in solution.lists)
assert list(solution.objectives) == [fleet, total_distance], "reported objectives match replay"
record = {
    "instance": name,
    "status": solution.status,
    "customers": len(customers),
    "vehicles": vehicles,
    "capacity": capacity,
    "objectives": [fleet, total_distance],
    "objective_convention": "fleet_then_dimacs_trunc1_distance",
    "distance_scale": scale,
    "rounding": args.rounding,
    "dual_bound": solution.dual_bound,
    "absolute_gap": solution.absolute_gap,
    "relative_gap": solution.relative_gap,
    "bound_method": solution.bound_method,
    "alns_iterations": solution.alns_iterations,
    "candidates_evaluated": solution.candidates_evaluated,
    "candidates_per_second": solution.candidates_per_second,
    "full_recompute_percentage": solution.full_recompute_percentage,
    "distance": total_distance / scale,
    "elapsed_seconds": elapsed,
    "seed": args.seed,
    "threads": args.threads,
    "engine": args.engine,
    "routing_two_way": args.routing_two_way,
    "routing_nearest_neighbor": args.routing_nearest_neighbor,
    "routing_warm_start": args.routing_warm_start,
    "routes": route_records,
    "verified": True,
}
if args.json:
    print(json.dumps(record, sort_keys=True))
else:
    certified = (
        f"  dual: {solution.dual_bound}  gap: {100 * solution.relative_gap:.2f}%  bound: {solution.bound_method}"
        if solution.dual_bound is not None
        else "  dual: unavailable"
    )
    print(f"instance: {name}  customers: {len(customers)}  vehicles<={vehicles}  capacity: {capacity}")
    print(f"status: {solution.status}  fleet: {fleet}  distance: {total_distance / scale:g}{certified}  elapsed: {elapsed:.3f}s")
    for index, route in enumerate(route_records):
        if route["nodes"]:
            print(f"  route {index}: load {route['load']:3d}  {[node_ids[depot], *route['nodes'], node_ids[depot]]}")
