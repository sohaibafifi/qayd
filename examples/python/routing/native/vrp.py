"""Capacitated VRP with an optional CVRPLIB instance.

Without a positional file this generates a deterministic CVRP. Pass a CVRPLIB
file to solve it directly:

    uv run examples/python/routing/native/vrp.py X-n101-k25.vrp --threads 4
"""

import argparse
import contextlib
import json
import math
import os
import sys
import time
from random import Random

import qayd as cp
from qayd.datasets import read_cvrplib, read_vrp_solution


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("instance", nargs="?", default=os.environ.get("QAYD_VRP_INSTANCE"))
parser.add_argument("--customers", type=int, default=int(os.environ.get("QAYD_VRP_N", "30")))
parser.add_argument("--vehicles", type=int, default=None)
parser.add_argument("--capacity", type=int, default=40, help="generated-instance capacity")
parser.add_argument("--time-limit", type=int, default=int(os.environ.get("QAYD_VRP_T", "10")))
parser.add_argument("--threads", type=int, default=int(os.environ.get("QAYD_VRP_THREADS", "1")))
parser.add_argument("--seed", type=int, default=int(os.environ.get("QAYD_VRP_SEED", "0")))
parser.add_argument("--engine", choices=("auto", "exact", "ls"), default="ls")
parser.add_argument("--solution", help="VRP solution file used as an LS warm start")
parser.add_argument("--verbose", action="store_true", default=os.environ.get("QAYD_VERBOSE") == "1")
parser.add_argument("--json", action="store_true", help="emit one machine-readable result")
args = parser.parse_args()

if args.threads <= 0 or args.time_limit < 0 or args.seed < 0:
    raise SystemExit("threads must be positive; time limit and seed must be non-negative")

instance = read_cvrplib(args.instance) if args.instance else None
if instance is None:
    n = args.customers
    rng = Random(args.seed)
    coordinates = [(50, 50)] + [(rng.randint(0, 100), rng.randint(0, 100)) for _ in range(n)]
    demand = [0] + [rng.randint(1, 9) for _ in range(n)]
    distance = [
        [round(math.hypot(x1 - x2, y1 - y2)) for x2, y2 in coordinates]
        for x1, y1 in coordinates
    ]
    depot = 0
    customers = list(range(1, n + 1))
    capacity = args.capacity
    vehicles = args.vehicles or (-(-sum(demand) // capacity) + 1)
    name = f"generated-cvrp-n{n}"
    node_ids = list(range(n + 1))
    best_known = None
else:
    demand = list(instance.demands)
    distance = [list(row) for row in instance.edge_weights]
    depot = instance.depot
    customers = list(instance.customers)
    capacity = instance.capacity
    vehicles = args.vehicles or instance.vehicles
    if vehicles is None:
        raise SystemExit("instance name has no -kN fleet; pass --vehicles")
    name = instance.name
    node_ids = list(instance.node_ids)
    best_known = instance.best_known

if vehicles <= 0 or capacity <= 0:
    raise SystemExit("vehicle count and capacity must be positive")
if args.solution and (instance is None or args.engine != "ls"):
    raise SystemExit("--solution requires a real instance and --engine ls")

D, Q = cp.matrix(distance), cp.array(demand)
model = cp.Model()
routes = model.list_vars(customers, count=vehicles)
model.minimize(cp.sum(cp.sum_edges(route, lambda i, j: D[i][j], start=depot, end=depot) for route in routes))
for route in routes:
    model.add(cp.sum(route, lambda customer: Q[customer]) <= capacity)

hint = None
if args.solution:
    hint = [list(route) for route in read_vrp_solution(args.solution, instance=instance).routes]
solve_options = {
    "engine": args.engine,
    "threads": args.threads,
    "time_limit": args.time_limit,
    "seed": args.seed,
    "verbose": args.verbose,
}
if hint is not None:
    solve_options["list_hint"] = hint
started = time.perf_counter()
output = contextlib.redirect_stdout(sys.stderr) if args.json else contextlib.nullcontext()
with output:
    solution = model.solve(**solve_options)
elapsed = time.perf_counter() - started

if solution.lists is None:
    record = {"instance": name, "status": solution.status, "elapsed_seconds": elapsed, "objectives": []}
    print(json.dumps(record, sort_keys=True) if args.json else f"instance: {name}  status: {solution.status}")
    raise SystemExit(0)

served = sorted(customer for route in solution.lists for customer in route)
assert served == sorted(customers), "every customer served exactly once"
route_records = []
total_distance = 0
for route in solution.lists:
    load = sum(demand[customer] for customer in route)
    assert load <= capacity, "capacity respected"
    sequence = [depot, *route, depot]
    route_distance = sum(distance[before][after] for before, after in zip(sequence, sequence[1:]))
    total_distance += route_distance
    route_records.append({"nodes": [node_ids[customer] for customer in route], "load": load, "distance": route_distance})
assert list(solution.objectives) == [total_distance], "reported objective matches replay"

record = {
    "instance": name,
    "status": solution.status,
    "customers": len(customers),
    "vehicles": vehicles,
    "vehicles_used": sum(bool(route) for route in solution.lists),
    "capacity": capacity,
    "objectives": [total_distance],
    "best_known": best_known,
    "elapsed_seconds": elapsed,
    "seed": args.seed,
    "threads": args.threads,
    "engine": args.engine,
    "routes": route_records,
    "verified": True,
}
if args.json:
    print(json.dumps(record, sort_keys=True))
else:
    gap = f"  known optimum: {best_known}" if best_known is not None else ""
    print(f"instance: {name}  customers: {len(customers)}  vehicles: {vehicles}  capacity: {capacity}")
    print(f"status: {solution.status}  distance: {total_distance}{gap}  elapsed: {elapsed:.3f}s")
    for index, route in enumerate(route_records):
        if route["nodes"]:
            print(f"  route {index}: load {route['load']:5d}  {[node_ids[depot], *route['nodes'], node_ids[depot]]}")
