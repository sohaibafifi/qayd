"""Capacitated VRP through the routing convenience API.

The command is identical to ``routing/native/vrp.py``. Without a positional
file it generates the same deterministic CVRP. With a CVRPLIB file it solves
that instance directly:

    uv run examples/python/routing/api/vrp.py X-n101-k25.vrp --threads 4
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
parser.add_argument("instance", nargs="?")
parser.add_argument("--customers", type=int, default=30)
parser.add_argument("--vehicles", type=int)
parser.add_argument("--capacity", type=int, default=40, help="generated-instance capacity")
parser.add_argument("--time-limit", type=int, default=10)
parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1), help="portfolio workers (default: up to 8 cores)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max-iterations", type=int)
parser.add_argument("--profile", action="store_true")
parser.add_argument("--routing-two-way", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--routing-nearest-neighbor", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--routing-warm-start", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--linear-backend", choices=("auto", "native", "amthal"), default="auto")
parser.add_argument("--lp-root-ms", type=int, default=1000, help="route-master LP budget in milliseconds")
parser.add_argument("--lp-max-variables", type=int, default=2000)
parser.add_argument("--lp-max-rows", type=int, default=1000)
parser.add_argument("--lp-max-nonzeros", type=int, default=100000)
parser.add_argument("--engine", choices=("auto", "exact", "ls"), default="ls")
parser.add_argument("--solution", help="VRP solution file used as an LS warm start")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--json", action="store_true", help="emit one machine-readable result")
args = parser.parse_args()

if (
    args.threads <= 0
    or args.time_limit < 0
    or args.seed < 0
    or args.lp_root_ms < 0
    or args.lp_max_variables <= 0
    or args.lp_max_rows <= 0
    or args.lp_max_nonzeros <= 0
    or (args.max_iterations is not None and args.max_iterations < 0)
):
    raise SystemExit("threads and LP size limits must be positive; time limits and seed must be non-negative")

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
    customer_ids = list(range(1, n + 1))
    capacity = args.capacity
    vehicles = args.vehicles or (-(-sum(demand) // capacity) + 1)
    minimum_vehicles = None
    name = f"generated-cvrp-n{n}"
    node_ids = list(range(n + 1))
    best_known = None
else:
    demand = list(instance.demands)
    distance = [list(row) for row in instance.edge_weights]
    depot = instance.depot
    customer_ids = list(instance.customers)
    capacity = instance.capacity
    minimum_vehicles = instance.vehicles
    vehicles = args.vehicles or len(customer_ids)
    name = instance.name
    node_ids = list(instance.node_ids)
    best_known = instance.best_known

if vehicles <= 0 or capacity <= 0:
    raise SystemExit("vehicle count and capacity must be positive")
objective_convention = (
    "cvrplib_unlimited_fleet_distance"
    if instance is not None and args.vehicles is None
    else "distance_with_vehicle_limit"
)
if args.solution and (instance is None or args.engine != "ls"):
    raise SystemExit("--solution requires a real instance and --engine ls")

model = cp.Model()
customers = model.customers(customer_ids)
for customer in customers:
    customer.demand = demand[customer.id]
routes = model.routes(customers, vehicles=vehicles, depot=depot, travel=distance)
for route in routes:
    model.add(route.sum(lambda customer: customer.demand) <= capacity)
model.minimize(routes.total_distance())

hint = None
if args.solution:
    raw_hint = [list(route) for route in read_vrp_solution(args.solution).routes]
    universe = set(customer_ids)
    hint = (
        raw_hint
        if all(customer in universe for route in raw_hint for customer in route)
        else [[instance.normalize_node(customer) for customer in route] for route in raw_hint]
    )
    if len(hint) > vehicles:
        raise SystemExit(f"solution uses {len(hint)} routes but the model allows only {vehicles}")
    hint.extend([] for _ in range(vehicles - len(hint)))
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
    "linear_backend": args.linear_backend,
    "lp_root_ms": args.lp_root_ms,
    "lp_max_variables": args.lp_max_variables,
    "lp_max_rows": args.lp_max_rows,
    "lp_max_nonzeros": args.lp_max_nonzeros,
}
if hint is not None:
    solve_options["list_hint"] = hint
started = time.perf_counter()
output = contextlib.redirect_stdout(sys.stderr) if args.json else contextlib.nullcontext()
with output:
    solution = model.solve(**solve_options)
elapsed = time.perf_counter() - started
construction_record = {
    "backend_build_seconds": solution.backend_build_seconds,
    "construction_seconds": solution.construction_seconds,
    "time_to_first_feasible": solution.time_to_first_feasible,
    "construction_candidates": solution.construction_candidates,
    "estimated_backend_bytes": solution.estimated_backend_bytes,
    "constructor": solution.constructor,
    "constructor_fleet": solution.constructor_fleet,
    "constructor_cost": solution.constructor_cost,
    "anytime_checkpoints": None
    if solution.anytime_checkpoints is None
    else [
        {
            "target_nanos": target,
            "observed_nanos": observed,
            "feasible": feasible,
            "objectives": objectives,
            "fleet": fleet,
            "candidates": candidates,
        }
        for target, observed, feasible, objectives, fleet, candidates in solution.anytime_checkpoints
    ],
    "neighborhood_profile": None
    if solution.neighborhood_profile is None
    else [
        {
            "name": name,
            "uses": uses,
            "generated": generated,
            "evaluated": evaluated,
            "cpu_nanos": cpu_nanos,
            "improvements": improvements,
            "global_bests": global_bests,
            "positive_rewards": positive_rewards,
            "weight": weight,
        }
        for name, uses, generated, evaluated, cpu_nanos, improvements, global_bests, positive_rewards, weight in solution.neighborhood_profile
    ],
    "routing_counters": None if solution.routing_counters is None else dict(solution.routing_counters),
    "lp_root_bound": solution.stats.lp_root_bound,
    "lp_model_status": solution.stats.lp_model_status,
    "lp_solves": solution.stats.lp_solves,
    "lp_certified": solution.stats.lp_certified,
    "lp_simplex_time_ms": solution.stats.lp_micros / 1000,
}

if solution.lists is None:
    record = {
        **construction_record,
        "instance": name,
        "status": solution.status,
        "elapsed_seconds": elapsed,
        "objectives": [],
        "objective_convention": objective_convention,
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
        "linear_backend": args.linear_backend,
        "lp_root_ms": args.lp_root_ms,
    }
    print(json.dumps(record, sort_keys=True) if args.json else f"instance: {name}  status: {solution.status}")
    raise SystemExit(0)

served = sorted(customer for route in solution.lists for customer in route)
assert served == sorted(customer_ids), "every customer served exactly once"
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
    **construction_record,
    "instance": name,
    "status": solution.status,
    "customers": len(customer_ids),
    "vehicles": vehicles,
    "minimum_vehicles": minimum_vehicles,
    "vehicles_used": sum(bool(route) for route in solution.lists),
    "capacity": capacity,
    "objectives": [total_distance],
    "objective_convention": objective_convention,
    "dual_bound": solution.dual_bound,
    "absolute_gap": solution.absolute_gap,
    "relative_gap": solution.relative_gap,
    "bound_method": solution.bound_method,
    "alns_iterations": solution.alns_iterations,
    "candidates_evaluated": solution.candidates_evaluated,
    "candidates_per_second": solution.candidates_per_second,
    "full_recompute_percentage": solution.full_recompute_percentage,
    "best_known": best_known,
    "elapsed_seconds": elapsed,
    "seed": args.seed,
    "threads": args.threads,
    "engine": args.engine,
    "routing_two_way": args.routing_two_way,
    "routing_nearest_neighbor": args.routing_nearest_neighbor,
    "routing_warm_start": args.routing_warm_start,
    "linear_backend": args.linear_backend,
    "lp_root_ms": args.lp_root_ms,
    "routes": route_records,
    "verified": True,
}
if args.json:
    print(json.dumps(record, sort_keys=True))
else:
    known = f"  known optimum: {best_known}" if best_known is not None else ""
    certified = (
        f"  dual: {solution.dual_bound}  gap: {100 * solution.relative_gap:.2f}%  bound: {solution.bound_method}"
        if solution.dual_bound is not None
        else "  dual: unavailable"
    )
    print(f"instance: {name}  customers: {len(customer_ids)}  vehicles: {vehicles}  capacity: {capacity}")
    print(f"status: {solution.status}  distance: {total_distance}{known}{certified}  elapsed: {elapsed:.3f}s")
    if solution.stats.lp_model_status != "not-attempted":
        print(
            f"  route LP: {solution.stats.lp_root_bound}  solves: {solution.stats.lp_solves}"
            f"  certified: {solution.stats.lp_certified}  simplex: {solution.stats.lp_micros / 1000:.3f} ms"
        )
    for index, route in enumerate(route_records):
        if route["nodes"]:
            print(f"  route {index}: load {route['load']:5d}  {[node_ids[depot], *route['nodes'], node_ids[depot]]}")
