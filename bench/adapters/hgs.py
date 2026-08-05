#!/usr/bin/env python3
"""Verified HGS-CVRP command adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time

from qayd.datasets import read_cvrplib, read_vrp_solution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1, help="recorded for fairness; HGS-CVRP is single-threaded")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not args.binary.is_file():
        raise SystemExit(f"HGS-CVRP binary not found: {args.binary}")
    if args.time_limit <= 0 or args.seed < 0 or args.threads <= 0:
        raise SystemExit("time limit and threads must be positive; seed must be non-negative")

    instance = read_cvrplib(args.instance)
    with tempfile.TemporaryDirectory(prefix="qayd_hgs_") as directory:
        solution_path = Path(directory) / "solution.sol"
        command = [
            str(args.binary.resolve()), str(Path(args.instance).resolve()), str(solution_path),
            "-t", str(args.time_limit), "-seed", str(args.seed), "-log", "0",
        ]
        started = time.perf_counter()
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        elapsed = time.perf_counter() - started
        record = {
            "instance": instance.name,
            "status": "ERROR" if result.returncode else "UNKNOWN",
            "objectives": [],
            "dual_bound": None,
            "absolute_gap": None,
            "relative_gap": None,
            "bound_method": None,
            "elapsed_seconds": elapsed,
            "objective_convention": "cvrplib_unlimited_fleet_distance",
            "solver_engine": "HGS-CVRP",
            "effective_threads": 1,
            "seed_effective": True,
            "verified": False,
        }
        if result.returncode or not solution_path.is_file():
            record["diagnostic"] = (result.stderr or result.stdout)[-2000:]
        else:
            raw_solution = read_vrp_solution(solution_path)
            try:
                normalized_routes = tuple(
                    tuple(instance.customers[node - 1] for node in route)
                    for route in raw_solution.routes
                )
            except (IndexError, TypeError):
                raise AssertionError("HGS solution contains a customer index outside the instance") from None
            served = sorted(node for route in normalized_routes for node in route)
            assert served == sorted(instance.customers)
            total = 0
            routes = []
            for route in normalized_routes:
                load = sum(instance.demands[node] for node in route)
                assert load <= instance.capacity
                sequence = [instance.depot, *route, instance.depot]
                distance = sum(instance.edge_weights[a][b] for a, b in zip(sequence, sequence[1:]))
                total += distance
                routes.append({
                    "nodes": [instance.node_ids[node] for node in route],
                    "load": load,
                    "distance": distance,
                })
            if raw_solution.cost is not None:
                assert math_isclose(total, raw_solution.cost)
            record.update({
                "status": "SATISFIABLE",
                "objectives": [total],
                "routes": routes,
                "vehicles_used": len(normalized_routes),
                "minimum_vehicles": instance.vehicles,
                "best_known": instance.best_known,
                "verified": True,
            })
    print(json.dumps(record, sort_keys=True) if args.json else record)


def math_isclose(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-6 * max(1.0, abs(left), abs(right))


if __name__ == "__main__":
    main()
