"""Capacitated vehicle routing with time windows.

Adds time windows to the CVRP model. The end-of-service time along each route is
a prefix scan: ``end = max(earliest[cust], arrival) + service[cust]``, threaded
with ``cp.scan_sum``. Lateness ``max(0, end - latest[cust])`` is summed per route
and constrained to 0, so the violation-driven search drives windows to feasible
with the same feasibility-first scoring used by the list-domain engine. The
objective is lexicographic: fewest vehicles, then shortest distance.

Tune via ``QAYD_CVRPTW_N`` / ``QAYD_CVRPTW_T``; trace with ``QAYD_VERBOSE=1``.
"""

import math
import os
from random import Random

import qayd as cp

n = int(os.environ.get("QAYD_CVRPTW_N", "15"))
time_limit = int(os.environ.get("QAYD_CVRPTW_T", "10"))

rng = Random(0)
# Node 0 is the depot; 1..n are customers with a demand, service time, window.
coords = [(50, 50)] + [(rng.randint(0, 100), rng.randint(0, 100)) for _ in range(n)]
demand = [0] + [rng.randint(1, 9) for _ in range(n)]
service = [0] + [10 for _ in range(n)]
dist = [[round(math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])) for j in range(n + 1)] for i in range(n + 1)]
# Each customer opens at a random time and stays open for a generous span; the
# depot-to-customer travel keeps some routes window-constrained.
earliest = [0] + [rng.randint(0, 60) for _ in range(n)]
latest = [0] + [earliest[i] + 80 for i in range(1, n + 1)]
capacity = 40
min_k = -(-sum(demand) // capacity)
k = min_k + 3

D, Q, E, L, S = cp.matrix(dist), cp.array(demand), cp.array(earliest), cp.array(latest), cp.array(service)

model = cp.Model()
routes = model.list_vars(list(range(1, n + 1)), count=k)
model.minimize(cp.sum(cp.used(r) for r in routes))                                  # fewest vehicles
model.then_minimize(cp.sum(cp.sum_edges(r, lambda i, j: D[i][j], start=0, end=0) for r in routes))  # then distance
for r in routes:
    model.add(cp.sum(r, lambda i: Q[i]) <= capacity)
    lateness = cp.scan_sum(
        r,
        step=lambda cur, acc, prev: cp.max(E[cur], acc + D[prev][cur]) + S[cur],
        emit=lambda cur, end, prev: cp.max(0, end - L[cur]),
        init=0,
        boundary=0,
    )
    model.add(lateness <= 0)

solution = model.solve(time_limit=time_limit, verbose=os.environ.get("QAYD_VERBOSE") == "1")

print(f"customers: {n}  vehicles<={k}  capacity: {capacity}  status: {solution.status}")
if solution.lists is None:
    raise SystemExit(f"status: {solution.status} - no feasible plan within {time_limit}s")
fleet, distance = solution.objectives
print(f"fleet: {fleet}  total distance: {distance}")
for r, route in enumerate(solution.lists):
    if not route:
        continue
    # Replay the schedule to show every stop is on time.
    acc, prev = 0, 0
    for c in route:
        acc = max(earliest[c], acc + dist[prev][c]) + service[c]
        assert acc <= latest[c], "every stop served within its window"
        prev = c
    print(f"  route {r}: load {sum(demand[c] for c in route):3d}  {[0, *route, 0]}")

assert sorted(c for route in solution.lists for c in route) == list(range(1, n + 1))
