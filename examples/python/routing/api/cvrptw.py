"""Capacitated vehicle routing with time windows, using per-customer visit views.

Tune via ``QAYD_CVRPTW_N`` / ``QAYD_CVRPTW_T``; trace with ``QAYD_VERBOSE=1``.
"""

import math
import os
from random import Random

import qayd as cp

n = int(os.environ.get("QAYD_CVRPTW_N", "15"))
time_limit = int(os.environ.get("QAYD_CVRPTW_T", "10"))

rng = Random(0)
coords = [(50, 50)] + [(rng.randint(0, 100), rng.randint(0, 100)) for _ in range(n)]
demand = [0] + [rng.randint(1, 9) for _ in range(n)]
service = [0] + [10 for _ in range(n)]
dist = [[round(math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])) for j in range(n + 1)] for i in range(n + 1)]
earliest = [0] + [rng.randint(0, 60) for _ in range(n)]
latest = [0] + [earliest[i] + 80 for i in range(1, n + 1)]
capacity = 40
min_k = -(-sum(demand) // capacity)
k = min_k + 3

model = cp.Model()
customers = model.customers(range(1, n + 1))
for customer in customers:
    customer.demand = demand[customer.id]
    customer.service = service[customer.id]
    customer.earliest = earliest[customer.id]
    customer.latest = latest[customer.id]

routes = model.routes(customers, vehicles=k, depot=0, travel=dist)
for customer in customers:
    visit = routes[customer]
    model.add(visit.start >= customer.earliest)
    model.add(visit.start <= customer.latest)
for route in routes:
    model.add(route.sum(lambda customer: customer.demand) <= capacity)

model.minimize(routes.used_count())
model.then_minimize(routes.sum(lambda route: route.distance()))

solution = model.solve(time_limit=time_limit, verbose=os.environ.get("QAYD_VERBOSE") == "1")

print(f"customers: {n}  vehicles<={k}  capacity: {capacity}  status: {solution.status}")
if solution.lists is None:
    raise SystemExit(f"status: {solution.status} - no feasible plan within {time_limit}s")
fleet, distance = solution.objectives
print(f"fleet: {fleet}  total distance: {distance}")
for r, route in enumerate(solution.lists):
    if not route:
        continue
    acc, prev = 0, 0
    for c in route:
        start = max(earliest[c], acc + dist[prev][c])
        assert start <= latest[c], "every stop starts within its window"
        acc = start + service[c]
        prev = c
    print(f"  route {r}: load {sum(demand[c] for c in route):3d}  {[0, *route, 0]}")

assert sorted(c for route in solution.lists for c in route) == list(range(1, n + 1))
