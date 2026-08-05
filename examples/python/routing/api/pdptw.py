"""Pickup and delivery with capacity and precedence, using visit views.

Use ``--requests`` and ``--time-limit`` to control the generated instance.
"""

import argparse
import math
from random import Random

import qayd as cp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--requests", type=int, default=6)
parser.add_argument("--time-limit", type=int, default=8)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()
reqs = args.requests
time_limit = args.time_limit

rng = Random(0)
n = 2 * reqs
coords = [(50, 50)] + [(rng.randint(0, 100), rng.randint(0, 100)) for _ in range(n)]
load = [rng.randint(5, 15) for _ in range(reqs)]
demand = [0] + load + [-q for q in load]
capacity = 30
dist = [[round(math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])) for j in range(n + 1)] for i in range(n + 1)]
pairs = [(p, p + reqs) for p in range(1, reqs + 1)]
k = reqs

model = cp.Model()
customers = model.customers(range(1, n + 1))
for customer in customers:
    customer.demand = demand[customer.id]

routes = model.routes(customers, vehicles=k, depot=0, travel=dist)
for customer in customers:
    visit = routes[customer]
    model.add(visit.load_after >= 0)
    model.add(visit.load_after <= capacity)
for pickup, delivery in pairs:
    model.add(routes[pickup].route == routes[delivery].route)
    model.add(routes[pickup].position < routes[delivery].position)

model.minimize(routes.used_count())
model.then_minimize(routes.sum(lambda route: route.distance()))

solution = model.solve(time_limit=time_limit, verbose=args.verbose)

print(f"requests: {reqs}  capacity: {capacity}  status: {solution.status}")
if solution.lists is None:
    raise SystemExit(f"status: {solution.status} - no feasible plan within {time_limit}s")
fleet, distance = solution.objectives
print(f"fleet: {fleet}  total distance: {distance}")
list_of = {c: idx for idx, route in enumerate(solution.lists) for c in route}
for r, route in enumerate(solution.lists):
    if route:
        print(f"  route {r}: {[0, *route, 0]}")
for pickup, delivery in pairs:
    assert list_of[pickup] == list_of[delivery], f"pair {pickup}/{delivery} same vehicle"
    route = solution.lists[list_of[pickup]]
    assert route.index(pickup) < route.index(delivery), f"pickup {pickup} before delivery"
    acc = 0
    for c in route:
        acc += demand[c]
        assert 0 <= acc <= capacity, "load within capacity"
