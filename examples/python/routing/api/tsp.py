"""Travelling salesman, modeled with the routing convenience API.

Size and budget via ``QAYD_TSP_N`` / ``QAYD_TSP_T``.
"""

import math
import os
from random import Random

import qayd as cp

n = int(os.environ.get("QAYD_TSP_N", "15"))
time_limit = int(os.environ.get("QAYD_TSP_T", "3"))

rng = Random(0)
coords = [(50, 50)] + [(rng.randint(0, 100), rng.randint(0, 100)) for _ in range(n)]
dist = [[round(math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])) for j in range(n + 1)] for i in range(n + 1)]

model = cp.Model()
customers = model.customers(range(1, n + 1))
routes = model.routes(customers, vehicles=1, depot=0, travel=dist)
model.minimize(routes.sum(lambda route: route.distance()))

solution = model.solve(time_limit=time_limit, verbose=os.environ.get("QAYD_VERBOSE") == "1")

print(f"cities: {n}  status: {solution.status}")
if solution.lists is None:
    raise SystemExit(f"status: {solution.status} - no solution within {time_limit}s")
print(f"tour length: {solution.objective}")
print(f"tour: {[0, *solution.lists[0], 0]}")
assert sorted(solution.lists[0]) == list(range(1, n + 1)), "every city visited once"
