"""Team orienteering with the routing convenience API.

Tune via ``QAYD_TOP_N`` / ``QAYD_TOP_K`` / ``QAYD_TOP_T``; trace
``QAYD_VERBOSE=1``.
"""

import math
import os
from random import Random

import qayd as cp

n = int(os.environ.get("QAYD_TOP_N", "20"))
k = int(os.environ.get("QAYD_TOP_K", "2"))
time_limit = int(os.environ.get("QAYD_TOP_T", "8"))

rng = Random(0)
coords = [(50, 50)] + [(rng.randint(0, 100), rng.randint(0, 100)) for _ in range(n)]
profit = [0] + [rng.randint(1, 10) for _ in range(n)]
dist = [[round(math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])) for j in range(n + 1)] for i in range(n + 1)]
budget = 200

model = cp.Model()
customers = model.customers(range(1, n + 1))
for customer in customers:
    customer.profit = profit[customer.id]

routes = model.routes(customers, vehicles=k, depot=0, travel=dist, optional=True)
model.maximize(routes.sum(lambda route: route.profit()))
for route in routes:
    model.add(route.distance() <= budget)

solution = model.solve(time_limit=time_limit, verbose=os.environ.get("QAYD_VERBOSE") == "1")

print(f"customers: {n}  vehicles: {k}  budget: {budget}  status: {solution.status}")
if solution.lists is None:
    raise SystemExit(f"status: {solution.status} - infeasible within {time_limit}s")
served = solution.lists[:k]
pool = solution.lists[k]
collected = sum(profit[c] for route in served for c in route)
print(f"collected profit: {solution.objective} (check {collected})  visited: {sum(len(r) for r in served)}/{n}")
for r, route in enumerate(served):
    length = sum(dist[a][b] for a, b in zip([0, *route], [*route, 0]))
    print(f"  route {r}: length {length:3d}  profit {sum(profit[c] for c in route):3d}  {[0, *route, 0]}")
    assert length <= budget, "route within budget"

assert solution.objective == collected
assert sorted(c for route in solution.lists for c in route) == list(range(1, n + 1)), "every customer placed once"
