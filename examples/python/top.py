"""Team orienteering: k vehicles, each with a route-length budget, collect as
much customer profit as possible. Not every customer is visited (prize
collecting), so the routes are created with ``optional=True``, which adds a
hidden pool list holding the unvisited customers. Maximise total collected
profit subject to each route's length cap.

Tune via ``QAYD_TOP_N`` / ``QAYD_TOP_K`` / ``QAYD_TOP_T``; trace ``QAYD_VERBOSE=1``.
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
budget = 200  # max length per route

D, P = cp.matrix(dist), cp.array(profit)
model = cp.Model()
routes = model.list_vars(list(range(1, n + 1)), count=k, optional=True)  # unvisited go to the pool
model.maximize(cp.sum(cp.sum(r, lambda i: P[i]) for r in routes))
for r in routes:
    model.add(cp.sum_edges(r, lambda i, j: D[i][j], start=0, end=0) <= budget)

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
