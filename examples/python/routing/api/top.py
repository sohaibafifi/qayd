"""Team orienteering with the routing convenience API.

Use ``--customers``, ``--vehicles`` and ``--time-limit`` to control the example.
"""

import argparse
import math
from random import Random

import qayd as cp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--customers", type=int, default=20)
parser.add_argument("--vehicles", type=int, default=2)
parser.add_argument("--time-limit", type=int, default=8)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()
n = args.customers
k = args.vehicles
time_limit = args.time_limit

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

solution = model.solve(time_limit=time_limit, verbose=args.verbose)

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
