"""Travelling salesman, modeled the usual way: build a Model, add a list
variable for the visiting order, set the objective, call solve(). The solver
picks the list-domain engine because the model uses a list variable.

Use ``--cities`` and ``--time-limit`` to control the generated instance.
"""

import argparse
import math
from random import Random

import qayd as cp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cities", type=int, default=15)
parser.add_argument("--time-limit", type=int, default=3)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()
n = args.cities
time_limit = args.time_limit

rng = Random(0)
# Node 0 is the start/end depot; nodes 1..n are the cities to order.
coords = [(50, 50)] + [(rng.randint(0, 100), rng.randint(0, 100)) for _ in range(n)]
dist = [[round(math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])) for j in range(n + 1)] for i in range(n + 1)]

D = cp.matrix(dist)                                    # constant table, indexable inside lambdas

model = cp.Model()
(tour,) = model.list_vars(list(range(1, n + 1)), count=1)    # one list = one tour over the cities
# Closed-tour distance: sum dist[i][j] over the edges, depot at both ends.
model.minimize(cp.sum_edges(tour, lambda i, j: D[i][j], start=0, end=0))

solution = model.solve(time_limit=time_limit, verbose=args.verbose)

print(f"cities: {n}  status: {solution.status}")
if solution.lists is None:
    raise SystemExit(f"status: {solution.status} - no solution within {time_limit}s")
print(f"tour length: {solution.objective}")
print(f"tour: {[0, *solution.lists[0], 0]}")
assert sorted(solution.lists[0]) == list(range(1, n + 1)), "every city visited once"
