"""Quadratic assignment: assign n facilities to n locations to minimise the
total flow-times-distance cost ``sum a[i][j] * b[p[i]][p[j]]`` over location
pairs, where ``a`` is the location distance matrix and ``b`` the facility flow.

Modeled with a single list variable that is a permutation of the facilities over
the locations (``list_vars(facilities, count=1)`` puts every facility in the one list).
The quadratic objective is a ``pos_pairs`` reduction over ordered position pairs.

Use ``--size`` and ``--time-limit`` to control the generated instance.
"""

import argparse
from random import Random

import qayd as cp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--size", type=int, default=10)
parser.add_argument("--time-limit", type=int, default=5)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()
n = args.size
time_limit = args.time_limit

rng = Random(0)
dist = [[0 if i == j else rng.randint(1, 20) for j in range(n)] for i in range(n)]
flow = [[0 if i == j else rng.randint(1, 20) for j in range(n)] for i in range(n)]

A = cp.matrix(dist)  # location-to-location distances, indexed by positions
B = cp.matrix(flow)  # facility-to-facility flows, indexed by assigned facilities

model = cp.Model()
(p,) = model.list_vars(list(range(n)), count=1)  # p[i] = facility placed at location i
model.minimize(cp.pos_pairs(p, lambda a, b, i, j: A[i][j] * B[a][b]))

solution = model.solve(time_limit=time_limit, verbose=args.verbose)

print(f"facilities/locations: {n}  status: {solution.status}")
if solution.lists is None:
    raise SystemExit(f"status: {solution.status} - no solution within {time_limit}s")
assignment = solution.lists[0]
print(f"cost: {solution.objective}")
print(f"assignment (location -> facility): {assignment}")
assert sorted(assignment) == list(range(n)), "every facility assigned once"
