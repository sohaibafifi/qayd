"""Quadratic assignment: assign n facilities to n locations to minimise the
total flow-times-distance cost ``sum a[i][j] * b[p[i]][p[j]]`` over location
pairs, where ``a`` is the location distance matrix and ``b`` the facility flow.

Modeled with a single list variable that is a permutation of the facilities over
the locations (``list_vars(facilities, count=1)`` puts every facility in the one list).
The quadratic objective is a ``pos_pairs`` reduction over ordered position pairs.

Tune via ``QAYD_QAP_N`` / ``QAYD_QAP_T``; trace with ``QAYD_VERBOSE=1``.
"""

import os
from random import Random

import qayd as cp

n = int(os.environ.get("QAYD_QAP_N", "10"))
time_limit = int(os.environ.get("QAYD_QAP_T", "5"))

rng = Random(0)
dist = [[0 if i == j else rng.randint(1, 20) for j in range(n)] for i in range(n)]
flow = [[0 if i == j else rng.randint(1, 20) for j in range(n)] for i in range(n)]

A = cp.matrix(dist)  # location-to-location distances, indexed by positions
B = cp.matrix(flow)  # facility-to-facility flows, indexed by assigned facilities

model = cp.Model()
(p,) = model.list_vars(list(range(n)), count=1)  # p[i] = facility placed at location i
model.minimize(cp.pos_pairs(p, lambda a, b, i, j: A[i][j] * B[a][b]))

solution = model.solve(time_limit=time_limit, verbose=os.environ.get("QAYD_VERBOSE") == "1")

print(f"facilities/locations: {n}  status: {solution.status}")
if solution.lists is None:
    raise SystemExit(f"status: {solution.status} - no solution within {time_limit}s")
assignment = solution.lists[0]
print(f"cost: {solution.objective}")
print(f"assignment (location -> facility): {assignment}")
assert sorted(assignment) == list(range(n)), "every facility assigned once"
