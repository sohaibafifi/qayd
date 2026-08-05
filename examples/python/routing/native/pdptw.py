"""Pickup and delivery with time windows (capacity + precedence core).

Each request is a pickup node and a delivery node that must be served by the
same vehicle (``model.same_list``) with the pickup before the delivery (a
``pos_pairs`` order check). The vehicle load is a prefix scan that rises on a
pickup and falls on a delivery and must stay within ``[0, capacity]``. Objective:
fewest vehicles, then shortest distance.

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
# Nodes 1..reqs are pickups, reqs+1..2*reqs the matching deliveries; node 0 depot.
coords = [(50, 50)] + [(rng.randint(0, 100), rng.randint(0, 100)) for _ in range(n)]
load = [rng.randint(5, 15) for _ in range(reqs)]
demand = [0] + load + [-q for q in load]            # pickup +q, delivery -q
capacity = 30
dist = [[round(math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])) for j in range(n + 1)] for i in range(n + 1)]
pairs = [(p, p + reqs) for p in range(1, reqs + 1)]  # (pickup, delivery)
# is_delivery_of[a][b] = 1 iff a is the delivery of pickup b.
is_del = [[1 if (a, b) in [(d, p) for p, d in pairs] else 0 for b in range(n + 1)] for a in range(n + 1)]

D, Q, M = cp.matrix(dist), cp.array(demand), cp.matrix(is_del)
k = reqs  # an upper bound on vehicles

model = cp.Model()
routes = model.list_vars(list(range(1, n + 1)), count=k)
model.minimize(cp.sum(cp.used(r) for r in routes))
model.then_minimize(cp.sum(cp.sum_edges(r, lambda i, j: D[i][j], start=0, end=0) for r in routes))
for r in routes:
    # Load stays within [0, capacity] along the route.
    model.add(cp.scan_sum(r, step=lambda cur, acc, prev: acc + Q[cur], emit=lambda cur, acc, prev: cp.max(0, acc - capacity) + cp.max(0, -acc), init=0, boundary=0) <= 0)
    # A delivery may not appear before its pickup in the same route.
    model.add(cp.pos_pairs(r, lambda a, b, i, j: (i < j) * M[a][b]) <= 0)
for p, d in pairs:
    model.same_list(p, d)  # pickup and delivery on the same vehicle

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
for p, d in pairs:
    assert list_of[p] == list_of[d], f"pair {p}/{d} same vehicle"
    route = solution.lists[list_of[p]]
    assert route.index(p) < route.index(d), f"pickup {p} before delivery {d}"
    acc = 0
    for c in route:
        acc += demand[c]
        assert 0 <= acc <= capacity, "load within capacity"
