"""Bin packing: pack items of given weights into the fewest bins of capacity C.

Modeled with one list variable per candidate bin (an upper bound of bins), a
per-bin capacity constraint, and a single objective: minimise the number of
non-empty bins via ``cp.used``. This is the same partition-plus-capacity shape
as CVRP, with the objective counting used lists instead of distance.

Use ``--items`` and ``--time-limit`` to control the generated instance.
"""

import argparse
import math
from random import Random

import qayd as cp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--items", type=int, default=20)
parser.add_argument("--time-limit", type=int, default=5)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()
n = args.items
time_limit = args.time_limit
capacity = 100

rng = Random(0)
# Item ids are 1..n; index 0 is unused so weight[i] reads by item id.
weight = [0] + [rng.randint(20, 60) for _ in range(n)]
items = list(range(1, n + 1))
lower_bound = math.ceil(sum(weight) / capacity)

W = cp.array(weight)
model = cp.Model()
bins = model.list_vars(items, count=n)                       # up to n bins available
model.minimize(cp.sum(cp.used(b) for b in bins))  # minimise non-empty bins
for b in bins:
    model.add(cp.sum(b, lambda i: W[i]) <= capacity)   # each bin within capacity

solution = model.solve(time_limit=time_limit, verbose=args.verbose)

print(f"items: {n}  capacity: {capacity}  status: {solution.status}")
if solution.lists is None:
    raise SystemExit(f"status: {solution.status} - no feasible packing within {time_limit}s")
used_bins = [b for b in solution.lists if b]
print(f"bins used: {len(used_bins)}  (lower bound {lower_bound})")
for k, b in enumerate(used_bins):
    print(f"  bin {k}: load {sum(weight[i] for i in b):3d}  {b}")

assert sorted(i for b in solution.lists for i in b) == items, "every item packed once"
for b in solution.lists:
    assert sum(weight[i] for i in b) <= capacity, "capacity respected"
