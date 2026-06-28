"""Aircraft landing: order the planes on a runway to minimise total
earliness/tardiness cost around each plane's target time, respecting separation
times between consecutive landings and a latest-landing deadline.

The landing order is one list variable (a permutation of the planes). A prefix
scan computes each plane's landing time ``max(earliest, prev_landing +
separation[prev][cur])``; the objective sums the earliness/tardiness penalty via
a ternary, and a second scan constrains landings to their deadline.

Tune via ``QAYD_ALP_N`` / ``QAYD_ALP_T``; trace with ``QAYD_VERBOSE=1``.
"""

import os
from random import Random

import qayd as cp

n = int(os.environ.get("QAYD_ALP_N", "8"))
time_limit = int(os.environ.get("QAYD_ALP_T", "5"))

rng = Random(0)
earliest = [rng.randint(0, 20) for _ in range(n)]
target = [earliest[i] + rng.randint(10, 40) for i in range(n)]
latest = [target[i] + rng.randint(20, 60) for i in range(n)]
early_cost = [rng.randint(1, 3) for _ in range(n)]
tardy_cost = [rng.randint(2, 5) for _ in range(n)]
# Separation gap between consecutive planes; row/col n is the (zero) boundary.
sep = [[0 if i == n or j == n else rng.randint(3, 8) for j in range(n + 1)] for i in range(n + 1)]
boundary = n

E, T, L, EC, TC, SEP = (cp.array(earliest), cp.array(target), cp.array(latest), cp.array(early_cost), cp.array(tardy_cost), cp.matrix(sep))

model = cp.Model()
(order,) = model.list_vars(list(range(n)), count=1)

def land(cur, acc, prev):
    return cp.max(E[cur], acc + SEP[prev][cur])

# Earliness/tardiness cost objective.
model.minimize(
    cp.scan_sum(
        order,
        step=land,
        emit=lambda cur, t, prev: cp.if_(t < T[cur], EC[cur] * (T[cur] - t), TC[cur] * (t - T[cur])),
        init=0,
        boundary=boundary,
    )
)
# Hard latest-landing deadline.
model.add(cp.scan_sum(order, step=land, emit=lambda cur, t, prev: cp.max(0, t - L[cur]), init=0, boundary=boundary) <= 0)

solution = model.solve(time_limit=time_limit, verbose=os.environ.get("QAYD_VERBOSE") == "1")

print(f"planes: {n}  status: {solution.status}")
if solution.lists is None:
    raise SystemExit(f"status: {solution.status} - no on-time order within {time_limit}s")
seq = solution.lists[0]
print(f"total earliness/tardiness cost: {solution.objective}")
acc, prev, total = 0, boundary, 0
for c in seq:
    acc = max(earliest[c], acc + sep[prev][c])
    assert acc <= latest[c], "lands within its deadline"
    pen = early_cost[c] * (target[c] - acc) if acc < target[c] else tardy_cost[c] * (acc - target[c])
    total += pen
    prev = c
print(f"landing order: {seq}")
assert total == solution.objective, "reported cost matches the replayed schedule"
assert sorted(seq) == list(range(n)), "every plane scheduled once"
