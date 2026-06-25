"""Mining scheduling with a risk-quantile objective. Order n blocks for
extraction (one permutation list). A block extracted at 1-indexed position p
realises value ``g - rate*p`` (later extraction is discounted). The objective is
risk-averse: maximise the value-at-risk, i.e. the k-th smallest realised value
across the blocks, computed by sorting them.

``cp.select_kth`` is the new general order-statistic reduction: it scans the list
(here the scan accumulator is the position) and returns the k-th smallest of the
per-step emitted values. No float variables are involved.

Tune via ``QAYD_MSP_N`` / ``QAYD_MSP_T``.
"""

import os
from itertools import permutations
from random import Random

import qayd as cp

n = int(os.environ.get("QAYD_MSP_N", "7"))
time_limit = int(os.environ.get("QAYD_MSP_T", "5"))

rng = Random(0)
g = [rng.randint(1, 20) for _ in range(n)]
rate = 2
k = n // 4  # a low quantile: value-at-risk

G = cp.array(g)
model = cp.Model()
(seq,) = model.list_vars(1, list(range(n)))
# scan accumulator = 1-indexed position; emit the realised (discounted) value.
model.maximize(
    cp.select_kth(
        seq,
        k,
        step=lambda cur, acc, prev: acc + 1,
        emit=lambda cur, acc, prev: G[cur] - acc * rate,
        init=0,
        boundary=n,
    ),
)

solution = model.solve(time_limit=time_limit, verbose=os.environ.get("QAYD_VERBOSE") == "1")

print(f"blocks: {n}  rate: {rate}  quantile k: {k}  status: {solution.status}")
if solution.routes is None:
    raise SystemExit(f"status: {solution.status} - no ordering within {time_limit}s")
order = solution.routes[0]
var = solution.objective
print(f"value-at-risk (k-th smallest realised value): {var}")


def quantile(perm):
    realised = sorted(g[block] - rate * (pos + 1) for pos, block in enumerate(perm))
    return realised[k]


# Replay the reported ordering, then brute-force the optimum for small n.
assert quantile(order) == var, "reported quantile matches the replayed ordering"
assert sorted(order) == list(range(n)), "every block scheduled once"
best = max(quantile(p) for p in permutations(range(n)))
print(f"optimum (brute force): {best}")
assert var == best, "local search reaches the optimal value-at-risk"
