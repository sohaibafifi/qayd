"""Resource-constrained project scheduling (RCPSP): minimise the makespan of
tasks with durations, precedences, and renewable resources of fixed capacity.

One interval variable per task; ``model.precedes`` for each precedence edge;
``model.resource`` caps the total demand of overlapping tasks. Same interval
primitives as the job shop.

Tune via ``QAYD_RCPSP_N`` (tasks) / ``QAYD_RCPSP_R`` (resources) / ``QAYD_RCPSP_T``.
"""

import os
from random import Random

import qayd as cp

n = int(os.environ.get("QAYD_RCPSP_N", "12"))
resources = int(os.environ.get("QAYD_RCPSP_R", "2"))
time_limit = int(os.environ.get("QAYD_RCPSP_T", "8"))

rng = Random(0)
duration = [rng.randint(2, 6) for _ in range(n)]
capacity = [10 for _ in range(resources)]
demand = [[rng.randint(0, 5) for _ in range(resources)] for _ in range(n)]
# Each task may depend on one earlier task (a random DAG).
precedences = [(rng.randint(0, t - 1), t) for t in range(1, n) if rng.random() < 0.4]
horizon = sum(duration)

model = cp.Model()
ivs = model.interval_vars(duration, horizon)
for a, b in precedences:
    model.precedes(ivs[a], ivs[b])
for r in range(resources):
    model.resource([(ivs[t], demand[t][r]) for t in range(n) if demand[t][r] > 0], capacity[r])

solution = model.solve(time_limit=time_limit, verbose=os.environ.get("QAYD_VERBOSE") == "1")

print(f"tasks: {n}  resources: {resources}  precedences: {len(precedences)}  status: {solution.status}")
if not solution.starts:
    raise SystemExit(f"status: {solution.status} - no schedule within {time_limit}s")
starts = solution.starts
end = [starts[t] + duration[t] for t in range(n)]
print(f"makespan: {solution.objective}")
for a, b in precedences:
    assert end[a] <= starts[b], f"precedence {a} -> {b}"
for r in range(resources):
    for t in range(max(end)):
        used = sum(demand[i][r] for i in range(n) if starts[i] <= t < end[i])
        assert used <= capacity[r], f"resource {r} within capacity at t={t}"
assert max(end) == solution.objective, "reported makespan matches the schedule"
