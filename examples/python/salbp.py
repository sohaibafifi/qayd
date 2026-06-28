"""Simple assembly line balancing (SALBP-1): assign tasks to ordered stations
to minimise the number of stations, respecting a station cycle time and task
precedences (a task lands in a station no later than each of its successors).

One list per station; the objective counts non-empty stations via ``cp.used``;
``model.precedence(a, b)`` posts the cross-station precedence ``station(a) <=
station(b)``.

Tune via ``QAYD_SALBP_N`` / ``QAYD_SALBP_T``; trace with ``QAYD_VERBOSE=1``.
"""

import os
from random import Random

import qayd as cp

n = int(os.environ.get("QAYD_SALBP_N", "12"))
time_limit = int(os.environ.get("QAYD_SALBP_T", "5"))

rng = Random(0)
# Task ids are 1..n; index 0 unused so proc[i] reads by id.
proc = [0] + [rng.randint(2, 6) for _ in range(n)]
cycle = 10
# A random precedence DAG: each task may depend on one earlier task.
precedences = [(j, i) for i in range(2, n + 1) for j in (rng.randint(1, i - 1),) if rng.random() < 0.5]
tasks = list(range(1, n + 1))
k = n  # at most one station per task

P = cp.array(proc)
model = cp.Model()
stations = model.list_vars(tasks, count=k)
model.minimize(cp.sum(cp.used(s) for s in stations))  # minimise stations
for s in stations:
    model.add(cp.sum(s, lambda i: P[i]) <= cycle)
for a, b in precedences:
    model.precedence(a, b)  # station(a) <= station(b)

solution = model.solve(time_limit=time_limit, verbose=os.environ.get("QAYD_VERBOSE") == "1")

print(f"tasks: {n}  cycle time: {cycle}  precedences: {len(precedences)}  status: {solution.status}")
if solution.lists is None:
    raise SystemExit(f"status: {solution.status} - no feasible line within {time_limit}s")
print(f"stations used: {solution.objective}")
station_of = {t: s for s, station in enumerate(solution.lists) for t in station}
for s, station in enumerate(solution.lists):
    if station:
        print(f"  station {s}: load {sum(proc[t] for t in station):2d}  {station}")

for a, b in precedences:
    assert station_of[a] <= station_of[b], f"precedence {a} -> {b} respected"
for station in solution.lists:
    assert sum(proc[t] for t in station) <= cycle, "cycle time respected"
assert sorted(station_of) == tasks, "every task assigned once"
