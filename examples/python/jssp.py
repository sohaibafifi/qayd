"""Job-shop scheduling via interval variables: minimise the makespan. Each job is
a chain of operations (one per machine, fixed order); each machine runs one
operation at a time.

One interval variable per operation; ``model.precedes`` posts the job-order
chain; ``model.disjunctive`` posts each machine's no-overlap. No per-problem
code -- the same interval/precedence/disjunctive blocks compose RCPSP too.

Tune via ``QAYD_JSSP_J`` / ``QAYD_JSSP_M`` / ``QAYD_JSSP_T``.
"""

import os
from random import Random

import qayd as cp

jobs = int(os.environ.get("QAYD_JSSP_J", "4"))
machines = int(os.environ.get("QAYD_JSSP_M", "3"))
time_limit = int(os.environ.get("QAYD_JSSP_T", "8"))

rng = Random(0)
order = [rng.sample(range(machines), machines) for _ in range(jobs)]
ptime = [[rng.randint(2, 9) for _ in range(machines)] for _ in range(jobs)]

# op = j*machines + k is the k-th operation of job j.
duration = [ptime[j][k] for j in range(jobs) for k in range(machines)]
machine_of = [order[j][k] for j in range(jobs) for k in range(machines)]
horizon = sum(duration)

model = cp.Model()
ivs = model.interval_vars(duration, horizon)
for j in range(jobs):
    for k in range(1, machines):
        model.precedes(ivs[j * machines + k - 1], ivs[j * machines + k])
for mc in range(machines):
    model.disjunctive([ivs[op] for op in range(jobs * machines) if machine_of[op] == mc])

solution = model.solve(time_limit=time_limit, verbose=os.environ.get("QAYD_VERBOSE") == "1")

print(f"jobs: {jobs}  machines: {machines}  status: {solution.status}")
if not solution.starts:
    raise SystemExit(f"status: {solution.status} - no schedule within {time_limit}s")
starts = solution.starts
print(f"makespan: {solution.objective}")
# Verify precedence, machine no-overlap, and the reported makespan.
end = [starts[op] + duration[op] for op in range(jobs * machines)]
for j in range(jobs):
    for k in range(1, machines):
        assert end[j * machines + k - 1] <= starts[j * machines + k], "job order respected"
for mc in range(machines):
    ops = sorted((op for op in range(jobs * machines) if machine_of[op] == mc), key=lambda o: starts[o])
    for a, b in zip(ops, ops[1:]):
        assert end[a] <= starts[b], f"machine {mc} no overlap"
assert max(end) == solution.objective, "reported makespan matches the schedule"
