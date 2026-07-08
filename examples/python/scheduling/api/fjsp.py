"""Flexible job-shop scheduling with the scheduling convenience API.

Each task stores its eligible ``(machine, duration)`` modes. The schedule builds
the moded interval model and machine no-overlap is posted with ``schedule.no_overlap()``.

Tune via ``QAYD_FJSP_J`` / ``QAYD_FJSP_M`` / ``QAYD_FJSP_T``.
"""

import os
from random import Random

import qayd as cp

jobs = int(os.environ.get("QAYD_FJSP_J", "4"))
machines = int(os.environ.get("QAYD_FJSP_M", "3"))
ops_per_job = machines
time_limit = int(os.environ.get("QAYD_FJSP_T", "10"))

rng = Random(0)
modes = []
for _ in range(jobs):
    for _ in range(ops_per_job):
        elig = rng.sample(range(machines), rng.randint(1, machines))
        modes.append([(machine, rng.randint(2, 9)) for machine in elig])

horizon = sum(max(duration for _, duration in op) for op in modes)
proc = [{machine: duration for machine, duration in op} for op in modes]
n = jobs * ops_per_job

model = cp.Model()
tasks = model.tasks(range(n))
for task in tasks:
    task.modes = modes[task.id]

schedule = model.schedule(tasks, horizon=horizon)
for j in range(jobs):
    for k in range(1, ops_per_job):
        model.add(schedule[j * ops_per_job + k - 1].end <= schedule[j * ops_per_job + k].start)
model.add(schedule.no_overlap())
model.minimize(schedule.makespan())

solution = model.solve(time_limit=time_limit, verbose=os.environ.get("QAYD_VERBOSE") == "1")

print(f"jobs: {jobs}  machines: {machines}  ops: {n}  status: {solution.status}")
if not solution.starts:
    raise SystemExit(f"status: {solution.status} - no schedule within {time_limit}s")
starts, chosen = solution.starts, solution.machines
print(f"makespan: {solution.objective}")
for op in range(n):
    assert chosen[op] in proc[op], f"op {op} on an eligible machine"
end = [starts[op] + proc[op][chosen[op]] for op in range(n)]
for j in range(jobs):
    for k in range(1, ops_per_job):
        assert end[j * ops_per_job + k - 1] <= starts[j * ops_per_job + k], "job order respected"
for machine in range(machines):
    ops = sorted((op for op in range(n) if chosen[op] == machine), key=lambda op: starts[op])
    for a, b in zip(ops, ops[1:]):
        assert end[a] <= starts[b], f"machine {machine} no overlap"
assert max(end) == solution.objective, "reported makespan matches the schedule"
