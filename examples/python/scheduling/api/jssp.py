"""Job-shop scheduling via the scheduling convenience API.

Each operation stores its own duration and machine. The API builds the interval
variables from those task attributes.

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

n = jobs * machines
duration = [ptime[j][k] for j in range(jobs) for k in range(machines)]
machine_of = [order[j][k] for j in range(jobs) for k in range(machines)]
horizon = sum(duration)

model = cp.Model()
tasks = model.tasks(range(n))
for task in tasks:
    task.duration = duration[task.id]
    task.machine = machine_of[task.id]

schedule = model.schedule(tasks, horizon=horizon)
for j in range(jobs):
    for k in range(1, machines):
        model.add(schedule[j * machines + k - 1].end <= schedule[j * machines + k].start)
model.add(schedule.no_overlap(lambda task: task.machine))
model.minimize(schedule.makespan())

solution = model.solve(time_limit=time_limit, verbose=os.environ.get("QAYD_VERBOSE") == "1")

print(f"jobs: {jobs}  machines: {machines}  status: {solution.status}")
if not solution.starts:
    raise SystemExit(f"status: {solution.status} - no schedule within {time_limit}s")
starts = solution.starts
print(f"makespan: {solution.objective}")
end = [starts[op] + duration[op] for op in range(n)]
for j in range(jobs):
    for k in range(1, machines):
        assert end[j * machines + k - 1] <= starts[j * machines + k], "job order respected"
for machine in range(machines):
    ops = sorted((op for op in range(n) if machine_of[op] == machine), key=lambda op: starts[op])
    for a, b in zip(ops, ops[1:]):
        assert end[a] <= starts[b], f"machine {machine} no overlap"
assert max(end) == solution.objective, "reported makespan matches the schedule"
