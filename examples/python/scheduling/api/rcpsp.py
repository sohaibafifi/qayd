"""Resource-constrained project scheduling with the scheduling convenience API.

Durations and resource demands are attached to each task. Renewable resource
constraints use ``schedule.resource(lambda task: ...)``.

Tune via ``QAYD_RCPSP_N`` / ``QAYD_RCPSP_R`` / ``QAYD_RCPSP_T``.
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
precedences = [(rng.randint(0, task - 1), task) for task in range(1, n) if rng.random() < 0.4]
horizon = sum(duration)

model = cp.Model()
tasks = model.tasks(range(n))
for task in tasks:
    task.duration = duration[task.id]
    task.demand = demand[task.id]

schedule = model.schedule(tasks, horizon=horizon)
for before, after in precedences:
    model.add(schedule[before].end <= schedule[after].start)
for resource in range(resources):
    model.add(schedule.resource(lambda task, resource=resource: task.demand[resource]) <= capacity[resource])
model.minimize(schedule.makespan())

solution = model.solve(time_limit=time_limit, verbose=os.environ.get("QAYD_VERBOSE") == "1")

print(f"tasks: {n}  resources: {resources}  precedences: {len(precedences)}  status: {solution.status}")
if not solution.starts:
    raise SystemExit(f"status: {solution.status} - no schedule within {time_limit}s")
starts = solution.starts
end = [starts[task] + duration[task] for task in range(n)]
print(f"makespan: {solution.objective}")
for before, after in precedences:
    assert end[before] <= starts[after], f"precedence {before} -> {after}"
for resource in range(resources):
    for time in range(max(end)):
        used = sum(demand[task][resource] for task in range(n) if starts[task] <= time < end[task])
        assert used <= capacity[resource], f"resource {resource} within capacity at t={time}"
assert max(end) == solution.objective, "reported makespan matches the schedule"
