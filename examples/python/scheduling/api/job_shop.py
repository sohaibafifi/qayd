"""Small job-shop model using the scheduling convenience API."""

import qayd as cp


jobs = [
    [(0, 3), (1, 2), (2, 2)],
    [(0, 2), (2, 1), (1, 4)],
    [(1, 4), (2, 3), (0, 2)],
]

ops = [(job_id, machine, duration) for job_id, job in enumerate(jobs) for machine, duration in job]
horizon = sum(duration for _, _, duration in ops)

model = cp.Model()
tasks = model.tasks(range(len(ops)))
for task in tasks:
    job_id, machine, duration = ops[task.id]
    task.job = job_id
    task.machine = machine
    task.duration = duration

schedule = model.schedule(tasks, horizon=horizon)

offset = 0
for job in jobs:
    for k in range(1, len(job)):
        model.add(schedule[offset + k - 1].end <= schedule[offset + k].start)
    offset += len(job)

model.add(schedule.no_overlap(lambda task: task.machine))
model.minimize(schedule.makespan())

solution = model.solve(verbose=True)

print("status:", solution.status)
print("makespan:", solution.objective)
print("starts:", solution.starts)

assert solution.status == "OPTIMAL"
assert solution.objective == 11
