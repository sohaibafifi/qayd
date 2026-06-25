import qayd as cp
from qayd import schedule


jobs = [
    [(0, 3), (1, 2), (2, 2)],
    [(0, 2), (2, 1), (1, 4)],
    [(1, 4), (2, 3), (0, 2)],
]

horizon = sum(duration for job in jobs for _, duration in job)
n_machines = 1 + max(machine for job in jobs for machine, _ in job)

model = cp.Model()
operations = []
machine_tasks = [[] for _ in range(n_machines)]

for job_id, job in enumerate(jobs):
    row = []
    for op_id, (machine, duration) in enumerate(job):
        task = schedule.interval_var(
            model,
            start=(0, horizon),
            size=duration,
            name=f"J{job_id}O{op_id}_M{machine}",
        )
        row.append(task)
        machine_tasks[machine].append(task)
    operations.append(row)

for row in operations:
    for before, after in zip(row, row[1:]):
        model.add(schedule.end_before_start(before, after))

for machine, tasks in enumerate(machine_tasks):
    sequence = schedule.sequence_var(tasks, name=f"machine[{machine}]")
    schedule.no_overlap(model, sequence)

makespan = schedule.makespan_var(model, [row[-1] for row in operations], horizon)
model.objective(makespan)
solution = model.solve(verbose=True)

print("status:", solution.status)
print("makespan:", solution.objective)
for row in operations:
    print([schedule.interval_value(solution, task) for task in row])

assert solution.status == "OPTIMAL"
assert solution.objective == 11
