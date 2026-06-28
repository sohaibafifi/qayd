import qayd as cp


jobs = [
    [(0, 3), (1, 2), (2, 2)],
    [(0, 2), (2, 1), (1, 4)],
    [(1, 4), (2, 3), (0, 2)],
]

horizon = sum(duration for job in jobs for _, duration in job)
n_machines = 1 + max(machine for job in jobs for machine, _ in job)

model = cp.Model()
durations = [duration for job in jobs for _, duration in job]
machines = [machine for job in jobs for machine, _ in job]
operations = model.intervals(durations, horizon)

offset = 0
for job in jobs:
    for k in range(1, len(job)):
        model.precedence(operations[offset + k - 1], operations[offset + k])
    offset += len(job)

for machine in range(n_machines):
    model.no_overlap([op for op, mc in zip(operations, machines) if mc == machine])

model.minimize_makespan(operations)
solution = model.solve(verbose=True)

print("status:", solution.status)
print("makespan:", solution.objective)
print("starts:", solution.starts)

assert solution.status == "OPTIMAL"
assert solution.objective == 11
