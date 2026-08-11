"""A cumulative resource with a piecewise-constant capacity calendar."""

import qayd as cp


durations = [3, 3, 2]
demands = [2, 1, 2]
default_capacity = 3
calendar = [(3, 5, 1)]

model = cp.Model()
tasks = model.intervals(durations, horizon=15, name="task")
model.resource_calendar(list(zip(tasks, demands)), default_capacity, calendar)
model.minimize_makespan(tasks)

solution = model.solve(engine="exact", time_limit=5)
print("status:", solution.status)
print("makespan:", solution.objective)
print("starts:", solution.starts)

assert solution.status == "OPTIMAL"
ends = [solution.starts[index] + durations[index] for index in range(len(tasks))]
for time in range(max(ends)):
    capacity = next((value for start, end, value in calendar if start <= time < end), default_capacity)
    usage = sum(demands[index] for index in range(len(tasks)) if solution.starts[index] <= time < ends[index])
    assert usage <= capacity
