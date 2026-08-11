"""A first-class sequence with asymmetric setup times."""

import qayd as cp


durations = [2, 1, 2, 1]
setups = [
    [0, 4, 1, 3],
    [2, 0, 2, 1],
    [1, 3, 0, 1],
    [3, 1, 2, 0],
]

model = cp.Model()
tasks = model.intervals(durations, horizon=20, name="task")
sequence = model.sequence(tasks, setups)
model.minimize_makespan(tasks)

solution = model.solve(engine="exact", time_limit=5)
order = sequence.order(solution)
order_indices = [task.index for task in order]

print("status:", solution.status)
print("makespan:", solution.objective)
print("order:", order_indices)
print("starts:", solution.starts)

assert solution.status == "OPTIMAL"
for before, after in zip(order_indices, order_indices[1:]):
    required_start = solution.starts[before] + durations[before] + setups[before][after]
    assert required_start <= solution.starts[after]
