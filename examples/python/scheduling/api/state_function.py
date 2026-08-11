"""A state/resource function with setup times between production families."""

import qayd as cp


durations = [3, 2, 2, 1]
states = [0, 1, 0, 2]
transitions = [
    [0, 2, 3],
    [2, 0, 1],
    [3, 1, 0],
]

model = cp.Model()
batches = model.intervals(durations, horizon=20, name="batch")
model.state_function(list(zip(batches, states)), transitions)
model.minimize_makespan(batches)

solution = model.solve(engine="exact", time_limit=5)
starts = solution.starts
ends = [starts[index] + durations[index] for index in range(len(batches))]

print("status:", solution.status)
print("makespan:", solution.objective)
for index in sorted(range(len(batches)), key=lambda batch: (starts[batch], batch)):
    print(f"  batch {index}: state {states[index]}  [{starts[index]}, {ends[index]})")

assert solution.status == "OPTIMAL"
for left in range(len(batches)):
    for right in range(left + 1, len(batches)):
        if states[left] == states[right]:
            continue
        left_before = ends[left] + transitions[states[left]][states[right]] <= starts[right]
        right_before = ends[right] + transitions[states[right]][states[left]] <= starts[left]
        assert left_before or right_before
