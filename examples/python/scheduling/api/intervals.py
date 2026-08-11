"""First-class intervals, optional members and an alternative master."""

import qayd as cp


model = cp.Model()
cut = model.interval(3, 15, name="cut")
fast_machine = model.interval(2, 15, optional=True, name="fast_machine")
slow_machine = model.interval(4, 15, optional=True, name="slow_machine")
operation = model.alternative([fast_machine, slow_machine], name="operation")
pack = model.interval(2, 15, name="pack")

model.precedence(cut, operation)
model.precedence(operation, pack)
model.no_overlap([cut, operation, pack])
model.minimize_makespan([cut, operation, pack])

solution = model.solve(engine="exact", time_limit=5)
print("status:", solution.status)
print("makespan:", solution.objective)
print("starts:", solution.starts)

assert solution.status == "OPTIMAL"
assert solution.objective == 7
assert solution.presences[fast_machine.index]
assert not solution.presences[slow_machine.index]
assert solution.value(operation.start) == solution.starts[fast_machine.index]
