"""Schedule a mandatory setup followed by one optional processing mode.

The two processing intervals are optional. ``alternative`` requires exactly one
of them and channels its shared start. Precedence, no-overlap, and makespan all
honor interval presence, so the absent mode has no scheduling effect.
"""

import qayd as cp


horizon = 12
model = cp.Model()

setup = model.interval(3, horizon, name="setup")
fast = model.interval(2, horizon, optional=True, name="fast")
thorough = model.interval(5, horizon, optional=True, name="thorough")
processing = model.alternative([fast, thorough], name="processing")

# Posting precedence to both optional members is conditional on their presence.
model.precedence(setup, fast)
model.precedence(setup, thorough)
model.no_overlap([setup, fast, thorough])
model.minimize_makespan([setup, fast, thorough])

solution = model.solve(engine="exact", time_limit=5)

print(f"status: {solution.status}")
print(f"makespan: {solution.objective}")
print(f"starts: {solution.starts}")
print(f"presences: {solution.presences}")

assert solution.status == "OPTIMAL"
assert solution.objective == 5
assert solution.presences == [True, True, False]
assert solution.starts[0] == 0
assert solution.starts[1] == 3
assert solution.starts[2] is None
assert processing.start is not None
assert solution.value(processing.start) == solution.starts[1]
