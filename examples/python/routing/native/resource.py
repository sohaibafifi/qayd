"""Routing with a deterministic route resource using the native list API.

The resource is named as fuel here, but the same pattern applies to battery,
cash, inventory, risk, or any scalar state threaded along a route.
"""

import qayd as cp


customers = [1, 2]
travel = cp.matrix(
    [
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ]
)
energy = cp.matrix(
    [
        [0, 1, 1],
        [1, 0, 1],
        [1, 4, 0],
    ]
)
refill = cp.array([0, 0, 0])
capacity = 3

model = cp.Model()
route, = model.list_vars(customers, count=1)

fuel = cp.scan_resource(
    route,
    initial=capacity,
    bounds=(0, capacity),
    transition=lambda cur, state, prev: state - energy[prev][cur] + refill[cur],
    boundary=0,
)
fuel.view("before", lambda cur, state, prev: state - refill[cur])

model.add(fuel.within_bounds())
for customer in customers:
    model.add(fuel.before(customer) >= 0)

model.minimize(cp.sum_edges(route, lambda i, j: travel[i][j], start=0, end=0))
solution = model.solve(time_limit=2)

print(f"status: {solution.status}")
if solution.lists is None:
    raise SystemExit("no feasible route")
print("route:", [0, *solution.lists[0], 0])

assert solution.lists[0] == [1, 2]
