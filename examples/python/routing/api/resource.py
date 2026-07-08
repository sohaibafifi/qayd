"""Routing with a named deterministic resource using the routing API."""

import qayd as cp


travel = [
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
]
energy = cp.matrix(
    [
        [0, 1, 1],
        [1, 0, 1],
        [1, 4, 0],
    ]
)
capacity = 3

model = cp.Model()
customers = model.customers([1, 2])
for customer in customers:
    customer.refill = 0

routes = model.routes(customers, vehicles=1, depot=0, travel=travel)
fuel = routes.resource(
    "fuel",
    initial=capacity,
    bounds=(0, capacity),
    consume=lambda prev, cur: energy[prev.id][cur.id],
    refill=lambda cur: cur.refill,
)

model.add(fuel.within_bounds())
for customer in customers:
    model.add(routes[customer].fuel.before >= 0)

model.minimize(routes.total_distance())
solution = model.solve(time_limit=2)

print(f"status: {solution.status}")
if solution.lists is None:
    raise SystemExit("no feasible route")
print("route:", [0, *solution.lists[0], 0])

assert solution.lists[0] == [1, 2]
