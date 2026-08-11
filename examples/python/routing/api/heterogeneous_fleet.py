"""Capacitated routing with a different capacity for each vehicle."""

import math

import qayd as cp


coordinates = [(0, 0), (1, 4), (2, 1), (4, 4)]
demand = [0, 4, 3, 5]
capacities = [7, 10]
distance = [
    [round(math.hypot(ax - bx, ay - by)) for bx, by in coordinates]
    for ax, ay in coordinates
]

model = cp.Model()
customers = model.customers(range(1, len(coordinates)))
for customer in customers:
    customer.demand = demand[customer.id]

routes = model.routes(customers, vehicles=len(capacities), depot=0, travel=distance)
for route, capacity in zip(routes, capacities):
    model.add(route.sum(lambda customer: customer.demand) <= capacity)

model.minimize(routes.used_count())
model.then_minimize(routes.total_distance())

solution = model.solve(time_limit=2)
print("status:", solution.status)
if solution.lists is None:
    raise SystemExit("no feasible routing plan")

fleet, total_distance = solution.objectives
print(f"used vehicles: {fleet}  total distance: {total_distance}")
for index, route in enumerate(solution.lists):
    load = sum(demand[customer] for customer in route)
    print(f"  vehicle {index}: capacity {capacities[index]}  load {load}  {[0, *route, 0]}")
    assert load <= capacities[index]

assert sorted(customer for route in solution.lists for customer in route) == list(range(1, len(coordinates)))
