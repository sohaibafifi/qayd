"""Capacitated vehicle routing with the routing convenience API.

Instance: set ``QAYD_VRP_INSTANCE`` to any CVRPLIB ``.vrp`` file. If it is not
set, the script tries the local scratch path
``data/vrplib/CVRP/X-n101-k25.vrp``. More at http://vrp.galgos.inf.puc-rio.br.
Tune time via ``QAYD_VRP_T``; trace with ``QAYD_VERBOSE=1``.
"""

import os

import qayd as cp
from qayd.datasets import read_cvrplib

here = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
path = os.environ.get(
    "QAYD_VRP_INSTANCE",
    os.path.join(repo_root, "data", "vrplib", "CVRP", "X-n101-k25.vrp"),
)
time_limit = int(os.environ.get("QAYD_VRP_T", "10"))

if not os.path.exists(path):
    raise SystemExit("set QAYD_VRP_INSTANCE to a CVRPLIB .vrp file")

inst = read_cvrplib(path)
name = inst.name
depot = inst.depot
capacity = inst.capacity
demand = list(inst.demands)
customer_ids = list(inst.customers)
dist = [list(row) for row in inst.edge_weights]

min_k = inst.vehicles or -(-sum(demand) // capacity)
k = int(os.environ.get("QAYD_VRP_K", str(min_k)))

model = cp.Model()
customers = model.customers(customer_ids)
for customer in customers:
    customer.demand = demand[customer.id]

routes = model.routes(customers, vehicles=k, depot=depot, travel=dist)
for route in routes:
    model.add(route.sum(lambda customer: customer.demand) <= capacity)
model.minimize(routes.sum(lambda route: route.distance()))

solution = model.solve(
    time_limit=time_limit, verbose=os.environ.get("QAYD_VERBOSE") == "1"
)

gap = f"  (known optimum {inst.best_known})" if inst.best_known is not None else ""

print(
    f"instance: {name}  customers: {len(customers)}  vehicles: {k}  capacity: {capacity}"
)
if solution.lists is None:
    raise SystemExit(
        f"status: {solution.status} - no feasible solution within {time_limit}s"
    )
fleet = sum(1 for route in solution.lists if route)
distance = solution.objectives[-1]
print(f"status: {solution.status}  fleet: {fleet}  total distance: {distance}{gap}")
for r, route in enumerate(solution.lists):
    if not route:
        continue
    load = sum(demand[c] for c in route)
    print(f"  route {r}: load {load:5d}  {[depot, *route, depot]}")

served = sorted(c for route in solution.lists for c in route)
assert served == sorted(customer_ids), "every customer served exactly once"
for route in solution.lists:
    assert sum(demand[c] for c in route) <= capacity, "capacity respected"
