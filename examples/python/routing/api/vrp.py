"""Capacitated vehicle routing with the routing convenience API.

Instance: set ``QAYD_VRP_INSTANCE`` to any CVRPLIB ``.vrp`` file. If it is not
set, the script tries the local scratch path
``data/vrplib/CVRP/X-n101-k25.vrp``. More at http://vrp.galgos.inf.puc-rio.br.
Tune time via ``QAYD_VRP_T``; trace with ``QAYD_VERBOSE=1``.
"""

import os
import re

try:
    import vrplib
except ImportError as exc:
    raise SystemExit("install example dependencies with `uv run --extra examples ...`") from exc

import qayd as cp

here = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
path = os.environ.get("QAYD_VRP_INSTANCE", os.path.join(repo_root, "data", "vrplib", "CVRP", "X-n101-k25.vrp"))
time_limit = int(os.environ.get("QAYD_VRP_T", "10"))

if not os.path.exists(path):
    raise SystemExit("set QAYD_VRP_INSTANCE to a CVRPLIB .vrp file")

inst = vrplib.read_instance(path)
name = inst.get("name", os.path.basename(path))
dim = len(inst["demand"])
depot = int(inst["depot"][0])
capacity = int(inst["capacity"])
demand = [int(d) for d in inst["demand"]]
customer_ids = [i for i in range(dim) if i != depot]
dist = [[int(w + 0.5) for w in row] for row in inst["edge_weight"]]

m = re.search(r"-k(\d+)", name)
min_k = int(m.group(1)) if m else -(-sum(demand) // capacity)
k = int(os.environ.get("QAYD_VRP_K", str(min_k)))

model = cp.Model()
customers = model.customers(customer_ids)
for customer in customers:
    customer.demand = demand[customer.id]

routes = model.routes(customers, vehicles=k, depot=depot, travel=dist)
for route in routes:
    model.add(route.sum(lambda customer: customer.demand) <= capacity)
model.minimize(routes.sum(lambda route: route.distance()))

solution = model.solve(time_limit=time_limit, verbose=os.environ.get("QAYD_VERBOSE") == "1")

opt = re.search(r"Optimal value:\s*(\d+)", inst.get("comment", ""))
gap = f"  (known optimum {opt.group(1)})" if opt else ""

print(f"instance: {name}  customers: {len(customers)}  vehicles: {k}  capacity: {capacity}")
if solution.lists is None:
    raise SystemExit(f"status: {solution.status} - no feasible solution within {time_limit}s")
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
