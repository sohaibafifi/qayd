"""Capacitated vehicle routing on real CVRPLIB instances, read with vrplib.

Modeled the usual way: a Model with k list variables (one per vehicle), a total
distance objective, and a per-route capacity constraint. solve() picks the
collection engine because the model has list variables.

Instance: set ``QAYD_VRP_INSTANCE`` to any CVRPLIB ``.vrp`` file; defaults to the
bundled ``data/vrplib/CVRP/X-n101-k25.vrp`` (Uchoa et al.). More at
http://vrp.galgos.inf.puc-rio.br. Tune time via ``QAYD_VRP_T``; trace with
``QAYD_VERBOSE=1``.
"""

import os
import re

import vrplib

import qayd as cp

here = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(here, "..", ".."))
path = os.environ.get("QAYD_VRP_INSTANCE", os.path.join(repo_root, "data", "vrplib", "CVRP", "X-n101-k25.vrp"))
time_limit = int(os.environ.get("QAYD_VRP_T", "10"))

inst = vrplib.read_instance(path)
name = inst.get("name", os.path.basename(path))
dim = len(inst["demand"])                 # nodes including the depot
depot = int(inst["depot"][0])             # CVRPLIB depot is node 0
capacity = int(inst["capacity"])
demand = [int(d) for d in inst["demand"]]
customers = [i for i in range(dim) if i != depot]

# CVRPLIB optima use TSPLIB nearest-integer distances, so round EUC_2D weights.
dist = [[int(w + 0.5) for w in row] for row in inst["edge_weight"]]

# Available vehicles: CVRPLIB names encode the fixed benchmark fleet as "-kN".
# Override QAYD_VRP_K only when intentionally solving a different fleet size.
m = re.search(r"-k(\d+)", name)
min_k = int(m.group(1)) if m else -(-sum(demand) // capacity)
k = int(os.environ.get("QAYD_VRP_K", str(min_k)))

D = cp.matrix(dist)        # constant tables, indexed by node id inside lambdas
Q = cp.array(demand)

model = cp.Model()
routes = model.list_vars(k, customers)   # k vehicles partition the customers
model.minimize(cp.sum(cp.sum_edges(r, lambda i, j: D[i][j], start=depot, end=depot) for r in routes))
for r in routes:
    model.add(cp.sum(r, lambda i: Q[i]) <= capacity)  # each route within capacity

solution = model.solve(time_limit=time_limit, verbose=os.environ.get("QAYD_VERBOSE") == "1", local_search=False)

# Known optimum, if the instance comment records it (CVRPLIB convention).
opt = re.search(r"Optimal value:\s*(\d+)", inst.get("comment", ""))
gap = f"  (known optimum {opt.group(1)})" if opt else ""

print(f"instance: {name}  customers: {len(customers)}  vehicles: {k}  capacity: {capacity}")
if solution.routes is None:
    raise SystemExit(f"status: {solution.status} - no feasible solution within {time_limit}s")
fleet = sum(1 for route in solution.routes if route)
distance = solution.objectives[-1]
print(f"status: {solution.status}  fleet: {fleet}  total distance: {distance}{gap}")
for r, route in enumerate(solution.routes):
    if not route:
        continue
    load = sum(demand[c] for c in route)
    print(f"  route {r}: load {load:5d}  {[depot, *route, depot]}")

served = sorted(c for route in solution.routes for c in route)
assert served == sorted(customers), "every customer served exactly once"
for route in solution.routes:
    assert sum(demand[c] for c in route) <= capacity, "capacity respected"
