"""Explain an infeasibility at sub-constraint granularit.

`model.mus()` names *which* soft constraints conflict. `model.explain_mus()` goes
finer: for a core that refutes by propagation, it reports the *specific atoms*
each constraint reasoned about - not "the capacity constraint is involved" but
"the capacity failed because team_a ≥ 4 and team_b ≥ 4". That granularity comes
from qayd's lazy-clause-generation reasons: each nogood traces to a propagator's
structured explanation, which a CNF-level unsat-core tool cannot see.

Scenario: two teams each demand at least 4 seats, but the shared room holds at
most 5. The core is all three demands; the explanation pinpoints the two seat
counts that overflow the room.

Use ``--time-limit`` to bound explanation.
"""

import argparse

import qayd as cp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--time-limit", type=int, default=8)
args = parser.parse_args()

model = cp.Model()
team_a = model.int_var(0, 10, name="team_a")
team_b = model.int_var(0, 10, name="team_b")

with model.soft("demand_a"):
    model.sum([team_a], ">=", 4)
with model.soft("demand_b"):
    model.sum([team_b], ">=", 4)
with model.soft("room_capacity"):
    model.sum([team_a, team_b], "<=", 5)

time_limit = args.time_limit

core = model.mus(time_limit=time_limit)
if core is None:
    raise SystemExit("model is satisfiable - no MUS")
print(f"constraint-level MUS: {sorted(core)}")

explanation = model.explain_mus(time_limit=time_limit)
if explanation is None:
    raise SystemExit("this core needs search to refute - no sub-constraint explanation")

print("sub-constraint explanation:")
for name in sorted(explanation):
    atoms = explanation[name]
    detail = ", ".join(atoms) if atoms else "(asserted directly)"
    print(f"  {name}: {detail}")

# The capacity constraint's failure is pinned to exactly the two demands.
assert set(core) == {"demand_a", "demand_b", "room_capacity"}, core
assert set(explanation["room_capacity"]) == {"team_a >= 4", "team_b >= 4"}, explanation["room_capacity"]
print("\nverified: the room overflow is explained by team_a >= 4 and team_b >= 4")
