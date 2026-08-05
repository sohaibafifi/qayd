"""Partition people into unordered teams with rich inter-team constraints.

``set_vars`` assigns every person to exactly one team and returns each team in
canonical item order. The global constraints express co-location, separation,
distinct ownership, and a bound on the distance between team indices. Six
lexicographic objectives demonstrate that collection objectives have no fixed
tier limit.
"""

import qayd as cp


people = list(range(1, 7))
# Index 0 is unused so every table can be indexed directly by person id.
assignment_cost = [
    cp.array([0, 8, 7, 6, 3, 4, 9]),
    cp.array([0, 5, 6, 8, 7, 2, 4]),
    cp.array([0, 9, 8, 3, 4, 6, 2]),
]
workload = cp.array([0, 5, 4, 6, 3, 2, 7])

model = cp.Model()
teams = model.set_vars(people, count=3)

# People 1 and 2 must work together. People 1, 3, and 4 anchor three distinct
# teams, while 5 and 6 must also be separated.
model.all_same_list([1, 2])
model.all_different_lists([1, 3, 4])
model.different_list(5, 6)
model.list_distance(1, 3, min=1, max=2)

# First minimize each team's assignment cost in business-priority order, then
# use workload as three lower-priority tie breakers. This is intentionally six
# tiers, beyond the former four-tier ceiling.
cost_tiers = [
    cp.sum(team, lambda person, costs=costs: costs[person])
    for team, costs in zip(teams, assignment_cost)
]
load_tiers = [cp.sum(team, lambda person: workload[person]) for team in teams]
model.minimize(cost_tiers[0])
for tier in [*cost_tiers[1:], *load_tiers]:
    model.then_minimize(tier)

solution = model.solve(engine="exact", time_limit=5)

print(f"status: {solution.status}")
print(f"objectives: {solution.objectives}")
for team_id, members in enumerate(solution.lists or []):
    print(f"  team {team_id}: {members}")

assert solution.status == "OPTIMAL"
assert solution.lists is not None
assert len(solution.objectives) == 6
assert sorted(person for team in solution.lists for person in team) == people
assert all(team == sorted(team) for team in solution.lists), "sets use canonical order"

owner = {person: team_id for team_id, team in enumerate(solution.lists) for person in team}
assert owner[1] == owner[2]
assert len({owner[1], owner[3], owner[4]}) == 3
assert owner[5] != owner[6]
assert 1 <= abs(owner[1] - owner[3]) <= 2
