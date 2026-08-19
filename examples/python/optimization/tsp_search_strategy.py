"""Compare Qayd's default search with Choco's TSP search strategy."""

from dataclasses import dataclass
from math import hypot

import qayd as cp


CITIES = (
    (17, 62),
    (58, 29),
    (81, 69),
    (56, 57),
    (32, 27),
    (24, 65),
    (96, 31),
    (34, 77),
    (68, 51),
    (63, 16),
)


@dataclass(frozen=True)
class Run:
    objective: int
    nodes: int
    failures: int


def distance_matrix():
    return tuple(
        tuple(round(hypot(x1 - x2, y1 - y2)) for x2, y2 in CITIES)
        for x1, y1 in CITIES
    )


def solve(use_search_strategy: bool) -> Run:
    distances = distance_matrix()
    city_count = len(distances)

    model = cp.Model()
    successors = model.int_vars(city_count, 0, city_count - 1, name="succ")
    edge_costs = model.int_vars(city_count, 0, max(map(max, distances)), name="cost")
    for city in range(city_count):
        model.element_const(distances[city], successors[city], edge_costs[city])
    model.circuit(successors)
    model.minimize(cp.sum(edge_costs))

    policy = None
    if use_search_strategy:
        policy = cp.SearchPolicy([cp.SearchPhase(edge_costs, "max-regret", "min")])

    solution = model.solve(
        engine="exact",
        search_policy=policy,
        threads=1,
        seed=0,
    )
    assert solution.status == "OPTIMAL"

    selected = [solution.value(variable) for variable in successors]
    assert sorted(selected) == list(range(city_count))
    current, visited = 0, set()
    for _ in range(city_count):
        assert current not in visited
        visited.add(current)
        current = selected[current]
    assert current == 0 and len(visited) == city_count

    replayed_objective = sum(distances[city][selected[city]] for city in range(city_count))
    assert replayed_objective == solution.objective
    return Run(replayed_objective, solution.stats.nodes, solution.stats.failures)


def main() -> None:
    auto = solve(False)
    guided = solve(True)

    print(f"auto        objective={auto.objective} nodes={auto.nodes} failures={auto.failures}")
    print(
        f"max-regret  objective={guided.objective} "
        f"nodes={guided.nodes} failures={guided.failures}"
    )
    reduction = 100 * (auto.nodes - guided.nodes) / auto.nodes
    print(f"node reduction: {reduction:.1f}%")


if __name__ == "__main__":
    main()
