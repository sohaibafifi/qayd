"""CSPLib prob066: distance-based constrained clustering.

Specification: https://www.csplib.org/Problems/prob066/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "clusters": 2,
    "min_size": 2,
    "max_size": 2,
    "criterion": "within_sum",
    "distances": [[0, 1, 8, 9], [1, 0, 9, 8], [8, 9, 0, 1], [9, 8, 1, 0]],
    "must_link": [],
    "cannot_link": [[0, 2]],
}


@dataclass(frozen=True)
class ClusteringInstance:
    clusters: int
    min_size: int
    max_size: int
    criterion: str
    distances: tuple[tuple[int, ...], ...]
    must_link: tuple[tuple[int, int], ...]
    cannot_link: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ClusteringModel:
    model: cp.Model
    instance: ClusteringInstance
    assignment: list[cp.IntVar]


def parse_instance(data: str | bytes) -> ClusteringInstance:
    raw = json.loads(data)
    try:
        return ClusteringInstance(
            int(raw["clusters"]),
            int(raw["min_size"]),
            int(raw["max_size"]),
            str(raw.get("criterion", "within_sum")),
            tuple(tuple(int(value) for value in row) for row in raw["distances"]),
            tuple((int(pair[0]), int(pair[1])) for pair in raw.get("must_link", [])),
            tuple((int(pair[0]), int(pair[1])) for pair in raw.get("cannot_link", [])),
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError("invalid constrained-clustering JSON instance") from error


def load_instance(path: str | Path) -> ClusteringInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: ClusteringInstance) -> ClusteringModel:
    object_count = len(instance.distances)
    if object_count < 2 or instance.clusters < 1 or instance.clusters > object_count:
        raise ValueError("object and cluster counts are invalid")
    if not 1 <= instance.min_size <= instance.max_size:
        raise ValueError("cluster-size bounds are invalid")
    if any(
        len(row) != object_count or any(value < 0 for value in row)
        for row in instance.distances
    ):
        raise ValueError("distances must be a square non-negative matrix")
    if instance.criterion not in {"within_sum", "diameter", "split"}:
        raise ValueError("criterion must be within_sum, diameter, or split")
    pairs = (*instance.must_link, *instance.cannot_link)
    if any(
        min(pair) < 0 or max(pair) >= object_count or pair[0] == pair[1]
        for pair in pairs
    ):
        raise ValueError("a link constraint is invalid")
    model = cp.Model()
    assignment = model.int_vars(object_count, 0, instance.clusters - 1, name="cluster")
    model.cardinality(
        assignment,
        list(range(instance.clusters)),
        [instance.min_size] * instance.clusters,
        [instance.max_size] * instance.clusters,
        closed=True,
    )
    for left, right in instance.must_link:
        model.add(assignment[left] == assignment[right])
    for left, right in instance.cannot_link:
        model.add(assignment[left] != assignment[right])
    same_variables = []
    maximum_distance = max(value for row in instance.distances for value in row)
    diameter = model.int_var(0, maximum_distance, name="diameter")
    split = model.int_var(0, maximum_distance, name="split")
    for left in range(object_count):
        for right in range(left + 1, object_count):
            same = model.bool_var(name=f"same_{left}_{right}")
            model.table(
                [assignment[left], assignment[right], same],
                [
                    (first, second, int(first == second))
                    for first in range(instance.clusters)
                    for second in range(instance.clusters)
                ],
            )
            distance = instance.distances[left][right]
            model.add(diameter >= distance * same)
            model.add(split <= distance + maximum_distance * same)
            same_variables.append((distance, same))
    if instance.criterion == "within_sum":
        model.minimize(sum(distance * same for distance, same in same_variables))
    elif instance.criterion == "diameter":
        model.minimize(diameter)
    else:
        model.maximize(split)
    model.add(assignment[0] == 0)
    return ClusteringModel(model, instance, assignment)


def decode(built: ClusteringModel, solution: cp.Solution) -> list[list[int]]:
    clusters = [[] for _ in range(built.instance.clusters)]
    for item, cluster in enumerate(values(solution, built.assignment)):
        clusters[cluster].append(item)
    return clusters


def criterion_value(instance: ClusteringInstance, clusters: list[list[int]]) -> int:
    cluster_of = {
        item: cluster for cluster, items in enumerate(clusters) for item in items
    }
    within = [
        instance.distances[left][right]
        for left in range(len(instance.distances))
        for right in range(left + 1, len(instance.distances))
        if cluster_of[left] == cluster_of[right]
    ]
    between = [
        instance.distances[left][right]
        for left in range(len(instance.distances))
        for right in range(left + 1, len(instance.distances))
        if cluster_of[left] != cluster_of[right]
    ]
    if instance.criterion == "within_sum":
        return sum(within)
    if instance.criterion == "diameter":
        return max(within, default=0)
    return min(between, default=0)


def validate(
    built: ClusteringModel, clusters: list[list[int]], objective: int | None
) -> None:
    objects = [item for cluster in clusters for item in cluster]
    if sorted(objects) != list(range(len(built.instance.distances))):
        raise AssertionError("objects are not assigned exactly once")
    if any(
        not built.instance.min_size <= len(cluster) <= built.instance.max_size
        for cluster in clusters
    ):
        raise AssertionError("a cluster violates size bounds")
    cluster_of = {
        item: cluster for cluster, items in enumerate(clusters) for item in items
    }
    if any(
        cluster_of[left] != cluster_of[right]
        for left, right in built.instance.must_link
    ):
        raise AssertionError("a must-link constraint is violated")
    if any(
        cluster_of[left] == cluster_of[right]
        for left, right in built.instance.cannot_link
    ):
        raise AssertionError("a cannot-link constraint is violated")
    if objective is not None and criterion_value(built.instance, clusters) != objective:
        raise AssertionError("the objective does not match the clustering criterion")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON constrained-clustering instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob066 objects={len(instance.distances)} status={solution.status}")
    if not solution.is_sat():
        return 1
    clusters = decode(built, solution)
    validate(built, clusters, solution.objective)
    print(f"objective={solution.objective} clusters={clusters}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
