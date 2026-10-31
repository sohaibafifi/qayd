"""Replay the SINTEF route format emitted by LKH-3.0.13."""
from argparse import Namespace
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("lkh_adapter", ROOT / "bench/cluster/lib/adapters/lkh.py")
lkh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lkh)


@pytest.mark.parametrize("problem, instance, routes, objectives, customers", [
    ("cvrp", "tiny-cvrp.vrp", "Route 1 : 1 2\nRoute 2 : 3 4\n", [20], [2, 3, 4, 5]),
    ("cvrptw", "tiny-vrptw.txt", "Route 1 : 4 3\nRoute 2 : 2 1\n", [2, 200], [1, 2, 3, 4]),
])
def test_lkh_sintef_node_ids_replay_against_the_original_instance(
    tmp_path, problem, instance, routes, objectives, customers,
):
    # Recorded from the pinned 3.0.13 binary: SINTEF_WriteSolution emits N->Id - 1.
    solution = tmp_path / "solution.txt"
    solution.write_text("Instance name : smoke\nSolution\n" + routes)
    args = Namespace(instance=ROOT / "examples/instances/routing" / instance, distance_scale=10)
    replay = getattr(lkh, "solve_" + problem)(args, lkh.parse_sintef(solution))
    assert replay["verified"] is True
    assert replay["objectives"] == objectives
    assert sorted(node for route in replay["routes"] for node in route["nodes"]) == customers
