from argparse import Namespace
import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "bench" / "routing_lp_ablation.py"


def _load_module():
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec = importlib.util.spec_from_file_location("qayd_routing_lp_ablation", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


ablation = _load_module()


def _args():
    return Namespace(
        threads=4,
        launcher="api",
        max_iterations=None,
        lp_max_variables=2000,
        lp_max_rows=1000,
        lp_max_nonzeros=100000,
    )


def _record(arm, primal, dual, lp_bound=None, errors=None):
    return {
        "schema": "qayd.routing-lp-ablation.result/v1",
        "run_id": f"{arm}-{primal}-{dual}",
        "arm": arm,
        "instance_path": "data/vrplib/CVRP/X-n101-k25.vrp",
        "search_budget_seconds": 10,
        "seed": 0,
        "status": "SATISFIABLE",
        "verified": True,
        "objectives": [primal],
        "dual_bound": dual,
        "lp_root_bound": lp_bound,
        "lp_model_status": "ready" if lp_bound is not None else "not-attempted",
        "lp_simplex_time_ms": 25.0 if lp_bound is not None else 0.0,
        "lp_solves": 4 if lp_bound is not None else 0,
        "lp_certified": 4 if lp_bound is not None else 0,
        "bound_method": "certified q-route LP relaxation" if lp_bound is not None else "directed assignment relaxation",
        "wall_seconds": 10.1,
        "peak_memory_mb": 100.0,
        "validation_errors": errors or [],
    }


def test_arm_order_is_rotated_without_changing_comparison_identity():
    arms = ablation.arm_specs([0, 250, 1000, 3000])
    instances = [
        {
            "path": None,
            "relative_path": f"X-{index}.vrp",
            "sha256": str(index),
            "known_upper_bound": None,
        }
        for index in range(4)
    ]
    jobs = ablation.ordered_jobs(instances, arms, [10], [0], _args())

    positions = {arm["name"]: set() for arm in arms}
    for job in jobs:
        positions[job["arm"]["name"]].add(job["order_position"])
    assert all(values == {0, 1, 2, 3} for values in positions.values())
    assert len({job["run_id"] for job in jobs}) == len(jobs)


def test_explicit_instance_suppresses_the_default_glob(tmp_path, monkeypatch):
    explicit = tmp_path / "chosen.vrp"
    explicit.write_text("NAME: chosen\n", encoding="utf-8")
    default = tmp_path / "data" / "vrplib" / "CVRP" / "X-default.vrp"
    default.parent.mkdir(parents=True)
    default.write_text("NAME: default\n", encoding="utf-8")
    monkeypatch.setattr(ablation, "ROOT", tmp_path)

    instances = ablation.discover_instances([str(explicit)], [], 0)

    assert [item["path"] for item in instances] == [explicit.resolve()]


def test_certification_checks_incumbent_and_known_solution():
    valid = _record("lp-250ms", 120, 100, 105)
    assert ablation.validate_record(valid, 110) == []

    invalid = _record("lp-250ms", 120, 121, 111)
    errors = ablation.validate_record(invalid, 110)
    assert "dual_bound crosses the verified incumbent" in errors
    assert "dual_bound crosses the known feasible solution" in errors
    assert "lp_root_bound crosses the known feasible solution" in errors


def test_summary_is_paired_and_keeps_primal_and_dual_directions_distinct():
    records = [
        _record("structural", 120, 80),
        _record("lp-250ms", 115, 100, 100),
        _record("lp-1000ms", 125, 110, 110),
    ]
    summary = ablation.summarize(records, ["structural", "lp-250ms", "lp-1000ms"], False)
    by_arm = {row["arm"]: row for row in summary["arms"]}

    assert summary["complete_groups"] == 1
    assert by_arm["lp-250ms"]["primal_wins"] == 1
    assert by_arm["lp-250ms"]["median_primal_delta"] == -5
    assert by_arm["lp-250ms"]["median_dual_delta"] == 20
    assert by_arm["lp-1000ms"]["primal_losses"] == 1
    assert by_arm["lp-1000ms"]["median_dual_delta"] == 30
    instance_rows = {row["arm"]: row for row in summary["instances"]}
    assert instance_rows["lp-250ms"]["median_dual_gain"] == 0.25
    assert instance_rows["lp-1000ms"]["median_lp_root_bound"] == 110


def test_summary_rejects_an_incomplete_experiment_by_default():
    with pytest.raises(ablation.CampaignError, match="incomplete arm groups"):
        ablation.summarize([_record("structural", 120, 80)], ["structural", "lp-250ms"], False)
