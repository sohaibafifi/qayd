"""Qualification tests for the staged Large-TA JSSP hybrid gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "bench" / "jssp_hybrid_gate.py"
SUITE = Path(__file__).resolve().parents[2] / "bench" / "suites" / "large-ta-i1-hybrid-pilot.json"


def load_module():
    spec = importlib.util.spec_from_file_location("qayd_jssp_hybrid_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_suite_stage_names_and_thresholds_match_gate_contract():
    gate = load_module()
    suite = json.loads(SUITE.read_text(encoding="utf-8"))

    assert suite["reference_protocol"]["stage_thresholds"] == gate.STAGE_THRESHOLDS


def record(*, objective=876_334, seed=0, checkpoint=600, digest="a" * 64):
    return {
        "run_id": f"candidate-{checkpoint}-{seed}",
        "solver": "qayd-native",
        "solver_version": "qayd synthetic",
        "problem": "jssp",
        "engine": "ls",
        "instance": "tai_j1000_m1000_1.data",
        "instance_path": "data/jssp/large-ta/tai_j1000_m1000_1.data",
        "instance_sha256": digest,
        "checkpoint_seconds": checkpoint,
        "seed": seed,
        "threads": 7,
        "requested_threads": 7,
        "status": "SATISFIABLE",
        "verified": True,
        "return_code": 0,
        "timed_out": False,
        "objectives": [objective],
        "wall_seconds": checkpoint + 1.5,
        "peak_memory_mb": 5000,
        "start_vector_length": 1_000_000,
        "start_vector_sha256": "b" * 64,
        "verification_counts": {
            "starts": 1_000_000,
            "job_precedence_pairs": 999_000,
            "machine_non_overlap_pairs": 999_000,
            "objective_checks": 1,
        },
        "schedule_oracle_mismatches": 0,
        "schedule_work_budget_overruns": 0,
        "schedule_incumbent_verification_rejections": 0,
        "schedule_incumbent_verification_interruptions": 0,
        "schedule_incumbent_incomplete_rejections": 0,
        "schedule_incumbent_verifications": 3,
        "schedule_moves_considered": 1000,
        "schedule_moves_accepted": 100,
        "schedule_dirty_cone_operations": 250_000_000,
        "schedule_search_elite_pool_size": 3,
        "schedule_search_elite_batches": 2,
        "schedule_search_elite_candidates": 6,
        "schedule_search_elite_insertions": 3,
        "schedule_search_elite_duplicates": 1,
        "schedule_search_elite_dominated": 2,
        "schedule_search_elite_evictions": 0,
        "schedule_search_elite_interruptions": 0,
        "schedule_search_elite_merge_errors": 0,
        "schedule_search_elite_snapshot_captures": 6,
        "schedule_search_elite_snapshot_interruptions": 0,
        "schedule_search_elite_snapshot_errors": 0,
        "schedule_search_elite_objectives": "876334,880000,890000",
        "schedule_search_elite_pairwise_distances_ppm": "0-1=100000,0-2=200000,1-2=300000",
        "schedule_search_elite_min_distance_ppm": 100_000,
        "schedule_search_elite_mean_distance_ppm": 200_000,
        "schedule_search_elite_max_distance_ppm": 300_000,
        "schedule_search_elite_capture_worker_seconds_sum": 0.6,
        "schedule_search_elite_merge_wall_seconds": 0.3,
        "schedule_search_elite_heap_lower_bound_bytes": 60 * 1024 * 1024,
        "schedule_search_elite_peak_heap_lower_bound_bytes": 100 * 1024 * 1024,
        "schedule_search_elite_batches_skipped_after_stop": 0,
        "schedule_reactive_restarts": 4,
        "schedule_path_relink_enabled": 1,
        "schedule_path_relink_active": 1,
        "schedule_path_relink_guide_requests": 3,
        "schedule_path_relink_best_guides": 2,
        "schedule_path_relink_diverse_guides": 1,
        "schedule_path_relink_guide_loads": 1,
        "schedule_path_relink_guide_incompatible": 0,
        "schedule_path_relink_guide_interruptions": 0,
        "schedule_path_relink_critical_operations_scanned": 100,
        "schedule_path_relink_candidates_generated": 10,
        "schedule_path_relink_candidates_positive_gain": 8,
        "schedule_path_relink_acyclicity_certified": 8,
        "schedule_path_relink_acyclicity_unknown": 0,
        "schedule_path_relink_prefilter_rejections": 0,
        "schedule_path_relink_candidates_retained": 8,
        "schedule_path_relink_candidates_refined": 6,
        "schedule_path_relink_candidates_shortlisted": 3,
        "schedule_path_relink_no_move": 0,
        "schedule_path_relink_guide_arc_gain_shortlisted": 4,
        "schedule_path_relink_oracle_attempts": 3,
        "schedule_path_relink_oracle_accepts": 2,
        "schedule_path_relink_cycle_rejections": 0,
        "schedule_path_relink_window_rejections": 0,
        "schedule_path_relink_other_rejections": 1,
        "schedule_path_relink_rollbacks": 1,
        "schedule_path_relink_elite_improvements": 1,
        "schedule_path_relink_guide_arc_gain_accepted": 2,
        "schedule_path_relink_worker_seconds_sum": 0.2,
        "schedule_path_relink_workspace_peak_bytes": 4 * 1024 * 1024,
        "schedule_lns_shadow_enabled": 1,
        "schedule_lns_shadow_active": 1,
        "schedule_lns_shadow_owner_worker_mask": 1,
        "schedule_lns_attempts": 4,
        "schedule_lns_selected_operations": 256,
        "schedule_lns_feasible": 2,
        "schedule_lns_reconstructed": 1,
        "schedule_lns_improvements": 1,
        "schedule_lns_shadow_improvements": 1,
        "schedule_lns_shadow_improvement_sum": 100,
        "schedule_lns_shadow_best_improvement": 100,
        "schedule_lns_timeouts": 1,
        "schedule_lns_interruptions": 0,
        "schedule_lns_infeasible": 0,
        "schedule_lns_non_improving": 1,
        "schedule_lns_reconstruction_rejections": 0,
        "schedule_lns_oracle_rejections": 0,
        "schedule_lns_exact_rejections": 1,
        "schedule_lns_worker_seconds_sum": 0.4,
        "schedule_lns_workspace_peak_bytes": 8 * 1024 * 1024,
        "schedule_profile_work_steps": "0=100,1=100,2=100,3=100,4=100,5=100",
        "schedule_island_profile_mask": 63,
        "schedule_baseline_island_profile_mask": 63,
        "schedule_scored_island_profile_mask": 0,
        "schedule_island_profile_count": 6,
        "schedule_restart_boundaries": 1,
        "schedule_constructor_workers_requested": 1,
        "schedule_constructor_multistart_enabled": 1,
        "schedule_constructor_multistart_active": 1,
        "schedule_constructor_multistart_owner_worker_mask": 64,
        "schedule_constructor_multistart_attempts": 4,
        "schedule_constructor_multistart_constructions": 4,
        "schedule_constructor_multistart_feasible": 4,
        "schedule_constructor_multistart_distinct_fingerprints": 2,
        "schedule_constructor_multistart_other_fingerprint_observations": 2,
        "schedule_constructor_multistart_initial_objective": 943_227,
        "schedule_constructor_multistart_best_objective": 943_000,
        "schedule_constructor_multistart_improvements": 1,
        "schedule_constructor_multistart_work_units": 256,
        "schedule_constructor_multistart_interruptions": 0,
        "schedule_constructor_multistart_failures": 0,
        "schedule_constructor_multistart_worker_seconds_sum": 0.5,
        "schedule_constructor_multistart_workspace_peak_bytes": 16 * 1024 * 1024,
        "schedule_constructor_multistart_next_ordinal": 10,
        "schedule_constructor_multistart_best_ordinal": 8,
        "schedule_constructor_multistart_best_seed": 123456,
        "schedule_constructor_multistart_best_fingerprint": 987654,
    }


def write_campaign(
    tmp_path,
    records,
    name="candidate",
    *,
    path_relink=False,
    lns_shadow=False,
    constructor_workers=0,
):
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    checkpoints = sorted({item["checkpoint_seconds"] for item in records})
    seeds = sorted({item["seed"] for item in records})
    provenance = {
        "schema_version": 1,
        "budgets": checkpoints,
        "seeds": seeds,
        "threads": 7,
        "qayd_engine": "ls",
        "profile_qayd": True,
        "qayd_prepared": True,
        "qayd_schedule_path_relink": path_relink,
        "qayd_schedule_lns_shadow": lns_shadow,
        "qayd_schedule_constructor_workers": constructor_workers,
        "qayd_artifact": {"path": "/tmp/qayd.so", "sha256": "c" * 64},
        "host": {
            "commit": "d" * 40,
            "source_tree_sha256": "e" * 64,
            "dirty": True,
        },
    }
    path.with_suffix(path.suffix + ".provenance.json").write_text(
        json.dumps(provenance) + "\n", encoding="utf-8"
    )
    return path


def write_c1_pair(tmp_path, candidate_record):
    checkpoint = candidate_record["checkpoint_seconds"]
    seed = candidate_record["seed"]
    baseline_record = record(objective=943_065, checkpoint=checkpoint, seed=seed)
    candidate = write_campaign(tmp_path, [candidate_record], name="candidate", constructor_workers=1)
    baseline = write_campaign(tmp_path, [baseline_record], name="baseline")
    return candidate, baseline


def test_target_stage_requires_verified_replayed_records(tmp_path):
    gate = load_module()
    candidate = write_campaign(tmp_path, [record(seed=0), record(seed=1)])

    summary = gate.build_summary(
        candidate,
        baseline_path=None,
        instance="tai_j1000_m1000_1",
        checkpoints={600.0},
        seeds={0, 1},
        threads=7,
    )

    assert summary["target_ready"] is True
    assert summary["shadow_ready"] is True
    assert summary["highest_stage"] == "target"
    assert summary["checkpoints"][0]["median_objective"] == 876_334
    assert summary["rows"][0]["candidate"]["dirty_operations_per_probe"] == 250_000
    assert summary["rows"][0]["candidate"]["metrics"]["schedule_lns_improvements"] == 1


def test_constructor_c1_separates_protocol_readiness_from_quality_signal(tmp_path):
    gate = load_module()
    candidate, baseline = write_c1_pair(tmp_path, record(objective=943_065, checkpoint=60))

    summary = gate.build_summary(
        candidate,
        baseline_path=baseline,
        instance="tai_j1000_m1000_1",
        checkpoints={60.0},
        seeds={0},
        threads=7,
        require_constructor_multistart=True,
    )

    assert summary["constructor_multistart_ready"] is True
    assert summary["constructor_multistart_quality_signal"] is True
    checkpoint = summary["checkpoints"][0]
    assert checkpoint["constructor_multistart_ls_profile_retention"] is True
    assert checkpoint["constructor_multistart_final_objective_ok"] is True
    assert checkpoint["constructor_multistart_best_objective_ok"] is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schedule_constructor_multistart_owner_worker_mask", 32, "wrong_owner"),
        ("schedule_constructor_multistart_interruptions", 1, "partition"),
        ("schedule_constructor_multistart_workspace_peak_bytes", 512 * 1024 * 1024, "workspace_limit"),
        ("schedule_constructor_multistart_distinct_fingerprints", 0, "partition"),
    ],
)
def test_constructor_c1_rejects_incoherent_or_unsafe_accounting(tmp_path, field, value, message):
    gate = load_module()
    value_record = record(objective=943_065, checkpoint=60)
    value_record[field] = value
    candidate, baseline = write_c1_pair(tmp_path, value_record)

    if message == "partition":
        with pytest.raises(gate.GateError, match=message):
            gate.build_summary(
                candidate,
                baseline_path=baseline,
                instance="tai_j1000_m1000_1",
                checkpoints={60.0},
                seeds={0},
                threads=7,
                require_constructor_multistart=True,
            )
    else:
        summary = gate.build_summary(
            candidate,
            baseline_path=baseline,
            instance="tai_j1000_m1000_1",
            checkpoints={60.0},
            seeds={0},
            threads=7,
            require_constructor_multistart=True,
        )
        assert summary["constructor_multistart_ready"] is False
        assert any(message in failure for failure in summary["rows"][0]["candidate"]["constructor_multistart_failures"])


def test_constructor_c1_quality_thresholds_do_not_change_readiness(tmp_path):
    gate = load_module()
    value_record = record(objective=943_066, checkpoint=60)
    value_record["schedule_constructor_multistart_initial_objective"] = 950_000
    value_record["schedule_constructor_multistart_best_objective"] = 943_228
    candidate, baseline = write_c1_pair(tmp_path, value_record)

    summary = gate.build_summary(
        candidate,
        baseline_path=baseline,
        instance="tai_j1000_m1000_1",
        checkpoints={60.0},
        seeds={0},
        threads=7,
        require_constructor_multistart=True,
    )

    assert summary["constructor_multistart_ready"] is True
    assert summary["constructor_multistart_quality_signal"] is False


def test_constructor_c1_checks_ls_profile_retention_when_baseline_exposes_it(tmp_path):
    gate = load_module()
    candidate_record = record(objective=943_065, checkpoint=60)
    candidate_record["schedule_profile_work_steps"] = "0=100,1=100,2=100,3=100,4=100,5=89"
    baseline_record = record(objective=943_065, checkpoint=60)
    baseline_record["schedule_profile_work_steps"] = "0=100,1=100,2=100,3=100,4=100,5=100"
    candidate = write_campaign(tmp_path, [candidate_record], name="candidate", constructor_workers=1)
    baseline = write_campaign(tmp_path, [baseline_record], name="baseline")

    summary = gate.build_summary(
        candidate,
        baseline_path=baseline,
        instance="tai_j1000_m1000_1",
        checkpoints={60.0},
        seeds={0},
        threads=7,
        require_constructor_multistart=True,
    )

    assert summary["checkpoints"][0]["constructor_multistart_ls_profile_retention"] is False
    assert summary["constructor_multistart_ready"] is False


def test_constructor_c1_requires_a_fresh_profiled_baseline(tmp_path):
    gate = load_module()
    candidate = write_campaign(
        tmp_path,
        [record(objective=943_065, checkpoint=60)],
        constructor_workers=1,
    )

    with pytest.raises(gate.GateError, match="fresh baseline"):
        gate.build_summary(
            candidate,
            baseline_path=None,
            instance="tai_j1000_m1000_1",
            checkpoints={60.0},
            seeds={0},
            threads=7,
            require_constructor_multistart=True,
        )


def test_constructor_c1_requires_identical_source_and_artifact_fingerprints(tmp_path):
    gate = load_module()
    candidate, baseline = write_c1_pair(tmp_path, record(objective=943_065, checkpoint=60))
    sidecar = baseline.with_suffix(baseline.suffix + ".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    provenance["qayd_artifact"]["sha256"] = "f" * 64
    sidecar.write_text(json.dumps(provenance) + "\n", encoding="utf-8")

    with pytest.raises(gate.GateError, match="identical source and artifact"):
        gate.build_summary(
            candidate,
            baseline_path=baseline,
            instance="tai_j1000_m1000_1",
            checkpoints={60.0},
            seeds={0},
            threads=7,
            require_constructor_multistart=True,
        )


def test_constructor_c1_requires_plain_a0_baseline_controls(tmp_path):
    gate = load_module()
    candidate, baseline = write_c1_pair(tmp_path, record(objective=943_065, checkpoint=60))
    sidecar = baseline.with_suffix(baseline.suffix + ".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    provenance["qayd_schedule_path_relink"] = True
    sidecar.write_text(json.dumps(provenance) + "\n", encoding="utf-8")

    with pytest.raises(gate.GateError, match="disable path relinking"):
        gate.build_summary(
            candidate,
            baseline_path=baseline,
            instance="tai_j1000_m1000_1",
            checkpoints={60.0},
            seeds={0},
            threads=7,
            require_constructor_multistart=True,
        )


def test_constructor_c1_enforces_four_attempts_per_restart_boundary(tmp_path):
    gate = load_module()
    value_record = record(objective=943_065, checkpoint=60)
    value_record.update(
        {
            "schedule_constructor_multistart_attempts": 5,
            "schedule_constructor_multistart_constructions": 5,
            "schedule_constructor_multistart_feasible": 5,
            "schedule_constructor_multistart_other_fingerprint_observations": 3,
            "schedule_constructor_multistart_work_units": 320,
            "schedule_constructor_multistart_next_ordinal": 11,
            "schedule_restart_boundaries": 1,
        }
    )
    candidate, baseline = write_c1_pair(tmp_path, value_record)

    with pytest.raises(gate.GateError, match="four-per-boundary"):
        gate.build_summary(
            candidate,
            baseline_path=baseline,
            instance="tai_j1000_m1000_1",
            checkpoints={60.0},
            seeds={0},
            threads=7,
            require_constructor_multistart=True,
        )


def test_constructor_c1_treats_coherent_constructor_interruption_as_readiness_failure(tmp_path):
    gate = load_module()
    value_record = record(objective=943_065, checkpoint=60)
    value_record.update(
        {
            "schedule_constructor_multistart_attempts": 5,
            "schedule_constructor_multistart_interruptions": 1,
            "schedule_constructor_multistart_work_units": 320,
            "schedule_restart_boundaries": 2,
        }
    )
    candidate, baseline = write_c1_pair(tmp_path, value_record)

    summary = gate.build_summary(
        candidate,
        baseline_path=baseline,
        instance="tai_j1000_m1000_1",
        checkpoints={60.0},
        seeds={0},
        threads=7,
        require_constructor_multistart=True,
    )

    assert summary["constructor_multistart_ready"] is False
    assert "constructor_multistart_interruption" in summary["rows"][0]["candidate"]["constructor_multistart_failures"]


def test_shadow_stage_is_observational_and_accepts_batches_skipped_after_stop(tmp_path):
    gate = load_module()
    value_record = record(objective=943_011)
    value_record["schedule_search_elite_batches_skipped_after_stop"] = 3
    candidate = write_campaign(tmp_path, [value_record])

    summary = gate.build_summary(
        candidate,
        baseline_path=None,
        instance="tai_j1000_m1000_1",
        checkpoints={600.0},
        seeds={0},
        threads=7,
    )

    assert summary["shadow_ready"] is True
    assert summary["highest_stage"] == "shadow"
    assert summary["highest_quality_stage"] is None
    assert summary["target_ready"] is False
    metrics = summary["rows"][0]["candidate"]["metrics"]
    assert metrics["schedule_search_elite_batches_skipped_after_stop"] == 3


def test_path_relink_gate_requires_provenance_activity_oracle_and_bounded_workspace(tmp_path):
    gate = load_module()
    candidate = write_campaign(tmp_path, [record(objective=943_011)], path_relink=True)

    summary = gate.build_summary(
        candidate,
        baseline_path=None,
        instance="tai_j1000_m1000_1",
        checkpoints={600.0},
        seeds={0},
        threads=7,
        require_path_relink=True,
    )

    assert summary["shadow_ready"] is True
    assert summary["path_relink_required"] is True
    assert summary["path_relink_ready"] is True
    assert summary["rows"][0]["candidate"]["path_relink_failures"] == []


@pytest.mark.parametrize(
    ("field", "value", "failure"),
    [
        ("schedule_path_relink_enabled", 0, "path_relink_not_enabled"),
        ("schedule_path_relink_active", 0, "path_relink_not_active"),
        ("schedule_path_relink_guide_incompatible", 1, "path_relink_incompatible_guide"),
        ("schedule_path_relink_guide_interruptions", 1, "path_relink_guide_interruption"),
        ("schedule_path_relink_oracle_attempts", 0, "path_relink_no_oracle_attempt"),
        ("schedule_path_relink_workspace_peak_bytes", 16 * 1024 * 1024, "path_relink_workspace_limit"),
    ],
)
def test_path_relink_gate_fails_closed_on_missing_activity_or_integrity(tmp_path, field, value, failure):
    gate = load_module()
    value_record = record(objective=943_011)
    value_record[field] = value
    if field == "schedule_path_relink_oracle_attempts":
        value_record["schedule_path_relink_oracle_accepts"] = 0
        value_record["schedule_path_relink_other_rejections"] = 0
        value_record["schedule_path_relink_rollbacks"] = 0
        value_record["schedule_path_relink_elite_improvements"] = 0
        value_record["schedule_path_relink_guide_arc_gain_accepted"] = 0
    candidate = write_campaign(tmp_path, [value_record], path_relink=True)

    summary = gate.build_summary(
        candidate,
        baseline_path=None,
        instance="tai_j1000_m1000_1",
        checkpoints={600.0},
        seeds={0},
        threads=7,
        require_path_relink=True,
    )

    assert summary["path_relink_ready"] is False
    assert failure in summary["rows"][0]["candidate"]["path_relink_failures"]


def test_path_relink_gate_rejects_unrecorded_campaign_protocol(tmp_path):
    gate = load_module()
    candidate = write_campaign(tmp_path, [record(objective=943_011)], path_relink=False)

    with pytest.raises(gate.GateError, match="provenance must enable"):
        gate.build_summary(
            candidate,
            baseline_path=None,
            instance="tai_j1000_m1000_1",
            checkpoints={600.0},
            seeds={0},
            threads=7,
            require_path_relink=True,
        )


def test_lns_shadow_gate_requires_safe_activity_and_reports_effectiveness(tmp_path):
    gate = load_module()
    candidate = write_campaign(
        tmp_path,
        [record(objective=943_011)],
        lns_shadow=True,
    )

    summary = gate.build_summary(
        candidate,
        baseline_path=None,
        instance="tai_j1000_m1000_1",
        checkpoints={600.0},
        seeds={0},
        threads=7,
        require_lns_shadow=True,
    )

    assert summary["lns_shadow_required"] is True
    assert summary["lns_shadow_ready"] is True
    assert summary["lns_effective"] is True
    assert summary["rows"][0]["candidate"]["lns_shadow_failures"] == []


@pytest.mark.parametrize(
    ("field", "value", "failure"),
    [
        ("schedule_lns_shadow_enabled", 0, "lns_shadow_not_enabled"),
        ("schedule_lns_shadow_active", 0, "lns_shadow_not_active"),
        ("schedule_lns_shadow_owner_worker_mask", 2, "lns_shadow_wrong_owner"),
        ("schedule_lns_attempts", 0, "lns_shadow_no_attempt"),
        ("schedule_lns_selected_operations", 0, "lns_shadow_no_selected_operation"),
        ("schedule_lns_interruptions", 1, "lns_shadow_interruption"),
        ("schedule_lns_oracle_rejections", 1, "lns_shadow_oracle_rejection"),
        ("schedule_lns_workspace_peak_bytes", 512 * 1024 * 1024, "lns_shadow_workspace_limit"),
    ],
)
def test_lns_shadow_gate_fails_closed_on_missing_activity_or_safety(tmp_path, field, value, failure):
    gate = load_module()
    value_record = record(objective=943_011)
    value_record[field] = value
    if field == "schedule_lns_attempts":
        value_record.update(
            {
                "schedule_lns_selected_operations": 0,
                "schedule_lns_feasible": 0,
                "schedule_lns_reconstructed": 0,
                "schedule_lns_improvements": 0,
                "schedule_lns_shadow_improvements": 0,
                "schedule_lns_shadow_improvement_sum": 0,
                "schedule_lns_shadow_best_improvement": 0,
                "schedule_lns_timeouts": 0,
                "schedule_lns_non_improving": 0,
                "schedule_lns_exact_rejections": 0,
                "schedule_lns_workspace_peak_bytes": 0,
            }
        )
    candidate = write_campaign(tmp_path, [value_record], lns_shadow=True)

    summary = gate.build_summary(
        candidate,
        baseline_path=None,
        instance="tai_j1000_m1000_1",
        checkpoints={600.0},
        seeds={0},
        threads=7,
        require_lns_shadow=True,
    )

    assert summary["lns_shadow_ready"] is False
    assert failure in summary["rows"][0]["candidate"]["lns_shadow_failures"]


def test_lns_shadow_all_timeouts_is_safe_but_not_effective(tmp_path):
    gate = load_module()
    value_record = record(objective=943_011)
    value_record.update(
        {
            "schedule_lns_timeouts": value_record["schedule_lns_attempts"],
            "schedule_lns_improvements": 0,
            "schedule_lns_shadow_improvements": 0,
            "schedule_lns_shadow_improvement_sum": 0,
            "schedule_lns_shadow_best_improvement": 0,
        }
    )
    candidate = write_campaign(tmp_path, [value_record], lns_shadow=True)

    summary = gate.build_summary(
        candidate,
        baseline_path=None,
        instance="tai_j1000_m1000_1",
        checkpoints={600.0},
        seeds={0},
        threads=7,
        require_lns_shadow=True,
    )

    assert summary["lns_shadow_ready"] is True
    assert summary["lns_effective"] is False


def test_lns_shadow_safe_non_improving_repairs_are_not_called_effective(tmp_path):
    gate = load_module()
    value_record = record(objective=943_011)
    value_record.update(
        {
            "schedule_lns_improvements": 0,
            "schedule_lns_shadow_improvements": 0,
            "schedule_lns_shadow_improvement_sum": 0,
            "schedule_lns_shadow_best_improvement": 0,
        }
    )
    candidate = write_campaign(tmp_path, [value_record], lns_shadow=True)

    summary = gate.build_summary(
        candidate,
        baseline_path=None,
        instance="tai_j1000_m1000_1",
        checkpoints={600.0},
        seeds={0},
        threads=7,
        require_lns_shadow=True,
    )

    assert summary["lns_shadow_ready"] is True
    assert summary["lns_effective"] is False


def test_lns_shadow_gate_rejects_unrecorded_campaign_protocol(tmp_path):
    gate = load_module()
    candidate = write_campaign(tmp_path, [record(objective=943_011)], lns_shadow=False)

    with pytest.raises(gate.GateError, match="provenance must enable"):
        gate.build_summary(
            candidate,
            baseline_path=None,
            instance="tai_j1000_m1000_1",
            checkpoints={600.0},
            seeds={0},
            threads=7,
            require_lns_shadow=True,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schedule_path_relink_acyclicity_certified", 7, "must partition"),
        ("schedule_path_relink_acyclicity_unknown", 1, "must partition"),
        ("schedule_path_relink_prefilter_rejections", 1, "must be rejected"),
        ("schedule_path_relink_candidates_retained", 9, "candidate funnel"),
    ],
)
def test_path_relink_gate_enforces_acyclicity_prefilter_accounting(tmp_path, field, value, message):
    gate = load_module()
    value_record = record(objective=943_011)
    value_record[field] = value
    candidate = write_campaign(tmp_path, [value_record], path_relink=True)

    with pytest.raises(gate.GateError, match=message):
        gate.build_summary(
            candidate,
            baseline_path=None,
            instance="tai_j1000_m1000_1",
            checkpoints={600.0},
            seeds={0},
            threads=7,
            require_path_relink=True,
        )


def test_path_relink_gate_rejects_any_post_prefilter_cycle(tmp_path):
    gate = load_module()
    value_record = record(objective=943_011)
    value_record["schedule_path_relink_cycle_rejections"] = 1
    value_record["schedule_path_relink_other_rejections"] = 0
    candidate = write_campaign(tmp_path, [value_record], path_relink=True)

    summary = gate.build_summary(
        candidate,
        baseline_path=None,
        instance="tai_j1000_m1000_1",
        checkpoints={600.0},
        seeds={0},
        threads=7,
        require_path_relink=True,
    )

    assert summary["path_relink_ready"] is False
    assert "path_relink_cycle_rejection" in summary["rows"][0]["candidate"]["path_relink_failures"]


def test_path_relink_gate_rejects_combined_oracle_outcomes_above_attempts(tmp_path):
    gate = load_module()
    value_record = record(objective=943_011)
    value_record["schedule_path_relink_oracle_accepts"] = 3
    value_record["schedule_path_relink_other_rejections"] = 1
    candidate = write_campaign(tmp_path, [value_record], path_relink=True)

    with pytest.raises(gate.GateError, match="oracle accounting"):
        gate.build_summary(
            candidate,
            baseline_path=None,
            instance="tai_j1000_m1000_1",
            checkpoints={600.0},
            seeds={0},
            threads=7,
            require_path_relink=True,
        )


def test_cli_lns_shadow_gate_returns_two_when_shadow_is_not_ready(tmp_path):
    value_record = record(objective=943_011)
    value_record["schedule_lns_interruptions"] = 1
    candidate = write_campaign(tmp_path, [value_record], lns_shadow=True)

    failed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(candidate),
            "--threads",
            "7",
            "--require-lns-shadow",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert failed.returncode == 2
    assert "LNS shadow ready: `no`" in failed.stdout


@pytest.mark.parametrize(
    ("field", "value", "failure"),
    [
        ("schedule_search_elite_snapshot_errors", 1, "snapshot_errors"),
        ("schedule_search_elite_merge_errors", 1, "merge_errors"),
        ("schedule_search_elite_interruptions", 1, "merge_interruptions"),
        ("schedule_search_elite_snapshot_interruptions", 1, "snapshot_interruptions"),
        ("peak_memory_mb", 12_288, "rss_limit"),
        (
            "schedule_search_elite_peak_heap_lower_bound_bytes",
            6_000 * 1024 * 1024,
            "peak_heap_lower_bound_exceeds_rss",
        ),
    ],
)
def test_shadow_stage_fails_on_errors_or_memory(tmp_path, field, value, failure):
    gate = load_module()
    value_record = record(objective=876_334)
    value_record[field] = value
    candidate = write_campaign(tmp_path, [value_record])

    summary = gate.build_summary(
        candidate,
        baseline_path=None,
        instance="tai_j1000_m1000_1",
        checkpoints={600.0},
        seeds={0},
        threads=7,
    )

    assert summary["shadow_ready"] is False
    assert summary["highest_stage"] is None
    assert summary["target_ready"] is False
    assert failure in summary["rows"][0]["candidate"]["shadow_failures"]


def test_shadow_stage_requires_two_diverse_entries(tmp_path):
    gate = load_module()
    value_record = record(objective=943_011)
    value_record.update(
        {
            "schedule_search_elite_pool_size": 1,
            "schedule_search_elite_insertions": 1,
            "schedule_search_elite_objectives": "943011",
            "schedule_search_elite_pairwise_distances_ppm": "none",
            "schedule_search_elite_min_distance_ppm": None,
            "schedule_search_elite_mean_distance_ppm": None,
            "schedule_search_elite_max_distance_ppm": None,
        }
    )
    candidate = write_campaign(tmp_path, [value_record])

    summary = gate.build_summary(
        candidate,
        baseline_path=None,
        instance="tai_j1000_m1000_1",
        checkpoints={600.0},
        seeds={0},
        threads=7,
    )

    assert summary["shadow_ready"] is False
    assert summary["rows"][0]["candidate"]["shadow_failures"] == ["pool_size_lt_2"]


def test_obsolete_elite_fields_do_not_satisfy_shadow_contract(tmp_path):
    gate = load_module()
    value_record = record()
    for field in list(value_record):
        if field.startswith("schedule_search_elite_"):
            del value_record[field]
    value_record["schedule_elite_pool_size"] = 3
    value_record["schedule_elite_insertions"] = 3
    candidate = write_campaign(tmp_path, [value_record])

    with pytest.raises(gate.GateError, match="schedule_search_elite_pool_size must be an integer"):
        gate.build_summary(
            candidate,
            baseline_path=None,
            instance="tai_j1000_m1000_1",
            checkpoints={600.0},
            seeds={0},
            threads=7,
        )


def test_inconsistent_search_elite_distance_summary_fails_closed(tmp_path):
    gate = load_module()
    value_record = record()
    value_record["schedule_search_elite_mean_distance_ppm"] = 200_001
    candidate = write_campaign(tmp_path, [value_record])

    with pytest.raises(gate.GateError, match="distance statistics do not match"):
        gate.build_summary(
            candidate,
            baseline_path=None,
            instance="tai_j1000_m1000_1",
            checkpoints={600.0},
            seeds={0},
            threads=7,
        )


def test_soft_deadline_verification_interruption_does_not_invalidate_replayed_incumbent(
    tmp_path,
):
    gate = load_module()
    value_record = record()
    value_record["schedule_incumbent_verification_interruptions"] = 1
    candidate = write_campaign(tmp_path, [value_record])

    summary = gate.build_summary(
        candidate,
        baseline_path=None,
        instance="tai_j1000_m1000_1",
        checkpoints={600.0},
        seeds={0},
        threads=7,
    )

    assert summary["target_ready"] is True
    assert summary["rows"][0]["candidate"]["verification_interruptions"] == 1


def test_baseline_pairing_reports_per_seed_gain(tmp_path):
    gate = load_module()
    candidate = write_campaign(tmp_path, [record(objective=926_000, seed=0)])
    baseline_record = record(objective=943_011, seed=0)
    for field in list(baseline_record):
        if field.startswith("schedule_search_elite_"):
            del baseline_record[field]
    baseline_record["schedule_elite_pool_size"] = 1
    baseline = write_campaign(tmp_path, [baseline_record], name="baseline")

    summary = gate.build_summary(
        candidate,
        baseline_path=baseline,
        instance="tai_j1000_m1000_1",
        checkpoints={600.0},
        seeds={0},
        threads=7,
    )

    assert summary["highest_stage"] == "portfolio"
    assert summary["rows"][0]["baseline_objective"] == 943_011
    assert summary["rows"][0]["objective_gain"] == 17_011


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("verified", False, "verified, successful"),
        ("schedule_oracle_mismatches", 1, "oracle_mismatches must be exactly 0"),
        ("schedule_work_budget_overruns", 1, "work_budget_overruns must be exactly 0"),
        (
            "schedule_incumbent_verification_rejections",
            1,
            "verification_rejections must be exactly 0",
        ),
        ("start_vector_length", 999_999, "one-million-start commitment"),
        (
            "verification_counts",
            {
                "starts": 1_000_000,
                "job_precedence_pairs": 999_000,
                "machine_non_overlap_pairs": 998_999,
                "objective_checks": 1,
            },
            "replay counts",
        ),
    ],
)
def test_invalid_candidate_fails_closed(tmp_path, field, value, message):
    gate = load_module()
    value_record = record()
    value_record[field] = value
    candidate = write_campaign(tmp_path, [value_record])

    with pytest.raises(gate.GateError, match=message):
        gate.build_summary(
            candidate,
            baseline_path=None,
            instance="tai_j1000_m1000_1",
            checkpoints={600.0},
            seeds={0},
            threads=7,
        )


def test_missing_duplicate_and_mismatched_instance_fail(tmp_path):
    gate = load_module()
    missing = write_campaign(tmp_path, [record(seed=0)])
    missing_provenance_path = missing.with_suffix(missing.suffix + ".provenance.json")
    missing_provenance = json.loads(missing_provenance_path.read_text(encoding="utf-8"))
    missing_provenance["seeds"] = [0, 1]
    missing_provenance_path.write_text(json.dumps(missing_provenance) + "\n", encoding="utf-8")
    with pytest.raises(gate.GateError, match="missing records"):
        gate.build_summary(
            missing,
            baseline_path=None,
            instance="tai_j1000_m1000_1",
            checkpoints={600.0},
            seeds={0, 1},
            threads=7,
        )

    duplicate = write_campaign(tmp_path, [record(seed=0), record(seed=0)], name="duplicate")
    with pytest.raises(gate.GateError, match="duplicate record"):
        gate.build_summary(
            duplicate,
            baseline_path=None,
            instance="tai_j1000_m1000_1",
            checkpoints={600.0},
            seeds={0},
            threads=7,
        )

    candidate = write_campaign(tmp_path, [record(digest="f" * 64)], name="different")
    baseline = write_campaign(tmp_path, [record(digest="a" * 64)], name="baseline")
    with pytest.raises(gate.GateError, match="instance SHA-256 differs"):
        gate.build_summary(
            candidate,
            baseline_path=baseline,
            instance="tai_j1000_m1000_1",
            checkpoints={600.0},
            seeds={0},
            threads=7,
        )


def test_cli_stage_gate_returns_two_until_threshold_is_met(tmp_path):
    candidate = write_campaign(tmp_path, [record(objective=926_341)])
    failed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(candidate),
            "--threads",
            "7",
            "--require-stage",
            "target",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    passed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(candidate),
            "--threads",
            "7",
            "--require-stage",
            "portfolio",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert failed.returncode == 2
    assert passed.returncode == 0
    assert "Highest stage: `portfolio`" in passed.stdout


def test_cli_shadow_gate_fails_without_a_clean_archive_trace(tmp_path):
    failed_record = record(objective=943_011)
    failed_record["schedule_search_elite_snapshot_interruptions"] = 1
    candidate = write_campaign(tmp_path, [failed_record])

    failed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(candidate),
            "--threads",
            "7",
            "--require-stage",
            "shadow",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert failed.returncode == 2
    assert "SHADOW ready: `no`" in failed.stdout
