"""Qualification tests for the bounded TSAB JSSP A/B gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "bench" / "jssp_tsab_gate.py"


def load_gate():
    spec = importlib.util.spec_from_file_location("qayd_jssp_tsab_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tsab_metrics(**overrides):
    values = {
        "schedule_tsab_owner_worker_mask": 0,
        "schedule_tsab_shortlists": 0,
        "schedule_tsab_n5_generated": 0,
        "schedule_tsab_ranked": 0,
        "schedule_tsab_delta_probes": 0,
        "schedule_tsab_additional_delta_probes": 0,
        "schedule_tsab_full_oracle_commits": 0,
        "schedule_tsab_selections": 0,
        "schedule_tsab_selected_shortlist_rank_sum": 0,
        "schedule_tsab_aspirations": 0,
        "schedule_tsab_tabu_rejections": 0,
        "schedule_tsab_tabu_resets": 0,
        "schedule_tsab_fingerprint_repeats": 0,
        "schedule_tsab_escape_signals": 0,
        "schedule_tsab_n1_kicks": 0,
        "schedule_tsab_kick_moves": 0,
        "schedule_tsab_elite_restarts": 0,
        "schedule_tsab_n6_kicks": 0,
        "schedule_tsab_restart_attempts": 0,
        "schedule_tsab_restart_global_rebases": 0,
        "schedule_tsab_restart_n6_generated": 0,
        "schedule_tsab_restart_delta_probes": 0,
        "schedule_tsab_restart_oracle_commits": 0,
        "schedule_tsab_restart_relink_attempts": 0,
        "schedule_tsab_restart_relink_commits": 0,
        "schedule_tsab_restart_relink_components_shortlisted": 0,
        "schedule_tsab_restart_relink_components_committed": 0,
        "schedule_tsab_restart_relink_guide_arc_gain_shortlisted": 0,
        "schedule_tsab_restart_relink_guide_arc_gain_accepted": 0,
        "schedule_tsab_restart_rejections": 0,
        "schedule_tsab_restart_interruptions": 0,
        "schedule_tsab_restart_work_units": 0,
        "schedule_tsab_restart_best_base_objective": None,
        "schedule_tsab_restart_best_kicked_objective": None,
        "schedule_tsab_post_restart_improvements": 0,
        "schedule_tsab_restart_shortlist_peak_bytes": 0,
        "schedule_tsab_ranking_audits": 0,
        "schedule_tsab_exact_best_matches": 0,
        "schedule_tsab_regret_sum": 0,
        "schedule_tsab_regret_max": 0,
        "schedule_tsab_workspace_peak_bytes": 0,
        "schedule_tsab_activations": 0,
        "schedule_tsab_activation_boundary": None,
        "schedule_tsab_legacy_warmup_work_steps": 0,
        "schedule_tsab_activation_rebases": 0,
        "schedule_tsab_activation_objective": None,
        "schedule_tsab_active_boundaries": 0,
        "schedule_tsab_burst_work_limit": 0,
        "schedule_tsab_burst_work_units": 0,
        "schedule_tsab_improving_commits": 0,
        "schedule_tsab_best_committed_objective": None,
        "schedule_tsab_fast_enabled": 0,
        "schedule_tsab_fast_eligible": 0,
        "schedule_tsab_fast_disabled": 0,
        "schedule_tsab_fast_attempts": 0,
        "schedule_tsab_fast_commits": 0,
        "schedule_tsab_fast_fallbacks": 0,
        "schedule_tsab_fast_work_cap_recoveries": 0,
        "schedule_tsab_fast_date_changes": 0,
        "schedule_tsab_fast_queue_pops": 0,
        "schedule_tsab_fast_full_validations": 0,
        "schedule_tsab_fast_oracle_mismatches": 0,
        "schedule_tsab_fast_pending_promotions": 0,
        "schedule_tsab_fast_pending_discards": 0,
        "schedule_tsab_fast_transitions": 0,
        "schedule_tsab_fast_work_units": 0,
        "schedule_tsab_fast_elapsed_seconds": 0.0,
        "schedule_tsab_fast_workspace_peak_bytes": 0,
    }
    values.update(overrides)
    return values


def record(*, candidate=False, objective=943_043, seed=0):
    metrics = tsab_metrics()
    if candidate:
        metrics.update(
            tsab_metrics(
                schedule_tsab_owner_worker_mask=0x40,
                schedule_tsab_shortlists=60,
                schedule_tsab_n5_generated=300,
                schedule_tsab_ranked=300,
                schedule_tsab_delta_probes=0,
                schedule_tsab_additional_delta_probes=0,
                schedule_tsab_full_oracle_commits=0,
                schedule_tsab_selections=60,
                schedule_tsab_selected_shortlist_rank_sum=0,
                schedule_tsab_aspirations=0,
                schedule_tsab_tabu_rejections=0,
                schedule_tsab_tabu_resets=1,
                schedule_tsab_workspace_peak_bytes=4096,
                schedule_tsab_activations=1,
                schedule_tsab_activation_boundary=2,
                schedule_tsab_legacy_warmup_work_steps=512,
                schedule_tsab_activation_rebases=1,
                schedule_tsab_activation_objective=943_200,
                schedule_tsab_active_boundaries=1,
                schedule_tsab_burst_work_limit=128,
                schedule_tsab_burst_work_units=60,
                schedule_tsab_improving_commits=2,
                schedule_tsab_best_committed_objective=943_100,
                schedule_tsab_escape_signals=2,
                schedule_tsab_kick_moves=2,
                schedule_tsab_elite_restarts=2,
                schedule_tsab_n6_kicks=2,
                schedule_tsab_restart_attempts=2,
                schedule_tsab_restart_global_rebases=1,
                schedule_tsab_restart_n6_generated=100,
                schedule_tsab_restart_delta_probes=8,
                schedule_tsab_restart_oracle_commits=2,
                schedule_tsab_restart_rejections=0,
                schedule_tsab_restart_interruptions=0,
                schedule_tsab_restart_work_units=16,
                schedule_tsab_restart_best_base_objective=943_180,
                schedule_tsab_restart_best_kicked_objective=943_150,
                schedule_tsab_post_restart_improvements=1,
                schedule_tsab_restart_shortlist_peak_bytes=512,
                schedule_tsab_fast_enabled=1,
                schedule_tsab_fast_eligible=1,
                schedule_tsab_fast_attempts=60,
                schedule_tsab_fast_commits=48,
                schedule_tsab_fast_fallbacks=3,
                schedule_tsab_fast_work_cap_recoveries=3,
                schedule_tsab_fast_date_changes=300,
                schedule_tsab_fast_queue_pops=600,
                schedule_tsab_fast_full_validations=2,
                schedule_tsab_fast_pending_promotions=1,
                schedule_tsab_fast_transitions=48,
                schedule_tsab_fast_work_units=15,
                schedule_tsab_fast_elapsed_seconds=0.25,
                schedule_tsab_fast_workspace_peak_bytes=8192,
            )
        )
    return {
        "run_id": f"{'candidate' if candidate else 'baseline'}-60-{seed}",
        "solver": "qayd-native",
        "problem": "jssp",
        "engine": "ls",
        "instance": "tai_j1000_m1000_1.data",
        "instance_sha256": "a" * 64,
        "checkpoint_seconds": 60,
        "seed": seed,
        "threads": 7,
        "requested_threads": 7,
        "status": "SATISFIABLE",
        "verified": True,
        "return_code": 0,
        "timed_out": False,
        "objectives": [objective],
        "peak_memory_mb": 5000,
        "start_vector_length": 1_000_000,
        "start_vector_sha256": "b" * 64,
        "verification_counts": {
            "starts": 1_000_000,
            "job_precedence_pairs": 999_000,
            "machine_non_overlap_pairs": 999_000,
            "objective_checks": 1,
        },
        "schedule_work_budget_overruns": 0,
        "schedule_oracle_mismatches": 0,
        "schedule_incumbent_incomplete_rejections": 0,
        "schedule_incumbent_verification_rejections": 0,
        "schedule_incumbent_verifications": 2,
        "schedule_incumbent_verification_seconds": 0.05,
        "schedule_incumbent_verification_max_seconds": 0.03,
        "schedule_tabu_forced_moves": 0,
        "schedule_island_profile_mask": 0x7F if candidate else 0x7F,
        "schedule_baseline_island_profile_mask": 0x3F,
        "schedule_scored_island_profile_mask": 0x40,
        "schedule_profile_work_steps": "0=100,1=100,2=100,3=100,4=100,5=100,6=100",
        "schedule_profile_initial_objectives": "0=950000,1=950000,2=950000,3=950000,4=950000,5=950000,6=950000",
        "schedule_profile_best_objectives": (
            "0=949000,1=949500,2=950000,3=950000,4=950000,5=950000,6=950000"
            if candidate
            else "0=949000,1=949500,2=950000,3=950000,4=950000,5=950000,6=950000"
        ),
        **metrics,
    }


def multi_candidate_record(*, objective=943_043, seed=0):
    item = record(candidate=True, objective=objective, seed=seed)
    item.update(
        schedule_profile_work_steps="0=521,1=521,2=521,3=521,4=520,5=520,6=520",
        schedule_tsab_owner_worker_mask=0x7F,
        schedule_tsab_activations=7,
        schedule_tsab_legacy_warmup_work_steps=3_584,
        schedule_tsab_activation_rebases=6,
        schedule_tsab_n6_kicks=1,
        schedule_tsab_kick_moves=1,
        schedule_tsab_restart_relink_attempts=1,
        schedule_tsab_restart_relink_commits=1,
        schedule_tsab_restart_relink_components_shortlisted=3,
        schedule_tsab_restart_relink_components_committed=3,
        schedule_tsab_restart_relink_guide_arc_gain_shortlisted=5,
        schedule_tsab_restart_relink_guide_arc_gain_accepted=4,
    )
    return item


def write_campaign(tmp_path, name, item, *, strategy, source="c", artifact="d", path_relink=False):
    path = tmp_path / f"{name}.jsonl"
    path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    provenance = {
        "schema_version": 1,
        "budgets": [60],
        "seeds": [item["seed"]],
        "threads": 7,
        "qayd_engine": "ls",
        "profile_qayd": True,
        "qayd_schedule_jssp_search": strategy,
        "qayd_schedule_constructor_workers": 0,
        "qayd_schedule_path_relink": path_relink,
        "qayd_schedule_lns_shadow": False,
        "qayd_artifact": {"sha256": artifact * 64},
        "host": {"source_tree_sha256": source * 64},
    }
    path.with_suffix(path.suffix + ".provenance.json").write_text(
        json.dumps(provenance) + "\n", encoding="utf-8"
    )
    return path


def write_pair(
    tmp_path,
    *,
    candidate=None,
    baseline=None,
    candidate_strategy="tsab-candidate",
    candidate_path_relink=False,
    **provenance,
):
    baseline = baseline or record(candidate=False, objective=943_228)
    candidate = candidate or record(candidate=True)
    baseline_path = write_campaign(
        tmp_path,
        "baseline",
        baseline,
        strategy="legacy",
        source=provenance.get("baseline_source", "c"),
        artifact=provenance.get("baseline_artifact", "d"),
    )
    candidate_path = write_campaign(
        tmp_path,
        "candidate",
        candidate,
        strategy=candidate_strategy,
        source=provenance.get("candidate_source", "c"),
        artifact=provenance.get("candidate_artifact", "d"),
        path_relink=candidate_path_relink,
    )
    return baseline_path, candidate_path


def test_ready_is_separate_from_quality_signal(tmp_path):
    gate = load_gate()
    baseline, candidate = write_pair(tmp_path)

    report = gate.evaluate(baseline, candidate)

    assert report["tsab_ready"] is True
    assert report["tsab_search_signal"] is True
    assert report["tsab_quality_signal"] is False
    assert report["rows"][0]["minimum_control_work_retention"] == 1.0


def test_multi_candidate_is_accepted_and_reports_relink_accounting(tmp_path):
    gate = load_gate()
    baseline, candidate = write_pair(
        tmp_path,
        candidate=multi_candidate_record(),
        candidate_strategy="tsab-multi-candidate",
        candidate_path_relink=True,
    )

    report = gate.evaluate(baseline, candidate)

    assert report["tsab_ready"] is True
    assert report["candidate_strategy"] == "tsab-multi-candidate"
    assert report["rows"][0]["restart_relink_attempts"] == 1
    assert report["rows"][0]["restart_relink_commits"] == 1
    assert report["rows"][0]["control_work_retention"] == {}
    assert report["rows"][0]["minimum_control_work_retention"] is None
    assert report["rows"][0]["multi_worker_work_balance"] == 520 / 521


def test_existing_mono_artifacts_may_omit_relink_metrics(tmp_path):
    gate = load_gate()
    baseline_record = record(candidate=False, objective=943_228)
    candidate_record = record(candidate=True)
    for field in gate.TSAB_RELINK_COUNTERS:
        baseline_record.pop(field)
        candidate_record.pop(field)
    baseline, candidate = write_pair(tmp_path, baseline=baseline_record, candidate=candidate_record)

    report = gate.evaluate(baseline, candidate)

    assert report["tsab_ready"] is True
    assert report["rows"][0]["restart_relink_attempts"] == 0


def test_multi_candidate_requires_every_relink_metric(tmp_path):
    gate = load_gate()
    candidate_record = multi_candidate_record()
    candidate_record.pop("schedule_tsab_restart_relink_attempts")
    baseline, candidate = write_pair(
        tmp_path,
        candidate=candidate_record,
        candidate_strategy="tsab-multi-candidate",
    )

    with pytest.raises(gate.GateError, match="missing TSAB relink metrics"):
        gate.evaluate(baseline, candidate)


@pytest.mark.parametrize(
    ("field", "value", "failure"),
    (
        ("schedule_tsab_restart_relink_attempts", 0, "restart_relink_commit_accounting"),
        ("schedule_tsab_restart_oracle_commits", 1, "restart_commit_accounting"),
        ("schedule_tsab_restart_relink_components_shortlisted", 1, "restart_relink_shortlist_accounting"),
        ("schedule_tsab_restart_relink_components_shortlisted", 9, "restart_relink_shortlist_accounting"),
        ("schedule_tsab_restart_relink_components_committed", 1, "restart_relink_committed_accounting"),
        ("schedule_tsab_restart_relink_components_committed", 9, "restart_relink_committed_accounting"),
        ("schedule_tsab_restart_relink_components_committed", 0, "restart_relink_committed_presence"),
        ("schedule_tsab_restart_relink_commits", 0, "restart_relink_committed_presence"),
        ("schedule_tsab_restart_relink_guide_arc_gain_accepted", 0, "restart_relink_gain_presence"),
        ("schedule_tsab_restart_relink_commits", 0, "restart_relink_gain_presence"),
        ("schedule_tsab_restart_relink_guide_arc_gain_accepted", 6, "restart_relink_gain_accounting"),
    ),
)
def test_multi_relink_invariants_fail_closed(tmp_path, field, value, failure):
    gate = load_gate()
    candidate_record = multi_candidate_record()
    candidate_record[field] = value
    baseline, candidate = write_pair(
        tmp_path,
        candidate=candidate_record,
        candidate_strategy="tsab-multi-candidate",
    )

    report = gate.evaluate(baseline, candidate)

    assert report["tsab_ready"] is False
    assert failure in report["failures"]


def test_compound_tail_corridor_commit_satisfies_restart_accounting_without_single_probes(tmp_path):
    gate = load_gate()
    candidate_record = record(candidate=True)
    candidate_record["schedule_tsab_restart_delta_probes"] = 0
    candidate_record["schedule_tsab_kick_moves"] = 8
    baseline, candidate = write_pair(tmp_path, candidate=candidate_record)

    report = gate.evaluate(baseline, candidate)

    assert "restart_probe_accounting" not in report["failures"]
    assert "restart_commit_accounting" not in report["failures"]


def test_rejected_tail_corridor_satisfies_restart_probe_accounting_without_single_fallback(tmp_path):
    gate = load_gate()
    candidate_record = record(candidate=True)
    candidate_record["schedule_tsab_restart_delta_probes"] = 0
    candidate_record["schedule_tsab_restart_oracle_commits"] = 0
    candidate_record["schedule_tsab_restart_rejections"] = 2
    candidate_record["schedule_tsab_n6_kicks"] = 0
    candidate_record["schedule_tsab_kick_moves"] = 0
    candidate_record["schedule_tsab_restart_best_kicked_objective"] = None
    baseline, candidate = write_pair(tmp_path, candidate=candidate_record)

    report = gate.evaluate(baseline, candidate)

    assert "restart_probe_accounting" not in report["failures"]
    assert "restart_outcome_accounting" not in report["failures"]


@pytest.mark.parametrize("kick_moves", [1, 9])
def test_compound_tail_corridor_component_bounds_fail_closed(tmp_path, kick_moves):
    gate = load_gate()
    candidate_record = record(candidate=True)
    candidate_record["schedule_tsab_kick_moves"] = kick_moves
    baseline, candidate = write_pair(tmp_path, candidate=candidate_record)

    report = gate.evaluate(baseline, candidate)

    assert "restart_commit_accounting" in report["failures"]


def test_fast_path_does_not_require_legacy_k4_probes_or_oracle_commits(tmp_path):
    gate = load_gate()
    candidate_record = record(candidate=True)
    assert candidate_record["schedule_tsab_delta_probes"] == 0
    assert candidate_record["schedule_tsab_full_oracle_commits"] == 0
    baseline, candidate = write_pair(tmp_path, candidate=candidate_record)

    report = gate.evaluate(baseline, candidate)

    assert report["tsab_ready"] is True


def test_fast_fallbacks_are_reported_but_not_a_readiness_failure(tmp_path):
    gate = load_gate()
    candidate_record = record(candidate=True)
    candidate_record["schedule_tsab_fast_fallbacks"] = candidate_record["schedule_tsab_fast_attempts"]
    baseline, candidate = write_pair(tmp_path, candidate=candidate_record)

    report = gate.evaluate(baseline, candidate)

    assert report["tsab_ready"] is True


def test_fast_work_cap_recovery_rate_is_diagnostic_not_blocking(tmp_path):
    gate = load_gate()
    candidate_record = record(candidate=True)
    candidate_record["schedule_tsab_fast_work_cap_recoveries"] = candidate_record["schedule_tsab_fast_attempts"]
    baseline, candidate = write_pair(tmp_path, candidate=candidate_record)

    report = gate.evaluate(baseline, candidate)

    assert report["tsab_ready"] is True


def test_legacy_arm_must_keep_every_fast_metric_zero(tmp_path):
    gate = load_gate()
    baseline_record = record(candidate=False, objective=943_228)
    baseline_record["schedule_tsab_fast_work_cap_recoveries"] = 1
    baseline, candidate = write_pair(tmp_path, baseline=baseline_record)

    report = gate.evaluate(baseline, candidate)

    assert report["tsab_ready"] is False
    assert "baseline_tsab_activity" in report["failures"]


def test_quality_signal_requires_ready_objective_at_or_below_limit(tmp_path):
    gate = load_gate()
    baseline, candidate = write_pair(tmp_path, candidate=record(candidate=True, objective=936_343))

    report = gate.evaluate(baseline, candidate)

    assert report["tsab_ready"] is True
    assert report["tsab_quality_signal"] is True


def test_quality_signal_also_requires_no_regression_against_fresh_legacy(tmp_path):
    gate = load_gate()
    baseline, candidate = write_pair(
        tmp_path,
        baseline=record(candidate=False, objective=930_000),
        candidate=record(candidate=True, objective=936_343),
    )

    report = gate.evaluate(baseline, candidate)

    assert report["tsab_ready"] is True
    assert report["tsab_search_signal"] is False
    assert report["tsab_quality_signal"] is False


@pytest.mark.parametrize(
    ("field", "value", "failure"),
    [
        ("schedule_tsab_owner_worker_mask", 0x1F, "owner_mask"),
        ("schedule_tsab_ranked", 99, "shortlist_activity"),
        ("schedule_tsab_aspirations", 1, "aspiration_accounting"),
        ("schedule_tsab_tabu_resets", 61, "tabu_reset_accounting"),
        ("schedule_tsab_n1_kicks", 1, "deferred_tsab_activity"),
        ("schedule_tsab_activation_boundary", 3, "phased_activation"),
        ("schedule_tsab_legacy_warmup_work_steps", 511, "phased_activation"),
        ("schedule_tsab_activation_rebases", 0, "phased_activation"),
        ("schedule_tsab_burst_work_units", 129, "burst_accounting"),
        ("schedule_tsab_restart_attempts", 0, "restart_activity"),
        ("schedule_tsab_restart_global_rebases", 3, "restart_global_rebase"),
        ("schedule_tsab_restart_delta_probes", 9, "restart_probe_accounting"),
        ("schedule_tsab_n6_kicks", 1, "restart_commit_accounting"),
        ("schedule_tsab_restart_rejections", 1, "restart_outcome_accounting"),
        ("schedule_tsab_restart_interruptions", 1, "restart_interruption"),
        ("schedule_tsab_restart_work_units", 61, "restart_work_accounting"),
        ("schedule_tsab_fast_enabled", 0, "fast_activity"),
        ("schedule_tsab_fast_eligible", 0, "fast_activity"),
        ("schedule_tsab_fast_attempts", 0, "fast_activity"),
        ("schedule_tsab_fast_transitions", 0, "fast_activity"),
        ("schedule_tsab_fast_work_units", 0, "fast_activity"),
        ("schedule_tsab_fast_full_validations", 0, "fast_activity"),
        ("schedule_tsab_fast_disabled", 1, "fast_disabled"),
        ("schedule_tsab_fast_oracle_mismatches", 1, "fast_oracle_mismatch"),
        ("schedule_tsab_fast_pending_discards", 1, "fast_pending_discard"),
        ("schedule_tsab_fast_commits", 47, "fast_transition_accounting"),
        ("schedule_tsab_fast_fallbacks", 61, "fast_fallback_accounting"),
        ("schedule_tsab_fast_work_cap_recoveries", 0, "fast_work_cap_recovery"),
        ("schedule_tsab_fast_date_changes", 601, "fast_queue_accounting"),
        ("schedule_tsab_fast_pending_promotions", 3, "fast_validation_accounting"),
        ("schedule_tsab_fast_workspace_peak_bytes", 0, "fast_workspace"),
    ],
)
def test_candidate_accounting_is_fail_closed(tmp_path, field, value, failure):
    gate = load_gate()
    candidate_record = record(candidate=True)
    candidate_record[field] = value
    baseline, candidate = write_pair(tmp_path, candidate=candidate_record)

    report = gate.evaluate(baseline, candidate)

    assert report["tsab_ready"] is False
    assert failure in report["failures"]


def test_restart_without_an_admissible_kick_is_ready_after_a_global_rebase(tmp_path):
    gate = load_gate()
    candidate_record = record(candidate=True, objective=943_039)
    candidate_record.update(
        schedule_tsab_restart_oracle_commits=0,
        schedule_tsab_restart_rejections=2,
        schedule_tsab_n6_kicks=0,
        schedule_tsab_kick_moves=0,
        schedule_tsab_restart_best_kicked_objective=None,
    )
    baseline, candidate = write_pair(
        tmp_path,
        baseline=record(candidate=False, objective=943_065),
        candidate=candidate_record,
    )

    report = gate.evaluate(baseline, candidate)

    assert report["tsab_ready"] is True
    assert report["tsab_search_signal"] is True


def test_restart_without_a_new_global_rebase_remains_ready(tmp_path):
    gate = load_gate()
    candidate_record = record(candidate=True)
    candidate_record["schedule_tsab_restart_global_rebases"] = 0
    baseline, candidate = write_pair(tmp_path, candidate=candidate_record)

    report = gate.evaluate(baseline, candidate)

    assert report["tsab_ready"] is True
    assert "restart_global_rebase" not in report["failures"]


def test_all_six_legacy_controls_must_retain_ninety_percent_of_fresh_work(tmp_path):
    gate = load_gate()
    candidate_record = record(candidate=True)
    candidate_record["schedule_profile_work_steps"] = "0=100,1=100,2=100,3=100,4=100,5=89,6=100"
    baseline, candidate = write_pair(tmp_path, candidate=candidate_record)

    report = gate.evaluate(baseline, candidate)

    assert report["tsab_ready"] is False
    assert "control_work_retention" in report["failures"]


def test_worker_six_is_not_part_of_control_work_retention(tmp_path):
    gate = load_gate()
    candidate_record = record(candidate=True)
    candidate_record["schedule_profile_work_steps"] = "0=100,1=100,2=100,3=100,4=100,5=100,6=1"
    baseline, candidate = write_pair(tmp_path, candidate=candidate_record)

    report = gate.evaluate(baseline, candidate)

    assert report["tsab_ready"] is True


def test_multi_candidate_requires_active_work_for_every_profile(tmp_path):
    gate = load_gate()
    candidate_record = multi_candidate_record()
    candidate_record["schedule_profile_work_steps"] = "0=512,1=530,2=521,3=521,4=520,5=520,6=520"
    baseline, candidate = write_pair(
        tmp_path,
        candidate=candidate_record,
        candidate_strategy="tsab-multi-candidate",
    )

    report = gate.evaluate(baseline, candidate)

    assert report["tsab_ready"] is False
    assert "multi_profile_activity" in report["failures"]
    assert "multi_profile_work_accounting" not in report["failures"]


def test_multi_candidate_profile_work_must_equal_warmup_plus_burst(tmp_path):
    gate = load_gate()
    candidate_record = multi_candidate_record()
    candidate_record["schedule_profile_work_steps"] = "0=522,1=521,2=521,3=521,4=520,5=520,6=520"
    baseline, candidate = write_pair(
        tmp_path,
        candidate=candidate_record,
        candidate_strategy="tsab-multi-candidate",
    )

    report = gate.evaluate(baseline, candidate)

    assert report["tsab_ready"] is False
    assert "multi_profile_work_accounting" in report["failures"]


def test_search_signal_is_separate_from_readiness(tmp_path):
    gate = load_gate()
    candidate_record = record(candidate=True)
    candidate_record["schedule_tsab_post_restart_improvements"] = 0
    candidate_record["schedule_tsab_fast_pending_promotions"] = 0
    baseline, candidate = write_pair(tmp_path, candidate=candidate_record)

    report = gate.evaluate(baseline, candidate)

    assert report["tsab_ready"] is True
    assert report["tsab_search_signal"] is False
    assert report["tsab_quality_signal"] is False


def test_improving_commit_metadata_is_fail_closed(tmp_path):
    gate = load_gate()
    candidate_record = record(candidate=True)
    candidate_record["schedule_tsab_best_committed_objective"] = None
    baseline, candidate = write_pair(tmp_path, candidate=candidate_record)

    report = gate.evaluate(baseline, candidate)

    assert report["tsab_ready"] is False
    assert "improving_commit_accounting" in report["failures"]


def test_verification_timing_is_fail_closed(tmp_path):
    gate = load_gate()
    candidate_record = record(candidate=True)
    candidate_record["schedule_incumbent_verification_max_seconds"] = 0.06
    baseline, candidate = write_pair(tmp_path, candidate=candidate_record)

    report = gate.evaluate(baseline, candidate)

    assert report["tsab_ready"] is False
    assert "verification_timing" in report["failures"]


@pytest.mark.parametrize(
    "provenance",
    [
        {"candidate_source": "e"},
        {"candidate_artifact": "e"},
    ],
)
def test_pair_must_use_identical_source_and_artifact(tmp_path, provenance):
    gate = load_gate()
    baseline, candidate = write_pair(tmp_path, **provenance)

    with pytest.raises(gate.GateError, match="provenance mismatch"):
        gate.evaluate(baseline, candidate)


def test_fresh_baseline_must_expose_phased_tsab_schema(tmp_path):
    gate = load_gate()
    baseline_record = record(candidate=False, objective=943_228)
    del baseline_record["schedule_tsab_activation_boundary"]
    baseline, candidate = write_pair(tmp_path, baseline=baseline_record)

    with pytest.raises(gate.GateError, match="missing TSAB metrics"):
        gate.evaluate(baseline, candidate)


def test_candidate_provenance_requires_all_other_experiments_off(tmp_path):
    gate = load_gate()
    baseline, candidate = write_pair(tmp_path)
    provenance_path = gate.provenance_path(candidate)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["qayd_schedule_path_relink"] = True
    provenance_path.write_text(json.dumps(provenance) + "\n", encoding="utf-8")

    with pytest.raises(gate.GateError, match="qayd_schedule_path_relink"):
        gate.evaluate(baseline, candidate)


def test_records_must_be_seven_thread_jssp_local_search(tmp_path):
    gate = load_gate()
    candidate_record = record(candidate=True)
    candidate_record["threads"] = 6
    baseline, candidate = write_pair(tmp_path, candidate=candidate_record)

    report = gate.evaluate(baseline, candidate)

    assert report["tsab_ready"] is False
    assert "replay" in report["failures"]


def test_gate_cli_can_require_quality_separately(tmp_path):
    baseline, candidate = write_pair(tmp_path)

    readiness = subprocess.run(
        [sys.executable, str(SCRIPT), "--baseline", str(baseline), "--candidate", str(candidate), "--require-readiness"],
        check=False,
        capture_output=True,
        text=True,
    )
    quality = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--require-quality-signal",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    search = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--require-search-signal",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert readiness.returncode == 0
    assert search.returncode == 0
    assert quality.returncode == 2
