"""Strict acceptance-gate checks for the large-scale routing search."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[2] / "bench" / "routing_search_report.py"


def load_module():
    spec = importlib.util.spec_from_file_location("qayd_routing_search_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def profile(report, *, productive=True):
    uses = {
        "relocate": 7,
        "swap": 5,
        "or-opt": 5,
        "two-opt-star": 5,
        "cross-exchange": 5,
        "reverse": 5,
        "route-elimination": 4,
        "ejection-chain": 4,
        "chain-relocate": 4,
        "guided-segment-exchange": 4,
        "path-relink": 2,
        "elite-archive": 5,
        "elite-selection": 2,
        "alns_destroy/shaw": 8,
        "alns_destroy/segments": 8,
        "alns_destroy/worst": 7,
        "alns_destroy/route": 7,
        "alns_repair/greedy": 8,
        "alns_repair/regret-2": 8,
        "alns_repair/regret-3": 7,
        "alns_repair/blink-regret-2": 7,
    }
    return [
        {
            "name": name,
            "uses": uses[name],
            "generated": uses[name] * 10,
            "evaluated": uses[name] * 8,
            "cpu_nanos": 1_000_000 if not productive and name == "relocate" else 1_000,
            "improvements": 1 if productive else 0,
            "global_bests": 1 if productive else 0,
            "positive_rewards": 1 if productive else 0,
            "weight": 1.0,
        }
        for name in sorted(report.REQUIRED_PROFILE_NAMES)
    ]


def routing_counters():
    return {
        "slices": 80,
        "descent_slices": 30,
        "alns_slices": 30,
        "relink_slices": 2,
        "global_scan_slices": 2,
        "route_elimination_attempts": 4,
        "ejection_chain_attempts": 4,
        "chain_relocate_attempts": 4,
        "guided_segment_exchange_attempts": 4,
        "macro_candidates_built": 4,
        "macro_budget_exhaustions": 8,
        "elite_insertions": 3,
        "elite_rejections": 2,
        "path_relink_attempts": 2,
        "path_relink_steps": 2,
        "path_relink_budget_exhaustions": 0,
    }


def write_inputs(report, tmp_path, *, lag_1s=10.0, productive=True):
    campaign = tmp_path / "campaign.jsonl"
    versions = {"qayd-api": "qayd test", "hgs": "hgs test"}
    records = []
    for instance, spec in report.TARGETS.items():
        best_known = spec["best_known"]
        objective = round(best_known * (1.05 if instance != "X-n1001-k43.vrp" else 1.08))
        for seed in report.REQUIRED_SEED_IDS:
            for solver in ("qayd-api", "hgs"):
                value = objective + seed if solver == "qayd-api" else best_known
                checkpoints = [
                    {
                        "target_nanos": target,
                        "observed_nanos": target + 10,
                        "feasible": True,
                        "objectives": [int(value)],
                        "fleet": 10,
                        "candidates": (index + 1) * 10,
                    }
                    for index, target in enumerate((250_000_000, 1_000_000_000, 5_000_000_000))
                ]
                records.append({
                    "run_id": f"{solver}-{instance}-{seed}",
                    "solver": solver,
                    "solver_version": versions[solver],
                    "problem": "cvrp",
                    "instance_path": spec["path"],
                    "instance_sha256": spec["sha256"],
                    "checkpoint_seconds": 10,
                    "seed": seed,
                    "requested_threads": 1,
                    "threads": 1,
                    "status": "SATISFIABLE",
                    "verified": True,
                    "objectives": [value],
                    "best_known": best_known,
                    "elapsed_seconds": 10.0,
                    "peak_memory_mb": 80.0,
                    "anytime_checkpoints": checkpoints if solver == "qayd-api" else None,
                    "neighborhood_profile": profile(report, productive=productive) if solver == "qayd-api" else None,
                    "routing_counters": routing_counters() if solver == "qayd-api" else None,
                })
    campaign.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    positioning_provenance = {
        "schema_version": 1,
        "suite": {"name": "positioning"},
        "solvers": versions,
        "budgets": [10],
        "seeds": list(report.REQUIRED_SEED_IDS),
        "threads": 1,
        "qayd_engine": "ls",
        "profile_qayd": True,
        "qayd_prepared": True,
        "qayd_artifact": {
            "path": "/tmp/qayd-test/_core.abi3.so",
            "sha256": "c" * 64,
        },
        "host": {
            "commit": "a" * 40,
            "dirty": True,
            "source_tree_sha256": "b" * 64,
        },
    }
    campaign.with_suffix(".jsonl.provenance.json").write_text(json.dumps(positioning_provenance), encoding="utf-8")

    probes = []
    factors = (
        "distance-scale", "index-permutation", "single-edge", "capacity-threshold",
    )
    for index in range(8):
        factor = factors[index // 2]
        path = tmp_path / f"probe-{index}.vrp"
        path.write_text(f"probe {index}\n", encoding="utf-8")
        probes.append({
            "probe_id": f"p{index}",
            "pair_id": f"pair{index // 2}",
            "factor": factor,
            "role": "control" if index % 2 == 0 else "treatment",
            "size": 200,
            "instance_path": str(path),
            "instance_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    pairs = []
    for probe in probes:
        for budget in (1, 5):
            for seed in report.REQUIRED_SEED_IDS:
                reference_objective = 100.0
                lag = lag_1s if budget == 1 else 3.0
                solver_objective = reference_objective * (1.0 + lag / 100.0)
                stem = f"{probe['probe_id']}-t{budget}-s{seed}"
                pairs.append({
                    "solver": "qayd-api",
                    "reference": "hexaly",
                    "probe_id": probe["probe_id"],
                    "pair_id": probe["pair_id"],
                    "factor": probe["factor"],
                    "role": probe["role"],
                    "size": 200,
                    "instance_path": probe["instance_path"],
                    "instance_sha256": probe["instance_sha256"],
                    "checkpoint_seconds": budget,
                    "seed": seed,
                    "campaign_threads": 1,
                    "solver_run_id": f"qayd-{stem}",
                    "solver_version": versions["qayd-api"],
                    "solver_instance_path": probe["instance_path"],
                    "solver_instance_sha256": probe["instance_sha256"],
                    "solver_requested_threads": 1,
                    "solver_threads": 1,
                    "solver_status": "SATISFIABLE",
                    "solver_verified": True,
                    "solver_normalized_objective": solver_objective,
                    "reference_run_id": f"hexaly-{stem}",
                    "reference_version": "hexaly behavior test",
                    "reference_instance_path": probe["instance_path"],
                    "reference_instance_sha256": probe["instance_sha256"],
                    "reference_requested_threads": 1,
                    "reference_threads": 1,
                    "reference_status": "SATISFIABLE",
                    "reference_verified": True,
                    "reference_normalized_objective": reference_objective,
                    "objective_lag_percent": lag,
                    "missing_reason": None,
                })
    behavior = tmp_path / "behavior.json"
    behavior.write_text(json.dumps({
        "schema_version": 2,
        "method": "clean-room-black-box",
        "generator": dict(report.REQUIRED_BEHAVIOR_GENERATOR),
        "campaign": {
            "solvers": ["qayd-api", "hexaly"],
            "budgets": [1, 5],
            "seeds": list(report.REQUIRED_SEED_IDS),
            "threads": 1,
        },
        "campaign_provenance": {
            **positioning_provenance,
            "solvers": {
                "qayd-api": versions["qayd-api"],
                "hexaly": "hexaly behavior test",
            },
            "budgets": [1, 5],
        },
        "probes": probes,
        "solver_lag_pairs": pairs,
        "solver_lag_rows": [
            {
                "solver": "qayd-api", "reference": "hexaly", "size": 200,
                "checkpoint_seconds": budget, "expected_pairs": 40,
                "verified_pairs": 40, "missing_pairs": 0,
                "median_objective_lag_percent": lag_1s if budget == 1 else 3.0,
            }
            for budget in (1, 5)
        ],
    }), encoding="utf-8")
    return campaign, behavior


def rewrite_jsonl(path, mutate):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mutate(records)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_complete_campaign_passes_gate(tmp_path):
    report = load_module()
    campaign, behavior = write_inputs(report, tmp_path)

    summary = report.build_summary(campaign, behavior)

    assert summary["ready"], summary["errors"]
    assert summary["profile"]["missing_profiles"] == 0
    assert summary["profile"]["missing_required_checkpoints"] == 0
    assert summary["profile"]["missing_or_inconsistent_counters"] == 0


def test_lag_and_unproductive_cpu_dominance_block_gate(tmp_path):
    report = load_module()
    campaign, behavior = write_inputs(report, tmp_path, lag_1s=25.0, productive=False)

    summary = report.build_summary(campaign, behavior)

    assert not summary["ready"]
    assert any("median lag" in error for error in summary["errors"])
    assert any("above 70% CPU" in error for error in summary["errors"])


def test_unproductive_elite_overhead_is_included_in_cpu_guard(tmp_path):
    report = load_module()
    campaign, behavior = write_inputs(report, tmp_path)

    def corrupt(records):
        target = next(record for record in records if record["solver"] == "qayd-api")
        archive = next(entry for entry in target["neighborhood_profile"] if entry["name"] == "elite-archive")
        archive["cpu_nanos"] = 1_000_000
        archive["improvements"] = 0
        archive["global_bests"] = 0
        archive["positive_rewards"] = 0

    rewrite_jsonl(campaign, corrupt)
    summary = report.build_summary(campaign, behavior)

    assert not summary["ready"]
    assert any("elite-archive" in entry for entry in summary["profile"]["unproductive_dominant_runs"])


def test_seed_thread_path_hash_and_bks_provenance_are_mandatory(tmp_path):
    report = load_module()
    campaign, behavior = write_inputs(report, tmp_path)

    def corrupt(records):
        target = next(record for record in records if record["solver"] == "qayd-api" and record["seed"] == 4)
        target["threads"] = 8
        target["instance_sha256"] = "0" * 64
        target["best_known"] = target["objectives"][0]

    rewrite_jsonl(campaign, corrupt)
    summary = report.build_summary(campaign, behavior)

    assert not summary["ready"]
    assert any("canonical provenance" in error for error in summary["errors"])


def test_duplicate_or_disjoint_complete_seeds_cannot_pass(tmp_path):
    report = load_module()
    campaign, behavior = write_inputs(report, tmp_path)

    def corrupt(records):
        first = next(record for record in records if record["solver"] == "qayd-api" and record["seed"] == 0)
        for record in records:
            if record["solver"] == "qayd-api" and record["instance_path"] == first["instance_path"] and record["seed"] != 0:
                record["verified"] = False
                record["status"] = "UNKNOWN"
                record["objectives"] = []
        records.extend(dict(first) for _ in range(4))

    rewrite_jsonl(campaign, corrupt)
    summary = report.build_summary(campaign, behavior)

    row = summary["qayd_targets"][0]
    assert row["verified_seed_ids"] == [0]
    assert not summary["ready"]


def test_profiles_counters_and_timeline_must_be_causal(tmp_path):
    report = load_module()
    campaign, behavior = write_inputs(report, tmp_path)

    def corrupt(records):
        target = next(record for record in records if record["solver"] == "qayd-api")
        target["neighborhood_profile"] = [{"name": "fake"}]
        target["routing_counters"]["alns_slices"] += 1
        checkpoints = target["anytime_checkpoints"]
        checkpoints[1]["candidates"] = 0
        checkpoints[2]["objectives"] = [target["objectives"][0] + 100]

    rewrite_jsonl(campaign, corrupt)
    summary = report.build_summary(campaign, behavior)

    assert not summary["ready"]
    assert summary["profile"]["missing_profiles"] == 1
    assert summary["profile"]["missing_required_checkpoints"] == 1
    assert summary["profile"]["missing_or_inconsistent_counters"] == 1


def test_profile_cannot_dilute_cpu_or_misreport_alns_slices(tmp_path):
    report = load_module()
    campaign, behavior = write_inputs(report, tmp_path)

    def corrupt(records):
        targets = [record for record in records if record["solver"] == "qayd-api"]
        targets[0]["neighborhood_profile"].append({
            "name": "padding",
            "uses": 1,
            "generated": 1,
            "evaluated": 1,
            "cpu_nanos": 1_000_000,
            "improvements": 1,
            "global_bests": 0,
            "positive_rewards": 1,
            "weight": 1.0,
        })
        targets[1]["neighborhood_profile"][0]["uses"] += 1

    rewrite_jsonl(campaign, corrupt)
    summary = report.build_summary(campaign, behavior)

    assert not summary["ready"]
    assert summary["profile"]["missing_profiles"] == 1
    assert summary["profile"]["missing_or_inconsistent_counters"] == 2


def test_selected_but_unevaluated_routing_operator_blocks_gate(tmp_path):
    report = load_module()
    campaign, behavior = write_inputs(report, tmp_path)

    def corrupt(records):
        for record in records:
            if record["solver"] != "qayd-api":
                continue
            operator = next(
                entry
                for entry in record["neighborhood_profile"]
                if entry["name"] == "guided-segment-exchange"
            )
            operator["generated"] = operator["uses"] * 10
            operator["evaluated"] = 0

    rewrite_jsonl(campaign, corrupt)
    summary = report.build_summary(campaign, behavior)

    assert not summary["ready"]
    assert summary["profile"]["unevaluated_required_operators"] == [
        "guided-segment-exchange"
    ]
    assert any("attempted no candidate evaluation" in error for error in summary["errors"])


def test_negative_rss_and_unpinned_behavior_generator_block_gate(tmp_path):
    report = load_module()
    campaign, behavior = write_inputs(report, tmp_path)

    def corrupt(records):
        target = next(record for record in records if record["solver"] == "qayd-api")
        target["peak_memory_mb"] = -1.0

    rewrite_jsonl(campaign, corrupt)
    document = json.loads(behavior.read_text(encoding="utf-8"))
    document["generator"]["instance_seed"] += 1
    behavior.write_text(json.dumps(document), encoding="utf-8")

    summary = report.build_summary(campaign, behavior)

    assert not summary["ready"]
    assert any("canonical provenance, BKS and RSS" in error for error in summary["errors"])
    assert any("generator parameters" in error for error in summary["errors"])


def test_source_artifact_and_cross_campaign_build_are_mandatory(tmp_path):
    report = load_module()
    campaign, behavior = write_inputs(report, tmp_path)
    provenance = campaign.with_suffix(".jsonl.provenance.json")
    sidecar = json.loads(provenance.read_text(encoding="utf-8"))
    sidecar["qayd_artifact"]["sha256"] = None
    provenance.write_text(json.dumps(sidecar), encoding="utf-8")
    document = json.loads(behavior.read_text(encoding="utf-8"))
    for pair in document["solver_lag_pairs"]:
        pair["solver_version"] = "different qayd build"
    behavior.write_text(json.dumps(document), encoding="utf-8")

    summary = report.build_summary(campaign, behavior)

    assert not summary["ready"]
    assert any("extension artifact" in error for error in summary["errors"])
    assert any("different qayd source trees" in error for error in summary["errors"])


def test_behavior_aggregates_without_exact_pair_evidence_cannot_pass(tmp_path):
    report = load_module()
    campaign, behavior = write_inputs(report, tmp_path)
    document = json.loads(behavior.read_text(encoding="utf-8"))
    document["solver_lag_pairs"].pop()
    behavior.write_text(json.dumps(document), encoding="utf-8")

    summary = report.build_summary(campaign, behavior)

    assert not summary["ready"]
    assert any("exact expected" in error or "underlying pair matrix" in error for error in summary["errors"])


def test_qayd_tail_and_hgs_reference_regressions_block_gate(tmp_path):
    report = load_module()
    campaign, behavior = write_inputs(report, tmp_path)

    def corrupt(records):
        for record in records:
            if record["instance_path"].endswith("X-n401-k29.vrp") and record["seed"] in {3, 4}:
                if record["solver"] == "qayd-api":
                    record["objectives"] = [record["best_known"] * 2]
                else:
                    record["objectives"] = [record["best_known"] * 1.2]

    rewrite_jsonl(campaign, corrupt)
    summary = report.build_summary(campaign, behavior)

    assert not summary["ready"]
    assert any("tail gap" in error for error in summary["errors"])
    assert any("HGS gap" in error for error in summary["errors"])
