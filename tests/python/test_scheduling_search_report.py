"""Strict Phase 13 scheduling-search qualification checks."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "bench" / "scheduling_search_report.py"


def load_module():
    spec = importlib.util.spec_from_file_location("qayd_scheduling_search_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def instance_hash(problem, instance):
    return hashlib.sha256(f"{problem}/{instance}".encode()).hexdigest()


def record(problem, instance, seed, objective, engine, *, best_known=None):
    value = {
        "run_id": f"{problem}-{engine}-{instance}-{seed}",
        "solver": "qayd-api",
        "solver_version": "qayd test",
        "problem": problem,
        "family": f"positioning-{problem}",
        "engine": engine,
        "instance": instance,
        "instance_path": f"fixtures/{problem}/{instance}",
        "instance_sha256": instance_hash(problem, instance),
        "checkpoint_seconds": 10,
        "seed": seed,
        "status": "SATISFIABLE",
        "verified": True,
        "objectives": [objective],
        "best_known": objective if best_known is None else best_known,
        "peak_memory_mb": 32.0,
        "threads": 1,
        "requested_threads": 1,
        "return_code": 0,
        "timed_out": False,
    }
    if engine == "ls":
        value.update({
            "constructor": (
                "giffler-thompson-critical-path"
                if problem == "jssp"
                else "resource-priority-sgs"
            ),
            "construction_candidates": 12,
            "candidates_evaluated": 10,
            "candidates_per_second": 1_000.0,
            "full_recompute_percentage": 10.0,
            "schedule_moves_considered": 10,
            "schedule_moves_accepted": 2,
            "schedule_moves_rejected": 8,
            "schedule_work_steps": 12 if problem == "rcpsp" else 10,
            "schedule_delta_evaluations": 10,
            "schedule_full_evaluations": 2,
            "schedule_full_fallbacks": 1,
            "schedule_oracle_validations": 2,
            "schedule_oracle_mismatches": 0,
            "schedule_workspace_growths": 1,
            "schedule_workspace_rollbacks": 3 if problem == "rcpsp" else 0,
            "schedule_alns_generation_attempts": 4 if problem == "rcpsp" else 0,
            "schedule_alns_moves_generated": 2 if problem == "rcpsp" else 0,
            "resource_candidate_scheduling_attempts": 20 if problem == "rcpsp" else 0,
            "resource_profile_checks": 18 if problem == "rcpsp" else 0,
            "resource_event_visits": 30 if problem == "rcpsp" else 0,
            "resource_peak_profile_events": 6 if problem == "rcpsp" else 0,
        })
    return value


def write_campaign(path, records, engine, seeds, *, artifact_hash="c" * 64):
    path.write_text("".join(json.dumps(value) + "\n" for value in records), encoding="utf-8")
    provenance = {
        "schema_version": 1,
        "budgets": [10],
        "seeds": list(seeds),
        "threads": 1,
        "memory_limit_mb": 256,
        "max_iterations": 512,
        "profile_qayd": True,
        "qayd_engine": engine,
        "solvers": {"qayd-api": "qayd test"},
        "host": {
            "commit": "a" * 40,
            "dirty": False,
            "source_tree_sha256": "b" * 64,
        },
        "qayd_artifact": {
            "path": "/tmp/qayd/_core.abi3.so",
            "sha256": artifact_hash,
        },
    }
    path.with_suffix(path.suffix + ".provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )


def write_best_known(path, values):
    path.write_text(
        json.dumps({
            "schema_version": 1,
            "problem": "rcpsp",
            "best_known": values,
        }),
        encoding="utf-8",
    )


def write_reference_campaign(path, records, seeds, solver="ortools-cp-sat"):
    path.write_text("".join(json.dumps(value) + "\n" for value in records), encoding="utf-8")
    provenance = {
        "schema_version": 1,
        "budgets": [10],
        "seeds": list(seeds),
        "threads": 1,
        "memory_limit_mb": 256,
        "solvers": {solver: f"{solver} test"},
    }
    path.with_suffix(path.suffix + ".provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )


def campaign_inputs(tmp_path, *, baseline=False, complete_jssp=False):
    jssp_instances = (
        ("ft06", "orb10", "la38", "swv10", "ta40", "ta80")
        if complete_jssp
        else ("ft06", "orb10")
    )
    rcpsp_instances = ("r1",)
    seeds = (0, 1)
    paths = {
        "jssp_auto": tmp_path / "jssp-auto.jsonl",
        "jssp_ls": tmp_path / "jssp-ls.jsonl",
        "rcpsp_auto": tmp_path / "rcpsp-auto.jsonl",
        "rcpsp_ls": tmp_path / "rcpsp-ls.jsonl",
    }
    best_known_path = tmp_path / "rcpsp-best-known.json"
    write_best_known(best_known_path, {instance: 100 for instance in rcpsp_instances})
    jssp_values = {
        "ft06": (55, 55),
        "orb10": (1_000, 1_020),
        "la38": (110, 110),
        "swv10": (110, 110),
        "ta40": (110, 110),
        "ta80": (110, 110),
    }
    for arm in ("auto", "ls"):
        records = [
            record("jssp", instance, seed, jssp_values[instance][seed], arm)
            for instance in jssp_instances
            for seed in seeds
        ]
        write_campaign(paths[f"jssp_{arm}"], records, arm, seeds)
        records = [record("rcpsp", instance, seed, 100, arm, best_known=100) for instance in rcpsp_instances for seed in seeds]
        write_campaign(paths[f"rcpsp_{arm}"], records, arm, seeds)

    baseline_paths = []
    if baseline:
        path = tmp_path / "baseline.jsonl"
        records = [
            record("jssp", instance, seed, jssp_values[instance][seed], "ls")
            for instance in jssp_instances
            for seed in seeds
        ] + [
            record("rcpsp", instance, seed, 100, "ls", best_known=100)
            for instance in rcpsp_instances
            for seed in seeds
        ]
        write_campaign(path, records, "ls", seeds, artifact_hash="d" * 64)
        baseline_paths.append(path)
    return paths, baseline_paths, jssp_instances, rcpsp_instances, seeds, best_known_path


def build(report, fixture, **kwargs):
    paths, baseline, jssp_instances, rcpsp_instances, seeds, best_known_path = fixture
    return report.build_summary(
        paths["jssp_auto"],
        paths["jssp_ls"],
        paths["rcpsp_auto"],
        paths["rcpsp_ls"],
        rcpsp_best_known_path=best_known_path,
        baseline_paths=baseline,
        jssp_instances=jssp_instances,
        rcpsp_instances=rcpsp_instances,
        seeds=seeds,
        **kwargs,
    )


def complete_inputs(tmp_path):
    fixture = campaign_inputs(tmp_path, complete_jssp=True)
    seeds = fixture[4]
    reference_path = tmp_path / "jssp-reference.jsonl"
    reference_records = []
    for instance in ("la38", "swv10", "ta40", "ta80"):
        for seed in seeds:
            value = record("jssp", instance, seed, 100, "reference")
            value["solver"] = "ortools-cp-sat"
            value["solver_version"] = "OR-Tools test"
            value["run_id"] = f"ortools-{instance}-{seed}"
            reference_records.append(value)
    write_reference_campaign(reference_path, reference_records, seeds)

    scale_paths = {}
    for group in ("j60", "j90", "j120", "mmlib"):
        path = tmp_path / f"rcpsp-{group}.jsonl"
        records = [
            record("rcpsp", f"{group}-case", seed, 200, "auto", best_known=200)
            for seed in seeds
        ]
        write_campaign(path, records, "auto", seeds)
        scale_paths[group] = path
    return fixture, reference_path, scale_paths


def rewrite(path, mutate):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mutate(records)
    path.write_text("".join(json.dumps(value) + "\n" for value in records), encoding="utf-8")


def test_complete_gate_passes_and_references_source_artifacts(tmp_path):
    report = load_module()
    summary = build(report, campaign_inputs(tmp_path, baseline=True))

    assert summary["accepted"], summary["errors"]
    assert len(summary["source_artifacts"]) == 5
    assert summary["provenance"]["current_build_fingerprint"] == ["a" * 40, "b" * 64, "c" * 64]
    assert summary["quality"]["ft06"]["pass"]
    assert summary["quality"]["orb10"]["median_objective"] == 1_010
    assert summary["quality"]["rcpsp_auto_bks"]["matches"] == 2
    assert summary["quality"]["rcpsp_ls_bks"]["matches"] == 2
    assert summary["regression"]["pairs"] == 6
    assert "All acceptance checks passed" in report.markdown(summary)


def test_missing_unexpected_and_duplicate_matrix_keys_fail(tmp_path):
    report = load_module()
    fixture = campaign_inputs(tmp_path)
    path = fixture[0]["jssp_auto"]

    def damage(records):
        records.pop(0)
        duplicate = dict(records[0])
        records.append(duplicate)
        extra = dict(records[0], instance="extra", instance_path="fixtures/jssp/extra")
        extra["instance_sha256"] = instance_hash("jssp", "extra")
        records.append(extra)

    rewrite(path, damage)
    summary = build(report, fixture)
    matrix = summary["matrices"]["jssp-auto"]

    assert not summary["accepted"]
    assert len(matrix["missing"]) == 1
    assert len(matrix["unexpected"]) == 1
    assert len(matrix["duplicates"]) == 1


@pytest.mark.parametrize(
    ("change", "diagnostic"),
    [
        ({"status": "ERROR", "error": "boom"}, "record carries an error"),
        ({"verified": False}, "verified is not true"),
        ({"timed_out": True}, "timed_out is true"),
        ({"objectives": []}, "missing or non-finite scalar incumbent"),
        ({"peak_memory_mb": None}, "peak_memory_mb is required"),
        ({"peak_memory_mb": 0.0}, "finite positive RSS"),
        ({"peak_memory_mb": 256.0}, "limit is strictly below 256 MB"),
        ({"threads": None}, "threads is required"),
        ({"threads": 2}, "expected 1 from provenance"),
        ({"return_code": 9}, "process return code is 9"),
        ({"engine": "auto"}, "expected 'ls'"),
    ],
)
def test_invalid_error_timeout_incumbent_rss_and_engine_fail(tmp_path, change, diagnostic):
    report = load_module()
    fixture = campaign_inputs(tmp_path)
    rewrite(fixture[0]["jssp_ls"], lambda records: records[0].update(change))

    summary = build(report, fixture)
    invalid = summary["matrices"]["jssp-ls"]["invalid"]

    assert not summary["accepted"]
    assert invalid
    assert diagnostic in " ".join(invalid[0]["issues"])


def test_empty_requested_family_fails_explicitly(tmp_path):
    report = load_module()
    fixture = campaign_inputs(tmp_path)
    path = fixture[0]["rcpsp_ls"]
    write_campaign(path, [record("jssp", "ft06", 0, 55, "ls")], "ls", fixture[4])

    summary = build(report, fixture)

    assert not summary["accepted"]
    assert "no records" in " ".join(summary["matrices"]["rcpsp-ls"]["errors"])


def test_provenance_requires_profile_and_max_iterations(tmp_path):
    report = load_module()
    fixture = campaign_inputs(tmp_path)
    auto_sidecar = fixture[0]["jssp_auto"].with_suffix(".jsonl.provenance.json")
    auto = json.loads(auto_sidecar.read_text(encoding="utf-8"))
    auto["profile_qayd"] = False
    auto_sidecar.write_text(json.dumps(auto), encoding="utf-8")
    ls_sidecar = fixture[0]["jssp_ls"].with_suffix(".jsonl.provenance.json")
    ls = json.loads(ls_sidecar.read_text(encoding="utf-8"))
    del ls["max_iterations"]
    ls_sidecar.write_text(json.dumps(ls), encoding="utf-8")

    summary = build(report, fixture)
    diagnostics = " ".join(summary["provenance"]["errors"])

    assert not summary["accepted"]
    assert "profile_qayd must be true" in diagnostics
    assert "must declare max_iterations" in diagnostics


def test_baseline_must_share_profile_and_iteration_configuration(tmp_path):
    report = load_module()
    fixture = campaign_inputs(tmp_path, baseline=True)
    sidecar = fixture[1][0].with_suffix(".jsonl.provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    provenance["max_iterations"] = 513
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")

    summary = build(report, fixture)

    assert not summary["accepted"]
    assert "baseline and current campaigns do not share" in " ".join(summary["errors"])


@pytest.mark.parametrize(
    ("problem", "change", "diagnostic"),
    (
        ("jssp", {"constructor": "generic"}, "expected 'giffler-thompson-critical-path'"),
        ("jssp", {"schedule_delta_evaluations": 0}, "schedule_delta_evaluations must be positive"),
        ("jssp", {"schedule_oracle_mismatches": 1}, "schedule_oracle_mismatches must be zero"),
        ("jssp", {"schedule_work_steps": 9}, "must be at least schedule_moves_considered"),
        ("jssp", {"candidates_evaluated": 9}, "must equal schedule_moves_considered"),
        ("jssp", {"schedule_moves_rejected": 7}, "plus schedule_moves_rejected must equal"),
        ("rcpsp", {"constructor": "generic"}, "expected 'resource-priority-sgs'"),
    ),
)
def test_forced_ls_requires_structured_constructor_and_coherent_profile(
    tmp_path, problem, change, diagnostic,
):
    report = load_module()
    fixture = campaign_inputs(tmp_path)
    rewrite(fixture[0][f"{problem}_ls"], lambda records: records[0].update(change))

    summary = build(report, fixture)
    invalid = summary["matrices"][f"{problem}-ls"]["invalid"]

    assert not summary["accepted"]
    assert diagnostic in " ".join(invalid[0]["issues"])


@pytest.mark.parametrize(
    "field",
    (
        "resource_candidate_scheduling_attempts",
        "resource_profile_checks",
        "resource_event_visits",
        "resource_peak_profile_events",
        "schedule_alns_generation_attempts",
        "schedule_alns_moves_generated",
        "schedule_workspace_rollbacks",
    ),
)
def test_forced_ls_rcpsp_requires_positive_resource_alns_and_rollback_counters(
    tmp_path, field,
):
    report = load_module()
    fixture = campaign_inputs(tmp_path)
    rewrite(fixture[0]["rcpsp_ls"], lambda records: records[0].update({field: 0}))

    summary = build(report, fixture)
    invalid = summary["matrices"]["rcpsp-ls"]["invalid"]

    assert not summary["accepted"]
    assert f"{field} must be a positive integer" in " ".join(invalid[0]["issues"])


def test_ft06_and_orb10_quality_regressions_fail(tmp_path):
    report = load_module()
    fixture = campaign_inputs(tmp_path)

    def damage(records):
        for value in records:
            if value["instance"] == "ft06" and value["seed"] == 0:
                value["objectives"] = [56]
            if value["instance"] == "orb10":
                value["objectives"] = [1_039]

    rewrite(fixture[0]["jssp_ls"], damage)
    summary = build(report, fixture)

    assert not summary["quality"]["ft06"]["pass"]
    assert not summary["quality"]["orb10"]["pass"]
    assert not summary["accepted"]


def test_rcpsp_auto_must_match_every_best_known_value(tmp_path):
    report = load_module()
    fixture = campaign_inputs(tmp_path)
    rewrite(fixture[0]["rcpsp_auto"], lambda records: records[0].update(objectives=[101]))

    summary = build(report, fixture)

    assert summary["quality"]["rcpsp_auto_bks"]["matches"] == 1
    assert not summary["quality"]["rcpsp_auto_bks"]["pass"]
    assert not summary["accepted"]


def test_rcpsp_forced_ls_must_match_every_best_known_value(tmp_path):
    report = load_module()
    fixture = campaign_inputs(tmp_path)
    rewrite(fixture[0]["rcpsp_ls"], lambda records: records[0].update(objectives=[101]))

    summary = build(report, fixture)

    assert summary["quality"]["rcpsp_ls_bks"]["matches"] == 1
    assert not summary["quality"]["rcpsp_ls_bks"]["pass"]
    assert not summary["accepted"]


def test_self_declared_best_known_cannot_override_external_manifest(tmp_path):
    report = load_module()
    fixture = campaign_inputs(tmp_path)
    rewrite(
        fixture[0]["rcpsp_auto"],
        lambda records: records[0].update(objectives=[101], best_known=101),
    )

    summary = build(report, fixture)
    row = summary["quality"]["rcpsp_auto_bks"]["rows"][0]

    assert not summary["accepted"]
    assert row["manifest_best_known"] == 100
    assert row["record_best_known"] == 101
    assert not row["objective_matches"]
    assert not row["record_matches"]


def test_record_best_known_must_concord_even_when_objective_matches_manifest(tmp_path):
    report = load_module()
    fixture = campaign_inputs(tmp_path)
    rewrite(fixture[0]["rcpsp_ls"], lambda records: records[0].update(best_known=101))

    summary = build(report, fixture)
    row = summary["quality"]["rcpsp_ls_bks"]["rows"][0]

    assert row["objective_matches"]
    assert not row["record_matches"]
    assert not summary["accepted"]


def test_best_known_manifest_is_hashed_and_requires_exact_instance_keys(tmp_path):
    report = load_module()
    fixture = campaign_inputs(tmp_path)
    passing = build(report, fixture)

    assert passing["rcpsp_best_known"]["pass"]
    assert passing["rcpsp_best_known"]["sha256"] == hashlib.sha256(
        fixture[5].read_bytes()
    ).hexdigest()

    write_best_known(fixture[5], {"extra": 100})
    failing = build(report, fixture)

    assert not failing["accepted"]
    assert failing["rcpsp_best_known"]["missing"] == ["r1"]
    assert failing["rcpsp_best_known"]["unexpected"] == ["extra"]


def test_missing_best_known_manifest_fails_without_trusting_records(tmp_path):
    report = load_module()
    fixture = campaign_inputs(tmp_path)
    fixture[5].unlink()

    summary = build(report, fixture)

    assert not summary["accepted"]
    assert summary["rcpsp_best_known"]["sha256"] is None
    assert "does not exist" in " ".join(summary["errors"])


def test_paired_forced_ls_regression_limit_is_enforced(tmp_path):
    report = load_module()
    fixture = campaign_inputs(tmp_path, baseline=True)
    rewrite(fixture[0]["rcpsp_ls"], lambda records: records[0].update(objectives=[121]))

    summary = build(report, fixture)

    assert not summary["regression"]["pass"]
    assert len(summary["regression"]["regressions"]) == 1
    assert summary["regression"]["regressions"][0]["regression_percent"] == 21.0
    assert not summary["accepted"]


def test_missing_or_incompatible_provenance_fails(tmp_path):
    report = load_module()
    fixture = campaign_inputs(tmp_path)
    fixture[0]["jssp_auto"].with_suffix(".jsonl.provenance.json").unlink()
    sidecar = fixture[0]["jssp_ls"].with_suffix(".jsonl.provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    provenance["qayd_artifact"]["sha256"] = "e" * 64
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")

    summary = build(report, fixture)

    assert not summary["provenance"]["pass"]
    assert "missing provenance sidecar" in " ".join(summary["errors"])
    assert "different Qayd source or extension artifacts" in " ".join(summary["errors"])


def test_inconsistent_instance_source_hash_fails(tmp_path):
    report = load_module()
    fixture = campaign_inputs(tmp_path)
    rewrite(fixture[0]["jssp_ls"], lambda records: records[0].update(instance_sha256="f" * 64))

    summary = build(report, fixture)

    assert not summary["instance_sources"]["pass"]
    assert not summary["accepted"]


def test_complete_acceptance_requires_references_and_all_scale_matrices(tmp_path):
    report = load_module()
    fixture = campaign_inputs(tmp_path)

    summary = build(report, fixture, require_complete_acceptance=True)

    assert not summary["accepted"]
    assert not summary["complete_qualification"]["pass"]
    diagnostics = " ".join(summary["errors"])
    assert "requires at least one --jssp-reference" in diagnostics
    assert "j60, j90, j120, mmlib" in diagnostics


def test_complete_acceptance_passes_matched_references_and_full_scale_coverage(tmp_path):
    report = load_module()
    fixture, reference_path, scale_paths = complete_inputs(tmp_path)

    summary = build(
        report,
        fixture,
        jssp_reference_paths=[reference_path],
        rcpsp_scale_paths=scale_paths,
        require_complete_acceptance=True,
    )

    assert summary["accepted"], summary["errors"]
    assert summary["complete_qualification"]["pass"]
    assert all(row["regression_percent"] == 10.0 for row in summary["jssp_reference"]["instances"])
    for group in ("j60", "j90", "j120", "mmlib"):
        matrix = summary["matrices"][f"rcpsp-scale-{group}"]
        assert matrix["incumbent_coverage_percent"] == 100.0
        assert matrix["pass"]


def test_complete_acceptance_rejects_reference_regression_and_partial_scale_coverage(tmp_path):
    report = load_module()
    fixture, reference_path, scale_paths = complete_inputs(tmp_path)
    rewrite(
        fixture[0]["jssp_ls"],
        lambda records: [
            value.update(objectives=[116])
            for value in records
            if value["instance"] == "la38"
        ],
    )
    rewrite(scale_paths["j60"], lambda records: records.pop())

    summary = build(
        report,
        fixture,
        jssp_reference_paths=[reference_path],
        rcpsp_scale_paths=scale_paths,
        require_complete_acceptance=True,
    )

    assert not summary["accepted"]
    la38 = next(row for row in summary["jssp_reference"]["instances"] if row["instance"] == "la38")
    assert la38["regression_percent"] == 16.0
    assert not la38["pass"]
    assert summary["matrices"]["rcpsp-scale-j60"]["incumbent_coverage_percent"] == 50.0
    assert not summary["matrices"]["rcpsp-scale-j60"]["pass"]


def test_require_acceptance_cli_cannot_report_complete_pass_without_extended_inputs(tmp_path):
    report = load_module()
    fixture = campaign_inputs(tmp_path)
    json_output = tmp_path / "report.json"
    markdown_output = tmp_path / "report.md"

    status = report.main([
        "--jssp-auto", str(fixture[0]["jssp_auto"]),
        "--jssp-ls", str(fixture[0]["jssp_ls"]),
        "--rcpsp-auto", str(fixture[0]["rcpsp_auto"]),
        "--rcpsp-ls", str(fixture[0]["rcpsp_ls"]),
        "--rcpsp-best-known", str(fixture[5]),
        "--jssp-instances", ",".join(fixture[2]),
        "--rcpsp-instances", ",".join(fixture[3]),
        "--seeds", ",".join(str(seed) for seed in fixture[4]),
        "--json", str(json_output),
        "--markdown", str(markdown_output),
        "--require-acceptance",
    ])

    summary = json.loads(json_output.read_text(encoding="utf-8"))
    assert status == 2
    assert not summary["accepted"]
    assert not summary["complete_qualification"]["pass"]


def test_require_acceptance_writes_both_reports_and_returns_two(tmp_path):
    report = load_module()
    fixture = campaign_inputs(tmp_path)
    rewrite(fixture[0]["jssp_ls"], lambda records: records[0].update(verified=False))
    json_output = tmp_path / "report.json"
    markdown_output = tmp_path / "report.md"

    status = report.main([
        "--jssp-auto", str(fixture[0]["jssp_auto"]),
        "--jssp-ls", str(fixture[0]["jssp_ls"]),
        "--rcpsp-auto", str(fixture[0]["rcpsp_auto"]),
        "--rcpsp-ls", str(fixture[0]["rcpsp_ls"]),
        "--rcpsp-best-known", str(fixture[5]),
        "--jssp-instances", ",".join(fixture[2]),
        "--rcpsp-instances", ",".join(fixture[3]),
        "--seeds", ",".join(str(seed) for seed in fixture[4]),
        "--json", str(json_output),
        "--markdown", str(markdown_output),
        "--require-acceptance",
    ])

    assert status == 2
    assert json.loads(json_output.read_text(encoding="utf-8"))["accepted"] is False
    assert "Gate: **FAIL**" in markdown_output.read_text(encoding="utf-8")
