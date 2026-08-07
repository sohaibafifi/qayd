from __future__ import annotations

import copy
import contextlib
import io
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "bench" / "parallel" / "harness.py"
SPEC = importlib.util.spec_from_file_location("qayd_parallel_harness", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
harness = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = harness
SPEC.loader.exec_module(harness)


def timing_report(samples: list[float]) -> dict:
    result_samples = [
        {
            "elapsed_seconds": sample,
            "result": {
                "status": "OPTIMAL",
                "objectives": [34],
                "objective": 34,
                "solution": {"x": 34},
                "metrics": {"split_jobs": 1},
                "stdout_sha256": "6" * 64,
            },
            "stderr_sha256": "7" * 64,
        }
        for sample in samples
    ]
    return {
        "schema": harness.TIMING_SCHEMA,
        "campaign": {
            "label": "test",
            "source_revision": harness.PINNED_PRE_REFACTOR_REVISION,
            "binary_path": "/tmp/qayd",
            "binary_sha256": "1" * 64,
            "manifest_path": "/tmp/manifest.json",
            "manifest_sha256": "2" * 64,
            "harness_sha256": "5" * 64,
            "corpus_version": "test-v1",
            "pre_refactor_revision": harness.PINNED_PRE_REFACTOR_REVISION,
            "captured_at": "2026-08-06T00:00:00+00:00",
        },
        "default_tolerance": 0.1,
        "cases": [
            {
                "scenario_id": "split",
                "feature": "split",
                "instance": {"path": "fixture.xml", "sha256": "3" * 64},
                "request": {
                    "args": ["--threads", "2", "--split", "{instance}"],
                    "seed": 0,
                    "threads": 2,
                    "budget": {"kind": "complete"},
                },
                "repetitions": len(samples),
                "warmups": 0,
                "samples": result_samples,
                "median_seconds": harness.median_seconds(samples),
            }
        ],
    }


def refresh_capture_id(pairing: dict) -> None:
    key = {field: pairing[field] for field in harness.PAIRING_KEY_FIELDS}
    pairing["capture_id"] = harness.sha256_bytes(harness.canonical_json(key).encode("utf-8"))


def paired_reports(repetitions: int = 4) -> tuple[dict, dict]:
    baseline = timing_report([1.0] * repetitions)
    candidate = timing_report([1.0] * repetitions)
    candidate["campaign"].update({
        "label": "candidate",
        "source_revision": "a" * 40,
        "binary_path": "/tmp/qayd-candidate",
        "binary_sha256": "4" * 64,
    })
    schedule = harness.expected_pair_schedule(["split"], repetitions)
    pairing = {
        "schema": harness.PAIRING_SCHEMA,
        "capture_nonce_ns": 1,
        "manifest_sha256": baseline["campaign"]["manifest_sha256"],
        "harness_sha256": baseline["campaign"]["harness_sha256"],
        "baseline_binary_sha256": baseline["campaign"]["binary_sha256"],
        "candidate_binary_sha256": candidate["campaign"]["binary_sha256"],
        "repetitions": repetitions,
        "warmups": 0,
        "schedule": schedule,
    }
    refresh_capture_id(pairing)
    baseline["pairing"] = copy.deepcopy(pairing)
    candidate["pairing"] = copy.deepcopy(pairing)
    for side, report in (("baseline", baseline), ("candidate", candidate)):
        for index, sample in enumerate(report["cases"][0]["samples"]):
            first = schedule[index]["first"]
            sample["pair"] = {
                "repetition": index + 1,
                "position": 1 if first == side else 2,
            }
    return baseline, candidate


class ManifestTests(unittest.TestCase):
    def test_checked_in_manifest_covers_all_parallel_features(self) -> None:
        manifest = harness.load_manifest(harness.DEFAULT_MANIFEST)
        self.assertEqual(
            {scenario["feature"] for scenario in manifest["scenarios"]},
            {"clause-sharing", "split", "probes", "lns"},
        )
        self.assertEqual(manifest["default_tolerance"], 0.1)

    def test_requests_are_complete_and_reproducible(self) -> None:
        manifest = harness.load_manifest(harness.DEFAULT_MANIFEST)
        for scenario in manifest["scenarios"]:
            request = scenario["request"]
            self.assertEqual(request["budget"]["kind"], "complete")
            self.assertIsInstance(request["seed"], int)
            self.assertGreater(request["threads"], 1)
            self.assertIn("{instance}", request["args"])


class ParserAndOracleTests(unittest.TestCase):
    def test_parser_retains_terminal_result_and_parallel_metrics(self) -> None:
        output = """\
o 40
o 34
c nodes 20 failures 3
c shared clauses 7 imported 4
c split jobs 2 completed 3
c probes attempts 5 unsat 2
c lns attempts 6 improved 1
s OPTIMUM FOUND
v <instantiation>
v   <list>
v x[0] x[1]
v   </list>
v   <values>
v 0 34
v   </values>
v </instantiation>
"""
        result = harness.parse_xcsp_output(output)
        self.assertEqual(result["status"], "OPTIMAL")
        self.assertEqual(result["objectives"], [40, 34])
        self.assertEqual(result["solution"], {"x[0]": 0, "x[1]": 34})
        self.assertEqual(result["metrics"]["imported_clauses"], 4)
        self.assertEqual(result["metrics"]["completed_jobs"], 3)
        self.assertEqual(result["metrics"]["probe_unsat"], 2)
        self.assertEqual(result["metrics"]["lns_improved"], 1)

    def test_golomb_oracle_replays_all_pairwise_distances(self) -> None:
        manifest = harness.load_manifest(harness.DEFAULT_MANIFEST)
        fixtures, scenarios = harness.manifest_maps(manifest)
        marks = [0, 1, 4, 9, 15, 22, 32, 34]
        distances = [marks[j] - marks[i] for i in range(len(marks)) for j in range(i + 1, len(marks))]
        solution = {"x[{}]".format(index): value for index, value in enumerate(marks)}
        solution.update({"d[{}]".format(index): value for index, value in enumerate(distances)})
        result = {
            "status": "OPTIMAL",
            "objective": 34,
            "solution": solution,
            "metrics": {"split_jobs": 1, "completed_jobs": 1},
        }
        self.assertEqual(harness.validate_result(fixtures["golomb-8"], scenarios["split"], result), [])
        result["solution"]["x[7]"] = 33
        self.assertTrue(harness.validate_semantics(fixtures["golomb-8"], scenarios["split"], result))

    def test_golomb_oracle_rejects_unassigned_distance_variables(self) -> None:
        manifest = harness.load_manifest(harness.DEFAULT_MANIFEST)
        fixtures, scenarios = harness.manifest_maps(manifest)
        marks = [0, 1, 4, 9, 15, 22, 32, 34]
        solution = {"x[{}]".format(index): value for index, value in enumerate(marks)}
        solution.update({"d[{}]".format(index): "*" for index in range(28)})
        result = {
            "status": "OPTIMAL",
            "objective": 34,
            "solution": solution,
            "metrics": {"split_jobs": 1, "completed_jobs": 1},
        }
        errors = harness.validate_result(fixtures["golomb-8"], scenarios["split"], result)
        self.assertTrue(any("unassigned distances" in error for error in errors))

    def test_missing_feature_evidence_is_rejected(self) -> None:
        manifest = harness.load_manifest(harness.DEFAULT_MANIFEST)
        fixtures, scenarios = harness.manifest_maps(manifest)
        result = {"status": "UNSATISFIABLE", "objective": None, "solution": None, "metrics": {}}
        errors = harness.validate_result(fixtures["pigeonhole-9-8"], scenarios["clause-sharing"], result)
        self.assertTrue(any("shared_clauses" in error for error in errors))


class TimingComparatorTests(unittest.TestCase):
    def test_median_uses_repeated_samples(self) -> None:
        self.assertEqual(harness.median_seconds([9.0, 1.0, 3.0]), 3.0)
        self.assertEqual(harness.median_seconds([1.0, 5.0, 3.0, 7.0]), 4.0)

    def test_exactly_ten_percent_slower_passes(self) -> None:
        comparison = harness.compare_reports(timing_report([10.0, 10.0, 10.0]), timing_report([11.0, 11.0, 11.0]), 0.1)
        self.assertTrue(comparison["passed"])
        self.assertFalse(comparison["cases"][0]["regression"])

    def test_more_than_ten_percent_slower_fails(self) -> None:
        comparison = harness.compare_reports(timing_report([10.0, 10.0, 10.0]), timing_report([11.01, 11.01, 11.01]), 0.1)
        self.assertFalse(comparison["passed"])
        self.assertEqual(comparison["regressions"], ["split"])

    def test_tolerance_is_configurable(self) -> None:
        baseline = timing_report([10.0, 10.0, 10.0])
        candidate = timing_report([11.5, 11.5, 11.5])
        self.assertFalse(harness.compare_reports(baseline, candidate, 0.1)["passed"])
        self.assertTrue(harness.compare_reports(baseline, candidate, 0.2)["passed"])

    def test_changed_instance_or_request_is_not_comparable(self) -> None:
        baseline = timing_report([1.0, 1.0, 1.0])
        candidate = copy.deepcopy(baseline)
        candidate["cases"][0]["instance"]["sha256"] = "4" * 64
        candidate["cases"][0]["request"]["seed"] = 1
        comparison = harness.compare_reports(baseline, candidate)
        self.assertFalse(comparison["comparable"])
        self.assertFalse(comparison["passed"])
        self.assertTrue(any("instance" in error for error in comparison["errors"]))
        self.assertTrue(any("request" in error for error in comparison["errors"]))

    def test_matching_but_missing_provenance_is_rejected(self) -> None:
        baseline = timing_report([1.0, 1.0, 1.0])
        candidate = copy.deepcopy(baseline)
        baseline["campaign"] = {}
        candidate["campaign"] = {}
        for report in (baseline, candidate):
            report["cases"][0].pop("instance")
            report["cases"][0].pop("request")
        comparison = harness.compare_reports(baseline, candidate)
        self.assertFalse(comparison["passed"])
        self.assertTrue(any("manifest_sha256" in error for error in comparison["errors"]))
        self.assertTrue(any("instance" in error for error in comparison["errors"]))
        self.assertTrue(any("request" in error for error in comparison["errors"]))

    def test_baseline_must_use_the_pinned_pre_refactor_revision(self) -> None:
        baseline = timing_report([1.0, 1.0, 1.0])
        candidate = copy.deepcopy(baseline)
        baseline["campaign"]["source_revision"] = "9" * 40
        comparison = harness.compare_reports(baseline, candidate)
        self.assertFalse(comparison["passed"])
        self.assertTrue(any("pinned pre-refactor" in error for error in comparison["errors"]))

    def test_reported_median_is_checked_against_samples(self) -> None:
        baseline = timing_report([1.0, 2.0, 3.0])
        candidate = copy.deepcopy(baseline)
        candidate["median_seconds"] = 99.0
        candidate["cases"][0]["median_seconds"] = 99.0
        comparison = harness.compare_reports(baseline, candidate)
        self.assertFalse(comparison["passed"])
        self.assertTrue(any("median" in error for error in comparison["errors"]))

    def test_truncated_or_differently_repeated_reports_are_rejected(self) -> None:
        baseline = timing_report([1.0, 1.0, 1.0])
        candidate = timing_report([1.0, 1.0, 1.0])
        baseline["cases"][0]["samples"] = baseline["cases"][0]["samples"][:1]
        candidate["cases"][0]["warmups"] = 1
        comparison = harness.compare_reports(baseline, candidate)
        self.assertFalse(comparison["passed"])
        self.assertTrue(any("fewer than three" in error for error in comparison["errors"]))
        self.assertTrue(any("warmup counts" in error for error in comparison["errors"]))


class PairedCaptureTests(unittest.TestCase):
    def test_pair_order_alternates_deterministically(self) -> None:
        self.assertEqual(harness.paired_order(0, 0), ("baseline", "candidate"))
        self.assertEqual(harness.paired_order(1, 0), ("candidate", "baseline"))
        self.assertEqual(harness.paired_order(0, 1), ("candidate", "baseline"))
        self.assertEqual(harness.paired_order(1, 1), ("baseline", "candidate"))

    def test_even_schedule_balances_first_position_per_scenario(self) -> None:
        schedule = harness.expected_pair_schedule(["split", "lns"], 6)
        for scenario_id in ("split", "lns"):
            first = [entry["first"] for entry in schedule if entry["scenario_id"] == scenario_id]
            self.assertEqual(first.count("baseline"), 3)
            self.assertEqual(first.count("candidate"), 3)

    def test_valid_paired_reports_compare(self) -> None:
        baseline, candidate = paired_reports()
        comparison = harness.compare_reports(baseline, candidate)
        self.assertTrue(comparison["passed"])
        self.assertEqual(comparison["errors"], [])

    def test_comparator_rejects_mismatched_pair_metadata(self) -> None:
        baseline, candidate = paired_reports()
        candidate["pairing"]["capture_id"] = "9" * 64
        comparison = harness.compare_reports(baseline, candidate)
        self.assertFalse(comparison["passed"])
        self.assertTrue(any("pairing metadata differs" in error for error in comparison["errors"]))
        self.assertTrue(any("capture_id does not match" in error for error in comparison["errors"]))

    def test_comparator_rejects_schedule_with_duplicate_coverage(self) -> None:
        baseline, candidate = paired_reports()
        for report in (baseline, candidate):
            report["pairing"]["schedule"][1] = copy.deepcopy(report["pairing"]["schedule"][0])
            refresh_capture_id(report["pairing"])
        comparison = harness.compare_reports(baseline, candidate)
        self.assertFalse(comparison["passed"])
        self.assertTrue(any("schedule does not match" in error for error in comparison["errors"]))

    def test_comparator_rejects_noncomplementary_sample_positions(self) -> None:
        baseline, candidate = paired_reports()
        candidate["cases"][0]["samples"][0]["pair"]["position"] = 1
        comparison = harness.compare_reports(baseline, candidate)
        self.assertFalse(comparison["passed"])
        self.assertTrue(any("positions do not match schedule" in error for error in comparison["errors"]))

    def test_comparator_rejects_pair_repetition_that_differs_from_sample(self) -> None:
        baseline, candidate = paired_reports()
        baseline["cases"][0]["samples"][1]["pair"]["repetition"] = 1
        comparison = harness.compare_reports(baseline, candidate)
        self.assertFalse(comparison["passed"])
        self.assertTrue(any("pair repetition does not match" in error for error in comparison["errors"]))

    def test_comparator_cross_checks_pair_binary_hashes(self) -> None:
        baseline, candidate = paired_reports()
        candidate["campaign"]["binary_sha256"] = "9" * 64
        comparison = harness.compare_reports(baseline, candidate)
        self.assertFalse(comparison["passed"])
        self.assertTrue(any("candidate pairing binary hash differs" in error for error in comparison["errors"]))

    def test_comparator_cross_checks_pair_counts(self) -> None:
        baseline, candidate = paired_reports()
        for report in (baseline, candidate):
            report["pairing"]["repetitions"] = 6
            report["pairing"]["schedule"] = harness.expected_pair_schedule(["split"], 6)
            refresh_capture_id(report["pairing"])
        comparison = harness.compare_reports(baseline, candidate)
        self.assertFalse(comparison["passed"])
        self.assertTrue(any("repetitions differ from pairing" in error for error in comparison["errors"]))

    def test_pair_cli_requires_explicit_revisions_and_defaults_to_six_repeats(self) -> None:
        base = [
            "measure-pair",
            "--baseline-binary", "/tmp/baseline",
            "--candidate-binary", "/tmp/candidate",
            "--baseline-out", "/tmp/baseline.json",
            "--candidate-out", "/tmp/candidate.json",
            "--baseline-label", "baseline",
            "--candidate-label", "candidate",
        ]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            harness.parse_args(base)
        args = harness.parse_args(base + [
            "--baseline-revision", harness.PINNED_PRE_REFACTOR_REVISION,
            "--candidate-revision", "a" * 40,
        ])
        self.assertEqual(args.repetitions, 6)

    def test_pair_capture_rejects_odd_repetition_count_before_io(self) -> None:
        with self.assertRaisesRegex(harness.HarnessError, "even repetition count"):
            harness.run_timing_pair(
                Path("/missing/manifest"),
                Path("/missing/baseline"),
                Path("/missing/candidate"),
                Path("/tmp/baseline.json"),
                Path("/tmp/candidate.json"),
                "baseline",
                "candidate",
                harness.PINNED_PRE_REFACTOR_REVISION,
                "a" * 40,
                [],
                5,
                0,
            )


class HarnessInputSafetyTests(unittest.TestCase):
    def test_timeout_override_must_be_positive(self) -> None:
        request = {"timeout_seconds": 30}
        self.assertEqual(harness.effective_timeout(request, None), 30)
        self.assertEqual(harness.effective_timeout(request, 10), 10)
        for invalid in (0, -1, True):
            with self.subTest(invalid=invalid), self.assertRaises(harness.HarnessError):
                harness.effective_timeout(request, invalid)

    def test_output_paths_cannot_alias_each_other_or_inputs(self) -> None:
        first = Path("/tmp/qayd-paired-first.json")
        with self.assertRaisesRegex(harness.HarnessError, "distinct"):
            harness.validate_output_paths([first, first], [])
        with self.assertRaisesRegex(harness.HarnessError, "protected input"):
            harness.validate_output_paths([first], [first])


if __name__ == "__main__":
    unittest.main()
