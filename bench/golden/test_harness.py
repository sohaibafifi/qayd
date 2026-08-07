from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import harness


class ManifestTests(unittest.TestCase):
    def test_checked_in_manifest_is_valid_and_covers_all_phase_a0_surfaces(self) -> None:
        manifest = harness.load_manifest(harness.DEFAULT_MANIFEST)
        surfaces = {case["surface"] for case in manifest["cases"]}
        self.assertTrue(
            {"rust-native", "python", "xcsp", "flatzinc", "sat", "pb", "lists", "routing", "scheduling"}.issubset(surfaces)
        )
        equivalent = {
            case["id"]
            for case in manifest["cases"]
            if case.get("equivalence_group") == "integer-minimize"
        }
        self.assertEqual(
            equivalent,
            {"rust-native-minimize", "python-integer-minimize", "xcsp-minimize", "flatzinc-minimize"},
        )

    def test_python_equivalence_fixture_matches_native_model_and_complete_request(self) -> None:
        manifest = harness.load_manifest(harness.DEFAULT_MANIFEST)
        cases = {case["id"]: case for case in manifest["cases"]}
        native_model = copy.deepcopy(cases["rust-native-minimize"]["oracle"]["model"])
        python_case = cases["python-integer-minimize"]
        python_spec = json.loads(harness.repo_path(python_case["instance"]["path"]).read_text(encoding="utf-8"))

        native_model["variables"] = [
            {
                "name": variable["name"],
                "domain": list(range(variable["domain"]["min"], variable["domain"]["max"] + 1)),
            }
            for variable in native_model["variables"]
        ]
        self.assertEqual(
            {key: python_spec[key] for key in ("variables", "constraints", "objectives")},
            native_model,
        )
        self.assertEqual(
            {
                "seed": python_case["request"]["seed"],
                "threads": python_case["request"]["threads"],
                "engine": python_case["request"]["engine"],
                "budget": python_case["request"]["budget"],
                "objective_senses": python_case["request"]["objective_senses"],
            },
            {
                "seed": 0,
                "threads": 1,
                "engine": "exact",
                "budget": {"kind": "complete", "wall_time_seconds": None, "max_iterations": None},
                "objective_senses": ["minimize"],
            },
        )

    def test_manifest_rejects_noncanonical_gap(self) -> None:
        manifest = json.loads(harness.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        manifest["cases"][0]["expected"]["gap"] = {"absolute": 1, "relative": 1.0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(harness.GoldenError, "expected gap is not canonical"):
                harness.load_manifest(path)

    def test_manifest_rejects_duplicate_json_keys(self) -> None:
        source = harness.DEFAULT_MANIFEST.read_text(encoding="utf-8")
        source = source.replace(
            '"schema": "qayd.golden.manifest/v1"',
            '"schema": "qayd.golden.manifest/v1", "schema": "duplicate"',
            1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(harness.GoldenError, "duplicate JSON key 'schema'"):
                harness.load_manifest(path)

    def test_manifest_has_no_regeneration_command(self) -> None:
        source = Path(harness.__file__).read_text(encoding="utf-8")
        self.assertNotIn("--update", source)
        self.assertNotIn("--bless", source)

    def test_protected_goldens_stay_unchanged(self) -> None:
        manifest = harness.load_manifest(harness.DEFAULT_MANIFEST)
        snapshot = harness.snapshot_goldens(harness.DEFAULT_MANIFEST, manifest)
        harness.assert_goldens_unchanged(snapshot)


class OracleTests(unittest.TestCase):
    def test_lexicographic_oracle_checks_all_objectives(self) -> None:
        model = {
            "variables": [
                {"name": "x", "domain": [0, 1]},
                {"name": "y", "domain": [0, 1]},
            ],
            "constraints": [{"terms": {"x": 1, "y": 1}, "relation": "eq", "rhs": 1}],
            "objectives": [
                {"sense": "minimize", "terms": {"x": 1}},
                {"sense": "maximize", "terms": {"y": 1}},
            ],
        }
        actual = {
            "status": "OPTIMAL",
            "solution": {"assignments": {"x": 0, "y": 1}},
        }
        outcome = harness.finite_domain_oracle(model, actual)
        self.assertTrue(outcome.valid_solution)
        self.assertEqual(outcome.actual_objectives, [0, 1])
        self.assertEqual(outcome.optimum, [0, 1])

    def test_checked_sat_proof_is_rup_refutation(self) -> None:
        instance = harness.repo_path("bench/golden/fixtures/sat_unsat.cnf")
        with tempfile.TemporaryDirectory() as directory:
            proof = Path(directory) / "proof.drat"
            proof.write_text("0\n", encoding="ascii")
            self.assertTrue(harness.verify_rup_drat(instance, proof))
            proof.write_text("1 0\n", encoding="ascii")
            self.assertFalse(harness.verify_rup_drat(instance, proof))


class SemanticComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = {
            "expected": {
                "status": "OPTIMAL",
                "senses": ["minimize"],
                "objectives": [1],
                "solution": {"assignments": {"x": 1}},
                "validity": True,
                "bound": {"values": [1], "source": "terminal-status"},
                "gap": {"absolute": 0, "relative": 0.0},
                "proof": {"claim": "optimality", "kind": "terminal-status", "verified": True},
            }
        }
        self.actual = copy.deepcopy(self.case["expected"])
        self.outcome = harness.OracleOutcome(True, True, [1], [1])

    def test_matching_semantics_pass(self) -> None:
        self.assertEqual(harness.compare_semantics(self.case, self.actual, self.outcome), [])

    def test_structured_result_rejects_duplicate_json_keys(self) -> None:
        marker = harness.RESULT_PREFIX + '{"status":"OPTIMAL","status":"UNKNOWN"}'
        with self.assertRaisesRegex(harness.GoldenError, "duplicate JSON key 'status'"):
            harness.decode_marker(marker)

    def test_unverified_optimality_is_rejected(self) -> None:
        self.actual["proof"]["verified"] = False
        errors = harness.compare_semantics(self.case, self.actual, self.outcome)
        self.assertTrue(any("verified proof" in error for error in errors))

    def test_invalid_candidate_is_rejected(self) -> None:
        self.actual["validity"] = False
        errors = harness.compare_semantics(self.case, self.actual, self.outcome)
        self.assertTrue(any("not valid" in error for error in errors))

    def test_canonical_json_types_are_not_coerced(self) -> None:
        self.actual["solution"]["assignments"]["x"] = True
        errors = harness.compare_semantics(self.case, self.actual, self.outcome)
        self.assertTrue(any("solution mismatch" in error for error in errors))

    def test_invalid_minimization_bound_is_rejected(self) -> None:
        self.actual["bound"]["values"] = [2]
        errors = harness.compare_semantics(self.case, self.actual, self.outcome)
        self.assertTrue(any("bound exceeds incumbent" in error for error in errors))

    def test_noncanonical_gap_is_rejected(self) -> None:
        self.actual["gap"] = {"absolute": 1, "relative": 1.0}
        errors = harness.compare_semantics(self.case, self.actual, self.outcome)
        self.assertTrue(any("gap is not canonical" in error for error in errors))


class GapTests(unittest.TestCase):
    def test_minimization_gap_uses_shared_absolute_scale(self) -> None:
        record = {
            "senses": ["minimize"],
            "objectives": [10],
            "bound": {"values": [7], "source": "test"},
        }
        self.assertEqual(harness.canonical_gap(record), {"absolute": 3, "relative": 0.3})

    def test_maximization_gap_respects_sense_and_negative_values(self) -> None:
        record = {
            "senses": ["maximize"],
            "objectives": [-10],
            "bound": {"values": [-7], "source": "test"},
        }
        self.assertEqual(harness.canonical_gap(record), {"absolute": 3, "relative": 0.3})

    def test_relative_gap_uses_larger_absolute_bound(self) -> None:
        record = {
            "senses": ["minimize"],
            "objectives": [2],
            "bound": {"values": [-8], "source": "test"},
        }
        self.assertEqual(harness.canonical_gap(record), {"absolute": 10, "relative": 1.25})

    def test_gap_is_null_without_a_bound(self) -> None:
        record = {"senses": ["minimize"], "objectives": [10], "bound": None}
        self.assertIsNone(harness.canonical_gap(record))

    def test_inconsistent_bound_has_no_gap(self) -> None:
        record = {
            "senses": ["minimize"],
            "objectives": [7],
            "bound": {"values": [10], "source": "test"},
        }
        self.assertIsNone(harness.canonical_gap(record))


class EquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cases = [
            {"id": "native", "surface": "rust-native", "equivalence_group": "same-model"},
            {"id": "xcsp", "surface": "xcsp", "equivalence_group": "same-model"},
        ]
        self.native = {
            "status": "OPTIMAL",
            "senses": ["minimize"],
            "objectives": [1],
            "solution": {"assignments": {"x": 1}},
            "validity": True,
            "bound": {"values": [1], "source": "complete-search"},
            "gap": {"absolute": 0, "relative": 0.0},
            "proof": {"claim": "optimality", "kind": "complete-search", "verified": True},
        }
        self.xcsp = copy.deepcopy(self.native)
        self.xcsp["bound"]["source"] = "terminal-status"
        self.xcsp["proof"]["kind"] = "terminal-status"

    def test_protocol_mechanism_names_are_ignored(self) -> None:
        records = {"native": self.native, "xcsp": self.xcsp}
        self.assertEqual(harness.compare_equivalence_groups(self.cases, records), [])

    def test_bound_value_difference_is_rejected(self) -> None:
        self.xcsp["bound"]["values"] = [0]
        records = {"native": self.native, "xcsp": self.xcsp}
        errors = harness.compare_equivalence_groups(self.cases, records)
        self.assertTrue(any("bound_values mismatch" in error for error in errors))

    def test_canonical_solution_types_are_compared_exactly(self) -> None:
        self.xcsp["solution"]["assignments"]["x"] = True
        records = {"native": self.native, "xcsp": self.xcsp}
        errors = harness.compare_equivalence_groups(self.cases, records)
        self.assertTrue(any("solution mismatch" in error for error in errors))

    def test_gap_difference_is_rejected(self) -> None:
        self.xcsp["gap"] = {"absolute": 1, "relative": 1.0}
        records = {"native": self.native, "xcsp": self.xcsp}
        errors = harness.compare_equivalence_groups(self.cases, records)
        self.assertTrue(any("gap mismatch" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
