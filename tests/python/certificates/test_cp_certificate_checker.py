"""Independent unit and mutation tests for the external certificate checker."""

import hashlib
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "check_cp_certificate.py"
SPEC = importlib.util.spec_from_file_location("qayd_cp_certificate_checker", SCRIPT)
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def model(domains=("ranges 1 0 1",), constraints=(), assumptions=(), objective="none"):
    lines = ["qayd-cp-model 1", f"vars {len(domains)}"]
    lines.extend(f"var {index} {domain}" for index, domain in enumerate(domains))
    lines.append(f"constraints {len(constraints)}")
    for index, constraint in enumerate(constraints):
        lines.extend(constraint.format(cid=index).split("\n"))
    lines.append(f"assumptions {len(assumptions)}")
    lines.extend(f"assume {assumption}" for assumption in assumptions)
    lines.extend([f"objective {objective}", "end-model"])
    return ("\n".join(lines) + "\n").encode("ascii")


def proof(source, body, claim="unsat", witness="none", bound="none"):
    digest = hashlib.sha256(source).hexdigest()
    return f"qayd-cp-proof 1 {digest} {claim}\nwitness {witness}\nbound {bound}\nbegin\n{body}\n".encode("ascii")


def verify(source, certificate, claim="unsat", **limits):
    return checker.verify_certificate(source, certificate, expected_model_sha256=hashlib.sha256(source).hexdigest(), expected_claim=claim, **limits)


PARITY_BODY = "\n".join([
    "split 1 0 0",
    "conflict 2 0 1 1",
    "learn 3 2 1 1",
    "right 4 1 3",
    "conflict 5 0 1 4",
    "learn 6 5 1 4",
    "resolve 7 1 6",
    "finish 7",
])
PARITY_MODEL = model(constraints=("linear {cid} eq 1 1 2 0",))
INFER_MODEL = model(constraints=("linear {cid} le 0 1 1 0", "linear {cid} ge 1 1 1 0"))
INFER_BODY = "infer 1 0 0 le 0 0\nconflict 2 1 1 1\nlearn 3 2 0\nfinish 3"


class CertificateCheckerTests(unittest.TestCase):
    def rejected(self, source, certificate, claim="unsat"):
        with self.assertRaises(checker.CertificateRejected):
            verify(source, certificate, claim)

    def test_sat_full_witness_and_exact_acknowledgement(self):
        source = model(constraints=("linear {cid} le 0 1 1 0",), assumptions=("0 ne 1",))
        result = verify(source, proof(source, "finish-sat", "sat", "1 0"), "sat")
        self.assertEqual(result.steps, 0)
        self.assertEqual(result.assumption_count, 1)
        self.assertEqual(result.acknowledgement(), f"VERIFIED qayd-cp-v1 {hashlib.sha256(source).hexdigest()} sat 0 1")
        self.rejected(source, proof(source, "finish-sat", "sat", "1 1"), "sat")
        self.rejected(source, proof(source, "finish-sat", "sat", "0"), "sat")
        self.rejected(source, proof(source, "finish-sat", "sat", "none"), "sat")
        self.rejected(source, proof(source, "conflict 1 0 0\nfinish-sat", "sat", "1 0"), "sat")

    def test_checked_inference_is_needed_as_an_explicit_premise(self):
        certificate = proof(INFER_MODEL, INFER_BODY)
        self.assertEqual(verify(INFER_MODEL, certificate).steps, 3)
        self.rejected(INFER_MODEL, certificate.replace(b"conflict 2 1 1 1", b"conflict 2 1 0"))
        self.rejected(INFER_MODEL, certificate.replace(b"infer 1 0 0 le 0 0", b"infer 1 0 0 ge 1 0"))
        self.rejected(INFER_MODEL, certificate.replace(b"infer 1 0", b"infer 1 D"))
        self.rejected(INFER_MODEL, certificate.replace(b"conflict 2 1", b"conflict 2 D"))

    def test_parity_requires_both_complementary_branches(self):
        certificate = proof(PARITY_MODEL, PARITY_BODY)
        self.assertEqual(verify(PARITY_MODEL, certificate).steps, 7)
        # An interval rule deliberately does not solve parity by enumeration.
        self.rejected(PARITY_MODEL, proof(PARITY_MODEL, "conflict 1 0 0\nlearn 2 1 0\nfinish 2"))
        for old, new in [
            (b"split 1 0 0", b"split 1 0 1"),
            (b"split 1 0 0", b"split 1 0 -1"),
            (b"right 4 1 3", b"right 4 1 2"),
            (b"right 4 1 3", b"right 4 9 3"),
            (b"right 4 1 3", b"resolve 4 1 3"),
            (b"learn 3 2 1 1", b"learn 3 2 0"),
            (b"learn 3 2 1 1", b"learn 3 1 1 1"),
            (b"learn 6 5 1 4", b"learn 6 5 1 1"),
            (b"resolve 7 1 6", b"resolve 7 1 3"),
            (b"resolve 7 1 6", b"resolve 7 9 6"),
            (b"finish 7", b"finish 3"),
        ]:
            with self.subTest(old=old, new=new):
                self.rejected(PARITY_MODEL, certificate.replace(old, new))

    def test_nested_resolution_restores_parent_facts_and_paths(self):
        source = model(domains=("ranges 1 0 1", "ranges 1 0 1"), constraints=("linear {cid} eq 1 2 2 0 2 1",))
        body = "\n".join([
            "split 1 0 0", "split 2 1 0", "conflict 3 0 2 1 2", "learn 4 3 2 1 2",
            "right 5 2 4", "conflict 6 0 2 1 5", "learn 7 6 2 1 5", "resolve 8 2 7",
            "right 9 1 8", "conflict 10 0 1 9", "learn 11 10 1 9", "resolve 12 1 11", "finish 12",
        ])
        certificate = proof(source, body)
        self.assertEqual(verify(source, certificate).steps, 12)
        for old, new in [
            (b"conflict 3 0 2 1 2", b"conflict 3 0 2 2 1"),
            (b"conflict 3 0 2 1 2", b"conflict 3 0 2 1 1"),
            (b"conflict 3 0 2 1 2", b"conflict 3 0 2 1 9"),
            (b"learn 7 6 2 1 5", b"learn 7 6 2 5 1"),
            (b"right 9 1 8", b"right 9 2 8"),
        ]:
            self.rejected(source, certificate.replace(old, new))

    def test_sibling_and_future_inference_facts_are_unavailable(self):
        body = "\n".join([
            "split 1 0 0", "infer 2 D 0 le 0 1 1", "conflict 3 0 1 2", "learn 4 3 1 1",
            "right 5 1 4", "conflict 6 0 1 5", "learn 7 6 1 5", "resolve 8 1 7", "finish 8",
        ])
        certificate = proof(PARITY_MODEL, body)
        self.assertEqual(verify(PARITY_MODEL, certificate).steps, 8)
        self.rejected(PARITY_MODEL, certificate.replace(b"conflict 6 0 1 5", b"conflict 6 0 1 2"))
        self.rejected(PARITY_MODEL, certificate.replace(b"infer 2 D 0 le 0 1 1", b"infer 2 D 0 le 0 1 2"))
        self.rejected(PARITY_MODEL, certificate.replace(b"infer 2 D 0 le 0 1 1", b"infer 2 D 0 le 0 1 8"))

    def test_conflict_immediacy_and_closed_branch_rules(self):
        certificate = proof(PARITY_MODEL, PARITY_BODY)
        self.rejected(PARITY_MODEL, certificate.replace(b"learn 3 2 1 1", b"infer 3 D 0 le 0 1 1"))
        self.rejected(PARITY_MODEL, certificate.replace(b"right 4 1 3", b"infer 4 D 0 le 0 1 1"))
        self.rejected(PARITY_MODEL, proof(PARITY_MODEL, "learn 1 0 0\nfinish 1"))
        self.rejected(PARITY_MODEL, proof(PARITY_MODEL, "right 1 0 0\nfinish 1"))
        self.rejected(PARITY_MODEL, proof(PARITY_MODEL, "resolve 1 0 0\nfinish 1"))
        self.rejected(PARITY_MODEL, proof(PARITY_MODEL, "finish 0"))

    def test_assumptions_bind_scope_and_support_domain_conflicts(self):
        source = model(assumptions=("0 eq 0", "0 eq 1"))
        certificate = proof(source, "conflict 1 D 0\nlearn 2 1 0\nfinish 2")
        self.assertEqual(verify(source, certificate).assumption_count, 2)
        changed = source.replace(b"assume 0 eq 1", b"assume 0 eq 0")
        with self.assertRaises(checker.CertificateRejected):
            checker.verify_certificate(changed, certificate, expected_model_sha256=hashlib.sha256(source).hexdigest(), expected_claim="unsat")
        self.rejected(changed, proof(changed, "conflict 1 D 0\nlearn 2 1 0\nfinish 2"))
        # Certified facts in a contradictory declared scope remain vacuous.
        verify(source, proof(source, "infer 1 D 0 gt 1 0\nconflict 2 D 0\nlearn 3 2 0\nfinish 3"))

    def test_min_and_max_optimality_require_valid_witness_and_fixed_cut(self):
        for direction, relation, cut in [("min", "ge", "lt"), ("max", "le", "gt")]:
            source = model(domains=("ranges 1 0 2",), constraints=(f"linear {{cid}} {relation} 1 1 1 0",), objective=f"{direction} var 0")
            body = f"infer 1 C 0 {cut} 1 0\nconflict 2 0 1 1\nlearn 3 2 0\nfinish 3"
            certificate = proof(source, body, "optimal", "1 1", f"{direction} 1")
            self.assertEqual(verify(source, certificate, "optimal").steps, 3)
            self.rejected(source, certificate.replace(f"bound {direction} 1".encode(), f"bound {direction} 0".encode()), "optimal")
            other = "max" if direction == "min" else "min"
            self.rejected(source, certificate.replace(f"bound {direction}".encode(), f"bound {other}".encode()), "optimal")
            self.rejected(source, certificate.replace(b"conflict 2 0 1 1", b"conflict 2 0 0"), "optimal")
            self.rejected(source, certificate.replace(b"witness 1 1", b"witness 1 3"), "optimal")
            self.rejected(source, proof(source, "conflict 1 C 0\nlearn 2 1 0\nfinish 2"))

    def test_constant_objective_i64_endpoints_need_no_sentinel_arithmetic(self):
        for direction in ["min", "max"]:
            for value in [-(1 << 63), (1 << 63) - 1]:
                source = model(domains=(), objective=f"{direction} const {value}")
                certificate = proof(source, "conflict 1 C 0\nlearn 2 1 0\nfinish 2", "optimal", "0", f"{direction} {value}")
                self.assertEqual(verify(source, certificate, "optimal").steps, 2)

    def test_sparse_extreme_domains_repeated_table_ids_and_literal_minimum(self):
        minimum, maximum = -(1 << 31), (1 << 31) - 1
        source = model(domains=(f"ranges 2 {minimum} {minimum} {maximum} {maximum}",), constraints=(f"table {{cid}} 2 2 0 0\nrow * {minimum}\nrow {maximum} {maximum}",))
        for value in [minimum, maximum]:
            verify(source, proof(source, "finish-sat", "sat", f"1 {value}"), "sat")
        self.rejected(source, proof(source, "finish-sat", "sat", "1 0"), "sat")
        inconsistent = model(constraints=("table {cid} 2 1 0 0\nrow 0 1",))
        verify(inconsistent, proof(inconsistent, "conflict 1 0 0\nlearn 2 1 0\nfinish 2"))
        wildcard = model(domains=(f"ranges 1 {minimum} {maximum}",), constraints=("table {cid} 1 1 0\nrow *",))
        self.rejected(wildcard, proof(wildcard, "conflict 1 0 0\nlearn 2 1 0\nfinish 2"))
        singleton = model(domains=(f"ranges 1 {minimum} {maximum}",), constraints=(f"table {{cid}} 1 1 0\nrow {minimum}",))
        self.rejected(singleton, proof(singleton, "finish-sat", "sat", f"1 {maximum}"), "sat")

    def test_table_inference_uses_only_cited_facts_and_original_rows(self):
        source = model(domains=("ranges 1 0 1", "ranges 1 0 1"), constraints=("table {cid} 2 2 0 1\nrow 0 1\nrow 1 0", "linear {cid} eq 0 1 1 0", "linear {cid} eq 0 1 1 1"))
        body = "infer 1 1 0 eq 0 0\ninfer 2 0 1 eq 1 1 1\nconflict 3 2 1 2\nlearn 4 3 0\nfinish 4"
        certificate = proof(source, body)
        verify(source, certificate)
        self.rejected(source, certificate.replace(b"infer 2 0 1 eq 1 1 1", b"infer 2 0 1 eq 1 0"))

    def test_all_predicate_relations_and_extreme_negations(self):
        for relation, value in [("eq", 1), ("ne", 0), ("le", 1), ("lt", 2), ("ge", 1), ("gt", 0)]:
            source = model(domains=("ranges 1 0 2",), constraints=("table {cid} 1 1 0\nrow 1", "linear {cid} ne 1 1 1 0"))
            body = f"infer 1 0 0 {relation} {value} 0\ninfer 2 0 0 eq 1 0\nconflict 3 1 1 2\nlearn 4 3 0\nfinish 4"
            verify(source, proof(source, body))
            invalid = body.replace(f"infer 1 0 0 {relation}", f"infer 1 0 0 {checker.NEGATED[relation]}")
            self.rejected(source, proof(source, invalid))
            scoped = model(domains=("ranges 1 0 2",), assumptions=(f"0 {relation} {value}",))
            verify(scoped, proof(scoped, "finish-sat", "sat", "1 1"), "sat")
        minimum, maximum = -(1 << 31), (1 << 31) - 1
        for witness, relation, bound in [(minimum + 1, "gt", minimum), (maximum - 1, "lt", maximum)]:
            source = model(domains=(f"ranges 1 {minimum} {maximum}",), constraints=(f"table {{cid}} 1 1 0\nrow {witness}", f"linear {{cid}} ne {witness} 1 1 0"))
            body = f"infer 1 0 0 {relation} {bound} 0\ninfer 2 0 0 eq {witness} 0\nconflict 3 1 1 2\nlearn 4 3 0\nfinish 4"
            verify(source, proof(source, body))

    def test_zero_arity_tables_and_zero_terms(self):
        for constraint in ["table {cid} 0 0", "linear {cid} eq 1 0", "linear {cid} eq 1 3 7 0 -7 0 0 0"]:
            source = model(constraints=(constraint,))
            verify(source, proof(source, "conflict 1 0 0\nlearn 2 1 0\nfinish 2"))
        source = model(domains=(), constraints=("table {cid} 0 1\nrow",))
        verify(source, proof(source, "finish-sat", "sat", "0"), "sat")
        self.rejected(source, proof(source, "conflict 1 0 0\nlearn 2 1 0\nfinish 2"))

    def test_checked_coefficient_aggregation_and_exact_large_activity(self):
        max128 = (1 << 127) - 1
        source = model(domains=("ranges 1 2147483647 2147483647",), constraints=(f"linear {{cid}} gt {max128} 1 {max128} 0",))
        verify(source, proof(source, "finish-sat", "sat", "1 2147483647"), "sat")
        overflow = model(constraints=(f"linear {{cid}} eq 0 2 {max128} 0 1 0",))
        self.rejected(overflow, proof(overflow, "finish-sat", "sat", "1 0"), "sat")
        cancelled = model(constraints=(f"linear {{cid}} eq 1 2 {max128} 0 {-max128} 0",))
        verify(cancelled, proof(cancelled, "conflict 1 0 0\nlearn 2 1 0\nfinish 2"))

    def test_every_linear_relation_uses_sound_interval_failure_rules(self):
        for relation, rhs in [("eq", 3), ("ne", 2), ("le", 1), ("lt", 2), ("ge", 3), ("gt", 2)]:
            source = model(domains=("ranges 1 2 2",), constraints=(f"linear {{cid}} {relation} {rhs} 1 1 0",))
            verify(source, proof(source, "conflict 1 0 0\nlearn 2 1 0\nfinish 2"))
        source = model(constraints=("linear {cid} ne 0 1 1 0",))
        self.rejected(source, proof(source, "conflict 1 0 0\nlearn 2 1 0\nfinish 2"))

    def test_hash_claim_headers_and_terminals_cannot_be_weakened(self):
        certificate = proof(PARITY_MODEL, PARITY_BODY)
        for changed in [
            certificate.replace(b"qayd-cp-proof 1", b"qayd-cp-proof 2"),
            certificate.replace(hashlib.sha256(PARITY_MODEL).hexdigest().encode(), b"0" * 64),
            certificate.replace(b" unsat\n", b" sat\n", 1),
            certificate.replace(b"witness none", b"witness 1 0"),
            certificate.replace(b"bound none", b"bound min 0"),
            certificate.replace(b"begin\n", b"start\n"),
            certificate.replace(b"finish 7", b"finish-sat"),
            certificate.replace(b"finish 7", b"finish 7 extra"),
            certificate + b"finish 7\n",
        ]:
            self.rejected(PARITY_MODEL, changed)
        with self.assertRaises(checker.CertificateRejected):
            checker.verify_certificate(PARITY_MODEL, certificate, expected_model_sha256="0" * 64, expected_claim="unsat")
        with self.assertRaises(checker.CertificateRejected):
            checker.verify_certificate(PARITY_MODEL, certificate, expected_model_sha256=hashlib.sha256(PARITY_MODEL).hexdigest(), expected_claim="optimal")

    def test_optional_proof_hash_binds_the_already_read_bytes(self):
        certificate = proof(PARITY_MODEL, PARITY_BODY)
        expected = hashlib.sha256(certificate).hexdigest()
        self.assertEqual(verify(PARITY_MODEL, certificate, expected_proof_sha256=expected).steps, 7)
        for wrong in ["0" * 64, "invalid", expected.upper()]:
            with self.assertRaises(checker.CertificateRejected):
                verify(PARITY_MODEL, certificate, expected_proof_sha256=wrong)
        with self.assertRaises(checker.CertificateRejected):
            verify(PARITY_MODEL, certificate.replace(b"finish 7", b"finish 6"), expected_proof_sha256=expected)

    def test_malformed_model_records_are_rejected(self):
        invalid = [
            PARITY_MODEL.replace(b"var 0", b"var 1"),
            PARITY_MODEL.replace(b"ranges 1 0 1", b"ranges 0"),
            PARITY_MODEL.replace(b"ranges 1 0 1", b"ranges 1 1 0"),
            PARITY_MODEL.replace(b"ranges 1 0 1", b"ranges 2 0 0 1 1"),
            PARITY_MODEL.replace(b"ranges 1 0 1", b"ranges 2 1 1 0 0"),
            PARITY_MODEL.replace(b"linear 0", b"linear 1"),
            PARITY_MODEL.replace(b"eq 1 1 2 0", b"eq 1 1 2 1"),
            PARITY_MODEL.replace(b"eq 1 1 2 0", b"eq 1 2 2 0"),
            PARITY_MODEL.replace(b"eq 1", b"bad 1"),
            PARITY_MODEL.replace(b"objective none", b"objective min expr 0"),
            PARITY_MODEL.replace(b"end-model", b"end"),
            PARITY_MODEL + b"end-model\n",
            PARITY_MODEL[:-1],
            PARITY_MODEL.replace(b"vars 1", b"vars +1"),
            PARITY_MODEL.replace(b"vars 1", b"vars nan"),
            PARITY_MODEL.replace(b"eq 1", b"eq " + b"9" * 5000),
        ]
        for source in invalid:
            with self.subTest(source=source[:100]):
                self.rejected(source, proof(source, PARITY_BODY))

    def test_malformed_steps_and_all_truncations_are_rejected(self):
        certificate = proof(INFER_MODEL, INFER_BODY)
        for prefix in range(len(certificate)):
            self.rejected(INFER_MODEL, certificate[:prefix])
        for old, new in [
            (b"infer 1", b"infer 0"), (b"infer 1", b"infer 2"), (b"infer 1", b"infer -1"),
            (b"infer 1 0", b"infer 1 99"), (b"infer 1 0", b"infer 1 unknown"),
            (b"infer 1 0 0 le 0 0", b"infer 1 0 0 le 0 1 9"),
            (b"infer 1 0 0 le 0 0", b"infer 1 0 0 le 0 0 extra"),
            (b"learn 3 2 0", b"learn 3 2 1"),
            (b"conflict 2", b"unknown 2"),
            (b"infer 1 0 0 le 0 0", b"infer 1 0 0 le 2147483648 0"),
            (b"infer 1 0 0 le 0 0", b"infer 1 0 0 le 0.0 0"),
        ]:
            self.rejected(INFER_MODEL, certificate.replace(old, new))
        for suffix in [b"garbage", b"\n", b"# comment\n"]:
            self.rejected(INFER_MODEL, certificate + suffix)
        for old, new in [(b" ", b"\t"), (b"\n", b"\r\n"), (b"infer 1", b" infer 1"), (b"infer 1", b"infer  1"), (b"infer 1", b"infer\x001")]:
            self.rejected(INFER_MODEL, certificate.replace(old, new, 1))

    def test_resources_and_invalid_numeric_limits_never_accept(self):
        certificate = proof(PARITY_MODEL, PARITY_BODY)
        for limits in [
            {"max_work": 0}, {"max_steps": 6}, {"max_depth": 0},
            {"max_model_bytes": len(PARITY_MODEL) - 1}, {"max_proof_bytes": len(certificate) - 1},
            {"time_limit": 0}, {"time_limit": math.nan}, {"time_limit": math.inf},
            {"time_limit": -1}, {"time_limit": 10 ** 1000}, {"max_work": -1}, {"max_steps": 1.5}, {"max_depth": True},
        ]:
            with self.subTest(limits=limits):
                with self.assertRaises(checker.CheckLimitExceeded):
                    verify(PARITY_MODEL, certificate, **limits)
        too_many = b"qayd-cp-model 1\nvars 65\n"
        with self.assertRaises(checker.CheckLimitExceeded):
            verify(too_many, proof(too_many, "finish 0"))
        # Check record count before building a huge split-line object graph.
        with self.assertRaises(checker.CheckLimitExceeded):
            verify(PARITY_MODEL, b"\n" * 100, max_steps=1)

    def test_cli_and_embedded_execution_acknowledgements_and_exit_codes(self):
        source = model(constraints=("linear {cid} le 0 1 1 0",))
        certificate = proof(source, "finish-sat", "sat", "1 0")
        with tempfile.TemporaryDirectory() as directory:
            model_path, proof_path = Path(directory) / "model.cp", Path(directory) / "proof.cp"
            model_path.write_bytes(source)
            proof_path.write_bytes(certificate)
            args = ["--model", str(model_path), "--proof", str(proof_path), "--expected-model-sha256", hashlib.sha256(source).hexdigest(), "--expected-claim", "sat"]
            expected = verify(source, certificate, "sat").acknowledgement() + "\n"
            for command in [[sys.executable, str(SCRIPT)], [sys.executable, "-I", "-c", SCRIPT.read_text()]]:
                result = subprocess.run(command + args, capture_output=True, text=True, check=False)
                self.assertEqual((result.returncode, result.stdout, result.stderr), (0, expected, ""))
                bound = subprocess.run(command + args + ["--expected-proof-sha256", hashlib.sha256(certificate).hexdigest()], capture_output=True, text=True, check=False)
                self.assertEqual((bound.returncode, bound.stdout, bound.stderr), (0, expected, ""))
                wrong_hash = subprocess.run(command + args + ["--expected-proof-sha256", "0" * 64], capture_output=True, text=True, check=False)
                self.assertEqual((wrong_hash.returncode, wrong_hash.stdout), (1, ""))
                rejected = subprocess.run(command + args + ["--expected-claim", "unsat"], capture_output=True, text=True, check=False)
                self.assertEqual(rejected.returncode, 1)
                self.assertEqual(rejected.stdout, "")
                for extra in [["--max-work", "0"], ["--time-limit", "nan"], ["--max-steps", "-1"], ["--time-limit", "inf"], ["--time-limit", "0"]]:
                    result = subprocess.run(command + args + extra, capture_output=True, text=True, check=False)
                    self.assertEqual(result.returncode, 2)
                    self.assertEqual(result.stdout, "")
                missing = subprocess.run(command + args + ["--model", str(Path(directory) / "absent")], capture_output=True, text=True, check=False)
                self.assertEqual(missing.returncode, 2)
                self.assertEqual(missing.stdout, "")


if __name__ == "__main__":
    unittest.main()
