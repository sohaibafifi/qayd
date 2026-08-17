import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.common import satpb_verify


def test_sat_assignment_is_replayed_against_every_clause():
    source = "p cnf 3 3\n1 2 0\n-1 3 0\n-3 2 0\n"

    accepted = satpb_verify.verify_sat(source, "s SATISFIABLE\nv 1 2 3 0\n")
    rejected = satpb_verify.verify_sat(source, "s SATISFIABLE\nv 1 -2 3 0\n")

    assert accepted == {
        "attempted": True,
        "valid": True,
        "reason": "accepted",
        "assigned_variables": 3,
    }
    assert rejected["valid"] is False
    assert rejected["reason"] == "unsatisfied-clause"
    assert rejected["constraint"] == 3


def test_sat_assignment_must_be_complete_and_consistent():
    source = "p cnf 2 1\n1 2 0\n"

    for stdout, message in [
        ("v 1 0\n", "omits 1 variable"),
        ("v 1 -1 2 0\n", "conflicting values"),
        ("v 1 2 3 0\n", "above x2"),
    ]:
        try:
            satpb_verify.verify_sat(source, stdout)
        except satpb_verify.VerificationError as error:
            assert message in str(error)
        else:
            raise AssertionError("invalid SAT assignment was accepted")


def test_linear_pb_replay_handles_negated_literals_and_objective():
    source = (
        "* #variable= 3 #constraint= 2\n"
        "min: 5 ~x1 2 x2 -3 x3 ;\n"
        "2 x1 3 ~x2 >= 2 ;\n"
        "1 x2 1 x3 = 1 ;\n"
    )

    accepted = satpb_verify.verify_pb(source, "s OPTIMUM FOUND\nv x1 x2 -x3\no 2\n", 2)
    bad_objective = satpb_verify.verify_pb(source, "v x1 x2 -x3\n", 3)
    infeasible = satpb_verify.verify_pb(source, "v -x1 x2 -x3\n", 7)

    assert accepted["valid"] is True
    assert accepted["computed_objective"] == 2
    assert bad_objective == {
        "attempted": True,
        "valid": False,
        "reason": "objective-mismatch",
        "computed_objective": 2,
        "reported_objective": 3,
    }
    assert infeasible["valid"] is False
    assert infeasible["reason"] == "unsatisfied-constraint"
    assert infeasible["constraint"] == 1


def test_pb_replay_supports_statements_split_across_lines_and_strict_relations():
    source = (
        "* #variable= 2 #constraint= 2\n"
        "1 x1\n"
        "1 x2 > 0 ;\n"
        "2 x1 -1 ~x2\n"
        "< 2 ;\n"
    )

    result = satpb_verify.verify_pb(source, "v x1 -x2\n", None)

    assert result["valid"] is True
    assert result["computed_objective"] is None


def test_verification_errors_are_inconclusive_not_solver_rejections(tmp_path):
    instance = tmp_path / "nonlinear.opb"
    instance.write_text("1 x1 x2 >= 1 ;\n", encoding="utf-8")

    result = satpb_verify.verify_solution(
        "pb", instance, "s SATISFIABLE\nv x1 x2\n", "SAT", None
    )

    assert result["attempted"] is True
    assert result["valid"] is None
    assert result["reason"] == "verification-error"
    assert "nonlinear" in result["message"]


def test_unknown_and_unsat_claims_do_not_require_assignment_replay(tmp_path):
    missing = tmp_path / "missing.cnf"

    for status in ("UNKNOWN", "UNSAT"):
        assert satpb_verify.verify_solution("sat", missing, "", status, None) == {
            "attempted": False,
            "valid": None,
            "reason": "no-feasible-claim",
        }
