import csv
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "bench" / "common" / "compare.py"


def write_csv(tmp_path, name, rows):
    p = tmp_path / name
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["instance", "status", "obj", "time", "timedout", "dir"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return str(p)


def n_contradictions(tmp_path, a_rows, b_rows):
    out = compare_output(tmp_path, a_rows, b_rows)
    line = next(l for l in out.splitlines() if "CONTRADICTIONS" in l)
    return int(line.rsplit(":", 1)[1])


def compare_output(tmp_path, a_rows, b_rows, timeout=10):
    a = write_csv(tmp_path, "a.csv", a_rows)
    b = write_csv(tmp_path, "b.csv", b_rows)
    return subprocess.run(
        [sys.executable, str(SCRIPT), a, b, "--timeout", str(timeout)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def row(inst, status, obj, d="min"):
    return {"instance": inst, "status": status, "obj": obj, "time": "1.0", "timedout": "0", "dir": d}


def test_optimum_differing_obj_is_contradiction(tmp_path):
    assert n_contradictions(tmp_path, [row("x", "OPTIMUM", 10)], [row("x", "OPTIMUM", 12)]) == 1


def test_optimum_equal_obj_not_contradiction(tmp_path):
    assert n_contradictions(tmp_path, [row("x", "OPTIMUM", 10)], [row("x", "OPTIMUM", 10)]) == 0


def test_sat_vs_unsat_still_contradiction(tmp_path):
    assert n_contradictions(tmp_path, [row("x", "SAT", "")], [row("x", "UNSAT", "")]) == 1


def test_sat_strictly_better_than_claimed_optimum_min(tmp_path):
    assert n_contradictions(tmp_path, [row("x", "OPTIMUM", 10, "min")], [row("x", "SAT", 8, "min")]) == 1


def test_sat_worse_than_optimum_not_contradiction(tmp_path):
    assert n_contradictions(tmp_path, [row("x", "OPTIMUM", 10, "min")], [row("x", "SAT", 12, "min")]) == 0


def test_sat_better_max_direction(tmp_path):
    assert n_contradictions(tmp_path, [row("x", "OPTIMUM", 10, "max")], [row("x", "SAT", 12, "max")]) == 1


def test_sat_better_unknown_direction_not_flagged(tmp_path):
    assert n_contradictions(tmp_path, [row("x", "OPTIMUM", 10, "?")], [row("x", "SAT", 8, "?")]) == 0


def test_decision_sat_counts_as_solved(tmp_path):
    out = compare_output(
        tmp_path,
        [row("decision", "SAT", "", "?")],
        [row("decision", "UNKNOWN", "", "?")],
    )

    assert "  qayd solved : 1" in out
    assert "  baseline solved : 0" in out
    assert "  PAR2 qayd : 1.0" in out


def test_optimization_sat_is_only_an_incumbent_and_is_charged_unsolved(tmp_path):
    out = compare_output(
        tmp_path,
        [row("cop", "SAT", 12, "min")],
        [row("cop", "UNKNOWN", "", "min")],
    )

    assert "  qayd solved : 0" in out
    assert "  qayd incumbent found : 1" in out
    assert "  PAR2 qayd : 20.0" in out


def test_optimum_and_unsat_count_as_solved(tmp_path):
    out = compare_output(
        tmp_path,
        [row("optimal", "OPTIMUM", 7), row("infeasible", "UNSAT", "")],
        [row("optimal", "UNKNOWN", ""), row("infeasible", "UNKNOWN", "")],
    )

    assert "  qayd solved : 2" in out
    assert "  baseline solved : 0" in out
    assert "  PAR2 qayd : 2.0" in out


def test_minimization_incumbent_quality_prefers_the_lower_value(tmp_path):
    out = compare_output(
        tmp_path,
        [row("min-cop", "SAT", 8, "min")],
        [row("min-cop", "SAT", 10, "min")],
    )

    assert "objective quality (incumbent by both): same 0 | qayd better 1 | qayd worse 0" in out


def test_maximization_incumbent_quality_prefers_the_higher_value(tmp_path):
    out = compare_output(
        tmp_path,
        [row("max-cop", "SAT", 12, "max")],
        [row("max-cop", "SAT", 10, "max")],
    )

    assert "objective quality (incumbent by both): same 0 | qayd better 1 | qayd worse 0" in out
