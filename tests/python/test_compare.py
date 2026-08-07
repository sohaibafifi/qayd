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
    a = write_csv(tmp_path, "a.csv", a_rows)
    b = write_csv(tmp_path, "b.csv", b_rows)
    out = subprocess.run(
        [sys.executable, str(SCRIPT), a, b], capture_output=True, text=True, check=True
    ).stdout
    line = next(l for l in out.splitlines() if "CONTRADICTIONS" in l)
    return int(line.rsplit(":", 1)[1])


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
