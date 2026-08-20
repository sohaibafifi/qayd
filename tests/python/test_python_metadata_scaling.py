"""Regression coverage for Python package metadata preparation."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

cp = pytest.importorskip("qayd")

ROOT = Path(__file__).resolve().parents[2]


def test_large_compact_schedule_metadata_preparation_does_not_regress_to_quadratic():
    """Keep the old output-membership scan from returning unnoticed.

    The subprocess isolates the deliberately large allocation from the pytest
    worker.  Ten seconds is intentionally much larger than the linear path
    needs, while two preparations of 150,000 outputs make the former repeated
    ``Vec.contains`` scan prohibitive.
    """

    program = textwrap.dedent(
        """
        import qayd as cp

        interval_count = 150_000
        model = cp.Model()
        model.schedule_intervals([1] * interval_count, interval_count)
        shape = (model.num_vars, model.num_constraints, model.objective_sense, model.objective_expr)

        first = model.solve(engine="ls", threads=1, seed=0, time_limit=0)
        second = model.solve(engine="ls", threads=1, seed=0, time_limit=0)

        assert first.status == "UNKNOWN"
        assert second.status == "UNKNOWN"
        assert (model.num_vars, model.num_constraints, model.objective_sense, model.objective_expr) == shape
        """
    )

    try:
        completed = subprocess.run(
            [sys.executable, "-c", program],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("large compact-schedule metadata preparation exceeded 10 seconds")

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_two_successive_schedule_solves_do_not_accumulate_semantic_state():
    model = cp.Model()
    intervals = model.schedule_intervals([2, 1, 3, 2], horizon=8)
    for before, after in zip(intervals, intervals[1:]):
        model.precedence(before, after)

    shape = (model.num_vars, model.num_constraints, model.objective_sense, model.objective_expr, repr(model))

    first = model.solve(engine="ls", threads=1, seed=11, max_iterations=1)
    after_first = (model.num_vars, model.num_constraints, model.objective_sense, model.objective_expr, repr(model))
    second = model.solve(engine="ls", threads=1, seed=11, max_iterations=1)
    after_second = (model.num_vars, model.num_constraints, model.objective_sense, model.objective_expr, repr(model))

    assert first.status == second.status == "SATISFIABLE"
    assert first.objective == second.objective == 8
    assert first.starts == second.starts == [0, 2, 3, 6]
    assert first.presences == second.presences == [True, True, True, True]
    assert shape == after_first == after_second
