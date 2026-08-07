"""Parsing and process-status checks for the linear backend benchmark."""

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[2] / "bench" / "linear_backend_probe.py"


def load_probe_module():
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec = importlib.util.spec_from_file_location("qayd_linear_backend_probe", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_highs_time_limit_with_incumbent_is_feasible_not_error():
    probe = load_probe_module()
    parsed = probe.parse_highs(
        """
Solving report
  Status            Time limit reached
  Primal bound      40405
  Dual bound        999
  Gap               97.53%
  Timing            10.00
  Nodes             34308
""",
        "",
    )

    assert parsed["status"] == "SATISFIABLE"
    assert parsed["objective"] == 40405
    assert parsed["dual_bound"] == 999
    assert not probe.process_failed(parsed, {"return_code": 1, "timed_out": False})


def test_missing_solver_status_with_nonzero_exit_is_error():
    probe = load_probe_module()
    parsed = probe.parse_highs("", "fatal: model could not be read")

    assert parsed["status"] == "UNKNOWN"
    assert probe.process_failed(parsed, {"return_code": 1, "timed_out": False})


def test_external_timeout_is_always_error():
    probe = load_probe_module()
    parsed = probe.parse_amthal("status: TIME_LIMIT\nobjective: 12\n", "")

    assert parsed["status"] == "SATISFIABLE"
    assert probe.process_failed(parsed, {"return_code": 0, "timed_out": True})
