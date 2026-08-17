"""CSPLib models implemented with the Qayd Python API.

Use ``python -m examples.python.csplib list`` to inspect coverage and
``python -m examples.python.csplib probNNN`` to run a model.
"""

from .catalog import ALL_PROBLEM_IDS, IMPLEMENTATIONS, normalize_problem_id

__all__ = ["ALL_PROBLEM_IDS", "IMPLEMENTATIONS", "normalize_problem_id"]
