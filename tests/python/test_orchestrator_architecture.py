"""The Python frontend must stay above the semantic orchestration boundary."""

from pathlib import Path


def test_python_frontend_uses_only_semantic_solver_services():
    source = (Path(__file__).parents[2] / "src" / "frontends" / "python.rs").read_text()

    for required in (
        "ModelPackage",
        "SolveRequest",
        "solve_model_with_external_stop",
        "SemanticSolveSession",
        "count_model_solutions_with_external_stop",
        "extract_model_mus_with_external_stop",
    ):
        assert required in source, f"Python frontend is missing canonical service {required}"

    for forbidden in (
        "crate::constraints",
        "crate::engines",
        "crate::problem",
        "crate::search",
        "crate::store",
        "Solver",
        "LocalSearchSpec",
        "VarId",
        "SolveBudget",
        "execute_specialized",
        "PhysicalSolve",
        "solve_physical",
        "materialize_objectives",
    ):
        assert forbidden not in source, f"Python frontend bypasses the semantic orchestrator through {forbidden}"
