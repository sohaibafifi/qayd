#[test]
fn frontend_builds_and_solves_only_the_canonical_model_package() {
    let model = include_str!("../src/model.rs");
    let parser = include_str!("../src/parse.rs");
    let solve = include_str!("../src/solve.rs");

    assert!(model.contains("ModelPackage"));
    assert!(model.contains("IntVarRef"));
    assert!(model.contains("Constraint::"));
    assert!(solve.contains("qayd::solve(&package"));
    assert!(solve.contains("SolveRequest"));
    assert!(solve.contains("SolveResult"));

    for (surface, source) in [("model", model), ("parser", parser), ("solve", solve)] {
        for forbidden in [
            "qayd::constraints",
            "qayd::engines",
            "qayd::problem",
            "qayd::Solver",
            "LocalSearchSpec",
            "solve_compiled_problem",
            "solve_physical",
            "execute_specialized",
        ] {
            assert!(!source.contains(forbidden), "FlatZinc {surface} bypasses the canonical package through {forbidden}");
        }
    }
}
