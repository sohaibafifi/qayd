use std::fs;
use std::process::{Command, Output};

/// Write `src` to a temp `.opb`, run the solver, and return its stdout.
fn run(tag: &str, src: &str) -> String {
    String::from_utf8(execute(tag, src, &[]).stdout).unwrap()
}

fn execute(tag: &str, src: &str, args: &[&str]) -> Output {
    let path = std::env::temp_dir().join(format!("qayd-pb-{tag}.opb"));
    fs::write(&path, src).unwrap();
    Command::new(env!("CARGO_BIN_EXE_qayd-pb")).args(args).arg(&path).output().unwrap()
}

#[test]
fn execution_is_routed_through_the_semantic_orchestrator() {
    let source = include_str!("../src/main.rs");
    assert!(source.contains("ModelPackage"));
    assert!(source.contains("solve_model_with_stop"));
    assert!(source.contains("SolveRequest"));
    assert!(source.contains("SolveResult"));
    assert!(!source.contains("Ok(SolveStatus::Unsupported)"));
    for forbidden in
        ["Problem", "Solver", "SolveBudget", "solve_compiled_problem", "first_solution", "optimize_with", "thread::spawn", "std::thread"]
    {
        assert!(!source.contains(forbidden), "PB CLI bypasses the semantic orchestrator through {forbidden}");
    }
}

#[test]
fn max_objective_is_maximized() {
    let out = run("max", "* #variable= 2 #constraint= 1\nmax: 1 x1 1 x2 ;\n1 x1 1 x2 >= 1 ;\n");
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.lines().any(|l| l == "o 2"), "{out}");
    assert!(out.contains("v x1 x2"), "{out}");
}

#[test]
fn min_objective_is_minimized() {
    let out = run("min", "* #variable= 2 #constraint= 1\nmin: 1 x1 1 x2 ;\n1 x1 1 x2 >= 1 ;\n");
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.lines().any(|l| l == "o 1"), "{out}");
}

#[test]
fn negated_literal_objective_restores_its_constant() {
    let out = run("objective-constant", "* #variable= 1 #constraint= 0\nmin: 5 ~x1 2 x1 ;\n");
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.lines().any(|line| line == "o 2"), "{out}");
    assert!(out.contains("v x1"), "{out}");
}

#[test]
fn constant_objective_remains_an_optimization_problem() {
    let out = run("constant-objective", "* #variable= 1 #constraint= 0\nmin: 5 ~x1 5 x1 ;\n");
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert_eq!(out.lines().filter(|line| *line == "o 5").count(), 1, "{out}");
}

#[test]
fn empty_objectives_remain_optimization_problems() {
    for (sense, tag) in [("min", "empty-min"), ("max", "empty-max")] {
        let source = format!("* #variable= 1 #constraint= 0\n{sense}: ;\n");
        let output = execute(tag, &source, &[]);
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert_eq!(output.status.code(), Some(30), "{stdout}");
        assert!(stdout.lines().any(|line| line == "s OPTIMUM FOUND"), "{stdout}");
        assert_eq!(stdout.lines().filter(|line| *line == "o 0").count(), 1, "{stdout}");
    }
}

#[test]
fn objective_values_are_not_limited_to_i32() {
    let out = run("wide-objective", "* #variable= 1 #constraint= 1\nmin: 3000000000 x1 ;\n1 x1 >= 1 ;\n");
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.lines().any(|line| line == "o 3000000000"), "{out}");
}

#[test]
fn zero_timeout_is_reported_as_unknown() {
    let output = execute("zero-timeout", "* #variable= 1 #constraint= 0\n1 x1 >= 0 ;\n", &["-t", "0"]);
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(output.status.code(), Some(0), "{stdout}");
    assert!(stdout.lines().any(|line| line == "s UNKNOWN"), "{stdout}");
    assert!(!stdout.lines().any(|line| line.starts_with('v')), "{stdout}");
}

#[test]
fn competition_exit_codes_follow_orchestrator_status() {
    let sat = execute("sat-status", "* #variable= 1 #constraint= 1\n1 x1 >= 1 ;\n", &[]);
    assert_eq!(sat.status.code(), Some(10), "{}", String::from_utf8_lossy(&sat.stdout));

    let unsat = execute("unsat-status", "* #variable= 1 #constraint= 2\n1 x1 >= 1 ;\n1 x1 <= 0 ;\n", &[]);
    assert_eq!(unsat.status.code(), Some(20), "{}", String::from_utf8_lossy(&unsat.stdout));

    let optimum = execute("optimum-status", "* #variable= 1 #constraint= 0\nmin: 1 x1 ;\n", &[]);
    assert_eq!(optimum.status.code(), Some(30), "{}", String::from_utf8_lossy(&optimum.stdout));
}
