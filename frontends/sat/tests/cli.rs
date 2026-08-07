use std::io::Write;
use std::process::{Command, Output, Stdio};

fn execute(args: &[&str], dimacs: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_qayd-sat"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(dimacs.as_bytes()).unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn competition_exit_codes_follow_orchestrator_status() {
    let sat = execute(&["--competition", "--preprocess", "full"], "p cnf 1 1\n1 0\n");
    assert_eq!(sat.status.code(), Some(10), "{}", String::from_utf8_lossy(&sat.stderr));
    assert!(String::from_utf8_lossy(&sat.stdout).contains("s SATISFIABLE"));

    let unsat = execute(&["--competition", "--no-preprocess"], "p cnf 1 2\n1 0\n-1 0\n");
    assert_eq!(unsat.status.code(), Some(20), "{}", String::from_utf8_lossy(&unsat.stderr));
    assert!(String::from_utf8_lossy(&unsat.stdout).contains("s UNSATISFIABLE"));

    let unknown = execute(&["--competition", "--time", "0"], "p cnf 1 1\n1 0\n");
    assert_eq!(unknown.status.code(), Some(0), "{}", String::from_utf8_lossy(&unknown.stderr));
    assert!(String::from_utf8_lossy(&unknown.stdout).contains("s UNKNOWN"));
}

#[test]
fn verbose_zero_budget_reports_unknown_without_an_engine_report() {
    let output = execute(&["--verbose", "--competition", "--time", "0"], "p cnf 1 1\n1 0\n");
    assert_eq!(output.status.code(), Some(0), "{}", String::from_utf8_lossy(&output.stderr));
    assert!(String::from_utf8_lossy(&output.stdout).contains("s UNKNOWN"));
    assert!(!String::from_utf8_lossy(&output.stderr).contains("panicked"));
}

#[test]
fn linear_backend_remains_available() {
    let output = execute(&["--linear"], "p cnf 2 2\n1 2 0\n-1 2 0\n");
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert!(String::from_utf8_lossy(&output.stdout).contains("s SATISFIABLE"));
}

#[test]
fn proof_still_requires_the_native_backend() {
    let output = execute(&["--linear", "--proof", "unused.drat"], "p cnf 1 1\n1 0\n");
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("SAT proof output requires the native SAT backend"));
}
