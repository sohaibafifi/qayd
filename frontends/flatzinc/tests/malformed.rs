use std::fs;
use std::process::Command;

/// Write `src` to a temp `.fzn`, run the solver, and assert it reports a
/// wrong-arity error (typed message, non-zero exit) instead of panicking.
fn err_on(tag: &str, constraint: &str) {
    let src = format!("var 1..3: x;\nvar 1..3: y;\n{constraint}\nsolve satisfy;\n");
    let path = std::env::temp_dir().join(format!("qayd-fzn-bad-{tag}.fzn"));
    fs::write(&path, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_qayd-fzn")).arg(&path).output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "expected failure for `{constraint}`, stdout: {}", String::from_utf8_lossy(&out.stdout));
    assert!(!stderr.contains("panicked"), "panic for `{constraint}`: {stderr}");
    assert!(stderr.contains("expects") || stderr.contains("error"), "no typed error for `{constraint}`: {stderr}");
}

#[test]
fn binary_handler_with_one_arg_errors() {
    err_on("int-eq-1", "constraint int_eq(x);");
    err_on("bool-le-1", "constraint bool_le(x);");
    err_on("int-abs-1", "constraint int_abs(x);");
}

#[test]
fn ternary_handler_with_two_args_errors() {
    err_on("int-plus-2", "constraint int_plus(x, y);");
    err_on("bool-and-2", "constraint bool_and(x, y);");
    err_on("precede-2", "constraint fzn_int_precede(x, 1);");
}

#[test]
fn set_in_without_set_errors() {
    err_on("set-in-1", "constraint set_in(x);");
}
