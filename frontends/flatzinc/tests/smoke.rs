use std::fs;
use std::process::Command;

/// Write `src` to a temp `.fzn`, run the solver, and return its stdout.
/// Output is the MiniZinc solver protocol (`name = value;`, `----------`, ...).
fn run(tag: &str, src: &str) -> String {
    run_args(tag, src, &[])
}

/// Like [`run`], but passes extra command-line `flags` before the file.
fn run_args(tag: &str, src: &str, flags: &[&str]) -> String {
    let path = std::env::temp_dir().join(format!("qayd-fzn-{tag}.fzn"));
    fs::write(&path, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_qayd-fzn")).args(flags).arg(&path).output().unwrap();
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn optimum_emits_solution_and_proof_markers() {
    let out = run("opt-markers", "var 0..5: x :: output_var;\nsolve maximize x;\n");
    assert!(out.contains("x = 5;"), "{out}");
    assert!(out.contains("----------"), "{out}");
    assert!(out.contains("=========="), "{out}");
}

#[test]
fn verbose_streams_progress_and_stats_as_comments() {
    let out = run_args("verbose", "var 0..5: x :: output_var;\nsolve maximize x;\n", &["-v"]);
    assert!(out.contains("% o "), "{out}");
    assert!(out.contains("% time="), "{out}");
    assert!(out.contains("x = 5;"), "{out}");
    assert!(out.contains("=========="), "{out}");
}

#[test]
fn time_limit_flag_allows_normal_completion() {
    // A generous limit must not prevent a quick problem from finishing.
    let out = run_args("tlimit", "var 0..5: x :: output_var;\nsolve maximize x;\n", &["-t", "10000"]);
    assert!(out.contains("x = 5;"), "{out}");
    assert!(out.contains("=========="), "{out}");
}

#[test]
fn output_array_prints_bools() {
    let out = run(
        "out-arr",
        "array [1..4] of var bool: b :: output_array([1..2, 1..2]);\n\
         constraint int_eq(b[1], 1);\nconstraint int_eq(b[2], 0);\n\
         constraint int_eq(b[3], 0);\nconstraint int_eq(b[4], 1);\n\
         solve satisfy;\n",
    );
    assert!(out.contains("b = array2d(1..2, 1..2, [true, false, false, true]);"), "{out}");
    assert!(out.contains("----------"), "{out}");
}

#[test]
fn unsat_prints_marker() {
    let out = run("unsat", "var 1..2: x;\nconstraint int_lt(x, 1);\nsolve satisfy;\n");
    assert!(out.contains("=====UNSATISFIABLE====="), "{out}");
}

#[test]
fn reified_eq_is_entailed() {
    let out = run(
        "reif-eq",
        "var 1..3: x;\nvar 0..1: b :: output_var;\nconstraint int_eq(x, 2);\nconstraint int_eq_reif(x, 2, b);\nsolve satisfy;\n",
    );
    assert!(out.contains("b = 1;"), "{out}");
}

#[test]
fn half_reified_eq_forces_consequent() {
    // r -> (x == 5), with r fixed true, must pin x.
    let out = run("imp-eq", "var 1..10: x :: output_var;\nvar 1..1: r;\nconstraint int_eq_imp(x, 5, r);\nsolve satisfy;\n");
    assert!(out.contains("x = 5;"), "{out}");
}

#[test]
fn reified_linear_disentails() {
    // b <-> (x <= 5); x is 7, so b must be 0.
    let out = run(
        "lin-reif",
        "var 0..10: x;\nvar 0..1: b :: output_var;\nconstraint int_eq(x, 7);\nconstraint int_lin_le_reif([1], [x], 5, b);\nsolve satisfy;\n",
    );
    assert!(out.contains("b = 0;"), "{out}");
}

#[test]
fn boolean_logic_predicates() {
    let out = run(
        "bool-ops",
        "var 0..1: a;\nvar 0..1: na :: output_var;\nvar 0..1: c :: output_var;\n\
         constraint int_eq(a, 1);\nconstraint bool_not(a, na);\nconstraint array_bool_and([a, na], c);\nsolve satisfy;\n",
    );
    assert!(out.contains("na = 0;"), "{out}");
    assert!(out.contains("c = 0;"), "{out}");
}

#[test]
fn set_membership_literal_and_reif() {
    let out = run(
        "set-in",
        "var 0..10: x :: output_var;\nvar 0..1: b :: output_var;\n\
         constraint set_in(x, 3..5);\nconstraint int_eq(x, 4);\nconstraint set_in_reif(x, {1, 2}, b);\nsolve satisfy;\n",
    );
    assert!(out.contains("x = 4;"), "{out}");
    assert!(out.contains("b = 0;"), "{out}");
}

#[test]
fn functional_arithmetic() {
    let out = run(
        "arith",
        "var 0..20: p :: output_var;\nvar 0..20: m :: output_var;\n\
         constraint int_times(3, 4, p);\nconstraint int_max(5, 9, m);\nsolve satisfy;\n",
    );
    assert!(out.contains("p = 12;"), "{out}");
    assert!(out.contains("m = 9;"), "{out}");
}

#[test]
fn count_with_variable_target() {
    // fzn_count_eq with a variable count: c = #{ xs == 1 } = 3.
    let out = run(
        "count-var",
        "array [1..4] of var 0..1: xs = [1, 0, 1, 1];\nvar 0..4: c :: output_var;\nconstraint fzn_count_eq(xs, 1, c);\nsolve satisfy;\n",
    );
    assert!(out.contains("c = 3;"), "{out}");
}

#[test]
fn constant_array_as_variable_array() {
    // A parameter array passed where a var array is expected (gecode_int_element).
    let out = run(
        "const-array",
        "array [1..3] of int: a = [5, 6, 7];\nvar 1..3: i;\nvar 0..10: y :: output_var;\n\
         constraint int_eq(i, 3);\nconstraint gecode_int_element(i, 1, a, y);\nsolve satisfy;\n",
    );
    assert!(out.contains("y = 7;"), "{out}");
}

#[test]
fn table_constraint() {
    // (a, b) must be a row of {(1,2), (2,3)}; a = 2 forces b = 3.
    let out = run(
        "table",
        "var 1..3: a;\nvar 1..3: b :: output_var;\nconstraint gecode_table_int([a, b], [1, 2, 2, 3]);\nconstraint int_eq(a, 2);\nsolve satisfy;\n",
    );
    assert!(out.contains("b = 3;"), "{out}");
}

#[test]
fn set_variable_membership_conflict() {
    // 3 in S (set var) forces the membership flag true, contradicting in3 = 0.
    let out = run(
        "set-var",
        "var set of 1..5: s;\nvar 0..1: in3;\nconstraint set_in(3, s);\nconstraint set_in_reif(3, s, in3);\nconstraint int_eq(in3, 0);\nsolve satisfy;\n",
    );
    assert!(out.contains("=====UNSATISFIABLE====="), "{out}");
}

#[test]
fn solves_tiny_fzn() {
    let out = run(
        "smoke",
        "var 1..3: x :: output_var;\nvar 1..3: y;\nconstraint int_ne(x, y);\nconstraint int_lin_eq([1, 1], [x, y], 4);\nsolve satisfy;\n",
    );
    assert!(out.contains("x = "), "{out}");
    assert!(out.contains("----------"), "{out}");
}

#[test]
fn accepts_solve_annotations() {
    let out = run(
        "solve-ann",
        "var 1..3: x :: output_var;\nconstraint int_eq(x, 2);\nsolve :: int_search([x], first_fail, indomain_min, complete) satisfy;\n",
    );
    assert!(out.contains("x = 2;"), "{out}");
}

#[test]
fn supports_explicit_set_domain() {
    let out = run("set-domain", "var {0,2}: x :: output_var;\nconstraint int_gt(x, 0);\nsolve satisfy;\n");
    assert!(out.contains("x = 2;"), "{out}");
}

#[test]
fn supports_array_explicit_set_domain() {
    let out = run("array-set-domain", "array [1..2] of var {0,2}: xs;\nconstraint int_lin_eq([1, 1], xs, 1);\nsolve satisfy;\n");
    assert!(out.contains("=====UNSATISFIABLE====="), "{out}");
}

#[test]
fn supports_array_int_extrema() {
    let out = run(
        "array-extrema",
        "array [1..3] of var int: xs = [1, 4, 2];\nvar 0..10: lo :: output_var;\nvar 0..10: hi :: output_var;\n\
         constraint array_int_minimum(lo, xs);\nconstraint array_int_maximum(hi, xs);\nsolve satisfy;\n",
    );
    assert!(out.contains("lo = 1;"), "{out}");
    assert!(out.contains("hi = 4;"), "{out}");
}

#[test]
fn supports_gecode_element_with_var_array() {
    let out = run(
        "gecode-var-element",
        "array [1..3] of var int: xs = [1, 4, 2];\nvar 1..3: i;\nvar 0..10: y :: output_var;\n\
         constraint int_eq(i, 2);\nconstraint gecode_int_element(i, 1, xs, y);\nsolve satisfy;\n",
    );
    assert!(out.contains("y = 4;"), "{out}");
}

#[test]
fn bool_sum_with_variable_target() {
    // n = number of true bits; forced bits give n = 2.
    let out = run(
        "bool-sum",
        "var 0..1: a;\nvar 0..1: b;\nvar 0..1: c;\nvar 0..3: n :: output_var;\n\
         constraint int_eq(a, 1);\nconstraint int_eq(b, 1);\nconstraint int_eq(c, 0);\n\
         constraint bool_sum_eq([a, b, c], n);\nsolve satisfy;\n",
    );
    assert!(out.contains("n = 2;"), "{out}");
}

#[test]
fn chuffed_value_precede_orders_first_occurrences() {
    // value_precede(1, 2, x): the first 2 cannot appear before the first 1.
    let out = run(
        "value-precede",
        "array [1..3] of var 1..2: x;\nconstraint int_eq(x[1], 2);\n\
         constraint chuffed_value_precede(1, 2, x);\nsolve satisfy;\n",
    );
    assert!(out.contains("=====UNSATISFIABLE====="), "{out}");
}

#[test]
fn array_var_bool_element_lookup() {
    let out = run(
        "bool-element",
        "array [1..3] of var int: xs = [0, 1, 0];\nvar 1..3: i;\nvar 0..1: y :: output_var;\n\
         constraint int_eq(i, 2);\nconstraint array_var_bool_element(i, xs, y);\nsolve satisfy;\n",
    );
    assert!(out.contains("y = 1;"), "{out}");
}

#[test]
fn fzn_circuit_is_one_based() {
    // A 3-node circuit over 1-based successors: x[i] != i and one cycle.
    let out = run(
        "fzn-circuit",
        "array [1..3] of var 1..3: x :: output_array([1..3]);\nconstraint int_eq(x[1], 2);\n\
         constraint fzn_circuit(x);\nsolve satisfy;\n",
    );
    assert!(out.contains("x = array1d(1..3, [2, 3, 1]);"), "{out}");
}

#[test]
fn connected_accepts_a_connected_selection() {
    // Path 1-2-3, all nodes selected: both edges can be selected.
    let out = run(
        "connected-sat",
        "array [1..2] of int: from = [1, 2];\narray [1..2] of int: to = [2, 3];\n\
         array [1..3] of var 0..1: ns;\narray [1..2] of var 0..1: es :: output_array([1..2]);\n\
         constraint int_eq(ns[1], 1);\nconstraint int_eq(ns[2], 1);\nconstraint int_eq(ns[3], 1);\n\
         constraint chuffed_connected(from, to, ns, es);\nsolve satisfy;\n",
    );
    assert!(out.contains("es = array1d(1..2, [1, 1]);"), "{out}");
    assert!(out.contains("----------"), "{out}");
}

#[test]
fn connected_rejects_a_split_selection() {
    // Path 1-2-3 with the middle node removed: nodes 1 and 3 cannot connect.
    let out = run(
        "connected-unsat",
        "array [1..2] of int: from = [1, 2];\narray [1..2] of int: to = [2, 3];\n\
         array [1..3] of var 0..1: ns;\narray [1..2] of var 0..1: es;\n\
         constraint int_eq(ns[1], 1);\nconstraint int_eq(ns[2], 0);\nconstraint int_eq(ns[3], 1);\n\
         constraint chuffed_connected(from, to, ns, es);\nsolve satisfy;\n",
    );
    assert!(out.contains("=====UNSATISFIABLE====="), "{out}");
}

#[test]
fn fzn_member_int_requires_presence() {
    let out = run(
        "member",
        "array [1..3] of var 1..3: x;\nvar 5..9: y;\n\
         constraint int_eq(y, 5);\nconstraint fzn_member_int(x, y);\nsolve satisfy;\n",
    );
    assert!(out.contains("=====UNSATISFIABLE====="), "{out}");
}
