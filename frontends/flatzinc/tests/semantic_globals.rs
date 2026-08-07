use std::fs;
use std::process::Command;

fn run(tag: &str, source: &str) -> String {
    let path = std::env::temp_dir().join(format!("qayd-fzn-semantic-{tag}.fzn"));
    fs::write(&path, source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_qayd-fzn")).arg(path).output().unwrap();
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn all_different_and_ordered_use_semantic_globals() {
    let output = run(
        "ordered",
        "array [1..3] of var 1..3: x :: output_array([1..3]);\n\
         constraint all_different_int(x);\n\
         constraint fzn_increasing_int(x);\n\
         solve satisfy;\n",
    );
    assert!(output.contains("x = array1d(1..3, [1, 2, 3]);"), "{output}");
}

#[test]
fn regular_uses_the_semantic_automaton() {
    let output = run(
        "regular",
        "array [1..2] of var 1..2: x = [1, 2] :: output_array([1..2]);\n\
         constraint fzn_regular(x, 2, 2, [2, 0, 0, 1], 1, {1});\n\
         solve satisfy;\n",
    );
    assert!(output.contains("x = array1d(1..2, [1, 2]);"), "{output}");
}

#[test]
fn cumulative_and_no_overlap_keep_infeasibility() {
    let cumulative = run(
        "cumulative-unsat",
        "array [1..2] of var 0..0: s;\n\
         array [1..2] of int: d = [2, 2];\n\
         array [1..2] of int: r = [1, 1];\n\
         constraint fzn_cumulative(s, d, r, 1);\n\
         solve satisfy;\n",
    );
    assert!(cumulative.contains("=====UNSATISFIABLE====="), "{cumulative}");

    let disjunctive = run(
        "disjunctive-unsat",
        "array [1..2] of var 0..0: s;\n\
         array [1..2] of int: d = [2, 2];\n\
         constraint fzn_disjunctive(s, d);\n\
         solve satisfy;\n",
    );
    assert!(disjunctive.contains("=====UNSATISFIABLE====="), "{disjunctive}");
}

#[test]
fn cardinality_and_bin_loads_keep_variable_results() {
    let cardinality = run(
        "gcc",
        "array [1..3] of var 1..2: x = [1, 1, 2];\n\
         array [1..2] of var 0..3: counts :: output_array([1..2]);\n\
         constraint fzn_global_cardinality(x, [1, 2], counts);\n\
         solve satisfy;\n",
    );
    assert!(cardinality.contains("counts = array1d(1..2, [2, 1]);"), "{cardinality}");

    let loads = run(
        "bin-loads",
        "array [1..2] of var 1..2: bins = [1, 2];\n\
         array [1..2] of int: weights = [3, 4];\n\
         array [1..2] of var 0..7: loads :: output_array([1..2]);\n\
         constraint fzn_bin_packing_load(loads, bins, weights);\n\
         solve satisfy;\n",
    );
    assert!(loads.contains("loads = array1d(1..2, [3, 4]);"), "{loads}");
}

#[test]
fn arg_max_preserves_first_maximum_tie_breaking() {
    let output = run(
        "arg-max",
        "array [1..3] of var 0..10: x = [4, 7, 7];\n\
         var 1..3: i :: output_var;\n\
         constraint gecode_maximum_arg_int_offset(x, 1, i);\n\
         solve satisfy;\n",
    );
    assert!(output.contains("i = 2;"), "{output}");
}

#[test]
fn lex_and_boolean_clause_keep_conflicts() {
    let lex = run(
        "lex-unsat",
        "array [1..2] of var 0..2: x = [2, 0];\n\
         array [1..2] of var 0..2: y = [1, 2];\n\
         constraint fzn_lex_lesseq_int(x, y);\n\
         solve satisfy;\n",
    );
    assert!(lex.contains("=====UNSATISFIABLE====="), "{lex}");

    let clause = run(
        "clause-unsat",
        "array [1..2] of var bool: b = [false, false];\n\
         constraint bool_clause(b, []);\n\
         solve satisfy;\n",
    );
    assert!(clause.contains("=====UNSATISFIABLE====="), "{clause}");
}
