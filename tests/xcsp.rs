//! End-to-end XCSP3 front-end tests: parse → build → solve, on hand-written
//! instances with known status.

use std::sync::atomic::AtomicBool;
use std::time::Duration;

use qayd::frontends::xcsp::{run, run_to_with_options, Mode, RunOptions};

#[test]
fn execution_is_routed_through_the_orchestrator() {
    let runtime = include_str!("../src/frontends/xcsp/mod.rs");
    let callback = include_str!("../src/frontends/xcsp/callback.rs");
    let source = format!("{runtime}\n{callback}");
    assert!(runtime.contains("solve_model_with_external_stop"));
    assert!(runtime.contains("SolveRequest"));
    assert!(runtime.contains("SolveResult"));
    assert!(runtime.contains("SolveMode::LocalSearch"));
    assert!(callback.contains("ModelPackage"));
    assert!(callback.contains("IntVarRef"));
    for forbidden in [
        "crate::parallel",
        "crate::constraints",
        "crate::engines",
        "crate::search",
        "solve_compiled_problem",
        "SolveBudget",
        "LocalSearchSpec",
        "Problem",
        "Solver",
        "Store",
        "VarId",
        "solve_cop(",
        "solve_csp(",
        "solve_ls(",
        "optimize_seeded(",
    ] {
        assert!(!source.contains(forbidden), "XCSP owns forbidden physical execution surface `{forbidden}`");
    }
}

#[test]
fn zero_workers_is_rejected_by_the_canonical_request_validator() {
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><var id="x"> 0..1 </var></variables>
        <constraints></constraints>
      </instance>"#;
    let error =
        run_to_with_options(xml, false, &AtomicBool::new(false), &mut Vec::new(), RunOptions { workers: 0, ..RunOptions::default() })
            .unwrap_err();
    assert!(error.contains("threads must be positive"), "{error}");
}

#[test]
fn local_search_mode_uses_the_canonical_incumbent_protocol() {
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables><var id="x"> 7 </var></variables>
        <constraints></constraints>
        <objectives><minimize>x</minimize></objectives>
      </instance>"#;
    let mut out = Vec::new();
    run_to_with_options(xml, true, &AtomicBool::new(false), &mut out, RunOptions { mode: Mode::Ls, ..RunOptions::default() }).unwrap();
    let out = String::from_utf8(out).unwrap();
    assert!(out.contains("c mode ls"), "{out}");
    assert!(out.contains("c effort incumbent"), "{out}");
    assert!(out.contains("s SATISFIABLE"), "{out}");
    assert!(!out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 7"), "{out}");
}

#[test]
fn local_search_mode_supports_a_satisfaction_problem() {
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><var id="x"> 0..1 </var></variables>
        <constraints></constraints>
      </instance>"#;
    let mut out = Vec::new();
    run_to_with_options(
        xml,
        false,
        &AtomicBool::new(false),
        &mut out,
        RunOptions { mode: Mode::Ls, time_limit: Some(Duration::from_millis(100)), ..RunOptions::default() },
    )
    .unwrap();
    let out = String::from_utf8(out).unwrap();
    assert!(out.contains("s SATISFIABLE"), "{out}");
    assert!(!out.contains("s OPTIMUM FOUND"), "{out}");
}

#[test]
fn alldifferent_sum_is_sat() {
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables>
          <array id="x" size="[3]"> 1..3 </array>
        </variables>
        <constraints>
          <allDifferent><list>x[]</list></allDifferent>
          <sum><list>x[]</list><condition>(eq,6)</condition></sum>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s SATISFIABLE"), "{out}");
    assert!(out.contains("v <instantiation>"), "{out}");
    assert!(out.contains("v <list>\nv x[0] x[1] x[2]\nv </list>"), "{out}");
    assert!(out.contains("v <values>"), "{out}");
    assert!(out.contains("v </instantiation>"), "{out}");
}

#[test]
fn pigeonhole_is_unsat() {
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><array id="x" size="[3]"> 1..2 </array></variables>
        <constraints><allDifferent><list>x[]</list></allDifferent></constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s UNSATISFIABLE"), "{out}");
}

#[test]
fn csp_learning_proves_unsat() {
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><array id="x" size="[3]"> 1..2 </array></variables>
        <constraints><allDifferent><list>x[]</list></allDifferent></constraints>
      </instance>"#;
    let mut out = Vec::new();
    run_to_with_options(xml, true, &AtomicBool::new(false), &mut out, RunOptions::default()).unwrap();
    let out = String::from_utf8(out).unwrap();
    assert!(out.contains("c csp learning true"), "{out}");
    assert!(out.contains("s UNSATISFIABLE"), "{out}");
}

#[test]
fn unsat_core_is_minimal_over_xcsp_source_constraints() {
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables>
          <var id="x"> 0..1 </var>
          <var id="y"> 0..1 </var>
        </variables>
        <constraints>
          <intension> eq(x,0) </intension>
          <intension> eq(x,1) </intension>
          <intension> eq(y,0) </intension>
        </constraints>
      </instance>"#;
    let mut out = Vec::new();
    run_to_with_options(xml, false, &AtomicBool::new(false), &mut out, RunOptions { core: true, ..RunOptions::default() }).unwrap();
    let out = String::from_utf8(out).unwrap();
    assert!(out.contains("s UNSATISFIABLE"), "{out}");
    assert!(out.contains("c core 2 constraint(s)"), "{out}");
    assert!(out.contains("c core-constraint #0 constraint"), "{out}");
    assert!(out.contains("c core-constraint #1 constraint"), "{out}");
    assert!(!out.contains("c core-constraint #2 constraint"), "{out}");
}

#[test]
fn core_option_does_not_emit_a_core_for_sat() {
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><var id="x"> 0..1 </var></variables>
        <constraints><intension> eq(x,0) </intension></constraints>
      </instance>"#;
    let mut out = Vec::new();
    run_to_with_options(xml, false, &AtomicBool::new(false), &mut out, RunOptions { core: true, ..RunOptions::default() }).unwrap();
    let out = String::from_utf8(out).unwrap();
    assert!(out.contains("s SATISFIABLE"), "{out}");
    assert!(!out.contains("c core"), "{out}");
}

#[test]
fn unsat_core_handles_a_refutation_that_requires_search() {
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables>
          <var id="a"> 0..1 </var>
          <var id="b"> 0..1 </var>
        </variables>
        <constraints>
          <intension> or(eq(a,1),eq(b,1)) </intension>
          <intension> or(eq(a,1),eq(b,0)) </intension>
          <intension> or(eq(a,0),eq(b,1)) </intension>
          <intension> or(eq(a,0),eq(b,0)) </intension>
        </constraints>
      </instance>"#;
    let mut out = Vec::new();
    run_to_with_options(xml, false, &AtomicBool::new(false), &mut out, RunOptions { core: true, ..RunOptions::default() }).unwrap();
    let out = String::from_utf8(out).unwrap();
    assert!(out.contains("s UNSATISFIABLE"), "{out}");
    assert!(out.contains("c core 4 constraint(s)"), "{out}");
}

#[test]
fn csp_no_learning_matches_learning_route() {
    // A unique-solution CSP: the default learning route and the `--no-learn-csp`
    // chronological-DFS route must agree on the solution.
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><array id="x" size="[3]"> 0..5 </array></variables>
        <constraints>
          <ordered><list>x[]</list><lengths>2 3</lengths><operator>le</operator></ordered>
        </constraints>
      </instance>"#;
    let solve = |no_learn_csp| {
        let mut out = Vec::new();
        run_to_with_options(xml, true, &AtomicBool::new(false), &mut out, RunOptions { no_learn_csp, ..RunOptions::default() }).unwrap();
        String::from_utf8(out).unwrap()
    };
    let learning = solve(false);
    let plain = solve(true);
    assert!(learning.contains("c csp learning true"), "{learning}");
    assert!(plain.contains("c csp learning false"), "{plain}");
    assert!(plain.contains("s SATISFIABLE"), "{plain}");
    assert!(plain.contains("v 0 2 5"), "{plain}");
    let values = |o: &str| o.lines().filter(|l| l.starts_with("v ")).collect::<Vec<_>>().join("\n");
    assert_eq!(values(&learning), values(&plain), "learning:\n{learning}\nplain:\n{plain}");
}

#[test]
fn csp_portfolio_is_sat() {
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables>
          <array id="x" size="[3]"> 1..3 </array>
        </variables>
        <constraints>
          <allDifferent><list>x[]</list></allDifferent>
          <sum><list>x[]</list><condition>(eq,6)</condition></sum>
        </constraints>
      </instance>"#;
    let mut out = Vec::new();
    run_to_with_options(xml, true, &AtomicBool::new(false), &mut out, RunOptions { workers: 4, ..RunOptions::default() }).unwrap();
    let out = String::from_utf8(out).unwrap();
    assert!(out.contains("c workers 4"), "{out}");
    assert!(out.contains("s SATISFIABLE"), "{out}");
    assert!(out.contains("v <instantiation>"), "{out}");
}

#[test]
fn csp_portfolio_proves_unsat() {
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><array id="x" size="[3]"> 1..2 </array></variables>
        <constraints><allDifferent><list>x[]</list></allDifferent></constraints>
      </instance>"#;
    let mut out = Vec::new();
    run_to_with_options(xml, false, &AtomicBool::new(false), &mut out, RunOptions { workers: 4, ..RunOptions::default() }).unwrap();
    let out = String::from_utf8(out).unwrap();
    assert!(out.contains("s UNSATISFIABLE"), "{out}");
}

#[test]
fn extension_supports_is_sat() {
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><var id="a"> 0..2 </var><var id="b"> 0..2 </var></variables>
        <constraints>
          <extension><list>a b</list><supports>(0,1)(1,2)(2,0)</supports></extension>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s SATISFIABLE"), "{out}");
}

#[test]
fn group_extension_supports_are_sat() {
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><array id="x" size="[3]"> 0..2 </array></variables>
        <constraints>
          <group>
            <extension><list>%0 %1</list><supports>(0,1)(1,2)</supports></extension>
            <args>x[0] x[1]</args>
            <args>x[1] x[2]</args>
          </group>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s SATISFIABLE"), "{out}");
    assert!(out.contains("v 0 1 2"), "{out}");
}

#[test]
fn grouped_sign_products_are_batched() {
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables>
          <array id="x" size="[3]"> -1 1 </array>
          <array id="y" size="[2]"> -1 1 </array>
        </variables>
        <constraints>
          <group>
            <intension> eq(%0,mul(%1,%2)) </intension>
            <args>y[0] x[0] x[1]</args>
            <args>y[1] x[1] x[2]</args>
          </group>
          <instantiation>
            <list>x[]</list>
            <values>1 -1 -1</values>
          </instantiation>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s SATISFIABLE"), "{out}");
    assert!(out.contains("v 1 -1 -1 -1 1"), "{out}");
}

#[test]
fn unconstrained_cells_use_stars_in_output() {
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><array id="x" size="[3]"> -1 1 </array></variables>
        <constraints>
          <instantiation><list>x[0]</list><values>1</values></instantiation>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s SATISFIABLE"), "{out}");
    assert!(out.contains("v 1 * *"), "{out}");
}

#[test]
fn unconstrained_wide_range_uses_bounds_storage() {
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><var id="x"> 0..20000001 </var></variables>
        <constraints></constraints>
      </instance>"#;
    let mut out = Vec::new();
    run_to_with_options(xml, true, &AtomicBool::new(false), &mut out, RunOptions::default()).unwrap();
    let out = String::from_utf8(out).unwrap();
    assert!(out.contains("c bounds domains 1"), "{out}");
    assert!(out.contains("s SATISFIABLE"), "{out}");
    assert!(out.contains("v *"), "{out}");
}

#[test]
fn parallel_wide_objective_uses_lazy_atoms() {
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables><var id="x"> 0..20000001 </var></variables>
        <constraints></constraints>
        <objectives><minimize>x</minimize></objectives>
      </instance>"#;
    let mut out = Vec::new();
    run_to_with_options(xml, true, &AtomicBool::new(false), &mut out, RunOptions { workers: 2, split: true, ..RunOptions::default() })
        .unwrap();
    let out = String::from_utf8(out).unwrap();
    assert!(out.contains("c bounds domains 1"), "{out}");
    assert!(out.contains("c workers 2"), "{out}");
    assert!(out.contains("c split jobs "), "{out}");
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 0"), "{out}");
    assert!(out.contains("c incumbent 0 source integer-exact"), "{out}");
}

fn queens_xml(n: usize) -> String {
    let mut cons = String::new();
    for i in 0..n {
        for j in (i + 1)..n {
            cons += &format!("<intension>ne(q[{i}],q[{j}])</intension>");
            let k = j - i;
            cons += &format!("<intension>ne(dist(q[{i}],q[{j}]),{k})</intension>");
        }
    }
    format!(
        r#"<instance format="XCSP3" type="CSP">
             <variables><array id="q" size="[{n}]"> 0..{hi} </array></variables>
             <constraints>{cons}</constraints>
           </instance>"#,
        hi = n - 1
    )
}

#[test]
fn intension_queens() {
    assert!(run(&queens_xml(4)).unwrap().contains("s SATISFIABLE"));
    assert!(run(&queens_xml(8)).unwrap().contains("s SATISFIABLE"));
    // 2- and 3-queens are unsatisfiable.
    assert!(run(&queens_xml(2)).unwrap().contains("s UNSATISFIABLE"));
    assert!(run(&queens_xml(3)).unwrap().contains("s UNSATISFIABLE"));
}

#[test]
fn minimize_sum_finds_optimum() {
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[3]"> 0..5 </array></variables>
        <constraints><allDifferent><list>x[]</list></allDifferent></constraints>
        <objectives><minimize type="sum"><list>x[]</list></minimize></objectives>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 3"), "{out}"); // 0 + 1 + 2
    assert!(out.contains("v <list>\nv x[0] x[1] x[2]\nv </list>"), "{out}");
}

#[test]
fn parallel_minimize_sum_finds_optimum() {
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[3]"> 0..5 </array></variables>
        <constraints><allDifferent><list>x[]</list></allDifferent></constraints>
        <objectives><minimize type="sum"><list>x[]</list></minimize></objectives>
      </instance>"#;
    let mut out = Vec::new();
    run_to_with_options(
        xml,
        true,
        &AtomicBool::new(false),
        &mut out,
        RunOptions {
            seed: 17,
            workers: 2,
            split: false,
            probes: 0,
            lns: 0,
            no_learn_csp: false,
            mem_limit: None,
            mode: Mode::Default,
            ..RunOptions::default()
        },
    )
    .unwrap();
    let out = String::from_utf8(out).unwrap();
    assert!(out.contains("c seed 17"), "{out}");
    assert!(out.contains("c workers 2"), "{out}");
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 3"), "{out}");
    assert!(out.contains("c incumbent 3 source integer-exact"), "{out}");
}

#[test]
fn parallel_probe_minimize_sum_finds_optimum() {
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[3]"> 0..5 </array></variables>
        <constraints><allDifferent><list>x[]</list></allDifferent></constraints>
        <objectives><minimize type="sum"><list>x[]</list></minimize></objectives>
      </instance>"#;
    let mut out = Vec::new();
    run_to_with_options(
        xml,
        true,
        &AtomicBool::new(false),
        &mut out,
        RunOptions { seed: 17, workers: 2, probes: 1, ..RunOptions::default() },
    )
    .unwrap();
    let out = String::from_utf8(out).unwrap();
    assert!(out.contains("c workers 2"), "{out}");
    assert!(out.contains("c probes attempts "), "{out}");
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 3"), "{out}");
}

#[test]
fn parallel_probe_maximize_sum_finds_optimum() {
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[3]"> 0..5 </array></variables>
        <constraints><allDifferent><list>x[]</list></allDifferent></constraints>
        <objectives><maximize type="sum"><list>x[]</list></maximize></objectives>
      </instance>"#;
    let mut out = Vec::new();
    run_to_with_options(
        xml,
        true,
        &AtomicBool::new(false),
        &mut out,
        RunOptions { seed: 17, workers: 2, probes: 1, ..RunOptions::default() },
    )
    .unwrap();
    let out = String::from_utf8(out).unwrap();
    assert!(out.contains("c workers 2"), "{out}");
    assert!(out.contains("c probes attempts "), "{out}");
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 12"), "{out}");
}

#[test]
fn parallel_group_extension_finds_optimum() {
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[3]"> 0..2 </array></variables>
        <constraints>
          <group>
            <extension><list>%0 %1</list><supports>(0,1)(1,2)</supports></extension>
            <args>x[0] x[1]</args>
            <args>x[1] x[2]</args>
          </group>
        </constraints>
        <objectives><minimize>x[0]</minimize></objectives>
      </instance>"#;
    let mut out = Vec::new();
    run_to_with_options(xml, false, &AtomicBool::new(false), &mut out, RunOptions { seed: 19, workers: 2, ..RunOptions::default() })
        .unwrap();
    let out = String::from_utf8(out).unwrap();
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 0"), "{out}");
    assert!(out.contains("v 0 1 2"), "{out}");
}

#[test]
fn parallel_pigeonhole_is_unsat() {
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><array id="x" size="[3]"> 1..2 </array></variables>
        <constraints><allDifferent><list>x[]</list></allDifferent></constraints>
      </instance>"#;
    let mut out = Vec::new();
    run_to_with_options(xml, false, &AtomicBool::new(false), &mut out, RunOptions { seed: 11, workers: 2, ..RunOptions::default() })
        .unwrap();
    let out = String::from_utf8(out).unwrap();
    assert!(out.contains("s UNSATISFIABLE"), "{out}");
}

#[test]
fn maximize_sum_finds_optimum() {
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[3]"> 0..5 </array></variables>
        <constraints><allDifferent><list>x[]</list></allDifferent></constraints>
        <objectives><maximize type="sum"><list>x[]</list></maximize></objectives>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 12"), "{out}"); // 5 + 4 + 3
}

#[test]
fn negative_table_cop_finds_optimum() {
    // Conflict (negative) table forbidding the diagonal: the local-search engine
    // must model this as a NegExtension violation. Run with 2 workers so the
    // LS fast-incumbent worker exercises it (under the debug score oracle) while
    // the exact worker proves the optimum.
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[2]"> 0..2 </array></variables>
        <constraints>
          <extension>
            <list> x[0] x[1] </list>
            <conflicts> (0,0)(1,1)(2,2) </conflicts>
          </extension>
        </constraints>
        <objectives><minimize type="sum"><list>x[]</list></minimize></objectives>
      </instance>"#;
    let mut out = Vec::new();
    run_to_with_options(xml, false, &AtomicBool::new(false), &mut out, RunOptions { seed: 7, workers: 2, ..RunOptions::default() })
        .unwrap();
    let out = String::from_utf8(out).unwrap();
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 1"), "{out}"); // (0,1) or (1,0); diagonal forbidden
}

#[test]
fn fortress2_starred_support_pattern_is_canonically_verified() {
    // Minimal form of Fortress2's repeated five-cell extension: once the
    // centre and one neighbour are zero, the other neighbours are wildcards.
    // Fixed domains make the expected candidate deterministic and ensure the
    // final canonical replay, rather than search alone, exercises the stars.
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables>
          <array id="x" size="[5]">
            <domain for="x[0] x[1]"> 0 </domain>
            <domain for="x[2]"> 1 </domain>
            <domain for="x[3]"> 10000 </domain>
            <domain for="x[4]"> 1 </domain>
          </array>
        </variables>
        <constraints>
          <extension>
            <list> x[] </list>
            <supports> (0,0,*,*,*)(0,*,0,*,*)(0,*,*,0,*)(0,*,*,*,0)(10000,*,*,*,*) </supports>
          </extension>
        </constraints>
        <objectives><minimize>x[0]</minimize></objectives>
      </instance>"#;

    let out = run(xml).unwrap();
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 0"), "{out}");
    assert!(out.contains("v 0 0 1 10000 1"), "{out}");
}

fn wide_weighted_objective_xml() -> &'static str {
    r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[2]"> 0 1 </array></variables>
        <constraints></constraints>
        <objectives>
          <maximize type="sum"><list>x[]</list><coeffs>1500000000 1500000000</coeffs></maximize>
        </objectives>
      </instance>"#
}

#[test]
fn wide_weighted_objective_stays_symbolic() {
    let out = run(wide_weighted_objective_xml()).unwrap();
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 3000000000"), "{out}");
}

#[test]
fn wide_weighted_minimize_objective_stays_symbolic() {
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[2]"> 0 1 </array></variables>
        <constraints><sum><list>x[]</list><condition>(ge,1)</condition></sum></constraints>
        <objectives>
          <minimize type="sum"><list>x[]</list><coeffs>1500000000 1500000000</coeffs></minimize>
        </objectives>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 1500000000"), "{out}");
}

#[test]
fn narrow_objective_range_outside_i32_stays_symbolic() {
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables><var id="x"> 2 </var></variables>
        <constraints></constraints>
        <objectives>
          <minimize type="sum"><list>x</list><coeffs>1500000000</coeffs></minimize>
        </objectives>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 3000000000"), "{out}");
}

#[test]
fn affine_objective_bound_accumulation_does_not_hide_overflow() {
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables><var id="x"> 2147483647 </var></variables>
        <constraints></constraints>
        <objectives>
          <minimize type="sum">
            <list>x x x x x x</list>
            <coeffs>2147483647 2147483647 2147483647 -2147483647 -2147483647 -4</coeffs>
          </minimize>
        </objectives>
      </instance>"#;
    let out = run(xml).unwrap();
    let maximum = i64::from(i32::MAX);
    let expected = (maximum - 4) * maximum;
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains(&format!("o {expected}")), "{out}");
}

#[test]
fn wide_affine_sum_keeps_its_materialized_propagation_view() {
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[40]"> 0 1 </array></variables>
        <constraints></constraints>
        <objectives><minimize type="sum"><list>x[]</list></minimize></objectives>
      </instance>"#;
    let mut output = Vec::new();
    run_to_with_options(xml, true, &AtomicBool::new(false), &mut output, RunOptions::default()).unwrap();
    let output = String::from_utf8(output).unwrap();

    assert!(output.contains("c variables 41"), "wide sum did not retain its auxiliary view:\n{output}");
    assert!(output.contains("c propagators 1"), "wide sum identity was not posted:\n{output}");
    assert!(output.contains("s OPTIMUM FOUND"), "{output}");
    assert!(output.contains("o 0"), "{output}");
}

#[test]
fn wide_weighted_objective_rejects_unsupported_probe_workers() {
    let mut out = Vec::new();
    let error = run_to_with_options(
        wide_weighted_objective_xml(),
        true,
        &AtomicBool::new(false),
        &mut out,
        RunOptions { seed: 17, workers: 2, probes: 1, ..RunOptions::default() },
    )
    .unwrap_err();
    assert!(error.contains("materialized variable"), "{error}");
}

#[test]
fn sum_of_square_expressions_stays_symbolic() {
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="c" size="[2]"> -2..2 </array></variables>
        <constraints></constraints>
        <objectives>
          <maximize type="sum"> mul(c[0],c[0]) mul(c[1],c[1]) </maximize>
        </objectives>
      </instance>"#;
    let mut out = Vec::new();
    run_to_with_options(xml, true, &AtomicBool::new(false), &mut out, RunOptions::default()).unwrap();
    let out = String::from_utf8(out).unwrap();
    assert!(out.contains("c variables 2"), "{out}");
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 8"), "{out}");
    assert!(out.contains("c incumbent 8 source integer-exact"), "{out}");
}

#[test]
fn weighted_equality_objective_publishes_the_rewarded_values_first() {
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[2]"> 0..5 </array></variables>
        <constraints></constraints>
        <objectives>
          <maximize type="sum">
            <list> eq(x[0],5) eq(x[1],2) </list>
            <coeffs> 7 3 </coeffs>
          </maximize>
        </objectives>
      </instance>"#;
    let mut output = Vec::new();
    run_to_with_options(xml, true, &AtomicBool::new(false), &mut output, RunOptions::default()).unwrap();
    let output = String::from_utf8(output).unwrap();
    let objectives =
        output.lines().filter_map(|line| line.strip_prefix("o ")).map(|value| value.parse::<i64>().unwrap()).collect::<Vec<_>>();

    assert_eq!(objectives.first(), Some(&10), "weighted equality hint missed its targets:\n{output}");
    assert!(output.contains("s OPTIMUM FOUND"), "{output}");
}

#[test]
fn expression_objective_uses_portfolio_workers() {
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="c" size="[2]"> -2..2 </array></variables>
        <constraints></constraints>
        <objectives>
          <maximize type="sum"> mul(c[0],c[0]) mul(c[1],c[1]) </maximize>
        </objectives>
      </instance>"#;
    let mut out = Vec::new();
    run_to_with_options(xml, true, &AtomicBool::new(false), &mut out, RunOptions { workers: 2, ..RunOptions::default() }).unwrap();
    let out = String::from_utf8(out).unwrap();
    assert!(out.contains("c workers 2"), "{out}");
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 8"), "{out}");
}

#[test]
fn expression_objective_uses_lns_worker() {
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="c" size="[4]"> -2..2 </array></variables>
        <constraints></constraints>
        <objectives>
          <maximize type="sum">
            mul(c[0],c[0]) mul(c[1],c[1]) mul(c[2],c[2]) mul(c[3],c[3])
          </maximize>
        </objectives>
      </instance>"#;
    let mut out = Vec::new();
    run_to_with_options(xml, true, &AtomicBool::new(false), &mut out, RunOptions { workers: 2, lns: 1, ..RunOptions::default() }).unwrap();
    let out = String::from_utf8(out).unwrap();
    assert!(out.contains("c workers 2"), "{out}");
    assert!(out.contains("c lns attempts "), "{out}");
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 16"), "{out}");
    assert!(out.contains("c incumbent 16 source "), "{out}");
}

#[test]
fn per_element_domains() {
    // Array with <domain for="..."> children (the case that gave "empty domain").
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables>
          <array id="s" size="[3]">
            <domain for="s[0]"> 0 </domain>
            <domain for="s[1..2]"> 0..5 </domain>
          </array>
        </variables>
        <constraints>
          <intension> gt(s[1],s[0]) </intension>
          <intension> gt(s[2],s[1]) </intension>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s SATISFIABLE"), "{out}");
}

#[test]
fn value_count_repetition() {
    // instantiation values use the `vxk` repetition notation (2x3 = three 2s).
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><array id="a" size="[3]"> 0..5 </array></variables>
        <constraints>
          <instantiation><list>a[]</list><values> 2x3 </values></instantiation>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s SATISFIABLE"), "{out}");
    assert!(out.contains("v 2 2 2"), "{out}");
}

#[test]
fn element_over_constant_array() {
    // element list is integer constants; maximise the selected value.
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables><var id="idx"> 0..2 </var><var id="v"> 0..9 </var></variables>
        <constraints>
          <element><list> 5 8 3 </list><index>idx</index><value>v</value></element>
        </constraints>
        <objectives><maximize>v</maximize></objectives>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 8"), "{out}");
}

#[test]
fn circuit_refs_in_text() {
    // circuit scope given directly in the element text (no <list>).
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><array id="succ" size="[3]"> 0..2 </array></variables>
        <constraints><circuit> succ[] </circuit></constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s SATISFIABLE"), "{out}");
}

#[test]
fn group_and_2d_array_permutation_matrix() {
    // 2x2 binary matrix; each row and column sums to 1 (a permutation matrix).
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><array id="x" size="[2][2]"> 0 1 </array></variables>
        <constraints>
          <group>
            <sum><list>%...</list><condition>(eq,1)</condition></sum>
            <args> x[0][] </args>
            <args> x[1][] </args>
          </group>
          <group>
            <sum><list>%...</list><condition>(eq,1)</condition></sum>
            <args> x[][0] </args>
            <args> x[][1] </args>
          </group>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s SATISFIABLE"), "{out}");
}

#[test]
fn coins_grid_objective_block() {
    // A 2x2 CoinsGrid-style COP: row/col sums = 1, minimise Σ |i-j|^2 · x[i][j].
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[2][2]"> 0 1 </array></variables>
        <constraints>
          <group>
            <sum><list>%...</list><condition>(eq,1)</condition></sum>
            <args> x[0][] </args>
            <args> x[1][] </args>
          </group>
          <group>
            <sum><list>%...</list><condition>(eq,1)</condition></sum>
            <args> x[][0] </args>
            <args> x[][1] </args>
          </group>
        </constraints>
        <objectives>
          <minimize type="sum"><list> x[][] </list><coeffs> 0 1 1 0 </coeffs></minimize>
        </objectives>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 0"), "{out}"); // identity matrix: cost 0
}

#[test]
fn element_and_objective_expression() {
    // value = x[idx]; maximize that value.
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables>
          <array id="x" size="[3]"> 0..9 </array>
          <var id="idx"> 0..2 </var>
          <var id="v"> 0..9 </var>
        </variables>
        <constraints>
          <instantiation><list>x[]</list><values>3 7 5</values></instantiation>
          <element><list>x[]</list><index>idx</index><value>v</value></element>
        </constraints>
        <objectives><maximize>v</maximize></objectives>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 7"), "{out}"); // max entry of [3,7,5]
}

#[test]
fn parallel_roles_accept_a_representable_expression_objective() {
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[2]"> 0..2 </array></variables>
        <constraints></constraints>
        <objectives><minimize>add(x[0],x[1])</minimize></objectives>
      </instance>"#;

    for options in
        [RunOptions { workers: 2, split: true, ..RunOptions::default() }, RunOptions { workers: 2, probes: 1, ..RunOptions::default() }]
    {
        let mut out = Vec::new();
        run_to_with_options(xml, true, &AtomicBool::new(false), &mut out, options).unwrap();
        let out = String::from_utf8(out).unwrap();
        assert!(out.contains("s OPTIMUM FOUND"), "{out}");
        assert!(out.contains("o 0"), "{out}");
    }
}

#[test]
fn element_value_start_index_one() {
    // 1-based list: idx=1 must select x[0]=3 (not x[1]).
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables>
          <array id="x" size="[3]"> 0..9 </array>
          <var id="idx"> 1..3 </var>
          <var id="v"> 0..9 </var>
        </variables>
        <constraints>
          <instantiation><list>x[]</list><values>3 7 5</values></instantiation>
          <intension>eq(idx,1)</intension>
          <element><list startIndex="1">x[]</list><index>idx</index><value>v</value></element>
        </constraints>
        <objectives><maximize>v</maximize></objectives>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 3"), "{out}"); // x[idx-1] = x[0] = 3
}

#[test]
fn element_cond_start_index_one() {
    // 1-based condition form: x[idx-1]==7 holds only at idx=2 (x[1]=7).
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables>
          <array id="x" size="[3]"> 0..9 </array>
          <var id="idx"> 1..3 </var>
        </variables>
        <constraints>
          <instantiation><list>x[]</list><values>3 7 5</values></instantiation>
          <element><list startIndex="1">x[]</list><index>idx</index><condition>(eq,7)</condition></element>
        </constraints>
        <objectives><minimize>idx</minimize></objectives>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 2"), "{out}"); // idx-1 = 1
}

#[test]
fn element_value_start_index_zero_regression() {
    // 0-based (default): idx=1 selects x[1]=7.
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables>
          <array id="x" size="[3]"> 0..9 </array>
          <var id="idx"> 0..2 </var>
          <var id="v"> 0..9 </var>
        </variables>
        <constraints>
          <instantiation><list>x[]</list><values>3 7 5</values></instantiation>
          <intension>eq(idx,1)</intension>
          <element><list startIndex="0">x[]</list><index>idx</index><value>v</value></element>
        </constraints>
        <objectives><maximize>v</maximize></objectives>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 7"), "{out}"); // x[idx] = x[1] = 7
}

#[test]
fn objective_sum_extreme_coeffs_avoids_overflow() {
    // Three i32::MAX coeffs over symmetric near-full-width domains: the span
    // endpoints hi and lo each fit i64, but hi - lo overflows it. The naive
    // subtraction wrapped negative and slipped the "too wide to materialize"
    // guard; saturating_sub keeps it above the guard so the objective stays
    // the symbolic Linear form and the true optimum (all vars at their min) is
    // still found. domain width 2*1073741823+1 == i32::MAX, the parser ceiling.
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[3]"> -1073741823..1073741823 </array></variables>
        <constraints></constraints>
        <objectives>
          <minimize type="sum"><list>x[]</list><coeffs>2147483647 2147483647 2147483647</coeffs></minimize>
        </objectives>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o -6917529017977405443"), "{out}"); // 3 * 2147483647 * -1073741823
}

#[test]
fn no_overlap_2d_packs_four_unit_boxes() {
    // Four 1×1 boxes in a 2×2 grid (origins 0..1 in each axis): the four cells,
    // a feasible packing.
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables>
          <array id="x" size="[4]"> 0..1 </array>
          <array id="y" size="[4]"> 0..1 </array>
        </variables>
        <constraints>
          <noOverlap>
            <origins> (x[0],y[0])(x[1],y[1])(x[2],y[2])(x[3],y[3]) </origins>
            <lengths> (1,1)(1,1)(1,1)(1,1) </lengths>
          </noOverlap>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s SATISFIABLE"), "{out}");
}

#[test]
fn no_overlap_2d_five_boxes_is_unsat() {
    // Five 1×1 boxes in the same 2×2 grid: only four cells, so unsatisfiable.
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables>
          <array id="x" size="[5]"> 0..1 </array>
          <array id="y" size="[5]"> 0..1 </array>
        </variables>
        <constraints>
          <noOverlap>
            <origins> (x[0],y[0])(x[1],y[1])(x[2],y[2])(x[3],y[3])(x[4],y[4]) </origins>
            <lengths> (1,1)(1,1)(1,1)(1,1)(1,1) </lengths>
          </noOverlap>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s UNSATISFIABLE"), "{out}");
}

#[test]
fn channel_start_index_reaches_the_kernel() {
    // channel with startIndex="1": positions and values live in 1..3, so with
    // domains capped at 2 no involution exists (x=1 at position 3 would need
    // value 3 at position 1) — UNSAT. A kernel that drops the start index
    // treats this as 0-based over 0..2 and wrongly finds the identity.
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables>
          <array id="x" size="[3]"> 0..2 </array>
        </variables>
        <constraints>
          <channel> <list startIndex="1"> x[] </list> </channel>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s UNSATISFIABLE"), "{out}");
}

#[test]
fn channel_start_index_one_finds_shifted_involutions() {
    // Same constraint with domains 1..3: the shifted involutions exist
    // (identity 1,2,3 among them).
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables>
          <array id="x" size="[3]"> 1..3 </array>
        </variables>
        <constraints>
          <channel> <list startIndex="1"> x[] </list> </channel>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s SATISFIABLE"), "{out}");
}

#[test]
fn no_overlap_1d_zero_length_zero_ignored() {
    // Three unit tasks fill horizon 0..3 exactly; the fourth has length 0 and
    // zeroIgnored="true", so it may sit anywhere — satisfiable. This routes
    // through the diffn path (the plain propagator only takes positive
    // lengths), exercising its kernel and LS encodings end to end.
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables>
          <array id="s" size="[4]"> 0..2 </array>
        </variables>
        <constraints>
          <noOverlap zeroIgnored="true">
            <origins> s[] </origins>
            <lengths> 1 1 1 0 </lengths>
          </noOverlap>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s SATISFIABLE"), "{out}");
}

#[test]
fn no_overlap_1d_zero_length_not_ignored_still_packs() {
    // zeroIgnored="false": the zero-length task is a point that must not sit
    // strictly inside another task's interval; with unit tasks on 0..2 it can
    // always sit at a boundary, so this stays satisfiable — but four unit
    // tasks on the same horizon would not (packing regression guard).
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables>
          <array id="s" size="[4]"> 0..2 </array>
        </variables>
        <constraints>
          <noOverlap zeroIgnored="false">
            <origins> s[] </origins>
            <lengths> 1 1 1 0 </lengths>
          </noOverlap>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s SATISFIABLE"), "{out}");
}

#[test]
fn positive_fixed_no_overlap_keeps_the_compact_global_when_zero_is_ignored() {
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables>
          <array id="s" size="[3]"> 0..5 </array>
        </variables>
        <constraints>
          <noOverlap zeroIgnored="true">
            <origins> s[] </origins>
            <lengths> 2 2 2 </lengths>
          </noOverlap>
        </constraints>
      </instance>"#;
    let mut output = Vec::new();
    run_to_with_options(xml, true, &AtomicBool::new(false), &mut output, RunOptions::default()).unwrap();
    let output = String::from_utf8(output).unwrap();
    assert!(output.contains("c propagators 1"), "positive durations were expanded instead of using noOverlap:\n{output}");
    assert!(output.contains("s SATISFIABLE"), "{output}");
}

fn low_autocorrelation_10_xml() -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/XCSP25/MiniCOP25/LowAutocorrelation-010-mini_c25.xml.lzma");
    let bytes = std::fs::read(path).unwrap();
    let mut xml = Vec::new();
    lzma_rs::lzma_decompress(&mut &bytes[..], &mut xml).unwrap();
    String::from_utf8(xml).unwrap()
}

fn low_autocorrelation_10_output(seed: u64, workers: usize) -> String {
    low_autocorrelation_10_output_with(seed, workers, false, false, 1 << 14)
}

fn low_autocorrelation_10_output_with(seed: u64, workers: usize, verbose: bool, split: bool, shared_pool_capacity: usize) -> String {
    let mut out = Vec::new();
    run_to_with_options(
        &low_autocorrelation_10_xml(),
        verbose,
        &AtomicBool::new(false),
        &mut out,
        RunOptions { seed, workers, split, shared_pool_capacity, ..RunOptions::default() },
    )
    .unwrap();
    String::from_utf8(out).unwrap()
}

fn last_objective(out: &str) -> i32 {
    out.lines().filter_map(|line| line.strip_prefix("o ")).map(|value| value.parse().unwrap()).next_back().unwrap()
}

#[test]
fn low_autocorrelation_10_seed_regressions() {
    for seed in [5, 8, 13, 26, 37] {
        let out = low_autocorrelation_10_output(seed, 1);
        assert!(out.contains("s OPTIMUM FOUND"), "seed {seed}:\n{out}");
        assert_eq!(last_objective(&out), 13, "seed {seed}:\n{out}");
    }
}

#[test]
fn parallel_low_autocorrelation_10_seed_regression() {
    let out = low_autocorrelation_10_output(23, 4);
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert_eq!(last_objective(&out), 13, "{out}");
}

#[test]
fn parallel_low_autocorrelation_shares_clauses() {
    let out = low_autocorrelation_10_output_with(23, 4, true, true, 1 << 14);
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert_eq!(last_objective(&out), 13, "{out}");
    let sharing = out.lines().find(|line| line.starts_with("c shared clauses ")).unwrap();
    let sharing: Vec<u64> = sharing.split_whitespace().filter_map(|word| word.parse().ok()).collect();
    let jobs = out.lines().find(|line| line.starts_with("c split jobs ")).unwrap();
    let jobs: Vec<u64> = jobs.split_whitespace().filter_map(|word| word.parse().ok()).collect();
    assert!(sharing[0] > 0, "{out}");
    assert!(sharing[1] > 0, "{out}");
    assert!(jobs[0] > 0, "{out}");
    assert!(jobs[1] > 1, "{out}");
    assert!(out.contains(" source integer-exact"), "{out}");
}

#[test]
fn parallel_tiny_shared_pool_still_finds_optimum() {
    // Force a tiny shared-clause ring so lagging readers drop most shared
    // clauses. Losing shared clauses is sound: it only reduces cross-worker
    // pruning, so the portfolio must still prove the same optimum.
    let out = low_autocorrelation_10_output_with(23, 4, true, true, 8);
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert_eq!(last_objective(&out), 13, "{out}");
}

#[test]
fn parallel_split_interruption_is_not_a_proof() {
    let mut out = Vec::new();
    run_to_with_options(
        &low_autocorrelation_10_xml(),
        true,
        &AtomicBool::new(true),
        &mut out,
        RunOptions { seed: 23, workers: 4, split: true, ..RunOptions::default() },
    )
    .unwrap();
    let out = String::from_utf8(out).unwrap();
    assert!(out.contains("s UNKNOWN"), "{out}");
    assert!(!out.contains("s OPTIMUM FOUND"), "{out}");
}

#[test]
fn parallel_probe_interruption_is_not_a_proof() {
    let mut out = Vec::new();
    run_to_with_options(
        &low_autocorrelation_10_xml(),
        true,
        &AtomicBool::new(true),
        &mut out,
        RunOptions { seed: 23, workers: 4, probes: 1, ..RunOptions::default() },
    )
    .unwrap();
    let out = String::from_utf8(out).unwrap();
    assert!(out.contains("s UNKNOWN"), "{out}");
    assert!(!out.contains("s OPTIMUM FOUND"), "{out}");
}

// `ordered` with constant `lengths` (variant 2): chain x[i] + len[i]  rel  x[i+1].

#[test]
fn ordered_constant_lengths_v2_sat_is_unique() {
    // x0+2<=x1, x1+3<=x2 over 0..5 forces the single tuple [0,2,5].
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><array id="x" size="[3]"> 0..5 </array></variables>
        <constraints>
          <ordered><list>x[]</list><lengths>2 3</lengths><operator>le</operator></ordered>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s SATISFIABLE"), "{out}");
    assert!(out.contains("v 0 2 5"), "{out}");
}

#[test]
fn ordered_constant_lengths_v2_unsat_when_domain_too_tight() {
    // Same chain needs x2 >= 5, but 0..4 cannot host it.
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><array id="x" size="[3]"> 0..4 </array></variables>
        <constraints>
          <ordered><list>x[]</list><lengths>2 3</lengths><operator>le</operator></ordered>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s UNSATISFIABLE"), "{out}");
}

// `nValues` over an expression list (variant 3): materialise each expression to
// an aux var, then count distinct values.

#[test]
fn nvalues_expr_list_v3_sat_is_unique() {
    // x0 fixed to 1 => add(x0,1)=2; exactly-one distinct forces x1 = 2.
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><array id="x" size="[2]"> 0..3 </array></variables>
        <constraints>
          <instantiation><list>x[0]</list><values>1</values></instantiation>
          <nValues><list> add(x[0],1) x[1] </list><condition>(eq,1)</condition></nValues>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s SATISFIABLE"), "{out}");
    assert!(out.contains("v 1 2"), "{out}");
}

#[test]
fn nvalues_expr_list_v3_unsat_when_distinct_forced() {
    // {x0, x0+1} are always two distinct values, so "exactly one distinct" fails.
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><array id="x" size="[1]"> 0..3 </array></variables>
        <constraints>
          <nValues><list> x[0] add(x[0],1) </list><condition>(eq,1)</condition></nValues>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s UNSATISFIABLE"), "{out}");
}

// `sum` with variable `coeffs` (variant 3): each term is a var*var product,
// materialised to an aux var, then summed with unit coefficients.

#[test]
fn sum_variable_coeffs_v3_sat_is_unique() {
    // x=[1,2], c0=3 fixed; c0*x0 + c1*x1 = 3 + 2*c1 = 9 forces c1=3.
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables>
          <array id="x" size="[2]"> 0..3 </array>
          <array id="c" size="[2]"> 0..3 </array>
        </variables>
        <constraints>
          <instantiation><list>x[] c[0]</list><values>1 2 3</values></instantiation>
          <sum><list>x[]</list><coeffs>c[]</coeffs><condition>(eq,9)</condition></sum>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s SATISFIABLE"), "{out}");
    assert!(out.contains("v 1 2 3 3"), "{out}");
}

#[test]
fn sum_variable_coeffs_v3_unsat_when_product_too_small() {
    // max c0*x0 = 2*2 = 4 < 7, so the equality is unreachable.
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables>
          <array id="x" size="[1]"> 0..2 </array>
          <array id="c" size="[1]"> 0..2 </array>
        </variables>
        <constraints>
          <sum><list>x[0]</list><coeffs>c[0]</coeffs><condition>(eq,7)</condition></sum>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s UNSATISFIABLE"), "{out}");
}

// `minimize`/`maximize sum` with variable `coeffs` (objective variant 2): each
// term is a var*var product aggregated into the objective.

#[test]
fn minimize_variable_coeffs_objective_v2() {
    // min Σ y_i*x_i with x_i>=1, y_i>=1 is 1*1 + 1*1 = 2.
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables>
          <array id="x" size="[2]"> 1..3 </array>
          <array id="y" size="[2]"> 1..2 </array>
        </variables>
        <constraints></constraints>
        <objectives>
          <minimize type="sum"><list>x[]</list><coeffs>y[]</coeffs></minimize>
        </objectives>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 2"), "{out}");
}

#[test]
fn maximize_variable_coeffs_objective_v2() {
    // max Σ y_i*x_i with x_i<=2, y_i<=2 is 2*2 + 2*2 = 8.
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables>
          <array id="x" size="[2]"> 1..2 </array>
          <array id="y" size="[2]"> 1..2 </array>
        </variables>
        <constraints></constraints>
        <objectives>
          <maximize type="sum"><list>x[]</list><coeffs>y[]</coeffs></maximize>
        </objectives>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 8"), "{out}");
}

// `binPacking` with fixed integer `loads` (variant 4): each bin's load equals
// the given constant.

#[test]
fn bin_packing_fixed_loads_v4_sat_is_unique() {
    // loads [3,2] with item sizes [3,2] forces item0->bin0, item1->bin1.
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><array id="x" size="[2]"> 0..1 </array></variables>
        <constraints>
          <binPacking><list>x[]</list><sizes>3 2</sizes><loads>3 2</loads></binPacking>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s SATISFIABLE"), "{out}");
    assert!(out.contains("v 0 1"), "{out}");
}

#[test]
fn bin_packing_fixed_loads_v4_unsat_on_total_mismatch() {
    // Σ sizes = 5 but Σ loads = 4, so no assignment can match.
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><array id="x" size="[2]"> 0..1 </array></variables>
        <constraints>
          <binPacking><list>x[]</list><sizes>3 2</sizes><loads>2 2</loads></binPacking>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s UNSATISFIABLE"), "{out}");
}

#[test]
fn bin_packing_fixed_loads_v4_rejects_unlisted_bin_escape() {
    // Only bin 0 is declared by the single load. Assigning x=1 must not let the
    // item escape the load equation.
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><var id="x"> 0..1 </var></variables>
        <constraints>
          <binPacking><list>x</list><sizes>5</sizes><loads>0</loads></binPacking>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s UNSATISFIABLE"), "{out}");
}

// `binPacking` with variable `limits` (variant 3): each bin's load must stay
// within a per-bin variable capacity.

#[test]
fn bin_packing_variable_limits_v3_sat_is_unique() {
    // Limits [2,3]: the size-3 item cannot fit bin0, and bin1 cannot hold both,
    // forcing item0->bin1, item1->bin0.
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables>
          <array id="x" size="[2]"> 0..1 </array>
          <array id="y" size="[2]"> 0..5 </array>
        </variables>
        <constraints>
          <instantiation><list>y[]</list><values>2 3</values></instantiation>
          <binPacking><list>x[]</list><sizes>3 2</sizes><limits>y[]</limits></binPacking>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s SATISFIABLE"), "{out}");
    assert!(out.contains("v 1 0 2 3"), "{out}");
}

#[test]
fn bin_packing_variable_limits_v3_unsat_when_no_bin_fits() {
    // Both limits are 1, but the size-3 item fits in no bin.
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables>
          <array id="x" size="[2]"> 0..1 </array>
          <array id="y" size="[2]"> 0..5 </array>
        </variables>
        <constraints>
          <instantiation><list>y[]</list><values>1 1</values></instantiation>
          <binPacking><list>x[]</list><sizes>3 2</sizes><limits>y[]</limits></binPacking>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s UNSATISFIABLE"), "{out}");
}

#[test]
fn bin_packing_variable_limits_v3_rejects_unlisted_bin_escape() {
    // Only bin 0 has a capacity variable. Assigning x=1 would bypass the load
    // unless item domains are restricted to the declared bins.
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables>
          <var id="x"> 0..1 </var>
          <var id="cap"> 0 </var>
        </variables>
        <constraints>
          <binPacking><list>x</list><sizes>5</sizes><limits>cap</limits></binPacking>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s UNSATISFIABLE"), "{out}");
}

// `count` against variable target `values` (variant 4): indicator_i = list[i] is
// one of the (variable) targets, then the indicators are counted.

#[test]
fn count_variable_values_v4_sat_is_unique() {
    // v0 fixed to 2; "exactly two of a equal v0" forces a = [2,2].
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables>
          <array id="a" size="[2]"> 0..3 </array>
          <array id="v" size="[1]"> 0..3 </array>
        </variables>
        <constraints>
          <instantiation><list>v[0]</list><values>2</values></instantiation>
          <count><list>a[]</list><values>v[0]</values><condition>(eq,2)</condition></count>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s SATISFIABLE"), "{out}");
    assert!(out.contains("v 2 2 2"), "{out}");
}

#[test]
fn count_variable_values_v4_unsat_on_disjoint_domains() {
    // a in 0..1, v in 2..3 can never be equal, so "exactly one equal" fails.
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables>
          <array id="a" size="[1]"> 0..1 </array>
          <array id="v" size="[1]"> 2..3 </array>
        </variables>
        <constraints>
          <count><list>a[]</list><values>v[0]</values><condition>(eq,1)</condition></count>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s UNSATISFIABLE"), "{out}");
}

#[test]
fn count_variable_values_v4_multi_target_unsat() {
    // a0 = 5 is in neither target {1,2}, so "at least one in the set" is unsatisfiable.
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables>
          <array id="a" size="[1]"> 5..5 </array>
          <array id="v" size="[2]"> 1..2 </array>
        </variables>
        <constraints>
          <instantiation><list>v[]</list><values>1 2</values></instantiation>
          <count><list>a[]</list><values>v[]</values><condition>(ge,1)</condition></count>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s UNSATISFIABLE"), "{out}");
}

// Sparse / out-of-declaration-order arrays: cells declared via per-cell
// `<domain for=...>` must map by index tuple, not by a dense row-major position.

#[test]
fn sparse_out_of_order_array_maps_by_index() {
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables>
          <array id="a" size="[2][2]">
            <domain for="a[1][1]"> 5 </domain>
            <domain for="a[0][0]"> 1 </domain>
            <domain for="a[0][1] a[1][0]"> 2 3 </domain>
          </array>
        </variables>
        <constraints><intension> eq(a[1][1],5) </intension></constraints>
      </instance>"#;
    assert!(run(xml).unwrap().contains("s SATISFIABLE"));
}

#[test]
fn sparse_array_index_is_not_misresolved() {
    // a[1][1] has domain {5}; if it mis-mapped to a[0][0] ({1}) this would be SAT.
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables>
          <array id="a" size="[2][2]">
            <domain for="a[1][1]"> 5 </domain>
            <domain for="a[0][0]"> 1 </domain>
            <domain for="a[0][1] a[1][0]"> 2 3 </domain>
          </array>
        </variables>
        <constraints><intension> eq(a[1][1],1) </intension></constraints>
      </instance>"#;
    assert!(run(xml).unwrap().contains("s UNSATISFIABLE"));
}

#[test]
fn ranged_array_slice_in_expression_list() {
    // x[0..2] must expand to three cells; "one distinct" + x0=2 forces all 2.
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><array id="x" size="[3]"> 0..3 </array></variables>
        <constraints>
          <instantiation><list>x[0]</list><values>2</values></instantiation>
          <nValues><list> x[0..2] </list><condition>(eq,1)</condition></nValues>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s SATISFIABLE"), "{out}");
    assert!(out.contains("v 2 2 2"), "{out}");
}

// `notin` / set conditions on aggregates.

#[test]
fn sum_notin_set_condition() {
    // Σ ∉ {0,1,2,3} with Σ in [0,4] forces Σ = 4, i.e. x = [2,2].
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><array id="x" size="[2]"> 0..2 </array></variables>
        <constraints><sum><list>x[]</list><condition>(notin,{0,1,2,3})</condition></sum></constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s SATISFIABLE"), "{out}");
    assert!(out.contains("v 2 2"), "{out}");
}

#[test]
fn sum_notin_interval_condition() {
    // Σ ∉ [0,3] with Σ in [0,4] forces Σ = 4, i.e. x = [2,2].
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><array id="x" size="[2]"> 0..2 </array></variables>
        <constraints><sum><list>x[]</list><condition>(notin,0..3)</condition></sum></constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s SATISFIABLE"), "{out}");
    assert!(out.contains("v 2 2"), "{out}");
}

#[test]
fn sum_notin_interval_full_range_is_unsat() {
    // The whole reachable sum range is excluded. This must be UNSAT, not an
    // empty-domain panic while materialising the complement.
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><var id="x"> 0..2 </var></variables>
        <constraints><sum><list>x</list><condition>(notin,0..2)</condition></sum></constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s UNSATISFIABLE"), "{out}");
}

#[test]
fn nvalues_in_interval_condition_unsat() {
    // x = [0,0,1] has 2 distinct values, which is not in [0,1].
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><array id="x" size="[3]"> 0..2 </array></variables>
        <constraints>
          <instantiation><list>x[]</list><values>0 0 1</values></instantiation>
          <nValues><list>x[]</list><condition>(in,0..1)</condition></nValues>
        </constraints>
      </instance>"#;
    assert!(run(xml).unwrap().contains("s UNSATISFIABLE"));
}

// `lex` against a constant tuple: a `<list>` may be integer constants, not just
// variables (parser fork emits them as literals; resolved via var_or_constant).

#[test]
fn lex_against_constant_tuple_sat() {
    // x <=_lex [2,1,3,2]; the min tuple [0,0,0,0] satisfies it (x[0]=0 < 2).
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><array id="x" size="[4]"> 0..4 </array></variables>
        <constraints>
          <lex><list>x[]</list><list>2 1 3 2</list><operator>le</operator></lex>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s SATISFIABLE"), "{out}");
    assert!(out.contains("v 0 0 0 0"), "{out}");
}

#[test]
fn lex_against_constant_tuple_unsat() {
    // x <_lex [0,0] is impossible since [0,0] is the lexicographic minimum.
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><array id="x" size="[2]"> 0..2 </array></variables>
        <constraints>
          <lex><list>x[]</list><list>0 0</list><operator>lt</operator></lex>
        </constraints>
      </instance>"#;
    assert!(run(xml).unwrap().contains("s UNSATISFIABLE"));
}

#[test]
fn lex_constant_tuple_both_directions_pins_value() {
    // x >=_lex [0,1] and x <=_lex [0,1] together force x = [0,1].
    let xml = r#"
      <instance format="XCSP3" type="CSP">
        <variables><array id="x" size="[2]"> 0..1 </array></variables>
        <constraints>
          <lex><list>x[]</list><list>0 1</list><operator>ge</operator></lex>
          <lex><list>x[]</list><list>0 1</list><operator>le</operator></lex>
        </constraints>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s SATISFIABLE"), "{out}");
    assert!(out.contains("v 0 1"), "{out}");
}
