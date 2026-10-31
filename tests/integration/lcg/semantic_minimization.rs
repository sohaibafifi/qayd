//! Public correctness and paired-measurement gates for opt-in semantic
//! learned-clause minimization. Expected answers come from direct enumeration,
//! not from the disabled solver arm.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use qayd::constraints::linear::{linear, Relation};
use qayd::constraints::table::{extension, STAR};
use qayd::model::{Constraint, IntExpr, IntGlobalConstraint, Model, ModelPackage, Objective};
use qayd::orchestrator::{
    solve_model_silent, CpControls, IgnoreEvents, ProofRequest, ProvenConclusion, SearchPhase, SearchPolicy, SemanticAssumption,
    SemanticAssumptionOp, SemanticSolveSession, SolveMode, SolveRequest, SolveResult, SolveStatus, ValueSelector, VariableSelector,
    VerificationLevel,
};
use qayd::{optimize_var_assuming, optimize_with, Assumption, AssumptionOp, BoolLit, SolveStats, Solver, VarId};

#[path = "../../common/mod.rs"]
mod common;
use common::oracle;

static NEVER_STOP: AtomicBool = AtomicBool::new(false);

fn next(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

fn tuple_matches(tuple: &[i32], values: &[i32]) -> bool {
    tuple.iter().zip(values).all(|(&t, &value)| t == STAR || t == value)
}

fn assert_disabled_counters(stats: SolveStats, enabled: bool) {
    if !enabled {
        assert_eq!(stats.semantic_minimized_clauses, 0);
        assert_eq!(stats.semantic_minimized_lits, 0);
    }
    assert!(stats.semantic_minimized_clauses <= stats.semantic_minimized_lits);
}

#[test]
fn holey_tables_and_signed_linear_rows_preserve_both_optima() {
    let universe = [-3, -1, 0, 2, 4];
    let mut state = 0x1832_47A5_CDE6_F901;
    let mut feasible = 0;
    let mut infeasible = 0;
    for round in 0..32 {
        let n = 3 + (next(&mut state) % 3) as usize;
        let domains = (0..n)
            .map(|_| {
                let mut values = vec![-3, 4];
                values.extend(universe[1..4].iter().copied().filter(|_| next(&mut state).is_multiple_of(2)));
                values.sort_unstable();
                values
            })
            .collect::<Vec<_>>();
        let tuples = (0..(4 + next(&mut state) % 10))
            .map(|_| {
                (0..n)
                    .map(|_| {
                        let index = (next(&mut state) % 6) as usize;
                        if index == 5 {
                            STAR
                        } else {
                            universe[index]
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let coefficients = (0..n).map(|i| if i % 2 == 0 { 1i64 } else { -2 }).collect::<Vec<_>>();
        let rhs = (next(&mut state) % 19) as i64 - 9;
        let expected = oracle::solutions(&domains, |a| {
            tuples.iter().any(|tuple| tuple_matches(tuple, a))
                && a.iter().zip(&coefficients).map(|(&v, &c)| i64::from(v) * c).sum::<i64>() >= rhs
        });
        if expected.is_empty() {
            infeasible += 1;
        } else {
            feasible += 1;
        }
        for enabled in [false, true] {
            for minimizing in [false, true] {
                let mut solver = Solver::new();
                solver.set_semantic_minimization(enabled);
                let vars = domains.iter().map(|domain| solver.new_var_set(domain)).collect::<Vec<_>>();
                extension(&mut solver, &vars, &tuples, true);
                linear(&mut solver, &coefficients, &vars, Relation::Ge, rhs);
                let (best, stats) = optimize_with(&mut solver, &vars, vars[0], minimizing, &NEVER_STOP, |_| {});
                let optimum = if minimizing { expected.iter().map(|a| a[0]).min() } else { expected.iter().map(|a| a[0]).max() };
                assert_eq!(best.as_ref().map(|(_, value)| *value), optimum, "round={round}, enabled={enabled}, minimizing={minimizing}");
                if let Some((assignment, value)) = best {
                    assert!(expected.contains(&assignment), "invalid witness in round {round}: {assignment:?}");
                    assert_eq!(assignment[0], value);
                }
                assert_disabled_counters(stats, enabled);
            }
        }
    }
    assert!(feasible > 0 && infeasible > 0, "the deterministic family must cover both outcomes");
}

fn support(n: usize) -> Vec<i32> {
    (0..n as i32).map(|value| 2 * value - 3).collect()
}

fn permutation_solutions(n: usize, impossible: bool) -> Vec<Vec<i32>> {
    let values = support(n);
    let rhs = values.iter().map(|&v| i64::from(v)).sum::<i64>() + i64::from(impossible);
    oracle::solutions(&vec![values; n], |a| {
        a.iter().copied().collect::<BTreeSet<_>>().len() == n && a.iter().map(|&v| i64::from(v)).sum::<i64>() >= rhs
    })
}

fn permutation_solver(n: usize, impossible: bool, enabled: bool, scope_reasons: bool) -> (Solver, Vec<VarId>) {
    let mut solver = Solver::new();
    solver.set_semantic_minimization(enabled);
    solver.set_force_scope_reasons(scope_reasons);
    let values = support(n);
    let vars = (0..n).map(|_| solver.new_var_set(&values)).collect::<Vec<_>>();
    let neq = values.iter().flat_map(|&a| values.iter().filter(move |&&b| a != b).map(move |&b| vec![a, b])).collect::<Vec<_>>();
    for i in 0..n {
        for j in (i + 1)..n {
            extension(&mut solver, &[vars[i], vars[j]], &neq, true);
        }
    }
    let rhs = values.iter().map(|&v| i64::from(v)).sum::<i64>() + i64::from(impossible);
    linear(&mut solver, &vec![1; n], &vars, Relation::Ge, rhs);
    (solver, vars)
}

#[test]
fn learned_holey_permutation_conflicts_preserve_sat_and_unsat() {
    for impossible in [false, true] {
        let expected = permutation_solutions(5, impossible);
        for scope_reasons in [false, true] {
            for enabled in [false, true] {
                let (mut solver, vars) = permutation_solver(5, impossible, enabled, scope_reasons);
                let (best, stats) = optimize_with(&mut solver, &vars, vars[0], true, &NEVER_STOP, |_| {});
                assert_eq!(best.as_ref().map(|(_, value)| *value), expected.iter().map(|a| a[0]).min());
                if let Some((assignment, _)) = best {
                    assert!(expected.contains(&assignment));
                }
                if impossible {
                    assert!(stats.failures > 1, "UNSAT must exercise search conflicts: {stats:?}");
                    assert!(stats.learned_lits > 0, "UNSAT must exercise learning: {stats:?}");
                }
                assert_disabled_counters(stats, enabled);
            }
        }
    }
}

#[test]
fn temporary_holey_bound_and_value_assumptions_preserve_optima() {
    let expected = permutation_solutions(4, false);
    for enabled in [false, true] {
        for minimizing in [false, true] {
            for contradictory in [false, true] {
                let (mut solver, vars) = permutation_solver(4, false, enabled, true);
                let mut assumptions = vec![
                    Assumption { var: vars[0], op: AssumptionOp::Ge, value: -1 },
                    Assumption { var: vars[1], op: AssumptionOp::Ne, value: 1 },
                    Assumption::eq(vars[2], -3),
                ];
                if contradictory {
                    assumptions.push(Assumption::eq(vars[1], -3));
                }
                let candidates =
                    expected.iter().filter(|a| a[0] >= -1 && a[1] != 1 && a[2] == -3 && (!contradictory || a[1] == -3)).collect::<Vec<_>>();
                let optimum = if minimizing {
                    candidates.iter().map(|a| i64::from(a[0])).min()
                } else {
                    candidates.iter().map(|a| i64::from(a[0])).max()
                };
                let (best, stats, complete) =
                    optimize_var_assuming(&mut solver, &vars, &assumptions, vars[0], minimizing, &NEVER_STOP, |_, _| {});
                assert!(complete);
                assert_eq!(best.as_ref().map(|(_, value)| *value), optimum);
                if let Some((assignment, _)) = best {
                    assert!(candidates.contains(&&assignment));
                }
                assert_disabled_counters(stats, enabled);
            }
        }
    }
}

#[test]
fn native_boolean_proof_stream_is_unchanged_by_the_opt_in() {
    let mut streams = Vec::new();
    for enabled in [false, true] {
        let mut solver = Solver::new();
        solver.set_semantic_minimization(enabled);
        let x = solver.new_var_range(0, 1);
        let y = solver.new_var_range(0, 1);
        let clauses = [false, true]
            .into_iter()
            .flat_map(|a| [false, true].into_iter().map(move |b| vec![BoolLit { var: x, value: a }, BoolLit { var: y, value: b }]))
            .collect::<Vec<_>>();
        let mut proof = Vec::new();
        let (solution, stats, complete) =
            qayd::search::solve_bool_cnf_seeded_with_proof(&mut solver, &[x, y], &clauses, &NEVER_STOP, 7, |clause| {
                proof.push(clause.to_vec());
            })
            .unwrap();
        assert!(complete && solution.is_none());
        assert!(proof.iter().any(Vec::is_empty), "proof must contain its final contradiction");
        // Semantic rewrites are deliberately disabled with a Boolean logger.
        assert_disabled_counters(stats, false);
        streams.push(proof);
    }
    assert_eq!(streams[0], streams[1]);
}

fn permutation_model(n: usize, impossible: bool, objective: bool) -> ModelPackage {
    let mut model = Model::new();
    let values = support(n);
    let vars = (0..n).map(|_| model.int_set(values.clone())).collect::<Vec<_>>();
    let neq: std::sync::Arc<[Vec<i32>]> =
        values.iter().flat_map(|&a| values.iter().filter(move |&&b| a != b).map(move |&b| vec![a, b])).collect();
    for i in 0..n {
        for j in (i + 1)..n {
            model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Table {
                variables: vec![vars[i], vars[j]],
                tuples: neq.clone(),
                positive: true,
            }));
        }
    }
    model.add_constraint(Constraint::Linear {
        terms: vars.iter().map(|&v| (1, v)).collect(),
        relation: qayd::model::Relation::Ge,
        rhs: values.iter().map(|&v| i64::from(v)).sum::<i64>() + i64::from(impossible),
    });
    if objective {
        model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(vars[0]) });
    }
    ModelPackage::new(model)
}

fn request(n: usize, enabled: bool, seed: u64) -> SolveRequest {
    SolveRequest {
        mode: SolveMode::Exact,
        seed,
        threads: 1,
        cp: CpControls { semantic_minimization: enabled, ..CpControls::default() },
        search_policy: SearchPolicy::new(vec![SearchPhase::new((0..n).collect(), VariableSelector::Activity, ValueSelector::RandomSeeded)]),
        ..SolveRequest::default()
    }
}

fn assert_metadata(result: &SolveResult, enabled: bool) {
    let values = result
        .reports()
        .iter()
        .flat_map(|report| &report.metadata)
        .filter(|(key, _)| key == "semantic_minimization")
        .map(|(_, value)| value.as_str())
        .collect::<Vec<_>>();
    if enabled {
        assert!(!values.is_empty(), "enabled request must be visible in engine metadata");
        assert!(values.iter().all(|&value| value == "true"));
    } else {
        assert!(values.is_empty(), "default requests must preserve existing metadata");
    }
    assert_disabled_counters(result.aggregate_search_stats(), enabled);
}

fn assert_canonical_result(result: &SolveResult, expected: &[Vec<i32>], objective: bool) {
    result.validate_contract().unwrap();
    if expected.is_empty() {
        assert_eq!(result.status(), SolveStatus::Unsatisfiable);
        assert!(result.primal().is_none());
        assert_eq!(result.proof().map(|proof| proof.conclusion()), Some(ProvenConclusion::Unsatisfiable));
    } else {
        assert_eq!(result.status(), if objective { SolveStatus::Optimal } else { SolveStatus::Satisfiable });
        let candidate = result.primal().unwrap();
        assert_eq!(candidate.verification(), VerificationLevel::Final);
        let values = candidate.assignment().integers.iter().map(|value| i32::try_from(value.unwrap()).unwrap()).collect::<Vec<_>>();
        assert!(expected.contains(&values), "canonical witness failed the independent oracle: {values:?}");
        if objective {
            assert_eq!(candidate.objectives(), &[i64::from(expected.iter().map(|a| a[0]).min().unwrap())]);
            assert_eq!(result.proof().map(|proof| proof.conclusion()), Some(ProvenConclusion::Optimal));
        }
    }
}

#[test]
fn canonical_opt_in_preserves_replay_and_completion_claims() {
    assert!(!CpControls::default().semantic_minimization);
    for objective in [false, true] {
        for impossible in [false, true] {
            let expected = permutation_solutions(4, impossible);
            let package = permutation_model(4, impossible, objective);
            for enabled in [false, true] {
                for threads in [1, 2] {
                    let mut options = request(4, enabled, 7);
                    options.threads = threads;
                    if objective || impossible {
                        options.proof = ProofRequest::Require;
                    }
                    let result = solve_model_silent(&package, &options).unwrap();
                    assert_canonical_result(&result, &expected, objective);
                    assert_metadata(&result, enabled);
                }
            }
        }
    }
}

#[test]
fn persistent_session_can_toggle_minimization_between_assumptions() {
    let mut session = SemanticSolveSession::new(permutation_model(4, false, false)).unwrap();
    let all = permutation_solutions(4, false);
    for (enabled, value) in [(true, -3), (false, 1), (true, 3), (false, -1)] {
        let mut options = request(4, enabled, 7);
        options.assumptions = vec![SemanticAssumption { variable: 0, operation: SemanticAssumptionOp::Eq, value }];
        let result = session.solve_with_external_stop(&options, &NEVER_STOP, &mut IgnoreEvents).unwrap();
        let expected = all.iter().filter(|a| a[0] == value).cloned().collect::<Vec<_>>();
        assert_canonical_result(&result, &expected, false);
        assert_metadata(&result, enabled);
    }
}

struct TemporaryInstance(PathBuf);

impl Drop for TemporaryInstance {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn cli_opt_in_is_visible_and_preserves_the_solution_status() {
    let instance = TemporaryInstance(std::env::temp_dir().join(format!("qayd-semantic-minimization-{}.xml", std::process::id())));
    std::fs::write(
        &instance.0,
        r#"<instance format="XCSP3" type="CSP">
          <variables><var id="x"> -3 -1 1 3 </var><var id="y"> -3 -1 1 3 </var></variables>
          <constraints><intension> ne(x,y) </intension></constraints>
        </instance>"#,
    )
    .unwrap();
    for enabled in [false, true] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_qayd"));
        command.args(["--verbose", "--seed", "7", "--search-phase", "0,1:activity:random-seeded"]);
        if enabled {
            command.arg("--semantic-minimization");
        }
        let output = command.arg(&instance.0).output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.contains("s SATISFIABLE"), "{stdout}");
        assert_eq!(stdout.contains("c semantic minimization enabled clauses "), enabled, "{stdout}");
    }
}

/// Diagnostic paired sweep, with no runtime or improvement assertion. Run:
/// `cargo test --release --test semantic_minimization semantic_metrics_sweep -- --ignored --nocapture`
#[test]
#[ignore]
fn semantic_metrics_sweep() {
    eprintln!("family,n,impossible,seed,enabled,nodes,failures,learned_lits,minimized_clauses,removed_lits,elapsed_ms");
    for n in [4, 5, 6] {
        for impossible in [false, true] {
            let expected = permutation_solutions(n, impossible);
            let package = permutation_model(n, impossible, true);
            for seed in [0, 1, 7] {
                // Alternate arm order to reduce a systematic first-run bias.
                let arms = if seed % 2 == 0 { [false, true] } else { [true, false] };
                for enabled in arms {
                    let started = Instant::now();
                    let result = solve_model_silent(&package, &request(n, enabled, seed)).unwrap();
                    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                    assert_canonical_result(&result, &expected, true);
                    assert_metadata(&result, enabled);
                    let stats = result.aggregate_search_stats();
                    eprintln!(
                        "holey_table_permutation,{n},{impossible},{seed},{enabled},{},{},{},{},{},{elapsed_ms:.3}",
                        stats.nodes, stats.failures, stats.learned_lits, stats.semantic_minimized_clauses, stats.semantic_minimized_lits,
                    );
                }
            }
        }
    }
}
