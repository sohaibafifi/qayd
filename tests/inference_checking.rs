//! Public gates for the opt-in runtime inference diagnostic. Every expected
//! status and optimum comes from enumeration, independently of either arm.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use qayd::constraints::linear::{linear, Relation};
use qayd::constraints::table::{extension, STAR};
use qayd::inference::InferenceCheckConfig;
use qayd::model::{Constraint, IntExpr, IntGlobalConstraint, Model, ModelPackage, Objective};
use qayd::orchestrator::{
    solve_model_silent, CpControls, IgnoreEvents, ProofRequest, ProvenConclusion, SearchPhase, SearchPolicy, SemanticAssumption,
    SemanticAssumptionOp, SemanticSolveSession, SolveMode, SolveRequest, SolveResult, SolveStatus, ValueSelector, VariableSelector,
    VerificationLevel,
};
use qayd::{optimize_with, solve_search, SearchControl, Solver, VarId};

mod common;
use common::oracle;

static NEVER_STOP: AtomicBool = AtomicBool::new(false);

fn holds(lhs: i64, relation: Relation, rhs: i64) -> bool {
    match relation {
        Relation::Eq => lhs == rhs,
        Relation::Ne => lhs != rhs,
        Relation::Le => lhs <= rhs,
        Relation::Lt => lhs < rhs,
        Relation::Ge => lhs >= rhs,
        Relation::Gt => lhs > rhs,
    }
}

fn table_holds(tuples: &[Vec<i32>], assignment: &[i32]) -> bool {
    tuples.iter().any(|tuple| tuple.iter().zip(assignment).all(|(&t, &v)| t == STAR || t == v))
}

#[test]
fn checked_holey_signed_rows_and_star_tables_preserve_both_optima() {
    let domains = vec![vec![-3, -1, 2], vec![-2, 0, 3], vec![-3, 1, 2]];
    let tuples = vec![vec![-3, STAR, 1], vec![-1, -2, STAR], vec![2, 3, -3], vec![2, 0, 2]];
    let coefficients = [2, -3, 1];
    let mut checked_inferences = 0;
    let mut checked_conflicts = 0;
    let mut feasible = 0;
    let mut infeasible = 0;
    for relation in [Relation::Le, Relation::Eq, Relation::Ne, Relation::Ge] {
        for rhs in [-9, -2, 3, 12] {
            let expected = oracle::solutions(&domains, |a| {
                table_holds(&tuples, a)
                    && holds(a.iter().zip(coefficients).map(|(&value, coefficient)| i64::from(value) * coefficient).sum(), relation, rhs)
            });
            feasible += usize::from(!expected.is_empty());
            infeasible += usize::from(expected.is_empty());
            for minimize in [false, true] {
                for semantic_minimization in [false, true] {
                    let mut baseline = None;
                    for enabled in [false, true] {
                        let mut solver = Solver::new();
                        solver.set_semantic_minimization(semantic_minimization);
                        let vars = domains.iter().map(|domain| solver.new_var_set(domain)).collect::<Vec<_>>();
                        extension(&mut solver, &vars, &tuples, true);
                        linear(&mut solver, &coefficients, &vars, relation, rhs);
                        if enabled {
                            solver.set_inference_checking(Some(InferenceCheckConfig::default()));
                        }
                        let (best, stats) = optimize_with(&mut solver, &vars, vars[0], minimize, &NEVER_STOP, |_| {});
                        let optimum = if minimize { expected.iter().map(|a| a[0]).min() } else { expected.iter().map(|a| a[0]).max() };
                        assert_eq!(best.as_ref().map(|(_, value)| *value), optimum, "relation={relation:?}, rhs={rhs}, enabled={enabled}");
                        if let Some((assignment, value)) = best {
                            assert!(expected.contains(&assignment), "oracle rejected {assignment:?}");
                            assert_eq!(assignment[0], value);
                        }
                        let search = (stats.nodes, stats.failures, stats.learned_lits);
                        if enabled {
                            assert_eq!(Some(search), baseline, "diagnostics must preserve deterministic search");
                            let checking = solver.inference_check_stats();
                            checked_inferences += checking.checked_inferences;
                            checked_conflicts += checking.checked_conflicts;
                            assert_eq!(checking.resource_limited, 0);
                        } else {
                            baseline = Some(search);
                            assert!(solver.inference_check_handle().is_none());
                            assert_eq!(solver.inference_check_stats().checked_inferences, 0);
                        }
                    }
                }
            }
        }
    }
    assert!(feasible > 0 && infeasible > 0);
    assert!(checked_inferences > 0, "family must exercise successful pruning checks");
    assert!(checked_conflicts > 0, "family must exercise conflict checks");
}

#[test]
fn checked_boolean_linear_rows_preserve_the_complete_solution_set() {
    let domains = vec![vec![0, 1]; 4];
    let coefficients = [3, -2, 5, 1];
    for rhs in [-1, 3, 6] {
        let expected = oracle::solutions(&domains, |a| {
            a.iter().zip(coefficients).map(|(&value, coefficient)| i64::from(value) * coefficient).sum::<i64>() <= rhs
        })
        .into_iter()
        .collect::<BTreeSet<_>>();
        let mut solver = Solver::new();
        let vars = (0..4).map(|_| solver.new_var_range(0, 1)).collect::<Vec<_>>();
        linear(&mut solver, &coefficients, &vars, Relation::Le, rhs);
        solver.set_inference_checking(Some(InferenceCheckConfig::default()));
        let mut actual = BTreeSet::new();
        solve_search(&mut solver, &vars, |state| {
            actual.insert(vars.iter().map(|&v| state.store.value(v)).collect::<Vec<_>>());
            SearchControl::Continue
        });
        assert_eq!(actual, expected, "rhs={rhs}");
        assert!(solver.inference_check_stats().checked_inferences > 0);
        assert_eq!(solver.inference_check_stats().resource_limited, 0);
    }
}

#[test]
fn root_conflicts_are_checked_before_search() {
    for table in [false, true] {
        let mut solver = Solver::new();
        let x = solver.new_var_set(&[-3, -1, 2]);
        let y = solver.new_var_set(&[-3, -1, 2]);
        if table {
            extension(&mut solver, &[x, y], &[vec![0, 0]], true);
        } else {
            linear(&mut solver, &[1, 1], &[x, y], Relation::Le, -7);
        }
        solver.set_inference_checking(Some(InferenceCheckConfig::default()));
        assert!(solver.propagate().is_err());
        assert!(solver.inference_check_stats().checked_conflicts > 0, "table={table}");
        assert_eq!(solver.inference_check_stats().resource_limited, 0);
    }
}

fn support(n: usize) -> Vec<i32> {
    (0..n as i32).map(|v| 2 * v - 3).collect()
}

fn permutation_solutions(n: usize, impossible: bool) -> Vec<Vec<i32>> {
    let values = support(n);
    let rhs = values.iter().map(|&v| i64::from(v)).sum::<i64>() + i64::from(impossible);
    oracle::solutions(&vec![values; n], |a| {
        a.iter().copied().collect::<BTreeSet<_>>().len() == n && a.iter().map(|&v| i64::from(v)).sum::<i64>() >= rhs
    })
}

fn permutation_solver(n: usize, impossible: bool) -> (Solver, Vec<VarId>) {
    let mut solver = Solver::new();
    let values = support(n);
    let vars = (0..n).map(|_| solver.new_var_set(&values)).collect::<Vec<_>>();
    let neq = values.iter().flat_map(|&a| values.iter().filter(move |&&b| a != b).map(move |&b| vec![a, b])).collect::<Vec<_>>();
    for i in 0..n {
        for j in (i + 1)..n {
            extension(&mut solver, &[vars[i], vars[j]], &neq, true);
        }
    }
    linear(&mut solver, &vec![1; n], &vars, Relation::Ge, values.iter().map(|&v| i64::from(v)).sum::<i64>() + i64::from(impossible));
    (solver, vars)
}

#[test]
fn monitor_observes_work_done_by_solver_clones() {
    let (mut original, vars) = permutation_solver(4, true);
    original.set_inference_checking(Some(InferenceCheckConfig::default()));
    let handle = original.inference_check_handle().unwrap();
    assert_eq!(handle.stats().checked_inferences, 0);
    let mut worker = original.clone();
    let (solution, _) = optimize_with(&mut worker, &vars, vars[0], true, &NEVER_STOP, |_| {});
    assert!(solution.is_none());
    let stats = handle.stats();
    assert!(stats.checked_inferences > 0 && stats.checked_conflicts > 0);
    assert_eq!(original.inference_check_stats().checked_inferences, stats.checked_inferences);
    assert_eq!(worker.inference_check_stats().checked_conflicts, stats.checked_conflicts);
}

#[test]
fn unsupported_negative_table_coverage_is_visible() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 2);
    let y = solver.new_var_range(0, 2);
    let forbidden = vec![vec![0, STAR], vec![1, 1]];
    extension(&mut solver, &[x, y], &forbidden, false);
    solver.set_inference_checking(Some(InferenceCheckConfig::default()));
    let mut actual = BTreeSet::new();
    solve_search(&mut solver, &[x, y], |state| {
        actual.insert(vec![state.store.value(x), state.store.value(y)]);
        SearchControl::Continue
    });
    let expected = oracle::solutions(&[vec![0, 1, 2], vec![0, 1, 2]], |a| !table_holds(&forbidden, a)).into_iter().collect::<BTreeSet<_>>();
    assert_eq!(actual, expected);
    let stats = solver.inference_check_stats();
    assert!(stats.unsupported_calls > 0);
    assert!(stats.uncovered_inferences > 0);
    assert!(solver.inference_check_handle().unwrap().uncovered_families().iter().any(|(_, count)| *count > 0));
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

fn request(n: usize, enabled: bool, seed: u64, activity: bool) -> SolveRequest {
    SolveRequest {
        mode: SolveMode::Exact,
        seed,
        threads: 1,
        cp: CpControls { check_inferences: enabled, ..CpControls::default() },
        search_policy: SearchPolicy::new(vec![SearchPhase::new(
            (0..n).collect(),
            if activity { VariableSelector::Activity } else { VariableSelector::InputOrder },
            ValueSelector::RandomSeeded,
        )]),
        ..SolveRequest::default()
    }
}

fn metadata_number(result: &SolveResult, key: &str) -> u128 {
    result
        .reports()
        .iter()
        .flat_map(|report| &report.metadata)
        .filter(|(name, _)| name == key)
        .map(|(_, value)| value.parse::<u128>().unwrap())
        .sum()
}

fn assert_metadata(result: &SolveResult, enabled: bool) {
    let metadata = result.reports().iter().flat_map(|report| &report.metadata).collect::<Vec<_>>();
    if enabled {
        assert!(metadata.iter().any(|(key, value)| key == "inference_checking" && value == "true"));
        for key in [
            "inference_checked_inferences",
            "inference_checked_conflicts",
            "inference_uncovered_inferences",
            "inference_uncovered_conflicts",
            "inference_unsupported_calls",
            "inference_resource_limited",
            "inference_assignments",
            "inference_work",
            "inference_elapsed_nanos",
            "inference_peak_snapshot_bytes",
        ] {
            assert!(metadata.iter().any(|(name, _)| name == key), "missing diagnostic: {key}");
        }
        assert_eq!(metadata_number(result, "inference_resource_limited"), 0);
    } else {
        assert!(metadata.iter().all(|(key, _)| !key.starts_with("inference_")), "default requests must preserve metadata");
    }
}

fn assert_result(result: &SolveResult, expected: &[Vec<i32>], objective: bool) {
    result.validate_contract().unwrap();
    if expected.is_empty() {
        assert_eq!(result.status(), SolveStatus::Unsatisfiable);
        assert!(result.primal().is_none());
        assert_eq!(result.proof().map(|proof| proof.conclusion()), Some(ProvenConclusion::Unsatisfiable));
    } else {
        assert_eq!(result.status(), if objective { SolveStatus::Optimal } else { SolveStatus::Satisfiable });
        let candidate = result.primal().unwrap();
        assert_eq!(candidate.verification(), VerificationLevel::Final);
        let assignment = candidate.assignment().integers.iter().map(|value| i32::try_from(value.unwrap()).unwrap()).collect::<Vec<_>>();
        assert!(expected.contains(&assignment), "oracle rejected canonical witness {assignment:?}");
        if objective {
            assert_eq!(candidate.objectives(), &[i64::from(expected.iter().map(|a| a[0]).min().unwrap())]);
            assert_eq!(result.proof().map(|proof| proof.conclusion()), Some(ProvenConclusion::Optimal));
        }
    }
}

#[test]
fn canonical_reports_name_uncovered_families() {
    let domains = vec![vec![-1, 0, 1]; 2];
    let forbidden = vec![vec![0, 0], vec![1, 1]];
    let expected = oracle::solutions(&domains, |a| !table_holds(&forbidden, a));
    let mut model = Model::new();
    let vars = domains.iter().map(|domain| model.int_set(domain.clone())).collect::<Vec<_>>();
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Table {
        variables: vars,
        tuples: forbidden.into(),
        positive: false,
    }));
    let result = solve_model_silent(&ModelPackage::new(model), &request(2, true, 7, true)).unwrap();
    assert_result(&result, &expected, false);
    assert_metadata(&result, true);
    assert!(metadata_number(&result, "inference_unsupported_calls") > 0);
    assert!(result.reports().iter().flat_map(|report| &report.metadata).any(|(key, value)| {
        key == "inference_uncovered_family"
            && value.rsplit_once(':').is_some_and(|(family, count)| !family.is_empty() && count.parse::<u64>().unwrap() > 0)
    }));
}

#[test]
fn canonical_dfs_and_lcg_workers_preserve_replay_and_completion() {
    assert!(!CpControls::default().check_inferences);
    let mut checked_inferences = 0;
    let mut checked_conflicts = 0;
    for objective in [false, true] {
        for impossible in [false, true] {
            let expected = permutation_solutions(4, impossible);
            let package = permutation_model(4, impossible, objective);
            for activity in [false, true] {
                for threads in [1, 2] {
                    for enabled in [false, true] {
                        let mut options = request(4, enabled, 7, activity);
                        options.threads = threads;
                        options.cp.semantic_minimization = activity;
                        if objective || impossible {
                            options.proof = ProofRequest::Require;
                        }
                        let result = solve_model_silent(&package, &options).unwrap();
                        assert_result(&result, &expected, objective);
                        assert_metadata(&result, enabled);
                        if !objective && threads == 1 {
                            let expected_driver = if activity { "cdcl" } else { "chronological-dfs" };
                            assert!(result
                                .reports()
                                .iter()
                                .flat_map(|report| &report.metadata)
                                .any(|(key, value)| { key == "csp_search" && value == expected_driver }));
                        }
                        checked_inferences += metadata_number(&result, "inference_checked_inferences");
                        checked_conflicts += metadata_number(&result, "inference_checked_conflicts");
                    }
                }
            }
        }
    }
    assert!(checked_inferences > 0 && checked_conflicts > 0);
}

#[test]
fn session_toggle_keeps_temporary_assumptions_and_statistics_scoped() {
    let mut session = SemanticSolveSession::new(permutation_model(4, false, false)).unwrap();
    let all = permutation_solutions(4, false);
    for (enabled, value, contradictory) in [(true, -3, false), (false, 1, false), (true, 3, true), (true, -1, false), (false, 3, false)] {
        let mut options = request(4, enabled, 7, true);
        options.cp.semantic_minimization = enabled;
        options.assumptions = vec![
            SemanticAssumption { variable: 0, operation: SemanticAssumptionOp::Eq, value },
            SemanticAssumption { variable: 1, operation: SemanticAssumptionOp::Ne, value: 1 },
            SemanticAssumption { variable: 2, operation: SemanticAssumptionOp::Ge, value: -1 },
        ];
        if contradictory {
            options.assumptions.push(SemanticAssumption { variable: 1, operation: SemanticAssumptionOp::Eq, value });
        }
        let result = session.solve_with_external_stop(&options, &NEVER_STOP, &mut IgnoreEvents).unwrap();
        let expected = all
            .iter()
            .filter(|a| a[0] == value && a[1] != 1 && a[2] >= -1 && (!contradictory || a[1] == value))
            .cloned()
            .collect::<Vec<_>>();
        assert_result(&result, &expected, false);
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
fn cli_flag_reports_checks_and_retains_valid_status() {
    let instance = TemporaryInstance(std::env::temp_dir().join(format!("qayd-inference-checking-{}.xml", std::process::id())));
    std::fs::write(
        &instance.0,
        r#"<instance format="XCSP3" type="CSP">
          <variables><var id="x"> -3 -1 1 3 </var><var id="y"> -3 -1 1 3 </var></variables>
          <constraints><sum><list>x y</list><condition>(le,-2)</condition></sum></constraints>
        </instance>"#,
    )
    .unwrap();
    for enabled in [false, true] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_qayd"));
        command.args(["--verbose", "--seed", "7", "--search-phase", "0,1:activity:random-seeded"]);
        if enabled {
            command.args(["--check-inferences", "--semantic-minimization"]);
        }
        let output = command.arg(&instance.0).output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.contains("s SATISFIABLE"), "{stdout}");
        assert_eq!(stdout.contains("c inference checking checked "), enabled, "{stdout}");
    }
}

/// Paired diagnostic overhead sweep, with an independent oracle for every
/// result and no runtime threshold. Run with --release --ignored --nocapture.
#[test]
#[ignore]
fn inference_metrics_sweep() {
    eprintln!("family,n,impossible,seed,enabled,status,expected_optimum,nodes,failures,elapsed_ms,checked_inferences,checked_conflicts,uncovered_inferences,uncovered_conflicts,unsupported_calls,resource_limited,assignments,work,checker_elapsed_nanos,peak_snapshot_bytes");
    for n in [3, 4, 5] {
        for impossible in [false, true] {
            let expected = permutation_solutions(n, impossible);
            let package = permutation_model(n, impossible, true);
            let expected_optimum = expected.iter().map(|a| a[0]).min().map_or_else(|| "none".to_owned(), |value| value.to_string());
            for seed in [0, 1, 7] {
                let arms = if seed % 2 == 0 { [false, true] } else { [true, false] };
                for enabled in arms {
                    // Fresh solves prevent retained clauses from changing an arm.
                    let started = Instant::now();
                    let result = solve_model_silent(&package, &request(n, enabled, seed, true)).unwrap();
                    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                    assert_result(&result, &expected, true);
                    assert_metadata(&result, enabled);
                    let stats = result.aggregate_search_stats();
                    let counters = [
                        "inference_checked_inferences",
                        "inference_checked_conflicts",
                        "inference_uncovered_inferences",
                        "inference_uncovered_conflicts",
                        "inference_unsupported_calls",
                        "inference_resource_limited",
                        "inference_assignments",
                        "inference_work",
                        "inference_elapsed_nanos",
                        "inference_peak_snapshot_bytes",
                    ]
                    .map(|key| metadata_number(&result, key).to_string())
                    .join(",");
                    eprintln!(
                        "holey_table_permutation,{n},{impossible},{seed},{enabled},{:?},{expected_optimum},{},{},{elapsed_ms:.3},{counters}",
                        result.status(), stats.nodes, stats.failures,
                    );
                }
            }
        }
    }
}
