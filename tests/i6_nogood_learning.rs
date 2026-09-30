//! Public oracle and lifecycle checks for extended nogoods and CPIP learning.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use qayd::constraints::linear::{linear, Relation as LinearRelation};
use qayd::constraints::table::extension;
use qayd::inference::InferenceCheckConfig;
use qayd::model::{Constraint, IntVarRef, Relation};
use qayd::orchestrator::{
    solve_model_silent, CpControls, IgnoreEvents, ProofRequest, ProvenConclusion, SearchPhase, SearchPolicy, SemanticAssumption,
    SemanticAssumptionOp, SemanticSolveSession, SolveLimits, SolveMode, SolveRequest, SolveResult, SolveStatus, ValueSelector,
    VariableSelector, VerificationLevel,
};
use qayd::{optimize_var_assuming, solve_search, Assumption, AssumptionOp, BoolLit, SearchControl, SolveStats, Solver, VarId};

mod common;
#[path = "common/nogood_models.rs"]
pub mod nogood_models;

const ARMS: [(bool, bool); 4] = [(false, false), (true, false), (false, true), (true, true)];
static NEVER_STOP: AtomicBool = AtomicBool::new(false);

fn configure(solver: &mut Solver, extended: bool, cpip: bool) {
    solver.set_extended_nogoods(extended);
    solver.set_cpip(cpip);
    assert_eq!(solver.extended_nogoods_enabled(), extended);
    assert_eq!(solver.cpip_enabled(), cpip);
}

fn assert_baseline_counters(stats: SolveStats, extended: bool, cpip: bool) {
    if !extended && !cpip {
        assert_eq!(stats.nogoods, Default::default());
    }
}

#[derive(Clone, Debug)]
struct SmallCase {
    domains: Vec<Vec<i32>>,
    coefficients: Vec<i64>,
    rhs: i64,
    table: bool,
}

impl SmallCase {
    fn oracle(&self) -> BTreeSet<Vec<i32>> {
        common::oracle::solutions(&self.domains, |assignment| {
            assignment.iter().copied().collect::<BTreeSet<_>>().len() == assignment.len()
                && assignment
                    .iter()
                    .zip(&self.coefficients)
                    .map(|(&value, &coefficient)| i128::from(value) * i128::from(coefficient))
                    .sum::<i128>()
                    >= i128::from(self.rhs)
        })
        .into_iter()
        .collect()
    }

    fn solver(&self, extended: bool, cpip: bool) -> (Solver, Vec<VarId>) {
        let mut solver = Solver::new();
        configure(&mut solver, extended, cpip);
        let variables = self.domains.iter().map(|domain| solver.new_var_set(domain)).collect::<Vec<_>>();
        for left in 0..variables.len() {
            for right in left + 1..variables.len() {
                if self.table {
                    let tuples = self.domains[left]
                        .iter()
                        .flat_map(|&a| self.domains[right].iter().filter(move |&&b| a != b).map(move |&b| vec![a, b]))
                        .collect::<Vec<_>>();
                    extension(&mut solver, &[variables[left], variables[right]], &tuples, true);
                } else {
                    // i32::MIN is a wildcard in positive tables. Linear
                    // disequality preserves its ordinary endpoint meaning.
                    linear(&mut solver, &[1, -1], &[variables[left], variables[right]], LinearRelation::Ne, 0);
                }
            }
        }
        linear(&mut solver, &self.coefficients, &variables, LinearRelation::Ge, self.rhs);
        solver.set_inference_checking(Some(InferenceCheckConfig::default()));
        (solver, variables)
    }
}

#[test]
fn four_modes_preserve_sparse_signed_solutions_and_both_optima() {
    let cases = [
        SmallCase { domains: vec![vec![-3, -1, 2], vec![-2, 0, 3], vec![-4, 1, 4]], coefficients: vec![2, -3, 1], rhs: -2, table: true },
        SmallCase { domains: vec![vec![-3, -1, 2]; 3], coefficients: vec![1, 1, 1], rhs: -1, table: true },
        SmallCase { domains: vec![vec![i32::MIN, 0, i32::MAX]; 3], coefficients: vec![1, 1, 1], rhs: -1, table: false },
        SmallCase { domains: vec![vec![i32::MIN, i32::MAX]; 2], coefficients: vec![-1, 1], rhs: 1, table: false },
    ];
    for case in cases {
        let expected = case.oracle();
        for (extended, cpip) in ARMS {
            let (mut solver, variables) = case.solver(extended, cpip);
            let mut actual = BTreeSet::new();
            let stats = solve_search(&mut solver, &variables, |state| {
                actual.insert(variables.iter().map(|&variable| state.store.value(variable)).collect::<Vec<_>>());
                SearchControl::Continue
            });
            assert_eq!(actual, expected, "extended={extended}, cpip={cpip}, case={case:?}");
            assert_baseline_counters(stats, extended, cpip);

            for minimizing in [false, true] {
                let (mut solver, variables) = case.solver(extended, cpip);
                let (best, stats, complete) =
                    optimize_var_assuming(&mut solver, &variables, &[], variables[0], minimizing, &NEVER_STOP, |_, _| {});
                assert!(complete, "extended={extended}, cpip={cpip}, case={case:?}");
                let optimum =
                    if minimizing { expected.iter().map(|values| values[0]).min() } else { expected.iter().map(|values| values[0]).max() };
                assert_eq!(best.as_ref().map(|(_, value)| *value), optimum.map(i64::from));
                if let Some((assignment, _)) = best {
                    assert!(expected.contains(&assignment));
                }
                assert_baseline_counters(stats, extended, cpip);
                assert_eq!(solver.inference_check_stats().resource_limited, 0);
            }
        }
    }
}

#[test]
fn conflict_learning_on_table_permutations_matches_the_constructive_oracle() {
    for n in [4, 5] {
        for impossible in [false, true] {
            let expected = nogood_models::permutation_solutions(n, impossible);
            for (extended, cpip) in ARMS {
                let (mut solver, variables) = nogood_models::permutation_solver(n, impossible);
                configure(&mut solver, extended, cpip);
                solver.set_inference_checking(Some(InferenceCheckConfig::default()));
                let (best, stats, complete) =
                    optimize_var_assuming(&mut solver, &variables, &[], variables[0], true, &NEVER_STOP, |_, _| {});
                assert!(complete, "n={n}, impossible={impossible}, extended={extended}, cpip={cpip}");
                assert_eq!(best.as_ref().map(|(_, value)| *value), expected.iter().map(|values| i64::from(values[0])).min());
                if let Some((assignment, _)) = best {
                    assert!(expected.contains(&assignment));
                }
                if impossible {
                    assert!(stats.failures > 0 && stats.learned_lits > 0, "UNSAT must exercise the learning path: {stats:?}");
                }
                assert_baseline_counters(stats, extended, cpip);
            }
        }
    }
}

fn request(count: usize, extended: bool, cpip: bool, threads: usize) -> SolveRequest {
    SolveRequest {
        mode: SolveMode::Exact,
        seed: 7,
        threads,
        cp: CpControls { extended_nogoods: extended, cpip, check_inferences: true, ..CpControls::default() },
        search_policy: SearchPolicy::new(vec![SearchPhase::new(
            (0..count).collect(),
            VariableSelector::Activity,
            ValueSelector::RandomSeeded,
        )]),
        ..SolveRequest::default()
    }
}

fn assignment(result: &SolveResult) -> Vec<i32> {
    let candidate = result.primal().expect("a feasible result has a candidate");
    assert_eq!(candidate.verification(), VerificationLevel::Final);
    candidate.assignment().integers.iter().map(|value| i32::try_from(value.expect("complete integer assignment")).unwrap()).collect()
}

fn verify_case(result: &SolveResult, case: &nogood_models::ModelCase) {
    result.validate_contract().unwrap();
    if case.feasible {
        assert_eq!(result.status(), if case.objective { SolveStatus::Optimal } else { SolveStatus::Satisfiable });
        assert!(case.accepts(&assignment(result)));
        if case.objective {
            assert_eq!(result.primal().unwrap().objectives(), &[case.optimum.unwrap()]);
            assert_eq!(result.proof().map(|proof| proof.conclusion()), Some(ProvenConclusion::Optimal));
        }
    } else {
        assert_eq!(result.status(), SolveStatus::Unsatisfiable);
        assert!(result.primal().is_none());
        assert_eq!(result.proof().map(|proof| proof.conclusion()), Some(ProvenConclusion::Unsatisfiable));
    }
}

#[test]
fn canonical_four_modes_keep_verified_statuses_and_required_proofs_with_two_workers() {
    assert!(!CpControls::default().extended_nogoods);
    assert!(!CpControls::default().cpip);
    assert!(!Solver::new().extended_nogoods_enabled());
    assert!(!Solver::new().cpip_enabled());
    for impossible in [false, true] {
        for objective in [false, true] {
            for case in [nogood_models::permutation(4, impossible, objective), nogood_models::pigeonhole(2, impossible, objective)] {
                for (extended, cpip) in ARMS {
                    for threads in [1, 2] {
                        let mut options = request(case.variable_count, extended, cpip, threads);
                        if objective || impossible {
                            options.proof = ProofRequest::Require;
                        }
                        let result = solve_model_silent(&case.package, &options).unwrap_or_else(|error| {
                            panic!("impossible={impossible}, objective={objective}, extended={extended}, cpip={cpip}, threads={threads}: {error:?}")
                        });
                        verify_case(&result, &case);
                        assert_baseline_counters(result.aggregate_search_stats(), extended, cpip);
                    }
                }
            }
        }
    }
}

#[test]
fn cloned_solvers_and_temporary_bound_value_assumptions_preserve_optima() {
    let expected = nogood_models::permutation_solutions(4, false);
    for (extended, cpip) in ARMS {
        let (mut template, variables) = nogood_models::permutation_solver(4, false);
        configure(&mut template, extended, cpip);
        for contradictory in [false, true] {
            let mut solver = template.clone();
            assert_eq!(solver.extended_nogoods_enabled(), extended);
            assert_eq!(solver.cpip_enabled(), cpip);
            let mut assumptions = vec![
                Assumption { var: variables[0], op: AssumptionOp::Ge, value: -1 },
                Assumption { var: variables[1], op: AssumptionOp::Le, value: 1 },
                Assumption { var: variables[2], op: AssumptionOp::Ne, value: 3 },
            ];
            if contradictory {
                assumptions.extend([Assumption::eq(variables[0], -1), Assumption::eq(variables[1], -1)]);
            }
            let candidates = expected
                .iter()
                .filter(|values| {
                    values[0] >= -1 && values[1] <= 1 && values[2] != 3 && (!contradictory || (values[0] == -1 && values[1] == -1))
                })
                .collect::<Vec<_>>();
            let (best, stats, complete) =
                optimize_var_assuming(&mut solver, &variables, &assumptions, variables[3], false, &NEVER_STOP, |_, _| {});
            assert!(complete);
            assert_eq!(best.as_ref().map(|(_, value)| *value), candidates.iter().map(|values| i64::from(values[3])).max());
            if let Some((values, _)) = best {
                assert!(candidates.contains(&&values));
            }
            assert_baseline_counters(stats, extended, cpip);
        }
        for &variable in &variables {
            let actual = template.store.values(variable).collect::<BTreeSet<_>>();
            assert_eq!(actual, nogood_models::permutation_support(4).into_iter().collect());
        }
    }
}

#[test]
fn persistent_session_toggles_both_modes_without_retaining_temporary_assumptions() {
    let case = nogood_models::permutation(4, false, false);
    let expected = nogood_models::permutation_solutions(4, false);
    let mut session = SemanticSolveSession::new(case.package).unwrap();
    for (extended, cpip) in ARMS.into_iter().chain(ARMS.into_iter().rev()) {
        for contradictory in [false, true] {
            let mut options = request(4, extended, cpip, 1);
            options.assumptions = vec![SemanticAssumption { variable: 0, operation: SemanticAssumptionOp::Eq, value: -1 }];
            if contradictory {
                options.assumptions.push(SemanticAssumption { variable: 1, operation: SemanticAssumptionOp::Eq, value: -1 });
                options.proof = ProofRequest::Require;
            }
            let result = session.solve_with_external_stop(&options, &NEVER_STOP, &mut IgnoreEvents).unwrap();
            result.validate_contract().unwrap();
            if contradictory {
                assert_eq!(result.status(), SolveStatus::Unsatisfiable);
                assert_eq!(result.proof().map(|proof| proof.conclusion()), Some(ProvenConclusion::Unsatisfiable));
            } else {
                assert_eq!(result.status(), SolveStatus::Satisfiable);
                let values = assignment(&result);
                assert_eq!(values[0], -1);
                assert!(expected.contains(&values));
            }
        }
    }
    let mut fresh = request(4, true, true, 1);
    fresh.assumptions = vec![SemanticAssumption { variable: 0, operation: SemanticAssumptionOp::Eq, value: 3 }];
    let result = session.solve_with_external_stop(&fresh, &NEVER_STOP, &mut IgnoreEvents).unwrap();
    assert_eq!(result.status(), SolveStatus::Satisfiable);
    let values = assignment(&result);
    assert_eq!(values[0], 3);
    assert!(expected.contains(&values));
}

fn metadata_count(result: &SolveResult, key: &str) -> u64 {
    result
        .reports()
        .iter()
        .flat_map(|report| &report.metadata)
        .filter(|(name, _)| name == key)
        .map(|(_, value)| value.parse::<u64>().unwrap())
        .sum()
}

#[test]
fn all_research_controls_coexist_in_one_connected_cp_component() {
    let mut case = nogood_models::permutation(4, false, true);
    let model = &mut case.package.model;
    let booleans = (0..5).map(|_| model.int_range(0, 1)).collect::<Vec<_>>();
    model.add_constraint(Constraint::Linear {
        terms: booleans.iter().map(|&variable| (1, variable)).collect(),
        relation: Relation::Le,
        rhs: 1,
    });
    // Every permutation extends with all Booleans zero, while this live row
    // connects the scopes so decomposition cannot separate the four controls.
    model.add_constraint(Constraint::Linear { terms: vec![(1, IntVarRef(0)), (1, booleans[0])], relation: Relation::Le, rhs: 3 });
    let mut deferred = 0;
    let mut checked = 0;
    for (extended, cpip) in ARMS {
        for threads in [1, 2] {
            let mut options = request(9, extended, cpip, threads);
            options.cp.semantic_minimization = true;
            options.cp.deferred_explanations = true;
            options.cp.explanation_metrics = true;
            options.proof = ProofRequest::Require;
            options.search_policy =
                SearchPolicy::new(vec![SearchPhase::new((4..9).chain(0..4).collect(), VariableSelector::Activity, ValueSelector::Max)]);
            let result = solve_model_silent(&case.package, &options).unwrap();
            result.validate_contract().unwrap();
            assert_eq!(result.status(), SolveStatus::Optimal);
            let values = assignment(&result);
            assert!(case.accepts(&values[..4]));
            assert!(values[4..].iter().all(|&value| value == 0 || value == 1));
            assert!(values[4..].iter().sum::<i32>() <= 1);
            assert!(values[0] + values[4] <= 3);
            assert_eq!(result.primal().unwrap().objectives(), &[-3]);
            deferred += metadata_count(&result, "explanation_created_handles");
            checked += metadata_count(&result, "inference_checked_inferences");
            assert_eq!(metadata_count(&result, "inference_resource_limited"), 0);
        }
    }
    assert!(deferred > 0, "the coexistence fixture must actually create deferred reasons");
    assert!(checked > 0, "the coexistence fixture must actually check supported reasons");
}

#[test]
fn cancellation_and_zero_deadline_never_publish_a_false_completion() {
    let case = nogood_models::permutation(4, false, true);
    for (extended, cpip) in ARMS {
        let mut options = request(case.variable_count, extended, cpip, 2);
        options.limits = SolveLimits { time: Some(Duration::ZERO), ..SolveLimits::default() };
        let result = solve_model_silent(&case.package, &options).unwrap();
        result.validate_contract().unwrap();
        assert_eq!(result.status(), SolveStatus::Unknown);
        assert!(result.primal().is_none() && result.proof().is_none());

        let (mut solver, variables) = nogood_models::permutation_solver(4, false);
        configure(&mut solver, extended, cpip);
        let stop = AtomicBool::new(false);
        let (best, _, complete) = optimize_var_assuming(&mut solver, &variables, &[], variables[0], true, &stop, |_, values| {
            assert!(case.accepts(values));
            stop.store(true, Ordering::Relaxed);
        });
        assert!(stop.load(Ordering::Relaxed));
        assert!(!complete, "a callback cancellation must not be reported as exhaustive search");
        if let Some((values, _)) = best {
            assert!(case.accepts(&values));
        }
    }
}

#[test]
fn boolean_proof_logging_keeps_ordinary_resolution_in_every_mode() {
    let mut streams = Vec::new();
    for (extended, cpip) in ARMS {
        let mut solver = Solver::new();
        configure(&mut solver, extended, cpip);
        let variables = (0..6).map(|_| solver.new_var_range(0, 1)).collect::<Vec<_>>();
        let mut clauses =
            variables.chunks_exact(2).map(|row| row.iter().copied().map(BoolLit::positive).collect::<Vec<_>>()).collect::<Vec<_>>();
        for hole in 0..2 {
            for first in 0..3 {
                for second in first + 1..3 {
                    clauses.push(vec![BoolLit::negative(variables[first * 2 + hole]), BoolLit::negative(variables[second * 2 + hole])]);
                }
            }
        }
        let mut proof = Vec::new();
        let (solution, stats, complete) =
            qayd::search::solve_bool_cnf_seeded_with_proof(&mut solver, &variables, &clauses, &NEVER_STOP, 7, |clause| {
                proof.push(clause.to_vec())
            })
            .unwrap();
        assert!(complete && solution.is_none());
        assert!(proof.last().is_some_and(Vec::is_empty));
        assert_eq!(stats.nogoods.extended_scans, 0);
        assert_eq!(stats.nogoods.cpip_attempts, 0);
        if extended || cpip {
            assert!(stats.nogoods.skipped_proof_logging > 0);
        }
        assert_baseline_counters(stats, extended, cpip);
        streams.push(proof);
    }
    for proof in &streams[1..] {
        assert_eq!(proof, &streams[0], "integer semantic modes must be suppressed with a Boolean proof logger");
    }
}
