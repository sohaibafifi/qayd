use std::sync::atomic::{AtomicBool, Ordering};

use crate::engines::cp::portfolio::{
    normalize_options, solve_cop_with_progress, uses_diversified_dive, InitialIncumbent, RunOptions, SearchGuidance,
};
use crate::expr::Expr;
use crate::model::{BoolLiteral, CompiledCp, Constraint, Model};
use crate::orchestrator::{compile_cp_plan, solve_cp_plan, IgnoreEvents, SolveBudget, SolveLimits, SolveRequest, SolveStatus};
use crate::problem::{Objective as PhysicalObjective, Problem};
use crate::Solver;

fn pigeonhole(pigeons: usize, holes: usize) -> Model {
    let mut model = Model::new();
    let variables = (0..pigeons * holes).map(|_| model.bool_var()).collect::<Vec<_>>();
    for pigeon in 0..pigeons {
        model.add_constraint(Constraint::Clause(
            (0..holes).map(|hole| BoolLiteral { variable: variables[pigeon * holes + hole], positive: true }).collect(),
        ));
    }
    for hole in 0..holes {
        for left in 0..pigeons {
            for right in left + 1..pigeons {
                model.add_constraint(Constraint::Clause(vec![
                    BoolLiteral { variable: variables[left * holes + hole], positive: false },
                    BoolLiteral { variable: variables[right * holes + hole], positive: false },
                ]));
            }
        }
    }
    model
}

#[test]
fn one_decision_cannot_overshoot_a_hard_cdcl_conflict_budget() {
    let compiled = CompiledCp::compile_interruptible(&pigeonhole(5, 4), &AtomicBool::new(false)).unwrap().unwrap();
    let mut problem = compiled.problem().clone();
    let (solution, stats, complete) = crate::search::decide_sat_assuming_seeded(
        &mut problem.solver,
        &problem.search,
        &[],
        &AtomicBool::new(false),
        0,
        None,
        Some(1),
        Vec::new(),
        Vec::new(),
    );

    assert!(solution.is_none());
    assert_eq!(stats.failures, 1);
    assert!(!complete);
}

#[test]
fn a_cp_conflict_quota_does_not_cancel_the_shared_solve_budget() {
    let model = pigeonhole(5, 4);
    let request =
        SolveRequest { threads: 4, limits: SolveLimits { conflicts: Some(1), ..SolveLimits::default() }, ..SolveRequest::default() };
    let budget = SolveBudget::new(None);
    let plan = compile_cp_plan(&model, &request, &budget).unwrap();
    let result = solve_cp_plan(&model, &plan, &request, &budget, &mut IgnoreEvents).unwrap();

    assert_eq!(result.status(), SolveStatus::Unknown);
    assert_eq!(result.aggregate_search_stats().failures, 1);
    assert!(result.message().is_some_and(|message| message.contains("ConflictLimit")));
    assert!(!budget.expired());
}

#[test]
fn direct_portfolio_options_always_leave_a_complete_worker() {
    let normalized = normalize_options(true, true, RunOptions { workers: 2, probes: usize::MAX, lns: usize::MAX, ..RunOptions::default() });

    assert!(normalized.probes.saturating_add(normalized.lns) < normalized.workers);
}

fn symbolic_sum_problem(variable_count: usize) -> Problem {
    let mut solver = Solver::new();
    let search = (0..variable_count).map(|_| solver.new_var_range(0, 1)).collect::<Vec<_>>();
    Problem { solver, search: search.clone(), objective: Some(PhysicalObjective::Linear(false, vec![1; variable_count], search)) }
}

fn guarded_sum_problem(variable_count: usize) -> Problem {
    let mut solver = Solver::new();
    let search = (0..variable_count).map(|_| solver.new_var_range(0, 1)).collect::<Vec<_>>();
    let objective = Expr::Add(search.iter().map(|&variable| Expr::And(vec![Expr::Var(variable)])).collect());
    Problem { solver, search, objective: Some(PhysicalObjective::Expr(true, objective)) }
}

#[test]
fn diversified_dive_is_reserved_for_large_undirected_symbolic_cops() {
    let options = RunOptions::default();
    let guidance = SearchGuidance::default();
    assert!(!uses_diversified_dive(&symbolic_sum_problem(31), options, &guidance));
    assert!(uses_diversified_dive(&symbolic_sum_problem(64), options, &guidance));
    assert!(!uses_diversified_dive(&symbolic_sum_problem(64), RunOptions { conflict_limit: Some(10_000), ..options }, &guidance,));

    let directed_problem = symbolic_sum_problem(64);
    let directed = SearchGuidance { branch_order: directed_problem.search.clone(), ..SearchGuidance::default() };
    assert!(!uses_diversified_dive(&directed_problem, options, &directed));

    assert!(uses_diversified_dive(&guarded_sum_problem(8), options, &guidance));
    assert!(!uses_diversified_dive(&guarded_sum_problem(8), RunOptions { conflict_limit: Some(10_000), ..options }, &guidance,));
}

#[test]
fn diversified_dive_publishes_before_the_fresh_exact_proof() {
    let stop = AtomicBool::new(false);
    let mut output = Vec::new();
    let mut incumbents = Vec::new();
    let outcome = solve_cop_with_progress(
        symbolic_sum_problem(64),
        true,
        &stop,
        &mut output,
        RunOptions::default(),
        SearchGuidance::default(),
        None,
        false,
        &mut |value, _| incumbents.push(value),
    )
    .unwrap();

    assert!(outcome.proved);
    assert_eq!(outcome.best.as_ref().map(|(_, value)| *value), Some(64));
    assert!(incumbents.contains(&64));
    assert!(outcome.stats.solutions > 0, "the bounded dive statistics were not merged");
    let output = String::from_utf8(output).unwrap();
    assert!(output.contains("source dive worker 0"), "{output}");
}

#[test]
fn small_symbolic_cop_keeps_the_single_exact_trajectory() {
    let mut output = Vec::new();
    let outcome = solve_cop_with_progress(
        symbolic_sum_problem(8),
        true,
        &AtomicBool::new(false),
        &mut output,
        RunOptions::default(),
        SearchGuidance::default(),
        None,
        false,
        &mut |_, _| {},
    )
    .unwrap();

    assert!(outcome.proved);
    assert_eq!(outcome.best.as_ref().map(|(_, value)| *value), Some(8));
    let output = String::from_utf8(output).unwrap();
    assert!(output.contains("source portfolio worker 0"), "{output}");
    assert!(!output.contains("source dive"), "{output}");
}

#[test]
fn interruption_after_a_diversified_incumbent_keeps_the_incumbent() {
    let stop = AtomicBool::new(false);
    let mut output = Vec::new();
    let mut publications = 0;
    let outcome = solve_cop_with_progress(
        symbolic_sum_problem(2_048),
        false,
        &stop,
        &mut output,
        RunOptions::default(),
        SearchGuidance::default(),
        None,
        false,
        &mut |_, _| {
            publications += 1;
            stop.store(true, Ordering::Release);
        },
    )
    .unwrap();

    assert!(publications > 0);
    assert!(outcome.best.is_some());
    assert!(!outcome.proved);
}

#[test]
fn verified_symbolic_incumbent_seeds_the_cutoff_without_republication() {
    let variable_count = 64;
    let initial = InitialIncumbent { solution: vec![1; variable_count], value: variable_count as i64 };
    let mut output = Vec::new();
    let mut publications = 0;
    let outcome = solve_cop_with_progress(
        symbolic_sum_problem(variable_count),
        true,
        &AtomicBool::new(false),
        &mut output,
        RunOptions::default(),
        SearchGuidance::default(),
        Some(initial),
        true,
        &mut |_, _| publications += 1,
    )
    .unwrap();

    assert!(outcome.proved, "the optimal incumbent cutoff should fail at the root");
    assert_eq!(outcome.best, Some((vec![1; variable_count], variable_count as i64)));
    assert_eq!(outcome.stats.solutions, 0, "the exact worker searched without importing the symbolic cutoff");
    assert_eq!(publications, 0);
    assert!(String::from_utf8(output).unwrap().is_empty());
}

#[test]
fn malformed_initial_incumbent_is_rejected() {
    let result = solve_cop_with_progress(
        symbolic_sum_problem(8),
        false,
        &AtomicBool::new(false),
        &mut Vec::new(),
        RunOptions::default(),
        SearchGuidance::default(),
        Some(InitialIncumbent { solution: vec![0; 7], value: 0 }),
        false,
        &mut |_, _| {},
    );
    let error = match result {
        Ok(_) => panic!("malformed initial incumbent was accepted"),
        Err(error) => error,
    };

    assert!(error.contains("7 values for 8 search variables"), "{error}");
}

#[test]
fn pre_cancelled_portfolio_preserves_the_verified_incumbent() {
    let variable_count = 8;
    let outcome = solve_cop_with_progress(
        symbolic_sum_problem(variable_count),
        false,
        &AtomicBool::new(true),
        &mut Vec::new(),
        RunOptions::default(),
        SearchGuidance::default(),
        Some(InitialIncumbent { solution: vec![1; variable_count], value: variable_count as i64 }),
        false,
        &mut |_, _| {},
    )
    .unwrap();

    assert_eq!(outcome.best, Some((vec![1; variable_count], variable_count as i64)));
    assert!(!outcome.proved);
}
