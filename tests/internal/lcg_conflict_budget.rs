use std::sync::atomic::AtomicBool;

use crate::engines::cp::portfolio::{normalize_options, RunOptions};
use crate::model::{BoolLiteral, CompiledCp, Constraint, Model};
use crate::orchestrator::{compile_cp_plan, solve_cp_plan, IgnoreEvents, SolveBudget, SolveLimits, SolveRequest, SolveStatus};

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
