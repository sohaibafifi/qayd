use std::time::Duration;

use qayd::model::{Constraint, IntExpr, IntGlobalConstraint, Model, ModelPackage, Objective, Relation};
use qayd::orchestrator::{
    solve_model, solve_model_silent, EngineKind, EventCallback, EventControl, SearchPhase, SearchPolicy, SolveEvent, SolveLimits,
    SolveRequest, SolveStatus, ValueSelector, VariableSelector,
};

fn weighted_cycle(materialized_objective: bool) -> ModelPackage {
    let mut model = Model::new();
    let positions = (0..3).map(|_| model.int_range(0, 2)).collect::<Vec<_>>();
    let penalties = (0..3).map(|_| model.int_range(0, 2)).collect::<Vec<_>>();
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::AllDifferent { variables: positions.clone(), except: Vec::new() }));
    for (arc, (from, to)) in [(0, 1), (1, 2), (2, 0)].into_iter().enumerate() {
        model.add_constraint(Constraint::Intension(IntExpr::Eq(
            Box::new(penalties[arc].into()),
            Box::new(IntExpr::IfThenElse(
                Box::new(IntExpr::Lt(Box::new(positions[from].into()), Box::new(positions[to].into()))),
                Box::new(IntExpr::Constant(0)),
                Box::new(IntExpr::Sub(Box::new(positions[from].into()), Box::new(positions[to].into()))),
            )),
        )));
    }
    let terms = penalties.iter().copied().zip([2, 3, 1]).map(|(variable, coefficient)| (coefficient, variable)).collect::<Vec<_>>();
    let objective = if materialized_objective {
        let objective = model.int_range(7, 19);
        let mut definition = terms;
        definition.push((-1, objective));
        model.add_constraint(Constraint::Linear { terms: definition, relation: Relation::Eq, rhs: -7 });
        objective.into()
    } else {
        let mut summands = terms
            .into_iter()
            .map(|(coefficient, variable)| IntExpr::Mul(vec![IntExpr::Constant(coefficient), variable.into()]))
            .collect::<Vec<_>>();
        summands.push(IntExpr::Constant(7));
        IntExpr::Add(summands)
    };
    model.add_objective(Objective::IntExpr { minimize: true, expr: objective });
    ModelPackage::new(model)
}

fn has_ordering_warm_start(result: &qayd::orchestrator::SolveResult) -> bool {
    result.reports().iter().any(|report| {
        report.engine == Some(EngineKind::IntegerLocalSearch)
            && report.metadata.iter().any(|(key, value)| key == "ls_role" && value == "ordering_warm_start")
    })
}

#[test]
fn ordering_warm_start_replays_weighted_objectives_and_exact_search_proves_them() {
    // Enumerating the six permutations gives a minimum penalty of two: only
    // the weight-one arc goes backwards, spanning two positions.
    for materialized in [false, true] {
        let result = solve_model_silent(&weighted_cycle(materialized), &SolveRequest::default()).unwrap();
        assert_eq!(result.status(), SolveStatus::Optimal);
        assert_eq!(result.primal().unwrap().objectives(), [9]);
        assert_eq!(result.primal().unwrap().assignment().integers.len(), if materialized { 7 } else { 6 });
        assert!(has_ordering_warm_start(&result), "recognized ordering did not pass the warm-start replay boundary");
        assert!(result.reports().iter().any(|report| report.engine == Some(EngineKind::IntegerExact)));
        result.validate_contract().unwrap();
    }
}

#[test]
fn stopping_after_ordering_candidate_preserves_feasibility_without_proof() {
    let mut sink = EventCallback(|event| {
        Ok(if matches!(event, SolveEvent::Progress { engine: EngineKind::IntegerLocalSearch, .. }) {
            EventControl::Stop
        } else {
            EventControl::Continue
        })
    });
    let result = solve_model(&weighted_cycle(true), &SolveRequest::default(), &mut sink).unwrap();
    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert_eq!(result.primal().unwrap().objectives(), [9]);
    assert!(has_ordering_warm_start(&result));
    assert!(result.proof().is_none());
    result.validate_contract().unwrap();
}

#[test]
fn ordering_zero_budget_cannot_claim_infeasibility() {
    let request = SolveRequest { limits: SolveLimits { time: Some(Duration::ZERO), ..SolveLimits::default() }, ..SolveRequest::default() };
    let result = solve_model_silent(&weighted_cycle(true), &request).unwrap();
    assert_eq!(result.status(), SolveStatus::Unknown);
    assert!(result.proof().is_none());
    result.validate_contract().unwrap();
}

#[test]
fn explicit_value_policy_bypasses_ordering_warm_start() {
    for (variable, value) in [(VariableSelector::InputOrder, ValueSelector::Min), (VariableSelector::Auto, ValueSelector::Auto)] {
        let request = SolveRequest {
            search_policy: SearchPolicy::new(vec![SearchPhase::new((0..7).collect(), variable, value)]),
            ..SolveRequest::default()
        };
        let result = solve_model_silent(&weighted_cycle(true), &request).unwrap();
        assert_eq!(result.status(), SolveStatus::Optimal);
        assert_eq!(result.primal().unwrap().objectives(), [9]);
        assert!(!has_ordering_warm_start(&result));
        result.validate_contract().unwrap();
    }
}
