use std::time::Duration;

use qayd::model::{Constraint, IntExpr, IntGlobalConstraint, Model, ModelPackage, Objective, Relation};
use qayd::orchestrator::{
    solve_model, solve_model_silent, EngineKind, EventCallback, EventControl, SolveEvent, SolveLimits, SolveRequest, SolveStatus,
};

fn small_curriculum(separate_first_pair: bool) -> ModelPackage {
    let mut model = Model::new();
    let courses = (0..4).map(|_| model.int_range(0, 1)).collect::<Vec<_>>();
    let counts = (0..2).map(|_| model.int_range(1, 3)).collect::<Vec<_>>();
    let credits = (0..2).map(|_| model.int_range(1, 7)).collect::<Vec<_>>();
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::BinLoads {
        items: courses.clone(),
        sizes: vec![1; 4],
        loads: counts.clone(),
    }));
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::BinLoads {
        items: courses.clone(),
        sizes: vec![1, 3, 2, 2],
        loads: credits.clone(),
    }));
    model.add_constraint(Constraint::Intension(IntExpr::Lt(
        Box::new(courses[0].into()),
        Box::new(courses[if separate_first_pair { 1 } else { 2 }].into()),
    )));
    model.add_constraint(Constraint::Linear {
        terms: credits.iter().copied().map(|variable| (1, variable)).collect(),
        relation: Relation::Eq,
        rhs: 8,
    });
    model.add_constraint(Constraint::Linear {
        terms: counts.iter().copied().map(|variable| (1, variable)).collect(),
        relation: Relation::Eq,
        rhs: 4,
    });
    model.add_objective(Objective::IntExpr {
        minimize: true,
        expr: IntExpr::Add(credits.iter().copied().map(|variable| IntExpr::Mul(vec![variable.into(), variable.into()])).collect()),
    });
    ModelPackage::new(model)
}

#[test]
fn balanced_warm_start_is_replayed_and_proved_by_exact_search() {
    let result = solve_model_silent(&small_curriculum(false), &SolveRequest::default()).unwrap();
    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(result.primal().unwrap().objectives(), [32]);
    assert_eq!(result.primal().unwrap().assignment().integers.len(), 8, "physical objective must not leak into semantic output");
    assert!(result.reports().iter().any(|report| {
        report.engine == Some(EngineKind::IntegerLocalSearch)
            && report.metadata.iter().any(|(key, value)| key == "ls_role" && value == "load_balance_warm_start")
    }));
    assert!(result.reports().iter().any(|report| report.engine == Some(EngineKind::IntegerExact)));
    result.validate_contract().unwrap();
}

#[test]
fn balancing_lower_bound_does_not_override_prerequisites() {
    // A load of four requires pairing weights 1 and 3. The prerequisite puts
    // them in different periods, so the best feasible loads are three and five.
    for threads in [1, 2] {
        let request = SolveRequest { threads, ..SolveRequest::default() };
        let result = solve_model_silent(&small_curriculum(true), &request).unwrap();
        assert_eq!(result.status(), SolveStatus::Optimal);
        assert_eq!(result.primal().unwrap().objectives(), [34]);
        result.validate_contract().unwrap();
    }
}

#[test]
fn stopping_after_balanced_candidate_preserves_feasibility_without_proof() {
    let mut sink = EventCallback(|event| {
        Ok(if matches!(event, SolveEvent::Progress { engine: EngineKind::IntegerLocalSearch, .. }) {
            EventControl::Stop
        } else {
            EventControl::Continue
        })
    });
    let result = solve_model(&small_curriculum(false), &SolveRequest::default(), &mut sink).unwrap();
    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert_eq!(result.primal().unwrap().objectives(), [32]);
    assert!(result.proof().is_none());
    result.validate_contract().unwrap();
}

#[test]
fn balanced_model_zero_budget_cannot_claim_infeasibility() {
    let request = SolveRequest { limits: SolveLimits { time: Some(Duration::ZERO), ..SolveLimits::default() }, ..SolveRequest::default() };
    let result = solve_model_silent(&small_curriculum(false), &request).unwrap();
    assert_eq!(result.status(), SolveStatus::Unknown);
    assert!(result.proof().is_none());
    result.validate_contract().unwrap();
}
