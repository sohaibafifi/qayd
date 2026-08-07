use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use qayd::model::{Constraint, IntExpr, Model, ModelPackage, Objective, Relation};
use qayd::orchestrator::{solve_model, EventCallback, EventControl, ProvenConclusion, SolveEvent, SolveRequest, SolveStatus};

#[test]
fn canonical_cp_proves_mixed_direction_lexicographic_objectives() {
    let mut model = Model::new();
    let x = model.int_range(0, 4);
    let y = model.int_range(0, 10);
    model.add_constraint(Constraint::Linear { terms: vec![(1, x), (1, y)], relation: Relation::Ge, rhs: 4 });
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Add(vec![IntExpr::Variable(x), IntExpr::Variable(y)]) });
    model.add_objective(Objective::IntExpr { minimize: false, expr: IntExpr::Variable(y) });

    let progress = Arc::new(std::sync::Mutex::new(Vec::new()));
    let captured = Arc::clone(&progress);
    let mut sink = EventCallback(move |event| {
        if let SolveEvent::Progress { objectives, .. } = event {
            captured.lock().unwrap().push(objectives);
        }
        Ok(EventControl::Continue)
    });
    let request = SolveRequest { threads: 2, ..SolveRequest::default() };
    let result = solve_model(&ModelPackage::new(model), &request, &mut sink).unwrap();

    assert_eq!(result.status(), SolveStatus::Optimal);
    let candidate = result.primal().unwrap();
    assert_eq!(candidate.objectives(), &[4, 4]);
    assert_eq!(candidate.assignment().integers[x.0], Some(0));
    assert_eq!(candidate.assignment().integers[y.0], Some(4));
    assert_eq!(result.bounds().iter().map(|bound| (bound.tier, bound.value)).collect::<Vec<_>>(), vec![(0, 4), (1, 4)]);
    let proof = result.proof().expect("lexicographic optimum has a proof");
    assert_eq!(proof.conclusion(), ProvenConclusion::Optimal);
    assert_eq!(proof.objective_tiers(), Some(2));
    assert!(progress.lock().unwrap().iter().any(|objectives| objectives.len() == 2 && objectives[0] == 4));
    result.validate_contract().unwrap();
}

#[test]
fn requested_cp_incumbents_cross_the_bus_only_after_transfer_replay() {
    let mut model = Model::new();
    let value = model.int_range(0, 8);
    model.add_objective(Objective::IntExpr { minimize: false, expr: IntExpr::Variable(value) });

    let levels = Arc::new(std::sync::Mutex::new(Vec::new()));
    let captured = Arc::clone(&levels);
    let mut sink = EventCallback(move |event| {
        if let SolveEvent::Candidate(candidate) = event {
            captured.lock().unwrap().push(candidate.verification());
        }
        Ok(EventControl::Continue)
    });
    let request = SolveRequest { publish_incumbent_assignments: true, ..SolveRequest::default() };
    let result = solve_model(&ModelPackage::new(model), &request, &mut sink).unwrap();

    assert_eq!(result.status(), SolveStatus::Optimal);
    let levels = levels.lock().unwrap();
    assert!(levels.contains(&qayd::orchestrator::VerificationLevel::Transfer));
    assert_eq!(levels.last(), Some(&qayd::orchestrator::VerificationLevel::Final));
}

#[test]
fn interruption_keeps_only_the_proved_lexicographic_prefix() {
    let mut model = Model::new();
    let fixed = model.int_range(0, 0);
    let trailing = model.int_range(0, 100);
    // Keep this fixture on the single CP lexicographic path. Without an
    // explicit dependency, semantic decomposition correctly assigns the two
    // independent objective tiers to separate components.
    model.add_constraint(Constraint::Linear { terms: vec![(0, fixed), (0, trailing)], relation: Relation::Eq, rhs: 0 });
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(fixed) });
    model.add_objective(Objective::IntExpr { minimize: false, expr: IntExpr::Variable(trailing) });

    let stopped = Arc::new(AtomicBool::new(false));
    let stopped_by_sink = Arc::clone(&stopped);
    let mut sink = EventCallback(move |event| {
        if matches!(event, SolveEvent::Progress { ref objectives, .. } if objectives.len() == 2)
            && !stopped_by_sink.swap(true, Ordering::AcqRel)
        {
            return Ok(EventControl::Stop);
        }
        Ok(EventControl::Continue)
    });
    let result = solve_model(&ModelPackage::new(model), &SolveRequest::default(), &mut sink).unwrap();

    assert!(stopped.load(Ordering::Acquire));
    assert_eq!(
        result.status(),
        SolveStatus::Satisfiable,
        "primal={:?}, bounds={:?}, message={:?}, stats={:?}",
        result.primal(),
        result.bounds(),
        result.message(),
        result.aggregate_search_stats()
    );
    assert!(result.primal().is_some());
    assert_eq!(result.bounds().iter().map(|bound| (bound.tier, bound.value)).collect::<Vec<_>>(), vec![(0, 0)]);
    assert!(result.proof().is_none());
    assert!(result.message().is_some_and(|message| message.contains("tier 1")));
    result.validate_contract().unwrap();
}
