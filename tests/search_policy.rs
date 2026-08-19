use std::collections::BTreeSet;
use std::str::FromStr;
use std::sync::atomic::AtomicBool;

use qayd::model::{Constraint, IntExpr, Model, ModelPackage, Objective, Relation};
use qayd::orchestrator::{
    solve_model_silent, CpControls, ProofKind, SearchPhase, SearchPolicy, SemanticSolveSession, SolveError, SolveMode, SolveRequest,
    SolveStatus, ValueSelector, VariableSelector,
};
#[cfg(feature = "lp-relaxation")]
use qayd::orchestrator::{LinearBackendMode, LinearControls};

fn policy(scope: Vec<usize>, variable: VariableSelector, value: ValueSelector) -> SearchPolicy {
    SearchPolicy::new(vec![SearchPhase::new(scope, variable, value)])
}

#[test]
fn selector_spellings_are_canonical_and_diagnostic() {
    for (spelling, expected) in [
        ("auto", VariableSelector::Auto),
        ("input-order", VariableSelector::InputOrder),
        ("first-fail", VariableSelector::FirstFail),
        ("dom-wdeg", VariableSelector::DomWdeg),
        ("activity", VariableSelector::Activity),
    ] {
        assert_eq!(VariableSelector::from_str(spelling).unwrap(), expected);
        assert_eq!(expected.to_string(), spelling);
    }
    for (spelling, expected) in [
        ("auto", ValueSelector::Auto),
        ("min", ValueSelector::Min),
        ("max", ValueSelector::Max),
        ("median", ValueSelector::Median),
        ("random-seeded", ValueSelector::RandomSeeded),
        ("hint", ValueSelector::Hint),
    ] {
        assert_eq!(ValueSelector::from_str(spelling).unwrap(), expected);
        assert_eq!(expected.to_string(), spelling);
    }
    assert!(VariableSelector::from_str("firstfail").unwrap_err().contains("unknown variable selector"));
    assert!(VariableSelector::from_str("first_fail").unwrap_err().contains("unknown variable selector"));
    assert!(ValueSelector::from_str("random").unwrap_err().contains("unknown value selector"));
    assert!(ValueSelector::from_str("random_seeded").unwrap_err().contains("unknown value selector"));
}

fn solve(request: SolveRequest, model: Model) -> qayd::orchestrator::SolveResult {
    solve_model_silent(&ModelPackage::new(model), &request).unwrap()
}

fn unequal_domains_model() -> Model {
    let mut model = Model::new();
    let wide = model.int_range(0, 2);
    let narrow = model.int_range(0, 1);
    model.add_constraint(Constraint::Intension(IntExpr::Ne(Box::new(IntExpr::Variable(wide)), Box::new(IntExpr::Variable(narrow)))));
    model
}

#[test]
fn an_all_variable_auto_phase_preserves_the_legacy_single_worker_trace() {
    for seed in 0..8 {
        let legacy = solve(SolveRequest { mode: SolveMode::Exact, seed, ..SolveRequest::default() }, unequal_domains_model());
        let phased = solve(
            SolveRequest {
                mode: SolveMode::Exact,
                seed,
                search_policy: policy(vec![0, 1], VariableSelector::Auto, ValueSelector::Auto),
                ..SolveRequest::default()
            },
            unequal_domains_model(),
        );

        assert_eq!(phased.status(), legacy.status());
        assert_eq!(phased.primal().unwrap().assignment(), legacy.primal().unwrap().assignment());
        assert_eq!(phased.aggregate_search_stats().nodes, legacy.aggregate_search_stats().nodes);
        assert_eq!(phased.aggregate_search_stats().failures, legacy.aggregate_search_stats().failures);
    }
}

#[test]
fn ordered_variable_selectors_control_the_first_decision() {
    let cases = [
        (VariableSelector::InputOrder, vec![Some(0), Some(1)]),
        (VariableSelector::FirstFail, vec![Some(1), Some(0)]),
        (VariableSelector::DomWdeg, vec![Some(1), Some(0)]),
        (VariableSelector::Activity, vec![Some(0), Some(1)]),
    ];
    for (selector, expected) in cases {
        let request = SolveRequest {
            mode: SolveMode::Exact,
            search_policy: policy(vec![0, 1], selector, ValueSelector::Min),
            ..SolveRequest::default()
        };
        let result = solve(request, unequal_domains_model());
        assert_eq!(result.status(), SolveStatus::Satisfiable, "selector={selector}");
        assert_eq!(result.primal().unwrap().assignment().integers, expected, "selector={selector}");
    }
}

fn single_variable_model() -> Model {
    let mut model = Model::new();
    model.int_range(0, 10);
    model
}

#[test]
fn endpoint_and_midpoint_value_selectors_are_exact() {
    for (selector, expected) in [(ValueSelector::Min, 0), (ValueSelector::Max, 10), (ValueSelector::Median, 5)] {
        let request = SolveRequest {
            mode: SolveMode::Exact,
            search_policy: policy(vec![0], VariableSelector::InputOrder, selector),
            ..SolveRequest::default()
        };
        let result = solve(request, single_variable_model());
        assert_eq!(result.status(), SolveStatus::Satisfiable);
        assert_eq!(result.primal().unwrap().assignment().integers, [Some(expected)], "selector={selector}");
    }
}

#[test]
fn median_uses_the_first_supported_value_at_or_above_the_numeric_midpoint() {
    let mut model = Model::new();
    let value = model.int_set(vec![0, 3, 9]);
    let request = SolveRequest {
        mode: SolveMode::Exact,
        search_policy: policy(vec![value.0], VariableSelector::InputOrder, ValueSelector::Median),
        ..SolveRequest::default()
    };

    let result = solve(request, model);

    assert_eq!(result.primal().unwrap().assignment().integers[value.0], Some(9));

    let mut binary = Model::new();
    binary.bool_var();
    let binary = solve(
        SolveRequest {
            mode: SolveMode::Exact,
            search_policy: policy(vec![0], VariableSelector::InputOrder, ValueSelector::Median),
            ..SolveRequest::default()
        },
        binary,
    );
    assert_eq!(binary.primal().unwrap().assignment().integers[0], Some(1));
}

#[test]
fn seeded_value_selection_is_reproducible_and_seed_sensitive() {
    let mut observed = BTreeSet::new();
    for seed in 0..16 {
        let request = SolveRequest {
            mode: SolveMode::Exact,
            seed,
            search_policy: policy(vec![0], VariableSelector::InputOrder, ValueSelector::RandomSeeded),
            ..SolveRequest::default()
        };
        let first = solve(request.clone(), single_variable_model());
        let second = solve(request, single_variable_model());
        let first_value = first.primal().unwrap().assignment().integers[0];
        assert_eq!(first_value, second.primal().unwrap().assignment().integers[0]);
        observed.insert(first_value);
    }
    assert!(observed.len() > 1, "different seeds should explore more than one supported value: {observed:?}");
}

#[test]
fn seeded_values_keep_semantic_identity_across_decomposition_and_sessions() {
    let model = || {
        let mut model = Model::new();
        for _ in 0..17 {
            model.int_range(0, 20);
        }
        model
    };
    let request = SolveRequest {
        mode: SolveMode::Exact,
        seed: 42,
        search_policy: policy((0..17).collect(), VariableSelector::InputOrder, ValueSelector::RandomSeeded),
        ..SolveRequest::default()
    };

    let decomposed = solve(request.clone(), model());
    let monolithic = solve(
        SolveRequest {
            limits: qayd::orchestrator::SolveLimits { time: Some(std::time::Duration::from_secs(60)), ..request.limits },
            ..request.clone()
        },
        model(),
    );
    let mut session = SemanticSolveSession::new(ModelPackage::new(model())).unwrap();
    let session = session.solve_with_external_stop(&request, &AtomicBool::new(false), &mut qayd::orchestrator::IgnoreEvents).unwrap();

    assert_eq!(decomposed.primal().unwrap().assignment(), monolithic.primal().unwrap().assignment());
    assert_eq!(decomposed.primal().unwrap().assignment(), session.primal().unwrap().assignment());
}

#[test]
fn implicit_auto_fallback_completes_every_unscoped_variable() {
    let mut model = Model::new();
    let primary = model.bool_var();
    let completion = model.bool_var();
    model.add_constraint(Constraint::Linear { terms: vec![(0, primary), (0, completion)], relation: Relation::Eq, rhs: 0 });
    let request = SolveRequest {
        mode: SolveMode::Exact,
        search_policy: policy(vec![primary.0], VariableSelector::InputOrder, ValueSelector::Max),
        ..SolveRequest::default()
    };

    let result = solve(request, model);

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert_eq!(result.primal().unwrap().assignment().integers, [Some(1), Some(0)]);
}

#[test]
fn fallback_preserves_optimality_and_unsatisfiability_proofs() {
    let mut optimal = Model::new();
    let guide = optimal.bool_var();
    let objective = optimal.int_range(0, 4);
    optimal.add_constraint(Constraint::Linear { terms: vec![(0, guide), (0, objective)], relation: Relation::Eq, rhs: 0 });
    optimal.add_objective(Objective::IntExpr { minimize: false, expr: IntExpr::Variable(objective) });
    let request = SolveRequest {
        mode: SolveMode::Exact,
        search_policy: policy(vec![guide.0], VariableSelector::InputOrder, ValueSelector::Min),
        linear: qayd::orchestrator::LinearControls {
            root_time: std::time::Duration::ZERO,
            ..qayd::orchestrator::LinearControls::default()
        },
        ..SolveRequest::default()
    };
    let optimal = solve(request.clone(), optimal);
    assert_eq!(optimal.status(), SolveStatus::Optimal);
    assert_eq!(optimal.primal().unwrap().objectives(), [4]);
    assert!(matches!(optimal.proof().map(|proof| proof.kind()), Some(ProofKind::CompleteSearch { objective_tiers: 1, .. })));

    let mut unsat = Model::new();
    let guide = unsat.bool_var();
    let hidden = unsat.bool_var();
    unsat.add_constraint(Constraint::Linear { terms: vec![(0, guide), (1, hidden)], relation: Relation::Eq, rhs: 0 });
    unsat.add_constraint(Constraint::Linear { terms: vec![(1, hidden)], relation: Relation::Eq, rhs: 1 });
    let mut unsat_request = request;
    unsat_request.linear = qayd::orchestrator::LinearControls::default();
    let unsat = solve(unsat_request, unsat);
    assert_eq!(unsat.status(), SolveStatus::Unsatisfiable);
    assert!(matches!(unsat.proof().map(|proof| proof.kind()), Some(ProofKind::CompleteSearch { objective_tiers: 0, .. })));
}

#[test]
fn invalid_or_ambiguous_policies_are_rejected() {
    let invalid = [
        SearchPolicy::new(vec![SearchPhase::new(Vec::new(), VariableSelector::Auto, ValueSelector::Auto)]),
        SearchPolicy::new(vec![SearchPhase::new(vec![0, 0], VariableSelector::Auto, ValueSelector::Auto)]),
        SearchPolicy::new(vec![
            SearchPhase::new(vec![0], VariableSelector::Auto, ValueSelector::Auto),
            SearchPhase::new(vec![0], VariableSelector::FirstFail, ValueSelector::Min),
        ]),
    ];
    for search_policy in invalid {
        let request = SolveRequest { mode: SolveMode::Exact, search_policy, ..SolveRequest::default() };
        assert!(matches!(solve_model_silent(&ModelPackage::new(single_variable_model()), &request), Err(SolveError::InvalidRequest(_))));
    }

    let unknown = SolveRequest {
        mode: SolveMode::Exact,
        search_policy: policy(vec![1], VariableSelector::Auto, ValueSelector::Auto),
        ..SolveRequest::default()
    };
    assert!(matches!(
        solve_model_silent(&ModelPackage::new(single_variable_model()), &unknown),
        Err(SolveError::InvalidRequest(message)) if message.contains("unknown integer variable 1")
    ));

    let legacy_controls = [
        SolveRequest {
            hints: vec![(0, 1)],
            search_policy: policy(vec![0], VariableSelector::Auto, ValueSelector::Hint),
            ..SolveRequest::default()
        },
        SolveRequest {
            branch_order: vec![0],
            search_policy: policy(vec![0], VariableSelector::Auto, ValueSelector::Auto),
            ..SolveRequest::default()
        },
        SolveRequest {
            primary_branch_scope: Some(vec![0]),
            search_policy: policy(vec![0], VariableSelector::Auto, ValueSelector::Auto),
            ..SolveRequest::default()
        },
    ];
    for request in legacy_controls {
        assert!(matches!(request.validate(), Err(SolveError::InvalidRequest(message)) if message.contains("cannot be combined")));
    }
}

#[test]
fn local_search_and_activity_no_learn_do_not_silently_ignore_a_policy() {
    let local = SolveRequest {
        mode: SolveMode::LocalSearch,
        search_policy: policy(vec![0], VariableSelector::Auto, ValueSelector::Auto),
        ..SolveRequest::default()
    };
    assert!(matches!(local.validate(), Err(SolveError::InvalidRequest(message)) if message.contains("exact CP")));

    let no_learn = SolveRequest {
        mode: SolveMode::Exact,
        cp: CpControls { no_learn_csp: true, ..CpControls::default() },
        search_policy: policy(vec![0], VariableSelector::Activity, ValueSelector::Min),
        ..SolveRequest::default()
    };
    assert!(matches!(
        solve_model_silent(&ModelPackage::new(single_variable_model()), &no_learn),
        Err(SolveError::InvalidRequest(message)) if message.contains("Activity") && message.contains("no_learn_csp")
    ));

    let compatible = SolveRequest {
        mode: SolveMode::Exact,
        cp: CpControls { no_learn_csp: true, ..CpControls::default() },
        search_policy: policy(vec![0], VariableSelector::FirstFail, ValueSelector::Min),
        ..SolveRequest::default()
    };
    assert_eq!(solve(compatible, single_variable_model()).status(), SolveStatus::Satisfiable);
}

#[test]
fn sessions_can_change_policy_between_epochs_without_recompiling_the_model() {
    let mut model = Model::new();
    let left = model.int_range(0, 4);
    let right = model.int_range(0, 4);
    model.add_constraint(Constraint::Linear { terms: vec![(1, left), (1, right)], relation: Relation::Eq, rhs: 4 });
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(left) });
    let mut session = SemanticSolveSession::new(ModelPackage::new(model)).unwrap();

    for value in [ValueSelector::Max, ValueSelector::Min, ValueSelector::Median] {
        let request = SolveRequest {
            mode: SolveMode::Exact,
            threads: 2,
            seed: 31,
            search_policy: policy(vec![left.0, right.0], VariableSelector::InputOrder, value),
            ..SolveRequest::default()
        };
        let result = session.solve_with_external_stop(&request, &AtomicBool::new(false), &mut qayd::orchestrator::IgnoreEvents).unwrap();
        assert_eq!(result.status(), SolveStatus::Optimal);
        assert_eq!(result.primal().unwrap().objectives(), [0]);
        assert_eq!(result.primal().unwrap().assignment().integers[left.0], Some(0));
        assert_eq!(result.primal().unwrap().assignment().integers[right.0], Some(4));
    }
}

#[cfg(feature = "lp-relaxation")]
#[test]
fn a_certified_dual_closure_is_not_mislabeled_as_complete_search() {
    let mut model = Model::new();
    let left = model.int_range(0, 10);
    let right = model.int_range(0, 10);
    model.add_constraint(Constraint::Linear { terms: vec![(2, left), (2, right)], relation: Relation::Ge, rhs: 5 });
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Add(vec![IntExpr::Variable(left), IntExpr::Variable(right)]) });
    let request = SolveRequest {
        mode: SolveMode::Exact,
        search_policy: policy(vec![left.0, right.0], VariableSelector::InputOrder, ValueSelector::Max),
        linear: LinearControls {
            backend: LinearBackendMode::Amthal,
            root_time: std::time::Duration::from_secs(1),
            ..LinearControls::default()
        },
        ..SolveRequest::default()
    };

    let result = solve(request, model);

    assert_eq!(result.status(), SolveStatus::Optimal);
    assert!(matches!(
        result.proof().map(|proof| proof.kind()),
        Some(ProofKind::CertifiedBounds { methods, objective_tiers: 1, .. })
            if methods.len() == 1 && methods[0].contains("exact rational recertification")
    ));
}
