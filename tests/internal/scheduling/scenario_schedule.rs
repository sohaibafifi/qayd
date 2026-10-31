use std::sync::atomic::AtomicBool;

use crate::engines::ls::scenario_schedule::{ConstructionLimits, ScenarioSchedulePlan};
use crate::model::{Constraint, IntDomain, IntExpr, IntGlobalConstraint, IntVarRef, Model, ModelPackage, Objective, Relation};
use crate::orchestrator::{solve_model_silent, EngineKind, SolveRequest};

#[derive(Clone, Copy)]
enum ExactOneEncoding {
    Boolean,
    Count,
    Linear,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Corruption {
    None,
    MissingExactOne,
    IncompleteScenario,
    SelectorOnWrongOperation,
    DurationMismatch,
    Cycle,
    NonUnaryCapacity,
    NonPositiveObjective,
    Maximization,
    CrossConstraint,
    UnclassifiedVariable,
}

struct Fixture {
    model: Model,
    starts: Vec<Vec<IntVarRef>>,
    completions: Vec<IntVarRef>,
}

fn variable(variable: IntVarRef) -> IntExpr {
    IntExpr::Variable(variable)
}

fn equality(left: IntVarRef, value: i32) -> IntExpr {
    IntExpr::Eq(Box::new(variable(left)), Box::new(IntExpr::Constant(i64::from(value))))
}

fn precedence(start: IntVarRef, duration: IntVarRef, end: IntVarRef) -> Constraint {
    Constraint::Intension(IntExpr::Le(Box::new(IntExpr::Add(vec![variable(start), variable(duration)])), Box::new(variable(end))))
}

fn fixture(encoding: ExactOneEncoding, corruption: Corruption) -> Fixture {
    let mut model = Model::new();
    let starts = (0..2).map(|_| (0..3).map(|_| model.int_range(0, 50)).collect::<Vec<_>>()).collect::<Vec<_>>();
    let durations = (0..2).map(|_| (0..3).map(|_| model.int_range(1, 8)).collect::<Vec<_>>()).collect::<Vec<_>>();
    let completion = (0..2).map(|_| model.int_range(0, 50)).collect::<Vec<_>>();
    let selectors = (0..3).map(|_| (0..2).map(|_| model.bool_var()).collect::<Vec<_>>()).collect::<Vec<_>>();

    // Each tuple is (resource, duration in scenario 0, duration in scenario 1).
    let modes = [[(0usize, 3, 4), (1usize, 2, 3)], [(0usize, 4, 5), (1usize, 2, 2)], [(1usize, 3, 4), (0usize, 1, 2)]];
    for (operation, operation_modes) in modes.iter().enumerate() {
        for (mode, mode_data) in operation_modes.iter().enumerate() {
            for (scenario, scenario_durations) in durations.iter().enumerate() {
                if corruption == Corruption::IncompleteScenario && operation == 0 && mode == 1 && scenario == 1 {
                    continue;
                }
                model.add_constraint(Constraint::Intension(IntExpr::Imp(
                    Box::new(variable(selectors[operation][mode])),
                    Box::new(equality(scenario_durations[operation], if scenario == 0 { mode_data.1 } else { mode_data.2 })),
                )));
            }
        }
    }

    for (operation, operation_selectors) in selectors.iter().enumerate() {
        if corruption == Corruption::MissingExactOne && operation == 0 {
            continue;
        }
        match encoding {
            ExactOneEncoding::Boolean => {
                model.add_constraint(Constraint::Intension(IntExpr::Or(operation_selectors.iter().copied().map(variable).collect())));
                model.add_constraint(Constraint::Intension(IntExpr::Ne(
                    Box::new(variable(operation_selectors[0])),
                    Box::new(variable(operation_selectors[1])),
                )));
            }
            ExactOneEncoding::Count => {
                model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Count {
                    variables: operation_selectors.clone(),
                    value: 1,
                    relation: Relation::Eq,
                    count: 1,
                }));
            }
            ExactOneEncoding::Linear => {
                model.add_constraint(Constraint::Linear {
                    terms: operation_selectors.iter().map(|&selector| (1, selector)).collect(),
                    relation: Relation::Eq,
                    rhs: 1,
                });
            }
        }
    }

    for ((scenario_starts, scenario_durations), &scenario_completion) in starts.iter().zip(&durations).zip(&completion) {
        model.add_constraint(precedence(scenario_starts[0], scenario_durations[0], scenario_starts[2]));
        model.add_constraint(precedence(scenario_starts[1], scenario_durations[1], scenario_starts[2]));
        model.add_constraint(precedence(scenario_starts[2], scenario_durations[2], scenario_completion));
        if corruption == Corruption::Cycle {
            model.add_constraint(precedence(scenario_starts[2], scenario_durations[2], scenario_starts[0]));
        }
    }

    for (scenario, scenario_starts) in starts.iter().enumerate() {
        for resource in 0..2 {
            let mut resource_starts = Vec::new();
            let mut resource_durations = Vec::new();
            let mut resource_demands = Vec::new();
            for (operation, operation_modes) in modes.iter().enumerate() {
                for (mode, mode_data) in operation_modes.iter().enumerate() {
                    if mode_data.0 != resource {
                        continue;
                    }
                    resource_starts.push(scenario_starts[operation]);
                    let duration = if scenario == 0 { mode_data.1 } else { mode_data.2 };
                    let duration = if corruption == Corruption::DurationMismatch && scenario == 0 && resource == 0 && operation == 0 {
                        duration + 1
                    } else {
                        duration
                    };
                    resource_durations.push(model.int_range(duration, duration));
                    let selector = if corruption == Corruption::SelectorOnWrongOperation && scenario == 0 && resource == 0 && operation == 0
                    {
                        selectors[1][0]
                    } else {
                        selectors[operation][mode]
                    };
                    resource_demands.push(selector);
                }
            }
            let capacity = i32::from(corruption == Corruption::NonUnaryCapacity && scenario == 0 && resource == 0) + 1;
            let capacity = model.int_range(capacity, capacity);
            model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::CumulativeVar {
                starts: resource_starts,
                durations: resource_durations,
                demands: resource_demands,
                capacity,
            }));
        }
    }

    if corruption == Corruption::CrossConstraint {
        model.add_constraint(Constraint::Intension(IntExpr::Le(Box::new(variable(starts[0][0])), Box::new(variable(starts[1][0])))));
    }
    if corruption == Corruption::UnclassifiedVariable {
        model.bool_var();
    }
    let second_coefficient = if corruption == Corruption::NonPositiveObjective { 0 } else { 2 };
    model.add_objective(Objective::IntExpr {
        minimize: corruption != Corruption::Maximization,
        expr: IntExpr::Add(vec![
            variable(completion[0]),
            IntExpr::Mul(vec![IntExpr::Constant(second_coefficient), variable(completion[1])]),
        ]),
    });
    Fixture { model, starts, completions: completion }
}

fn many_mode_model(mode_count: usize, exact_count: i64) -> Model {
    let mut model = Model::new();
    let starts = [model.int_range(0, 100), model.int_range(0, 100)];
    let durations = [model.int_range(1, 100), model.int_range(1, 100)];
    let completions = [model.int_range(0, 100), model.int_range(0, 100)];
    let selectors = (0..mode_count).map(|_| model.bool_var()).collect::<Vec<_>>();

    for (mode, &selector) in selectors.iter().enumerate() {
        for (scenario, &duration_variable) in durations.iter().enumerate() {
            let duration = i32::try_from(mode + scenario + 1).unwrap();
            model.add_constraint(Constraint::Intension(IntExpr::Imp(
                Box::new(variable(selector)),
                Box::new(equality(duration_variable, duration)),
            )));
        }
    }
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Count {
        variables: selectors.clone(),
        value: 1,
        relation: Relation::Eq,
        count: exact_count,
    }));
    for (scenario, ((&start, &duration), &completion)) in starts.iter().zip(&durations).zip(&completions).enumerate() {
        model.add_constraint(precedence(start, duration, completion));
        let fixed_durations = (0..mode_count)
            .map(|mode| model.int_range(i32::try_from(mode + scenario + 1).unwrap(), i32::try_from(mode + scenario + 1).unwrap()))
            .collect::<Vec<_>>();
        let capacity = model.int_range(1, 1);
        model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::CumulativeVar {
            starts: vec![starts[scenario]; mode_count],
            durations: fixed_durations,
            demands: selectors.clone(),
            capacity,
        }));
    }
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Add(completions.into_iter().map(variable).collect()) });
    model
}

#[test]
fn equivalent_exact_one_encodings_compile_and_construct_the_same_schedule() {
    let stop = AtomicBool::new(false);
    let mut objectives = Vec::new();
    for encoding in [ExactOneEncoding::Boolean, ExactOneEncoding::Count, ExactOneEncoding::Linear] {
        let fixture = fixture(encoding, Corruption::None);
        let plan = ScenarioSchedulePlan::compile(&fixture.model, &stop).expect("the complete scheduling structure compiles");
        assert_eq!(plan.operation_count(), 3);
        assert_eq!(plan.scenario_count(), 2);
        let first = plan.construct(&fixture.model, &stop, ConstructionLimits::default()).expect("serial generation succeeds");
        let second = plan.construct(&fixture.model, &stop, ConstructionLimits::default()).expect("serial generation is repeatable");
        assert_eq!(first.values, second.values);
        assert_eq!(first.objective, second.objective);
        assert_eq!(first.configurations, second.configurations);
        objectives.push(first.objective);
    }
    assert!(objectives.windows(2).all(|pair| pair[0] == pair[1]));
    assert_eq!(objectives, vec![16, 16, 16]);
}

#[test]
fn direct_exact_one_proofs_do_not_depend_on_a_mode_count_cutoff() {
    let stop = AtomicBool::new(false);
    let model = many_mode_model(17, 1);
    let plan = ScenarioSchedulePlan::compile(&model, &stop).expect("the direct count identity is linear in the mode count");
    let solution = plan.construct(&model, &stop, ConstructionLimits::default()).expect("the large mode set is constructed");
    assert_eq!(solution.objective, 3);

    let near_match = many_mode_model(17, 2);
    assert!(ScenarioSchedulePlan::compile(&near_match, &stop).is_none());
}

#[test]
fn a_single_proved_mode_is_a_valid_operation() {
    let stop = AtomicBool::new(false);
    let model = many_mode_model(1, 1);
    let plan = ScenarioSchedulePlan::compile(&model, &stop).expect("a fixed mode is still a complete scheduling structure");
    let solution = plan.construct(&model, &stop, ConstructionLimits::default()).expect("the fixed mode is constructed");
    assert_eq!(solution.objective, 3);
}

#[test]
fn a_count_with_duplicate_variables_is_not_an_exact_one_proof() {
    let mut fixture = fixture(ExactOneEncoding::Count, Corruption::None);
    let count = fixture
        .model
        .constraints
        .iter_mut()
        .find_map(|constraint| match constraint {
            Constraint::IntegerGlobal(IntGlobalConstraint::Count { variables, .. }) if variables.len() > 1 => Some(variables),
            _ => None,
        })
        .expect("fixture contains an exact-one count");
    count[1] = count[0];
    assert!(ScenarioSchedulePlan::compile(&fixture.model, &AtomicBool::new(false)).is_none());
}

#[test]
fn an_exact_materialized_positive_sum_is_recovered_semantically() {
    let mut fixture = fixture(ExactOneEncoding::Count, Corruption::None);
    fixture.model.objectives.clear();
    let objective = fixture.model.int_range(0, 150);
    fixture.model.add_constraint(Constraint::Linear {
        terms: vec![(1, fixture.completions[0]), (2, fixture.completions[1]), (-1, objective)],
        relation: Relation::Eq,
        rhs: 0,
    });
    fixture.model.add_objective(Objective::IntExpr { minimize: true, expr: variable(objective) });

    let stop = AtomicBool::new(false);
    let plan = ScenarioSchedulePlan::compile(&fixture.model, &stop).expect("the exact affine identity compiles");
    let solution = plan.construct(&fixture.model, &stop, ConstructionLimits::default()).expect("the schedule is constructed");
    assert_eq!(solution.objective, 16);
    assert_eq!(solution.values[objective.0], 16);

    let result = solve_model_silent(&ModelPackage::new(fixture.model), &SolveRequest::default()).unwrap();
    assert_eq!(result.primal().unwrap().objectives(), [16]);
    assert!(result.reports().iter().any(|report| {
        report.engine == Some(EngineKind::IntegerLocalSearch)
            && report.metadata.iter().any(|(key, value)| key == "ls_role" && value == "scenario_schedule_warm_start")
    }));
}

#[test]
fn near_materialized_objective_relations_are_not_inferred() {
    let mut fixture = fixture(ExactOneEncoding::Count, Corruption::None);
    fixture.model.objectives.clear();
    let objective = fixture.model.int_range(0, 150);
    fixture.model.add_constraint(Constraint::Linear {
        terms: vec![(1, fixture.completions[0]), (2, fixture.completions[1]), (-2, objective)],
        relation: Relation::Eq,
        rhs: 0,
    });
    fixture.model.add_objective(Objective::IntExpr { minimize: true, expr: variable(objective) });
    assert!(ScenarioSchedulePlan::compile(&fixture.model, &AtomicBool::new(false)).is_none());
}

#[test]
fn a_materialized_objective_must_belong_to_its_declared_domain() {
    let mut fixture = fixture(ExactOneEncoding::Count, Corruption::None);
    fixture.model.objectives.clear();
    let objective = fixture.model.int_range(0, 15);
    fixture.model.add_constraint(Constraint::Linear {
        terms: vec![(1, fixture.completions[0]), (2, fixture.completions[1]), (-1, objective)],
        relation: Relation::Eq,
        rhs: 0,
    });
    fixture.model.add_objective(Objective::IntExpr { minimize: true, expr: variable(objective) });

    let stop = AtomicBool::new(false);
    let plan = ScenarioSchedulePlan::compile(&fixture.model, &stop).expect("the algebraic structure is exact");
    assert!(plan.construct(&fixture.model, &stop, ConstructionLimits::default()).is_none());
}

#[test]
fn every_unproved_near_match_is_declined() {
    let stop = AtomicBool::new(false);
    for corruption in [
        Corruption::MissingExactOne,
        Corruption::IncompleteScenario,
        Corruption::SelectorOnWrongOperation,
        Corruption::DurationMismatch,
        Corruption::Cycle,
        Corruption::NonUnaryCapacity,
        Corruption::NonPositiveObjective,
        Corruption::Maximization,
        Corruption::CrossConstraint,
        Corruption::UnclassifiedVariable,
    ] {
        let fixture = fixture(ExactOneEncoding::Count, corruption);
        assert!(ScenarioSchedulePlan::compile(&fixture.model, &stop).is_none(), "an incomplete structural proof was accepted");
    }
}

#[test]
fn compilation_construction_and_work_limits_observe_interrupts() {
    let fixture = fixture(ExactOneEncoding::Linear, Corruption::None);
    let stopped = AtomicBool::new(true);
    assert!(ScenarioSchedulePlan::compile(&fixture.model, &stopped).is_none());

    let running = AtomicBool::new(false);
    let plan = ScenarioSchedulePlan::compile(&fixture.model, &running).unwrap();
    assert!(plan.construct(&fixture.model, &stopped, ConstructionLimits::default()).is_none());
    assert!(plan.construct(&fixture.model, &running, ConstructionLimits { configurations: 1, candidate_visits: 71, passes: 32 }).is_none());
    let bounded = plan
        .construct(&fixture.model, &running, ConstructionLimits { configurations: 1, candidate_visits: 72, passes: 32 })
        .expect("the exact construction budget permits one configuration");
    assert_eq!(bounded.configurations, 1);
    assert_eq!(bounded.candidate_visits, 72);
}

#[test]
fn serial_generation_uses_supported_values_from_sparse_domains() {
    let mut fixture = fixture(ExactOneEncoding::Count, Corruption::None);
    let constrained_start = fixture.starts[0][2];
    fixture.model.int_vars[constrained_start.0] = IntDomain::Set(vec![0, 5, 11, 23, 50]);
    let stop = AtomicBool::new(false);
    let plan = ScenarioSchedulePlan::compile(&fixture.model, &stop).unwrap();
    let solution = plan.construct(&fixture.model, &stop, ConstructionLimits::default()).unwrap();
    assert!(matches!(solution.values[constrained_start.0], 5 | 11 | 23 | 50));
}

#[test]
fn exact_search_receives_only_the_canonically_replayed_warm_start() {
    let fixture = fixture(ExactOneEncoding::Count, Corruption::None);
    let result = solve_model_silent(&ModelPackage::new(fixture.model), &SolveRequest::default()).unwrap();
    assert_eq!(result.primal().unwrap().objectives(), [16]);
    assert!(result.reports().iter().any(|report| {
        report.engine == Some(EngineKind::IntegerLocalSearch)
            && report.metadata.iter().any(|(key, value)| key == "ls_role" && value == "scenario_schedule_warm_start")
    }));
    assert!(result.reports().iter().any(|report| report.engine == Some(EngineKind::IntegerExact)));
}
