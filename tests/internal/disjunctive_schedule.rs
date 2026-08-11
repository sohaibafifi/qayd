use std::sync::atomic::AtomicBool;

use crate::constraints::table::STAR;
use crate::engines::ls::disjunctive_schedule::{ConstructionBudget, DisjunctiveSchedulePlan, MAX_CONSTRUCTION_WORK, MAX_REPAIR_CANDIDATES};
use crate::model::{Constraint, IntDomain, IntExpr, IntGlobalConstraint, IntVarRef, Model, ModelPackage, Objective, Relation};
use crate::orchestrator::{solve_model_silent, EngineKind, SolveRequest, SolveStatus};

struct Fixture {
    model: Model,
    starts: Vec<IntVarRef>,
    durations: Vec<i64>,
    positions: Vec<IntVarRef>,
    selected_starts: Vec<IntVarRef>,
    selected_durations: Vec<IntVarRef>,
    all_different: usize,
    tables: Vec<usize>,
    chains: Vec<usize>,
}

fn variable(variable: IntVarRef) -> IntExpr {
    IntExpr::Variable(variable)
}

fn finish_before(start: IntVarRef, duration: i64, end: IntVarRef) -> Constraint {
    Constraint::Intension(IntExpr::Le(Box::new(IntExpr::Add(vec![variable(start), IntExpr::Constant(duration)])), Box::new(variable(end))))
}

fn selected_finish_before(start: IntVarRef, duration: IntVarRef, end: IntVarRef) -> Constraint {
    Constraint::Intension(IntExpr::Le(Box::new(IntExpr::Add(vec![variable(start), variable(duration)])), Box::new(variable(end))))
}

fn push(model: &mut Model, constraint: Constraint) -> usize {
    let index = model.constraints.len();
    model.add_constraint(constraint);
    index
}

fn fixture() -> Fixture {
    let mut model = Model::new();
    let starts = (0..6).map(|_| model.int_range(0, 50)).collect::<Vec<_>>();
    let durations = vec![2, 3, 1, 4, 2, 5];
    for (left, right) in [(0, 3), (1, 4), (2, 5)] {
        model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::NoOverlap {
            starts: vec![starts[left], starts[right]],
            durations: vec![durations[left], durations[right]],
        }));
    }

    let positions = (0..3).map(|_| model.int_range(0, 2)).collect::<Vec<_>>();
    let selected_starts = (0..3).map(|_| model.int_range(0, 50)).collect::<Vec<_>>();
    let selected_durations = (0..3).map(|_| model.int_set(vec![1, 2, 3])).collect::<Vec<_>>();
    let all_different =
        push(&mut model, Constraint::IntegerGlobal(IntGlobalConstraint::AllDifferent { variables: positions.clone(), except: Vec::new() }));
    let mut tables = Vec::new();
    for slot in 0..3 {
        push(
            &mut model,
            Constraint::IntegerGlobal(IntGlobalConstraint::Element {
                array: starts[..3].to_vec(),
                index: positions[slot],
                value: selected_starts[slot],
            }),
        );
        tables.push(push(
            &mut model,
            Constraint::IntegerGlobal(IntGlobalConstraint::Table {
                variables: vec![positions[slot], selected_durations[slot]],
                tuples: vec![vec![0, 2], vec![1, 3], vec![2, 1]],
                positive: true,
            }),
        ));
    }
    let chains = vec![
        push(&mut model, selected_finish_before(selected_starts[0], selected_durations[0], selected_starts[1])),
        push(&mut model, selected_finish_before(selected_starts[1], selected_durations[1], selected_starts[2])),
    ];

    let makespan = model.int_range(0, 100);
    for (&start, &duration) in starts.iter().zip(&durations) {
        model.add_constraint(finish_before(start, duration, makespan));
    }
    model.add_objective(Objective::IntExpr { minimize: true, expr: variable(makespan) });

    Fixture { model, starts, durations, positions, selected_starts, selected_durations, all_different, tables, chains }
}

fn single_resource_model(count: usize) -> (Model, Vec<IntVarRef>, Vec<i64>) {
    let mut model = Model::new();
    let starts = (0..count).map(|_| model.int_range(0, 100_000)).collect::<Vec<_>>();
    let durations = (0..count).map(|index| i64::try_from(index % 11 + 1).expect("small positive duration")).collect::<Vec<_>>();
    model
        .add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::NoOverlap { starts: starts.clone(), durations: durations.clone() }));
    let makespan = model.int_range(0, 1_000_000);
    for (&start, &duration) in starts.iter().zip(&durations) {
        model.add_constraint(finish_before(start, duration, makespan));
    }
    model.add_objective(Objective::IntExpr { minimize: true, expr: variable(makespan) });
    (model, starts, durations)
}

fn crossed_resources_model(objective_values: Vec<i32>) -> (Model, Vec<IntVarRef>, Vec<i64>) {
    let mut model = Model::new();
    let starts = (0..4).map(|_| model.int_range(0, 100)).collect::<Vec<_>>();
    let durations = vec![10, 1, 1, 10];
    for (left, right) in [(0, 1), (2, 3), (0, 2), (1, 3)] {
        model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::NoOverlap {
            starts: vec![starts[left], starts[right]],
            durations: vec![durations[left], durations[right]],
        }));
    }
    let objective = model.int_set(objective_values);
    for (&start, &duration) in starts.iter().zip(&durations) {
        model.add_constraint(finish_before(start, duration, objective));
    }
    model.add_objective(Objective::IntExpr { minimize: true, expr: variable(objective) });
    (model, starts, durations)
}

fn makespan(assignment: &[Option<i32>], starts: &[IntVarRef], durations: &[i64]) -> i64 {
    starts
        .iter()
        .zip(durations)
        .map(|(&start, &duration)| i64::from(assignment[start.0].expect("constructed start")) + duration)
        .max()
        .expect("at least one operation")
}

fn compile(model: &Model) -> Option<DisjunctiveSchedulePlan> {
    DisjunctiveSchedulePlan::compile(model, &AtomicBool::new(false))
}

#[test]
fn an_incidental_resource_disconnected_from_the_objective_is_declined() {
    let mut model = Model::new();
    let starts = vec![model.int_range(0, 20), model.int_range(0, 20)];
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::NoOverlap { starts, durations: vec![2, 3] }));
    let unrelated = model.int_range(0, 20);
    model.add_objective(Objective::IntExpr { minimize: true, expr: variable(unrelated) });
    assert!(compile(&model).is_none());
}

#[test]
fn syntactic_objective_occurrence_without_positive_dependency_is_declined() {
    let mut negated = fixture();
    let [Objective::IntExpr { expr: IntExpr::Variable(target), .. }] = negated.model.objectives.as_slice() else {
        panic!("fixture objective must be a variable");
    };
    let target = *target;
    negated.model.objectives.clear();
    negated.model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Neg(Box::new(variable(target))) });
    assert!(compile(&negated.model).is_none());

    let mut cancelled = fixture();
    let [Objective::IntExpr { expr: IntExpr::Variable(target), .. }] = cancelled.model.objectives.as_slice() else {
        panic!("fixture objective must be a variable");
    };
    let target = *target;
    cancelled.model.objectives.clear();
    cancelled
        .model
        .add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Sub(Box::new(variable(target)), Box::new(variable(target))) });
    assert!(compile(&cancelled.model).is_none());

    let mut surrogate_only = fixture();
    let [Objective::IntExpr { expr: IntExpr::Variable(target), .. }] = surrogate_only.model.objectives.as_slice() else {
        panic!("fixture objective must be a variable");
    };
    let target = *target;
    surrogate_only.model.objectives.clear();
    surrogate_only.model.add_objective(Objective::IntExpr {
        minimize: true,
        expr: IntExpr::Add(vec![variable(target), variable(surrogate_only.starts[0])]),
    });
    assert!(compile(&surrogate_only.model).is_none(), "a makespan surrogate must not rank a different objective");
}

#[test]
fn a_materialized_maximum_of_exact_completion_expressions_is_connected() {
    let mut fixture = fixture();
    fixture.model.objectives.clear();
    let mut completions = Vec::new();
    let selected_completion = fixture.model.int_range(0, 100);
    let selected_channel = fixture.model.constraints.len();
    fixture.model.add_constraint(Constraint::Linear {
        terms: vec![(1, fixture.selected_starts[2]), (1, fixture.selected_durations[2]), (-1, selected_completion)],
        relation: Relation::Eq,
        rhs: 0,
    });
    completions.push(selected_completion);
    for (&start, duration) in fixture.starts[3..].iter().zip([4i64, 2, 5]) {
        let completion = fixture.model.int_range(0, 100);
        fixture.model.add_constraint(Constraint::Linear {
            terms: vec![(1, start), (-1, completion)],
            relation: Relation::Eq,
            rhs: duration.checked_neg().unwrap(),
        });
        completions.push(completion);
    }
    let objective = fixture.model.int_range(0, 100);
    fixture.model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Maximum { target: objective, variables: completions }));
    fixture.model.add_objective(Objective::IntExpr { minimize: true, expr: variable(objective) });
    let plan = compile(&fixture.model).expect("the exact maximum has a faithful schedule evaluator");
    let candidates = plan.construct(&fixture.model, 0, &AtomicBool::new(false), plan.construction_budget().unwrap()).unwrap();
    let values = candidates
        .assignments
        .iter()
        .map(|assignment| {
            let selected = i64::from(assignment[fixture.selected_starts[2].0].unwrap())
                + i64::from(assignment[fixture.selected_durations[2].0].unwrap());
            fixture.starts[3..]
                .iter()
                .zip([4i64, 2, 5])
                .map(|(&start, duration)| i64::from(assignment[start.0].unwrap()) + duration)
                .fold(selected, i64::max)
        })
        .collect::<Vec<_>>();
    assert!(values.iter().min() != values.iter().max(), "the ranking check needs distinct exact objective values");
    assert!(values.windows(2).all(|window| window[0] <= window[1]));

    let Constraint::Linear { terms, .. } = &mut fixture.model.constraints[selected_channel] else { unreachable!() };
    terms[1].0 = 2;
    assert!(compile(&fixture.model).is_none());
}

#[test]
fn non_ordering_constraints_on_channel_values_remain_for_canonical_replay() {
    let mut fixture = fixture();
    fixture.model.add_constraint(Constraint::Intension(IntExpr::Ge(
        Box::new(IntExpr::Add(vec![variable(fixture.selected_starts[2]), variable(fixture.selected_durations[2])])),
        Box::new(IntExpr::Constant(1)),
    )));
    assert!(compile(&fixture.model).is_some());
}

#[test]
fn complete_permutation_channel_compiles_and_constructs_deterministically() {
    let fixture = fixture();
    let plan = compile(&fixture.model).expect("the complete structural proof compiles");
    assert_eq!(plan.operation_count(), 6);
    assert_eq!(plan.resource_count(), 4);
    assert_eq!(plan.implicit_resource_count(), 1);
    assert_eq!(plan.precedence_count(), 0);

    let budget = plan.construction_budget().unwrap();
    let first = plan.construct(&fixture.model, 17, &AtomicBool::new(false), budget).unwrap();
    let second = plan.construct(&fixture.model, 17, &AtomicBool::new(false), budget).unwrap();
    assert_eq!(first.assignments, second.assignments);
    assert_eq!(first.work, second.work);
    assert!(!first.assignments.is_empty());
    assert!(first.assignments.len() <= MAX_REPAIR_CANDIDATES);
    let scores = first.assignments.iter().map(|assignment| makespan(assignment, &fixture.starts, &fixture.durations)).collect::<Vec<_>>();
    assert!(scores.windows(2).all(|window| window[0] <= window[1]));
    for assignment in &first.assignments {
        assert!(fixture.starts.iter().all(|variable| assignment[variable.0].is_some()));
        assert!(fixture.positions.iter().all(|variable| assignment[variable.0].is_some()));
        assert!(fixture.selected_starts.iter().all(|variable| assignment[variable.0].is_some()));
        assert!(fixture.selected_durations.iter().all(|variable| assignment[variable.0].is_some()));
    }
}

#[test]
fn diversified_attempts_are_seeded_repeatably() {
    let (model, _, _) = single_resource_model(10);
    let plan = compile(&model).expect("the single-resource schedule compiles");
    let budget = plan.construction_budget().unwrap();
    let first = plan.construct(&model, 11, &AtomicBool::new(false), budget).unwrap();
    let repeated = plan.construct(&model, 11, &AtomicBool::new(false), budget).unwrap();
    let different = plan.construct(&model, 12, &AtomicBool::new(false), budget).unwrap();

    assert_eq!(first.assignments, repeated.assignments);
    assert!(first.assignments.len() > 1);
    assert!(first.assignments.len() <= MAX_REPAIR_CANDIDATES);
    assert_ne!(first.assignments, different.assignments);
}

#[test]
fn construction_attempts_scale_with_size_under_a_fixed_work_cap() {
    let (small_model, _, _) = single_resource_model(16);
    let small = compile(&small_model).unwrap();
    let small_budget = small.construction_budget().unwrap();
    let small_work_per_attempt = 16u64 * 16;
    assert_eq!(small_budget.work / small_work_per_attempt, 519);
    assert!(small_budget.work < MAX_CONSTRUCTION_WORK);
    let candidates = small.construct(&small_model, 0, &AtomicBool::new(false), small_budget).unwrap();
    assert_eq!(candidates.work / small_work_per_attempt, 519);
    assert_eq!(candidates.materializations, MAX_REPAIR_CANDIDATES);
    assert!(candidates.materializations < candidates.work as usize / small_work_per_attempt as usize);

    let (large_model, _, _) = single_resource_model(128);
    let large = compile(&large_model).unwrap();
    let large_budget = large.construction_budget().unwrap();
    let large_work_per_attempt = 128u64 * 128;
    assert_eq!(large_budget.work, MAX_CONSTRUCTION_WORK);
    assert_eq!(large_budget.work / large_work_per_attempt, 128);
    assert!(large_budget.work / large_work_per_attempt < small_budget.work / small_work_per_attempt);
}

#[test]
fn objective_domain_holes_filter_individual_schedules_without_aborting_construction() {
    let (model, starts, durations) = crossed_resources_model(vec![11]);
    let plan = compile(&model).expect("the sparse objective domain is structurally evaluable");
    let candidates = plan.construct(&model, 0, &AtomicBool::new(false), plan.construction_budget().unwrap()).unwrap();

    assert!(!candidates.assignments.is_empty());
    assert!(candidates.assignments.iter().all(|assignment| makespan(assignment, &starts, &durations) == 11));
    assert!(candidates.work > 4 * 4, "construction continued after schedules outside the objective domain");
}

#[test]
fn exact_precedences_are_recovered_but_near_relations_are_rejected() {
    let mut exact = fixture();
    exact.model.add_constraint(finish_before(exact.starts[0], 2, exact.starts[4]));
    assert_eq!(compile(&exact.model).unwrap().precedence_count(), 1);

    let mut wrong_offset = fixture();
    wrong_offset.model.add_constraint(finish_before(wrong_offset.starts[0], 3, wrong_offset.starts[4]));
    assert!(compile(&wrong_offset.model).is_none());

    let mut strict = fixture();
    strict.model.add_constraint(Constraint::Intension(IntExpr::Lt(
        Box::new(IntExpr::Add(vec![variable(strict.starts[0]), IntExpr::Constant(2)])),
        Box::new(variable(strict.starts[4])),
    )));
    assert!(compile(&strict.model).is_none());

    let mut cyclic = fixture();
    cyclic.model.add_constraint(finish_before(cyclic.starts[0], 2, cyclic.starts[4]));
    cyclic.model.add_constraint(finish_before(cyclic.starts[4], 2, cyclic.starts[0]));
    assert!(compile(&cyclic.model).is_none());
}

#[test]
fn explicit_resources_require_positive_consistent_fixed_durations() {
    let mut inconsistent = fixture();
    inconsistent.model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::NoOverlap {
        starts: vec![inconsistent.starts[0], inconsistent.starts[4]],
        durations: vec![3, 2],
    }));
    assert!(compile(&inconsistent.model).is_none());

    let mut zero = fixture();
    let Constraint::IntegerGlobal(IntGlobalConstraint::NoOverlap { durations, .. }) = &mut zero.model.constraints[0] else {
        unreachable!()
    };
    durations[0] = 0;
    assert!(compile(&zero.model).is_none());

    let mut oversized = fixture();
    let Constraint::IntegerGlobal(IntGlobalConstraint::NoOverlap { durations, .. }) = &mut oversized.model.constraints[0] else {
        unreachable!()
    };
    durations[0] = i64::MAX;
    assert!(compile(&oversized.model).is_none());

    let mut variable_resource = fixture();
    let duration = variable_resource.model.int_range(1, 3);
    let demand = variable_resource.model.int_range(1, 1);
    let capacity = variable_resource.model.int_range(1, 1);
    variable_resource.model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::CumulativeVar {
        starts: vec![variable_resource.starts[0]],
        durations: vec![duration],
        demands: vec![demand],
        capacity,
    }));
    assert!(compile(&variable_resource.model).is_none());
}

#[test]
fn permutation_channel_rejects_incomplete_or_ambiguous_proofs() {
    let mut excepted = fixture();
    let Constraint::IntegerGlobal(IntGlobalConstraint::AllDifferent { except, .. }) =
        &mut excepted.model.constraints[excepted.all_different]
    else {
        unreachable!()
    };
    except.push(0);
    assert!(compile(&excepted.model).is_none());

    let mut hole = fixture();
    hole.model.int_vars[hole.positions[0].0] = IntDomain::Set(vec![0, 2]);
    assert!(compile(&hole.model).is_none());

    let mut ambiguous_element = fixture();
    ambiguous_element.model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Element {
        array: ambiguous_element.starts[..3].to_vec(),
        index: ambiguous_element.positions[0],
        value: ambiguous_element.selected_starts[0],
    }));
    assert!(compile(&ambiguous_element.model).is_none());

    let mut negative_table = fixture();
    let Constraint::IntegerGlobal(IntGlobalConstraint::Table { positive, .. }) =
        &mut negative_table.model.constraints[negative_table.tables[0]]
    else {
        unreachable!()
    };
    *positive = false;
    assert!(compile(&negative_table.model).is_none());

    let mut incomplete_table = fixture();
    let Constraint::IntegerGlobal(IntGlobalConstraint::Table { tuples, .. }) =
        &mut incomplete_table.model.constraints[incomplete_table.tables[0]]
    else {
        unreachable!()
    };
    tuples.pop();
    assert!(compile(&incomplete_table.model).is_none());

    let mut wildcard_table = fixture();
    let Constraint::IntegerGlobal(IntGlobalConstraint::Table { tuples, .. }) =
        &mut wildcard_table.model.constraints[wildcard_table.tables[0]]
    else {
        unreachable!()
    };
    tuples[0][1] = STAR;
    assert!(compile(&wildcard_table.model).is_none());

    let mut duration_mismatch = fixture();
    let Constraint::IntegerGlobal(IntGlobalConstraint::Table { tuples, .. }) =
        &mut duration_mismatch.model.constraints[duration_mismatch.tables[0]]
    else {
        unreachable!()
    };
    tuples[0][1] = 3;
    assert!(compile(&duration_mismatch.model).is_none());
}

#[test]
fn channel_order_must_be_a_complete_unique_chain() {
    let mut missing = fixture();
    missing.model.constraints.remove(missing.chains[1]);
    assert!(compile(&missing.model).is_none());

    let mut non_unique = fixture();
    non_unique.model.constraints[non_unique.chains[0]] =
        selected_finish_before(non_unique.selected_starts[0], non_unique.selected_durations[0], non_unique.selected_starts[2]);
    assert!(compile(&non_unique.model).is_none());

    let mut cyclic = fixture();
    cyclic.model.constraints[cyclic.chains[1]] =
        selected_finish_before(cyclic.selected_starts[1], cyclic.selected_durations[1], cyclic.selected_starts[0]);
    assert!(compile(&cyclic.model).is_none());
}

#[test]
fn domains_objective_and_work_limits_are_generic_guards() {
    let mut sparse_start = fixture();
    sparse_start.model.int_vars[sparse_start.starts[0].0] = IntDomain::Set(vec![0, 2, 4]);
    let plan = compile(&sparse_start.model).expect("sparse domains have an exact successor operation");
    let candidates = plan.construct(&sparse_start.model, 0, &AtomicBool::new(false), plan.construction_budget().unwrap()).unwrap();
    assert!(candidates.assignments.iter().all(|assignment| matches!(assignment[sparse_start.starts[0].0], Some(0 | 2 | 4))));

    let mut maximizing = fixture();
    let Objective::IntExpr { minimize, .. } = &mut maximizing.model.objectives[0] else { unreachable!() };
    *minimize = false;
    assert!(compile(&maximizing.model).is_none());

    let fixture = fixture();
    assert!(DisjunctiveSchedulePlan::compile(&fixture.model, &AtomicBool::new(true)).is_none());
    let plan = compile(&fixture.model).unwrap();
    assert!(plan.construct(&fixture.model, 0, &AtomicBool::new(true), plan.construction_budget().unwrap()).is_none());
    assert!(plan.construct(&fixture.model, 0, &AtomicBool::new(false), ConstructionBudget::new(1)).is_none());
}

#[test]
fn rejected_constructive_assignments_leave_exact_search_unchanged() {
    let mut fixture = fixture();
    fixture.model.add_constraint(Constraint::Linear { terms: vec![(1, fixture.starts[0])], relation: Relation::Ge, rhs: 40 });
    assert!(compile(&fixture.model).is_some(), "a unary side constraint is left to canonical replay");

    let result = solve_model_silent(&ModelPackage::new(fixture.model), &SolveRequest::default()).unwrap();
    assert!(matches!(result.status(), SolveStatus::Optimal | SolveStatus::Satisfiable));
    assert!(result.primal().is_some());
    assert!(result.reports().iter().any(|report| report.engine == Some(EngineKind::IntegerExact)));
}

#[test]
fn verified_warm_start_crosses_the_orchestrator_boundary() {
    let fixture = fixture();
    let result = solve_model_silent(&ModelPackage::new(fixture.model), &SolveRequest::default()).unwrap();
    assert!(result.primal().is_some());
    assert!(result.reports().iter().any(|report| {
        report.engine == Some(EngineKind::IntegerLocalSearch)
            && report.metadata.iter().any(|(key, value)| key == "ls_role" && value == "disjunctive_schedule_warm_start")
    }));
    assert!(result.reports().iter().any(|report| report.engine == Some(EngineKind::IntegerExact)));
}
