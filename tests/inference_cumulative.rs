use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use qayd::ids::{PropId, VarId};
use qayd::inference::{
    check_inference, check_inference_until, Inference, InferenceCheckConfig, InferenceCheckOutcome, InferenceConclusion,
    InferenceConstraint, InferenceCumulativeTask, InferenceDomain, InferenceUncoveredReason,
};
use qayd::store::Premise;

fn domain(var: u32, min: i32, max: i32) -> InferenceDomain {
    InferenceDomain { var: VarId(var), min, max, values: None, holes: vec![] }
}

fn task(start: u32, duration: i64, demand: i64, presence: Option<u32>) -> InferenceCumulativeTask {
    InferenceCumulativeTask { start: VarId(start), duration, demand, presence: presence.map(VarId) }
}

fn check(
    constraint: &InferenceConstraint,
    roots: &[InferenceDomain],
    premises: &[Premise],
    conclusion: InferenceConclusion,
    snapshot: &[InferenceDomain],
) -> InferenceCheckOutcome {
    check_inference(
        &Inference { propagator: PropId(17), constraint, root_domains: roots, premises, conclusion, snapshot },
        InferenceCheckConfig::default(),
    )
}

fn holds(premise: Premise, values: &BTreeMap<VarId, i32>) -> bool {
    match premise {
        Premise::Ge { var, bound } => values[&var] >= bound,
        Premise::Le { var, bound } => values[&var] <= bound,
        Premise::Eq { var, val } => values[&var] == val,
        Premise::Ne { var, val } => values[&var] != val,
    }
}

fn conclusion_holds(conclusion: InferenceConclusion, values: &BTreeMap<VarId, i32>) -> bool {
    match conclusion {
        InferenceConclusion::Predicate(premise) => holds(premise, values),
        InferenceConclusion::Conflict => false,
    }
}

/// Independent small-instance oracle: reserve individual integer time slots.
/// It uses neither the checker's start-instant scan nor any production profile.
fn occupied_slots_feasible(tasks: &[InferenceCumulativeTask], capacity: i64, values: &BTreeMap<VarId, i32>) -> bool {
    let mut usage = BTreeMap::<i64, i128>::new();
    for task in tasks {
        if task.presence.is_some_and(|var| values[&var] != 1) {
            continue;
        }
        let start = i64::from(values[&task.start]);
        for time in start..start + task.duration {
            let load = usage.entry(time).or_default();
            *load += i128::from(task.demand);
            if *load > i128::from(capacity) {
                return false;
            }
        }
    }
    true
}

fn domain_values(domain: &InferenceDomain) -> Vec<i32> {
    let candidates = domain.values.as_ref().map_or_else(|| (domain.min..=domain.max).collect(), |values| values.to_vec());
    candidates
        .into_iter()
        .filter(|value| domain.min <= *value && *value <= domain.max && !domain.holes.contains(value))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn assignments(roots: &[InferenceDomain]) -> Vec<BTreeMap<VarId, i32>> {
    let mut result = vec![BTreeMap::new()];
    for root in roots {
        let mut next = Vec::new();
        for prefix in result {
            for value in domain_values(root) {
                let mut extended = prefix.clone();
                extended.insert(root.var, value);
                next.push(extended);
            }
        }
        result = next;
    }
    result
}

fn assert_matches_oracle(
    tasks: &[InferenceCumulativeTask],
    capacity: i64,
    roots: &[InferenceDomain],
    premises: &[Premise],
    conclusion: InferenceConclusion,
    snapshot: &[InferenceDomain],
) {
    let valid = assignments(roots).iter().all(|values| {
        !premises.iter().all(|&premise| holds(premise, values))
            || !occupied_slots_feasible(tasks, capacity, values)
            || conclusion_holds(conclusion, values)
    });
    let constraint = InferenceConstraint::Cumulative { tasks: tasks.to_vec(), capacity };
    match check(&constraint, roots, premises, conclusion, snapshot) {
        InferenceCheckOutcome::Checked { .. } => assert!(valid, "accepted invalid cumulative inference: {constraint:?}, {conclusion:?}"),
        InferenceCheckOutcome::Rejected(failure) => {
            assert!(!valid, "rejected valid cumulative inference: {failure}");
            let witness: BTreeMap<_, _> = failure.counterexample.expect("invalid conclusion needs a witness").into_iter().collect();
            assert!(roots.iter().all(|root| domain_values(root).contains(&witness[&root.var])));
            assert!(premises.iter().all(|&premise| holds(premise, &witness)));
            assert!(occupied_slots_feasible(tasks, capacity, &witness));
            assert!(!conclusion_holds(conclusion, &witness));
        }
        InferenceCheckOutcome::Uncovered(reason) => panic!("small cumulative inference must be covered: {reason:?}"),
    }
}

fn assert_checked(outcome: InferenceCheckOutcome) {
    assert!(matches!(outcome, InferenceCheckOutcome::Checked { .. }), "{outcome:?}");
}

fn assert_rejected(outcome: InferenceCheckOutcome) {
    assert!(matches!(outcome, InferenceCheckOutcome::Rejected(_)), "{outcome:?}");
}

#[test]
fn cumulative_inferences_match_exhaustive_slot_oracle() {
    let roots = [domain(0, -1, 1), domain(1, 0, 2), domain(2, 0, 1)];
    let conclusions = [
        InferenceConclusion::Conflict,
        InferenceConclusion::Predicate(Premise::Ge { var: VarId(0), bound: 0 }),
        InferenceConclusion::Predicate(Premise::Le { var: VarId(1), bound: 1 }),
        InferenceConclusion::Predicate(Premise::Ne { var: VarId(0), val: 0 }),
        InferenceConclusion::Predicate(Premise::Eq { var: VarId(2), val: 0 }),
    ];
    for first_duration in 0..=2 {
        for second_duration in 0..=2 {
            for first_demand in 0..=2 {
                for second_demand in 0..=2 {
                    for capacity in 0..=3 {
                        for shared_presence in [false, true] {
                            let tasks = [
                                task(0, first_duration, first_demand, shared_presence.then_some(2)),
                                task(1, second_duration, second_demand, Some(2)),
                            ];
                            for with_premise in [false, true] {
                                let premises = if with_premise { vec![Premise::Ge { var: VarId(1), bound: 1 }] } else { vec![] };
                                let mut snapshot = roots.clone();
                                if with_premise {
                                    snapshot[1].min = 1;
                                }
                                for conclusion in conclusions {
                                    assert_matches_oracle(&tasks, capacity, &roots, &premises, conclusion, &snapshot);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn sparse_roots_holes_and_repeated_occurrences_keep_their_exact_meaning() {
    let mut roots = [domain(0, -3, 3), domain(1, -2, 4), domain(2, 0, 1)];
    roots[0].values = Some(Arc::from([3, -3, 1, -1, 1]));
    roots[0].holes = vec![-1, -1];
    roots[1].values = Some(Arc::from([4, -2, 0, 2, 4]));
    roots[1].holes = vec![0];
    let tasks = [task(0, 2, 1, Some(2)), task(0, 2, 1, Some(2)), task(1, 3, 2, None)];
    for capacity in 0..=4 {
        for conclusion in [
            InferenceConclusion::Conflict,
            InferenceConclusion::Predicate(Premise::Eq { var: VarId(2), val: 0 }),
            InferenceConclusion::Predicate(Premise::Ne { var: VarId(0), val: 1 }),
            InferenceConclusion::Predicate(Premise::Le { var: VarId(1), bound: 2 }),
        ] {
            assert_matches_oracle(&tasks, capacity, &roots, &[], conclusion, &roots);
        }
    }
    let repeated = InferenceConstraint::Cumulative { tasks: vec![task(0, 1, 1, None), task(0, 1, 1, None)], capacity: 1 };
    assert_checked(check(&repeated, &roots[..1], &[], InferenceConclusion::Conflict, &roots[..1]));
}

#[test]
fn shared_start_and_presence_variables_use_one_assignment() {
    let roots = [domain(0, 0, 1), domain(1, 0, 1)];
    let tasks = [task(0, 2, 2, Some(0)), task(1, 2, 1, Some(0)), task(0, 1, 1, Some(1))];
    for capacity in 0..=4 {
        for conclusion in [
            InferenceConclusion::Conflict,
            InferenceConclusion::Predicate(Premise::Eq { var: VarId(0), val: 0 }),
            InferenceConclusion::Predicate(Premise::Ne { var: VarId(1), val: 1 }),
        ] {
            assert_matches_oracle(&tasks, capacity, &roots, &[], conclusion, &roots);
        }
    }
}

#[test]
fn optional_absence_requires_no_implicit_boolean_axiom() {
    let roots = [domain(0, 0, 0), domain(1, 0, 2), domain(2, -1, 2)];
    let constraint = InferenceConstraint::Cumulative { tasks: vec![task(0, 3, 2, None), task(1, 1, 1, Some(2))], capacity: 2 };
    assert_checked(check(&constraint, &roots, &[], InferenceConclusion::Predicate(Premise::Ne { var: VarId(2), val: 1 }), &roots));
    assert_rejected(check(&constraint, &roots, &[], InferenceConclusion::Predicate(Premise::Eq { var: VarId(2), val: 0 }), &roots));
    let mut boolean_roots = roots.clone();
    boolean_roots[2] = domain(2, 0, 1);
    assert_checked(check(
        &constraint,
        &boolean_roots,
        &[],
        InferenceConclusion::Predicate(Premise::Eq { var: VarId(2), val: 0 }),
        &boolean_roots,
    ));
}

#[test]
fn lifted_bound_and_presence_premises_are_each_necessary() {
    let roots = [domain(0, -2, 2), domain(1, 0, 4), domain(2, 0, 1)];
    let snapshot = [domain(0, -1, 0), domain(1, 0, 4), domain(2, 1, 1)];
    let constraint = InferenceConstraint::Cumulative { tasks: vec![task(0, 4, 2, Some(2)), task(1, 1, 1, None)], capacity: 2 };
    let premises =
        [Premise::Ge { var: VarId(0), bound: -1 }, Premise::Le { var: VarId(0), bound: 0 }, Premise::Eq { var: VarId(2), val: 1 }];
    let conclusion = InferenceConclusion::Predicate(Premise::Ge { var: VarId(1), bound: 3 });
    assert_checked(check(&constraint, &roots, &premises, conclusion, &snapshot));
    for missing in 0..premises.len() {
        let incomplete: Vec<_> =
            premises.iter().enumerate().filter_map(|(index, &premise)| (index != missing).then_some(premise)).collect();
        assert_rejected(check(&constraint, &roots, &incomplete, conclusion, &snapshot));
    }
    let mut conflict_premises = premises.to_vec();
    conflict_premises.push(Premise::Le { var: VarId(1), bound: 2 });
    let mut conflict_snapshot = snapshot.clone();
    conflict_snapshot[1].max = 2;
    assert_checked(check(&constraint, &roots, &conflict_premises, InferenceConclusion::Conflict, &conflict_snapshot));
    assert_rejected(check(&constraint, &roots, &premises, InferenceConclusion::Conflict, &conflict_snapshot));
    let false_premise = [Premise::Ge { var: VarId(0), bound: 1 }];
    assert_rejected(check(&constraint, &roots, &false_premise, conclusion, &snapshot));
}

#[test]
fn half_open_endpoints_and_widened_resource_sums_are_exact() {
    let roots = [domain(0, i32::MAX - 1, i32::MAX - 1), domain(1, i32::MAX, i32::MAX)];
    let touching =
        InferenceConstraint::Cumulative { tasks: vec![task(0, 1, i64::MAX, None), task(1, i64::MAX, i64::MAX, None)], capacity: i64::MAX };
    assert_rejected(check(&touching, &roots, &[], InferenceConclusion::Conflict, &roots));
    let overlapping =
        InferenceConstraint::Cumulative { tasks: vec![task(0, 2, i64::MAX, None), task(1, i64::MAX, i64::MAX, None)], capacity: i64::MAX };
    assert_checked(check(&overlapping, &roots, &[], InferenceConclusion::Conflict, &roots));
    let extremes = [domain(0, i32::MIN, i32::MIN), domain(1, i32::MAX, i32::MAX)];
    let long = InferenceConstraint::Cumulative { tasks: vec![task(0, i64::MAX, 1, None), task(1, i64::MAX, 1, None)], capacity: 1 };
    assert_checked(check(&long, &extremes, &[], InferenceConclusion::Conflict, &extremes));
}

#[test]
fn empty_zero_duration_and_zero_demand_tasks_consume_nothing() {
    let roots = [domain(0, -1, 1), domain(1, 0, 1)];
    let zero = InferenceConstraint::Cumulative { tasks: vec![task(0, 0, i64::MAX, Some(1)), task(0, i64::MAX, 0, None)], capacity: 0 };
    assert_rejected(check(&zero, &roots, &[], InferenceConclusion::Conflict, &roots));
    assert_rejected(check(&zero, &roots, &[], InferenceConclusion::Predicate(Premise::Eq { var: VarId(1), val: 0 }), &roots));
    let empty = InferenceConstraint::Cumulative { tasks: vec![], capacity: 0 };
    assert_rejected(check(&empty, &[], &[], InferenceConclusion::Conflict, &[]));
    assert_checked(check(&empty, &roots, &[], InferenceConclusion::Predicate(Premise::Ge { var: VarId(0), bound: -1 }), &roots));
}

#[test]
fn invalid_constants_and_missing_presence_roots_are_uncovered() {
    let roots = [domain(0, 0, 0), domain(1, 0, 1)];
    for (duration, demand, capacity) in [(-1, 1, 1), (1, -1, 1), (1, 1, -1)] {
        let constraint = InferenceConstraint::Cumulative { tasks: vec![task(0, duration, demand, Some(1))], capacity };
        assert!(matches!(
            check(&constraint, &roots, &[], InferenceConclusion::Conflict, &roots),
            InferenceCheckOutcome::Uncovered(InferenceUncoveredReason::InvalidConstraint(_))
        ));
    }
    let constraint = InferenceConstraint::Cumulative { tasks: vec![task(0, 1, 1, Some(1))], capacity: 1 };
    assert!(matches!(
        check(&constraint, &roots[..1], &[], InferenceConclusion::Conflict, &roots),
        InferenceCheckOutcome::Uncovered(InferenceUncoveredReason::MissingRootDomain { var: VarId(1) })
    ));
}

#[test]
fn work_assignment_and_interruption_limits_never_certify_partial_checks() {
    let roots = [domain(0, 0, 20), domain(1, 0, 20)];
    let constraint = InferenceConstraint::Cumulative { tasks: vec![task(0, 50, 1, None), task(1, 50, 1, None)], capacity: 1 };
    let inference = Inference {
        propagator: PropId(17),
        constraint: &constraint,
        root_domains: &roots,
        premises: &[],
        conclusion: InferenceConclusion::Conflict,
        snapshot: &roots,
    };
    assert!(matches!(
        check_inference(&inference, InferenceCheckConfig { max_assignments: 400, max_work: 100_000 }),
        InferenceCheckOutcome::Uncovered(InferenceUncoveredReason::AssignmentLimit)
    ));
    for max_work in [0, 1, 64] {
        assert!(matches!(
            check_inference(&inference, InferenceCheckConfig { max_assignments: 1_000, max_work }),
            InferenceCheckOutcome::Uncovered(InferenceUncoveredReason::WorkLimit)
        ));
    }
    let completed_polls = Cell::new(0usize);
    assert_checked(check_inference_until(&inference, InferenceCheckConfig::default(), &|| {
        completed_polls.set(completed_polls.get() + 1);
        false
    }));
    assert!(completed_polls.get() > 500);
    for cutoff in [0, 1, completed_polls.get() / 2, completed_polls.get() - 1] {
        let polls = Cell::new(0usize);
        let outcome = check_inference_until(&inference, InferenceCheckConfig::default(), &|| {
            let count = polls.get();
            polls.set(count + 1);
            count >= cutoff
        });
        assert!(matches!(outcome, InferenceCheckOutcome::Uncovered(InferenceUncoveredReason::Interrupted)), "{outcome:?}");
    }
    let wide_roots = [domain(0, i32::MIN, i32::MAX), domain(1, 0, 20)];
    let wide = Inference { root_domains: &wide_roots, snapshot: &wide_roots, ..inference };
    assert!(matches!(
        check_inference(&wide, InferenceCheckConfig { max_assignments: 100, max_work: 100_000 }),
        InferenceCheckOutcome::Uncovered(InferenceUncoveredReason::AssignmentLimit)
    ));
}
