use std::collections::BTreeSet;
use std::sync::Arc;

use qayd::ids::{PropId, VarId};
use qayd::inference::{
    check_inference, Inference, InferenceCheckConfig, InferenceCheckOutcome, InferenceConclusion, InferenceConstraint, InferenceDomain,
    InferenceUncoveredReason,
};
use qayd::store::Premise;

fn domain(var: u32, min: i32, max: i32) -> InferenceDomain {
    InferenceDomain { var: VarId(var), min, max, values: None, holes: vec![] }
}

fn check(
    constraint: &InferenceConstraint,
    roots: &[InferenceDomain],
    premises: &[Premise],
    conclusion: InferenceConclusion,
    snapshot: &[InferenceDomain],
) -> InferenceCheckOutcome {
    check_inference(
        &Inference { propagator: PropId(42), constraint, root_domains: roots, premises, conclusion, snapshot },
        InferenceCheckConfig::default(),
    )
}

fn holds(premise: Premise, assignment: &[i32]) -> bool {
    match premise {
        Premise::Ge { var, bound } => assignment[var.index()] >= bound,
        Premise::Le { var, bound } => assignment[var.index()] <= bound,
        Premise::Eq { var, val } => assignment[var.index()] == val,
        Premise::Ne { var, val } => assignment[var.index()] != val,
    }
}

/// Independent small-instance oracle: reserve each occupied integer time slot.
/// This does not use the checker's pairwise endpoint formula or any propagator.
fn occupied_slots_feasible(tasks: &[(VarId, i64)], assignment: &[i32]) -> bool {
    let mut occupied = BTreeSet::new();
    for &(variable, duration) in tasks {
        let start = i64::from(assignment[variable.index()]);
        for time in start..start + duration {
            if !occupied.insert(time) {
                return false;
            }
        }
    }
    true
}

fn assert_verdict(outcome: InferenceCheckOutcome, valid: bool) {
    match outcome {
        InferenceCheckOutcome::Checked { .. } => assert!(valid, "accepted an invalid no-overlap inference"),
        InferenceCheckOutcome::Rejected(failure) => assert!(!valid, "rejected a valid inference: {failure}"),
        InferenceCheckOutcome::Uncovered(reason) => panic!("unexpected coverage gap: {reason:?}"),
    }
}

#[test]
fn no_overlap_inferences_match_exhaustive_occupied_slot_oracle() {
    let mut roots = vec![domain(0, -2, 2), domain(1, -1, 3), domain(2, 0, 4)];
    roots[0].values = Some(Arc::from([2, -2, 0, 2]));
    roots[0].holes = vec![0, 0];
    roots[1].values = Some(Arc::from([3, -1, 1]));
    roots[2].values = Some(Arc::from([0, 2, 4]));
    let conclusions = [
        InferenceConclusion::Conflict,
        InferenceConclusion::Predicate(Premise::Ge { var: VarId(1), bound: 1 }),
        InferenceConclusion::Predicate(Premise::Ge { var: VarId(2), bound: 4 }),
        InferenceConclusion::Predicate(Premise::Le { var: VarId(0), bound: -2 }),
        InferenceConclusion::Predicate(Premise::Le { var: VarId(2), bound: 2 }),
        InferenceConclusion::Predicate(Premise::Eq { var: VarId(2), val: 2 }),
        InferenceConclusion::Predicate(Premise::Ne { var: VarId(0), val: -2 }),
        InferenceConclusion::Predicate(Premise::Ne { var: VarId(1), val: 1 }),
    ];
    for with_premise in [false, true] {
        let premises = if with_premise { vec![Premise::Ge { var: VarId(1), bound: 1 }] } else { vec![] };
        let mut snapshot = roots.clone();
        if with_premise {
            snapshot[1].min = 1;
        }
        for first in 1..=3 {
            for second in 1..=3 {
                for third in 1..=3 {
                    let tasks = vec![(VarId(0), first), (VarId(1), second), (VarId(2), third)];
                    let constraint = InferenceConstraint::NoOverlap { tasks: tasks.clone() };
                    for conclusion in conclusions {
                        let mut valid = true;
                        for x in [-2, 2] {
                            for y in [-1, 1, 3] {
                                for z in [0, 2, 4] {
                                    let assignment = [x, y, z];
                                    let proves = match conclusion {
                                        InferenceConclusion::Conflict => false,
                                        InferenceConclusion::Predicate(premise) => holds(premise, &assignment),
                                    };
                                    if occupied_slots_feasible(&tasks, &assignment)
                                        && premises.iter().all(|&premise| holds(premise, &assignment))
                                        && !proves
                                    {
                                        valid = false;
                                    }
                                }
                            }
                        }
                        let outcome = check(&constraint, &roots, &premises, conclusion, &snapshot);
                        if let InferenceCheckOutcome::Rejected(failure) = &outcome {
                            let counterexample = failure.counterexample.as_ref().expect("rejected conclusion must have a witness");
                            assert_eq!(counterexample.len(), 3);
                            let assignment: Vec<_> = counterexample.iter().map(|(_, value)| *value).collect();
                            assert!(occupied_slots_feasible(&tasks, &assignment));
                            assert!(premises.iter().all(|&premise| holds(premise, &assignment)));
                            if let InferenceConclusion::Predicate(premise) = conclusion {
                                assert!(!holds(premise, &assignment));
                            }
                        }
                        assert_verdict(outcome, valid);
                    }
                }
            }
        }
    }
}

#[test]
fn no_overlap_requires_explicit_historical_premises_for_pruning_and_conflicts() {
    let roots = vec![domain(0, 0, 4), domain(1, 0, 4)];
    let constraint = InferenceConstraint::NoOverlap { tasks: vec![(VarId(0), 2), (VarId(1), 2)] };
    let snapshot = vec![domain(0, 0, 0), domain(1, 2, 4)];
    let conclusion = InferenceConclusion::Predicate(Premise::Ge { var: VarId(1), bound: 2 });
    assert_verdict(check(&constraint, &roots, &[], conclusion, &snapshot), false);
    assert_verdict(check(&constraint, &roots, &[Premise::Eq { var: VarId(0), val: 0 }], conclusion, &snapshot), true);

    let conflicting = vec![domain(0, 0, 0), domain(1, 0, 0)];
    assert_verdict(check(&constraint, &roots, &[], InferenceConclusion::Conflict, &conflicting), false);
    assert_verdict(
        check(
            &constraint,
            &roots,
            &[Premise::Eq { var: VarId(0), val: 0 }, Premise::Eq { var: VarId(1), val: 0 }],
            InferenceConclusion::Conflict,
            &conflicting,
        ),
        true,
    );
    let false_premise = Premise::Eq { var: VarId(0), val: 1 };
    let outcome = check(&constraint, &roots, &[false_premise], InferenceConclusion::Predicate(false_premise), &snapshot);
    let InferenceCheckOutcome::Rejected(failure) = outcome else { panic!("false current premise must be rejected") };
    assert!(failure.detail.contains("premise is false"));
}

#[test]
fn no_overlap_retains_duplicate_tasks_but_shares_start_assignments() {
    let roots = vec![domain(0, -3, 3)];
    for tasks in [vec![(VarId(0), 1), (VarId(0), 1)], vec![(VarId(0), 2), (VarId(0), 3)]] {
        let constraint = InferenceConstraint::NoOverlap { tasks };
        assert_verdict(check(&constraint, &roots, &[], InferenceConclusion::Conflict, &roots), true);
    }
    let constraint = InferenceConstraint::NoOverlap { tasks: vec![(VarId(0), 1)] };
    assert_verdict(check(&constraint, &roots, &[], InferenceConclusion::Conflict, &roots), false);
    assert_verdict(check(&constraint, &roots, &[], InferenceConclusion::Predicate(Premise::Eq { var: VarId(0), val: 0 }), &roots), false);
}

#[test]
fn no_overlap_uses_exact_widened_endpoints_and_allows_touching() {
    let roots = vec![domain(0, i32::MIN, i32::MIN), domain(1, i32::MAX, i32::MAX)];
    let separation = i64::from(i32::MAX) - i64::from(i32::MIN);
    for (first_duration, infeasible) in [(separation, false), (separation + 1, true), (i64::MAX, true)] {
        let constraint = InferenceConstraint::NoOverlap { tasks: vec![(VarId(0), first_duration), (VarId(1), i64::MAX)] };
        assert_verdict(check(&constraint, &roots, &[], InferenceConclusion::Conflict, &roots), infeasible);
    }
    let mut sparse = vec![domain(0, i32::MIN, i32::MAX), domain(1, i32::MAX, i32::MAX)];
    sparse[0].values = Some(Arc::from([i32::MIN, i32::MAX]));
    let constraint = InferenceConstraint::NoOverlap { tasks: vec![(VarId(0), 1), (VarId(1), 1)] };
    assert_verdict(
        check(&constraint, &sparse, &[], InferenceConclusion::Predicate(Premise::Eq { var: VarId(0), val: i32::MIN }), &sparse),
        true,
    );
}

#[test]
fn no_overlap_external_premises_and_nullary_constraint_have_exact_semantics() {
    let roots = vec![domain(0, 0, 2), domain(1, 0, 2)];
    let constraint = InferenceConstraint::NoOverlap { tasks: vec![(VarId(0), 1)] };
    let conclusion = InferenceConclusion::Predicate(Premise::Eq { var: VarId(1), val: 1 });
    assert_verdict(check(&constraint, &roots, &[], conclusion, &roots), false);
    let snapshot = vec![roots[0].clone(), domain(1, 1, 1)];
    assert_verdict(check(&constraint, &roots, &[Premise::Eq { var: VarId(1), val: 1 }], conclusion, &snapshot), true);
    let empty = InferenceConstraint::NoOverlap { tasks: vec![] };
    assert_verdict(check(&empty, &[], &[], InferenceConclusion::Conflict, &[]), false);
}

#[test]
fn no_overlap_invalid_and_missing_inputs_are_uncovered() {
    let roots = vec![domain(0, 0, 2)];
    for duration in [0, -1, i64::MIN] {
        let constraint = InferenceConstraint::NoOverlap { tasks: vec![(VarId(0), duration)] };
        assert!(matches!(
            check(&constraint, &roots, &[], InferenceConclusion::Conflict, &roots),
            InferenceCheckOutcome::Uncovered(InferenceUncoveredReason::InvalidConstraint(_))
        ));
    }
    let missing = InferenceConstraint::NoOverlap { tasks: vec![(VarId(1), 1)] };
    assert!(matches!(
        check(&missing, &roots, &[], InferenceConclusion::Conflict, &roots),
        InferenceCheckOutcome::Uncovered(InferenceUncoveredReason::MissingRootDomain { var: VarId(1) })
    ));
}

#[test]
fn no_overlap_assignment_and_work_caps_never_accept_partial_checks() {
    let roots = vec![domain(0, 0, 10), domain(1, 0, 10)];
    let constraint = InferenceConstraint::NoOverlap { tasks: vec![(VarId(0), 1), (VarId(1), 1)] };
    let inference = Inference {
        propagator: PropId(0),
        constraint: &constraint,
        root_domains: &roots,
        premises: &[],
        conclusion: InferenceConclusion::Conflict,
        snapshot: &roots,
    };
    for config in
        [InferenceCheckConfig { max_assignments: 1, max_work: 10_000 }, InferenceCheckConfig { max_assignments: 0, max_work: 10_000 }]
    {
        assert!(matches!(check_inference(&inference, config), InferenceCheckOutcome::Uncovered(InferenceUncoveredReason::AssignmentLimit)));
    }
    assert!(matches!(
        check_inference(&inference, InferenceCheckConfig { max_assignments: 1000, max_work: 0 }),
        InferenceCheckOutcome::Uncovered(InferenceUncoveredReason::WorkLimit)
    ));
    let high_arity = InferenceConstraint::NoOverlap { tasks: vec![(VarId(0), 1); 10_000] };
    let inference = Inference { constraint: &high_arity, ..inference };
    assert!(matches!(
        check_inference(&inference, InferenceCheckConfig { max_assignments: 1000, max_work: 100 }),
        InferenceCheckOutcome::Uncovered(InferenceUncoveredReason::WorkLimit)
    ));

    // All starts are fixed and nonoverlapping. A rejection is legitimate only
    // after the checker has validated every pair, not merely an initial prefix.
    let points: Vec<_> = (0..100).map(|index| domain(index, index as i32, index as i32)).collect();
    let many_pairs = InferenceConstraint::NoOverlap { tasks: (0..100).map(|index| (VarId(index), 1)).collect() };
    let inference = Inference { constraint: &many_pairs, root_domains: &points, snapshot: &points, ..inference };
    assert!(matches!(
        check_inference(&inference, InferenceCheckConfig { max_assignments: 1, max_work: 1000 }),
        InferenceCheckOutcome::Uncovered(InferenceUncoveredReason::WorkLimit)
    ));
    assert_verdict(check(&many_pairs, &points, &[], InferenceConclusion::Conflict, &points), false);
}
