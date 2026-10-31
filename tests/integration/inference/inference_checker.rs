use std::sync::Arc;

use qayd::ids::{PropId, VarId};
use qayd::inference::{
    check_inference, Inference, InferenceCheckConfig, InferenceCheckOutcome, InferenceConclusion, InferenceConstraint, InferenceDomain,
    InferenceRelation, InferenceUncoveredReason, MAX_INFERENCE_ASSIGNMENTS,
};
use qayd::store::Premise;

fn interval(var: u32, min: i32, max: i32) -> InferenceDomain {
    InferenceDomain { var: VarId(var), min, max, values: None, holes: Vec::new() }
}

fn check(
    constraint: &InferenceConstraint,
    roots: &[InferenceDomain],
    premises: &[Premise],
    conclusion: InferenceConclusion,
    snapshot: &[InferenceDomain],
) -> InferenceCheckOutcome {
    check_inference(
        &Inference { propagator: PropId(7), constraint, root_domains: roots, premises, conclusion, snapshot },
        InferenceCheckConfig::default(),
    )
}

fn holds(premise: Premise, assignment: &[i32]) -> bool {
    match premise {
        Premise::Ge { var, bound } => assignment[var.0 as usize] >= bound,
        Premise::Le { var, bound } => assignment[var.0 as usize] <= bound,
        Premise::Eq { var, val } => assignment[var.0 as usize] == val,
        Premise::Ne { var, val } => assignment[var.0 as usize] != val,
    }
}

fn assert_verdict(outcome: InferenceCheckOutcome, valid: bool) {
    match outcome {
        InferenceCheckOutcome::Checked { .. } => assert!(valid, "accepted invalid inference"),
        InferenceCheckOutcome::Rejected(failure) => assert!(!valid, "rejected valid inference: {failure}"),
        InferenceCheckOutcome::Uncovered(reason) => panic!("unexpected coverage gap: {reason:?}"),
    }
}

#[test]
fn linear_implications_match_exhaustive_signed_holey_oracle() {
    let mut roots = vec![interval(0, -2, 2), interval(1, -1, 2)];
    roots[0].values = Some(Arc::from([2, -2, 1, -1, 2]));
    roots[1].holes = vec![0, 0];
    let conclusions = [
        InferenceConclusion::Conflict,
        InferenceConclusion::Predicate(Premise::Ge { var: VarId(0), bound: 1 }),
        InferenceConclusion::Predicate(Premise::Le { var: VarId(1), bound: -1 }),
        InferenceConclusion::Predicate(Premise::Eq { var: VarId(0), val: -1 }),
        InferenceConclusion::Predicate(Premise::Ne { var: VarId(1), val: 2 }),
    ];
    for with_premise in [false, true] {
        let premises = if with_premise { vec![Premise::Ge { var: VarId(0), bound: 1 }] } else { Vec::new() };
        let mut snapshot = roots.clone();
        if with_premise {
            snapshot[0].min = 1;
        }
        for a in -2..=2i128 {
            for b in -2..=2i128 {
                for rhs in -3..=3i128 {
                    for relation in [InferenceRelation::Le, InferenceRelation::Eq, InferenceRelation::Ne] {
                        let constraint = InferenceConstraint::Linear { terms: vec![(VarId(0), a), (VarId(1), b)], relation, rhs };
                        for conclusion in conclusions {
                            let mut valid = true;
                            for x in [-2, -1, 1, 2] {
                                for y in [-1, 1, 2] {
                                    let assignment = [x, y];
                                    let sum = a * i128::from(x) + b * i128::from(y);
                                    let satisfies = match relation {
                                        InferenceRelation::Le => sum <= rhs,
                                        InferenceRelation::Eq => sum == rhs,
                                        InferenceRelation::Ne => sum != rhs,
                                    };
                                    let proves = match conclusion {
                                        InferenceConclusion::Conflict => false,
                                        InferenceConclusion::Predicate(p) => holds(p, &assignment),
                                    };
                                    if satisfies && premises.iter().all(|&p| holds(p, &assignment)) && !proves {
                                        valid = false;
                                    }
                                }
                            }
                            assert_verdict(check(&constraint, &roots, &premises, conclusion, &snapshot), valid);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn missing_premise_is_rejected_even_when_conclusion_holds_in_snapshot() {
    let roots = vec![interval(0, 0, 2), interval(1, 0, 2)];
    let snapshot = vec![interval(0, 1, 2), interval(1, 0, 1)];
    let constraint = InferenceConstraint::Linear { terms: vec![(VarId(0), 1), (VarId(1), 1)], relation: InferenceRelation::Le, rhs: 2 };
    let conclusion = InferenceConclusion::Predicate(Premise::Le { var: VarId(1), bound: 1 });
    let outcome = check(&constraint, &roots, &[], conclusion, &snapshot);
    let InferenceCheckOutcome::Rejected(failure) = outcome else { panic!("missing premise escaped checking") };
    let counterexample = failure.counterexample.as_ref().unwrap();
    assert_eq!(counterexample, &vec![(VarId(0), 0), (VarId(1), 2)]);
    assert!(failure.to_string().contains("PropId(7)"));
    assert_eq!(failure.snapshot[0].min, 1);
    assert_verdict(check(&constraint, &roots, &[Premise::Ge { var: VarId(0), bound: 1 }], conclusion, &snapshot), true);
}

#[test]
fn false_premise_is_rejected_even_if_it_would_entail_conclusion() {
    let roots = vec![interval(0, 0, 2)];
    let constraint = InferenceConstraint::Linear { terms: vec![(VarId(0), 1)], relation: InferenceRelation::Le, rhs: 2 };
    let premise = Premise::Ge { var: VarId(0), bound: 1 };
    let outcome = check(&constraint, &roots, &[premise], InferenceConclusion::Predicate(premise), &roots);
    let InferenceCheckOutcome::Rejected(failure) = outcome else { panic!("false premise accepted") };
    assert!(failure.detail.contains("premise is false"));
    assert_eq!(failure.counterexample, Some(vec![(VarId(0), 0)]));
}

#[test]
fn repeated_linear_variables_share_one_assignment() {
    let roots = vec![interval(0, -2, 2)];
    let constraint = InferenceConstraint::Linear { terms: vec![(VarId(0), 1), (VarId(0), -1)], relation: InferenceRelation::Eq, rhs: 1 };
    assert_verdict(check(&constraint, &roots, &[], InferenceConclusion::Conflict, &roots), true);
    let constraint = InferenceConstraint::Linear { terms: vec![(VarId(0), 1), (VarId(0), 1)], relation: InferenceRelation::Eq, rhs: 2 };
    assert_verdict(check(&constraint, &roots, &[], InferenceConclusion::Predicate(Premise::Eq { var: VarId(0), val: 1 }), &roots), true);
    assert_verdict(check(&constraint, &roots, &[], InferenceConclusion::Conflict, &roots), false);
}

#[test]
fn positive_table_with_stars_and_repeated_columns_matches_oracle() {
    let roots = vec![interval(0, -1, 1), interval(1, -1, 1)];
    let star = i32::MIN;
    for rows in [vec![-1, -1, 0, 1, 1, star], vec![-1, 1, star, star, 0, 1], vec![star, star, star], Vec::new()] {
        let constraint = InferenceConstraint::PositiveTable {
            variables: vec![VarId(0), VarId(0), VarId(1)],
            tuples: rows.clone().into(),
            arity: 3,
            rows: rows.len() / 3,
        };
        for conclusion in [
            InferenceConclusion::Conflict,
            InferenceConclusion::Predicate(Premise::Ne { var: VarId(0), val: 0 }),
            InferenceConclusion::Predicate(Premise::Le { var: VarId(1), bound: 0 }),
            InferenceConclusion::Predicate(Premise::Eq { var: VarId(1), val: 1 }),
        ] {
            let valid = (-1..=1).all(|x| {
                (-1..=1).all(|y| {
                    let values = [x, x, y];
                    let satisfies = rows.chunks(3).any(|row| row.iter().zip(values).all(|(&cell, value)| cell == star || cell == value));
                    let proves = match conclusion {
                        InferenceConclusion::Conflict => false,
                        InferenceConclusion::Predicate(p) => holds(p, &[x, y]),
                    };
                    !satisfies || proves
                })
            });
            assert_verdict(check(&constraint, &roots, &[], conclusion, &roots), valid);
        }
    }
}

#[test]
fn wildcard_support_respects_sparse_roots_holes_and_external_premises() {
    let mut roots = vec![interval(0, -3, 3), interval(1, -3, 3)];
    roots[0].values = Some(Arc::from([-3, 1, 3]));
    roots[0].holes = vec![1];
    let constraint = InferenceConstraint::PositiveTable { variables: vec![VarId(0)], tuples: Arc::from([i32::MIN]), arity: 1, rows: 1 };
    assert_verdict(check(&constraint, &roots, &[], InferenceConclusion::Predicate(Premise::Ne { var: VarId(0), val: 1 }), &roots), true);
    assert_verdict(check(&constraint, &roots, &[], InferenceConclusion::Predicate(Premise::Ne { var: VarId(0), val: 2 }), &roots), true);
    assert_verdict(check(&constraint, &roots, &[], InferenceConclusion::Predicate(Premise::Eq { var: VarId(1), val: 1 }), &roots), false);
    let mut snapshot = roots.clone();
    snapshot[1].min = 1;
    snapshot[1].max = 1;
    assert_verdict(
        check(
            &constraint,
            &roots,
            &[Premise::Eq { var: VarId(1), val: 1 }],
            InferenceConclusion::Predicate(Premise::Eq { var: VarId(1), val: 1 }),
            &snapshot,
        ),
        true,
    );
}

#[test]
fn full_i32_endpoints_and_wide_wildcard_domains_are_checked_without_expansion() {
    let roots = vec![interval(0, i32::MIN, i32::MAX)];
    let constraint = InferenceConstraint::PositiveTable { variables: vec![VarId(0)], tuples: Arc::from([i32::MIN]), arity: 1, rows: 1 };
    for premise in [Premise::Ge { var: VarId(0), bound: i32::MIN }, Premise::Le { var: VarId(0), bound: i32::MAX }] {
        assert_verdict(check(&constraint, &roots, &[], InferenceConclusion::Predicate(premise), &roots), true);
    }
    assert_verdict(check(&constraint, &roots, &[], InferenceConclusion::Conflict, &roots), false);
    let constraint = InferenceConstraint::Linear {
        terms: vec![(VarId(0), i128::from(i64::MAX))],
        relation: InferenceRelation::Eq,
        rhs: i128::from(i64::MAX) * i128::from(i32::MIN),
    };
    let point = vec![interval(0, i32::MIN, i32::MIN)];
    assert_verdict(check(&constraint, &point, &[], InferenceConclusion::Conflict, &point), false);
}

#[test]
fn arithmetic_overflow_and_resource_limits_are_uncovered() {
    let point = vec![interval(0, 2, 2)];
    let constraint = InferenceConstraint::Linear { terms: vec![(VarId(0), i128::MAX)], relation: InferenceRelation::Eq, rhs: 0 };
    assert!(matches!(
        check(&constraint, &point, &[], InferenceConclusion::Conflict, &point),
        InferenceCheckOutcome::Uncovered(InferenceUncoveredReason::ArithmeticOverflow)
    ));
    let constraint = InferenceConstraint::Linear { terms: vec![(VarId(0), 0)], relation: InferenceRelation::Eq, rhs: 1 };
    let wide = vec![interval(0, i32::MIN, i32::MAX)];
    assert!(matches!(
        check(&constraint, &wide, &[], InferenceConclusion::Conflict, &wide),
        InferenceCheckOutcome::Uncovered(InferenceUncoveredReason::AssignmentLimit)
    ));
    let inference = Inference {
        propagator: PropId(0),
        constraint: &constraint,
        root_domains: &point,
        premises: &[],
        conclusion: InferenceConclusion::Conflict,
        snapshot: &point,
    };
    assert!(matches!(
        check_inference(&inference, InferenceCheckConfig { max_assignments: 1, max_work: 0 }),
        InferenceCheckOutcome::Uncovered(InferenceUncoveredReason::WorkLimit)
    ));
    assert!(matches!(
        check_inference(&inference, InferenceCheckConfig { max_assignments: 0, max_work: 100 }),
        InferenceCheckOutcome::Uncovered(InferenceUncoveredReason::AssignmentLimit)
    ));
    let beyond_ceiling = vec![interval(0, 0, MAX_INFERENCE_ASSIGNMENTS as i32)];
    let inference = Inference { root_domains: &beyond_ceiling, snapshot: &beyond_ceiling, ..inference };
    assert!(matches!(
        check_inference(&inference, InferenceCheckConfig { max_assignments: usize::MAX, max_work: usize::MAX }),
        InferenceCheckOutcome::Uncovered(InferenceUncoveredReason::AssignmentLimit)
    ));
}

#[test]
fn malformed_or_missing_inputs_are_uncovered() {
    let roots = vec![interval(0, 0, 1)];
    let malformed = InferenceConstraint::PositiveTable { variables: vec![VarId(0)], tuples: Arc::from([0, 1]), arity: 1, rows: 3 };
    assert!(matches!(
        check(&malformed, &roots, &[], InferenceConclusion::Conflict, &roots),
        InferenceCheckOutcome::Uncovered(InferenceUncoveredReason::InvalidConstraint(_))
    ));
    let missing = InferenceConstraint::Linear { terms: vec![(VarId(1), 1)], relation: InferenceRelation::Eq, rhs: 0 };
    assert!(matches!(
        check(&missing, &roots, &[], InferenceConclusion::Conflict, &roots),
        InferenceCheckOutcome::Uncovered(InferenceUncoveredReason::MissingRootDomain { var: VarId(1) })
    ));
    let constraint = InferenceConstraint::Linear { terms: Vec::new(), relation: InferenceRelation::Eq, rhs: 1 };
    assert!(matches!(
        check(&constraint, &roots, &[Premise::Ge { var: VarId(0), bound: 0 }], InferenceConclusion::Conflict, &[]),
        InferenceCheckOutcome::Uncovered(InferenceUncoveredReason::MissingSnapshotDomain { .. })
    ));
    let duplicate = vec![roots[0].clone(), roots[0].clone()];
    assert!(matches!(
        check(&constraint, &duplicate, &[], InferenceConclusion::Conflict, &roots),
        InferenceCheckOutcome::Uncovered(InferenceUncoveredReason::InvalidDomain { .. })
    ));
}

#[test]
fn nullary_constraints_and_empty_domains_have_exact_semantics() {
    for rhs in [0, 1] {
        let constraint = InferenceConstraint::Linear { terms: Vec::new(), relation: InferenceRelation::Eq, rhs };
        assert_verdict(check(&constraint, &[], &[], InferenceConclusion::Conflict, &[]), rhs != 0);
    }
    for rows in [0, 1] {
        let constraint = InferenceConstraint::PositiveTable { variables: Vec::new(), tuples: Arc::from([]), arity: 0, rows };
        assert_verdict(check(&constraint, &[], &[], InferenceConclusion::Conflict, &[]), rows == 0);
    }
    let roots = vec![InferenceDomain { values: Some(Arc::from([])), ..interval(0, 0, 10) }];
    let constraint = InferenceConstraint::Linear { terms: vec![(VarId(0), 1)], relation: InferenceRelation::Eq, rhs: 1 };
    assert_verdict(check(&constraint, &roots, &[], InferenceConclusion::Conflict, &roots), true);
}

#[test]
fn high_arity_and_large_sparse_inputs_respect_work_budget() {
    let root = vec![interval(0, 0, 1)];
    let table = InferenceConstraint::PositiveTable {
        variables: vec![VarId(0); 10_000],
        tuples: vec![i32::MIN; 10_000].into(),
        arity: 10_000,
        rows: 1,
    };
    let inference = Inference {
        propagator: PropId(0),
        constraint: &table,
        root_domains: &root,
        premises: &[],
        conclusion: InferenceConclusion::Conflict,
        snapshot: &root,
    };
    assert!(matches!(
        check_inference(&inference, InferenceCheckConfig { max_assignments: 100, max_work: 100 }),
        InferenceCheckOutcome::Uncovered(InferenceUncoveredReason::WorkLimit)
    ));
    let mut sparse = root.clone();
    sparse[0].values = Some(vec![2; 10_000].into());
    let linear = InferenceConstraint::Linear { terms: vec![(VarId(0), 1)], relation: InferenceRelation::Eq, rhs: 0 };
    let inference = Inference { constraint: &linear, root_domains: &sparse, snapshot: &sparse, ..inference };
    assert!(matches!(
        check_inference(&inference, InferenceCheckConfig { max_assignments: 100, max_work: 100 }),
        InferenceCheckOutcome::Uncovered(InferenceUncoveredReason::WorkLimit)
    ));
}
