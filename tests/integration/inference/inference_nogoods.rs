//! Independent semantic gates for clause deductions and conditional derivations.

use std::cell::Cell;
use std::sync::Arc;

use qayd::inference::{
    check_clause_entailment, check_clause_entailment_until, check_inference, ClauseCheckOutcome, ClauseEntailment, Inference,
    InferenceCheckConfig, InferenceCheckOutcome, InferenceConclusion, InferenceConstraint, InferenceDomain, InferenceLiteral,
    InferenceUncoveredReason,
};
use qayd::store::Premise;
use qayd::{PropId, VarId};

fn literal(predicate: Premise, positive: bool) -> InferenceLiteral {
    InferenceLiteral { predicate, positive }
}

fn eq(var: u32, val: i32) -> InferenceLiteral {
    literal(Premise::Eq { var: VarId(var), val }, true)
}

fn domain(var: u32, values: &[i32]) -> InferenceDomain {
    InferenceDomain {
        var: VarId(var),
        min: *values.iter().min().unwrap(),
        max: *values.iter().max().unwrap(),
        values: Some(Arc::from(values)),
        holes: Vec::new(),
    }
}

fn range(var: u32, min: i32, max: i32) -> InferenceDomain {
    InferenceDomain { var: VarId(var), min, max, values: None, holes: Vec::new() }
}

fn holds(predicate: Premise, values: &[i32]) -> bool {
    match predicate {
        Premise::Eq { var, val } => values[var.index()] == val,
        Premise::Ne { var, val } => values[var.index()] != val,
        Premise::Le { var, bound } => values[var.index()] <= bound,
        Premise::Ge { var, bound } => values[var.index()] >= bound,
    }
}

fn clause_holds(clause: &[InferenceLiteral], values: &[i32]) -> bool {
    clause.iter().any(|literal| holds(literal.predicate, values) == literal.positive)
}

fn assignments(domains: &[Vec<i32>]) -> Vec<Vec<i32>> {
    let mut all = vec![Vec::new()];
    for domain in domains {
        all = all
            .into_iter()
            .flat_map(|prefix| {
                domain.iter().map(move |&value| {
                    let mut assignment = prefix.clone();
                    assignment.push(value);
                    assignment
                })
            })
            .collect();
    }
    all
}

fn verify_entailment(roots: &[InferenceDomain], values: &[Vec<i32>], sources: &[&[InferenceLiteral]], target: &[InferenceLiteral]) {
    let counterexamples = assignments(values)
        .into_iter()
        .filter(|assignment| sources.iter().all(|source| clause_holds(source, assignment)) && !clause_holds(target, assignment))
        .collect::<Vec<_>>();
    match check_clause_entailment(&ClauseEntailment { root_domains: roots, sources, conclusion: target }, InferenceCheckConfig::default()) {
        ClauseCheckOutcome::Checked { .. } => assert!(counterexamples.is_empty(), "sources={sources:?}, target={target:?}"),
        ClauseCheckOutcome::Rejected(failure) => {
            assert!(!counterexamples.is_empty(), "{failure}");
            assert!(counterexamples
                .iter()
                .any(|assignment| failure.counterexample.iter().all(|&(var, value)| assignment[var.index()] == value)));
        }
        other => panic!("small exhaustive check was not completed: {other:?}"),
    }
}

#[test]
fn signed_eq_le_ge_derivations_match_exhaustive_integer_truth_tables() {
    let mut pool = Vec::new();
    for var in [VarId(0), VarId(1)] {
        for predicate in [Premise::Eq { var, val: 0 }, Premise::Le { var, bound: 0 }, Premise::Ge { var, bound: 0 }] {
            for positive in [false, true] {
                pool.push(literal(predicate, positive));
            }
        }
    }
    let domains = [vec![-1, 0, 1], vec![-1, 1], vec![0]];
    for left in &domains {
        for right in &domains {
            let roots = [domain(0, left), domain(1, right)];
            let values = [left.clone(), right.clone()];
            for (i, &first) in pool.iter().enumerate() {
                for (j, &second) in pool.iter().enumerate() {
                    let a = [first, pool[(i + 4) % pool.len()]];
                    let b = [second, pool[(j + 7) % pool.len()]];
                    for target in [vec![], vec![first], vec![pool[(i + j) % pool.len()]], vec![first, second]] {
                        verify_entailment(&roots, &values, &[&a, &b], &target);
                    }
                }
            }
        }
    }
}

#[test]
fn clause_descriptor_deductions_match_exhaustive_signed_predicates() {
    let roots = [domain(0, &[-1, 0, 1]), domain(1, &[-1, 0, 1])];
    let all = assignments(&[vec![-1, 0, 1], vec![-1, 0, 1]]);
    let mut pool = Vec::new();
    for var in [VarId(0), VarId(1)] {
        for predicate in
            [Premise::Eq { var, val: 0 }, Premise::Ne { var, val: 0 }, Premise::Le { var, bound: 0 }, Premise::Ge { var, bound: 0 }]
        {
            for positive in [false, true] {
                pool.push(literal(predicate, positive));
            }
        }
    }
    for &first in &pool {
        for &second in &pool {
            let clause = vec![first, second];
            let constraint = InferenceConstraint::Clause { literals: clause.clone() };
            for conclusion in [
                InferenceConclusion::Conflict,
                InferenceConclusion::Predicate(Premise::Eq { var: VarId(0), val: 0 }),
                InferenceConclusion::Predicate(Premise::Ne { var: VarId(0), val: 0 }),
                InferenceConclusion::Predicate(Premise::Ge { var: VarId(1), bound: 0 }),
                InferenceConclusion::Predicate(Premise::Le { var: VarId(1), bound: 0 }),
            ] {
                let valid = all.iter().all(|assignment| {
                    !clause_holds(&clause, assignment)
                        || match conclusion {
                            InferenceConclusion::Conflict => false,
                            InferenceConclusion::Predicate(predicate) => holds(predicate, assignment),
                        }
                });
                let inference = Inference {
                    propagator: PropId(0),
                    constraint: &constraint,
                    root_domains: &roots,
                    premises: &[],
                    conclusion,
                    snapshot: &roots,
                };
                match check_inference(&inference, InferenceCheckConfig::default()) {
                    InferenceCheckOutcome::Checked { .. } => assert!(valid, "{inference:?}"),
                    InferenceCheckOutcome::Rejected(_) => assert!(!valid, "{inference:?}"),
                    other => panic!("small inference was not checked: {other:?}"),
                }
            }
        }
    }
}

#[test]
fn generalized_nogood_removals_require_the_actual_outside_variable_premises() {
    let roots = [range(0, -2, 2), range(1, 0, 1)];
    let snapshot = [roots[0].clone(), range(1, 0, 0)];
    let clause = InferenceConstraint::Clause {
        literals: vec![
            literal(Premise::Le { var: VarId(0), bound: -1 }, true),
            literal(Premise::Ge { var: VarId(0), bound: 1 }, true),
            eq(1, 1),
        ],
    };
    let premises = [Premise::Ne { var: VarId(1), val: 1 }];
    let mut inference = Inference {
        propagator: PropId(4),
        constraint: &clause,
        root_domains: &roots,
        premises: &premises,
        conclusion: InferenceConclusion::Predicate(Premise::Ne { var: VarId(0), val: 0 }),
        snapshot: &snapshot,
    };
    assert!(matches!(check_inference(&inference, InferenceCheckConfig::default()), InferenceCheckOutcome::Checked { .. }));
    inference.premises = &[];
    assert!(
        matches!(check_inference(&inference, InferenceCheckConfig::default()), InferenceCheckOutcome::Rejected(_)),
        "a current domain fact cannot replace an omitted historical premise"
    );
    inference.premises = &premises;
    inference.snapshot = &roots;
    let InferenceCheckOutcome::Rejected(failure) = check_inference(&inference, InferenceCheckConfig::default()) else {
        panic!("false premise accepted")
    };
    assert!(failure.detail.contains("premise is false"));
    inference.snapshot = &snapshot;
    inference.conclusion = InferenceConclusion::Predicate(Premise::Ne { var: VarId(0), val: -1 });
    assert!(matches!(check_inference(&inference, InferenceCheckConfig::default()), InferenceCheckOutcome::Rejected(_)));
}

#[test]
fn cp_resolution_uses_shared_integer_semantics_and_rejects_overlapping_pivots() {
    let roots = [range(0, -2, 2), range(1, 0, 1), range(2, 0, 1)];
    let values = [vec![-2, -1, 0, 1, 2], vec![0, 1], vec![0, 1]];
    let first = [literal(Premise::Ge { var: VarId(0), bound: 1 }, true), eq(1, 1)];
    let target = [eq(1, 1), eq(2, 1)];
    for upper in [0, 1] {
        let second = [literal(Premise::Le { var: VarId(0), bound: upper }, true), eq(2, 1)];
        verify_entailment(&roots, &values, &[&first, &second], &target);
        let outcome = check_clause_entailment(
            &ClauseEntailment { root_domains: &roots, sources: &[&first, &second], conclusion: &target },
            InferenceCheckConfig::default(),
        );
        assert_eq!(matches!(outcome, ClauseCheckOutcome::Checked { .. }), upper == 0);
    }
}

#[test]
fn root_holes_and_sparse_supports_are_part_of_derivation_semantics() {
    for sparse in [false, true] {
        let mut root = if sparse { domain(0, &[4, 0, 3, 1, 3]) } else { range(0, 0, 4) };
        if !sparse {
            root.holes = vec![2, 2];
        }
        let roots = [root, range(1, 0, 1), range(2, 0, 1)];
        let first = [literal(Premise::Ge { var: VarId(0), bound: 2 }, true), eq(1, 1)];
        let second = [literal(Premise::Le { var: VarId(0), bound: 2 }, true), eq(2, 1)];
        let target = [eq(1, 1), eq(2, 1)];
        verify_entailment(&roots, &[vec![0, 1, 3, 4], vec![0, 1], vec![0, 1]], &[&first, &second], &target);
        assert!(matches!(
            check_clause_entailment(
                &ClauseEntailment { root_domains: &roots, sources: &[&first, &second], conclusion: &target },
                InferenceCheckConfig::default()
            ),
            ClauseCheckOutcome::Checked { .. }
        ));
    }
}

#[test]
fn root_units_must_be_explicit_sources_when_the_baseline_does_not_entail_them() {
    let roots = [range(0, 0, 2), range(1, 0, 1)];
    let source = [literal(Premise::Eq { var: VarId(0), val: 0 }, false), eq(1, 1)];
    let target = [eq(1, 1)];
    let root_unit = [eq(0, 0)];
    let absent = ClauseEntailment { root_domains: &roots, sources: &[&source], conclusion: &target };
    assert!(matches!(check_clause_entailment(&absent, InferenceCheckConfig::default()), ClauseCheckOutcome::Rejected(_)));
    let explicit = ClauseEntailment { root_domains: &roots, sources: &[&source, &root_unit], conclusion: &target };
    assert!(matches!(check_clause_entailment(&explicit, InferenceCheckConfig::default()), ClauseCheckOutcome::Checked { .. }));
    // This result is conditional on root_unit as an axiom. Its provenance must
    // be checked separately; an arbitrary current branch unit cannot justify it.
}

#[test]
fn wide_domains_and_both_i32_endpoints_use_exact_predicate_truth_cells() {
    let roots = [range(0, i32::MIN, i32::MAX)];
    for value in [i32::MIN, -1, 0, i32::MAX] {
        let source = [eq(0, value)];
        let target = [literal(Premise::Ge { var: VarId(0), bound: value }, true)];
        assert!(matches!(
            check_clause_entailment(
                &ClauseEntailment { root_domains: &roots, sources: &[&source], conclusion: &target },
                InferenceCheckConfig { max_assignments: 8, max_work: 1000 }
            ),
            ClauseCheckOutcome::Checked { .. }
        ));
        let false_target = [literal(Premise::Le { var: VarId(0), bound: value }, false)];
        let ClauseCheckOutcome::Rejected(failure) = check_clause_entailment(
            &ClauseEntailment { root_domains: &roots, sources: &[&source], conclusion: &false_target },
            InferenceCheckConfig { max_assignments: 8, max_work: 1000 },
        ) else {
            panic!("endpoint counterexample missed")
        };
        assert_eq!(failure.counterexample, vec![(VarId(0), value)]);
    }
    let tautology = [literal(Premise::Ge { var: VarId(0), bound: i32::MIN }, true)];
    assert!(matches!(
        check_clause_entailment(
            &ClauseEntailment { root_domains: &roots, sources: &[], conclusion: &tautology },
            InferenceCheckConfig::default()
        ),
        ClauseCheckOutcome::Checked { .. }
    ));
    let false_literal = [literal(Premise::Le { var: VarId(0), bound: i32::MAX }, false)];
    assert!(matches!(
        check_clause_entailment(
            &ClauseEntailment { root_domains: &roots, sources: &[], conclusion: &false_literal },
            InferenceCheckConfig::default()
        ),
        ClauseCheckOutcome::Rejected(_)
    ));
}

#[test]
fn empty_clauses_duplicates_and_missing_domains_are_not_ambiguous() {
    let roots = [range(0, 0, 1)];
    let empty = [];
    let positive = [eq(0, 0), eq(0, 0)];
    verify_entailment(&roots, &[vec![0, 1]], &[&empty], &positive);
    verify_entailment(&roots, &[vec![0, 1]], &[], &empty);
    verify_entailment(&roots, &[vec![0, 1]], &[&positive], &[eq(0, 0)]);
    let missing = [eq(9, 0)];
    assert!(matches!(
        check_clause_entailment(
            &ClauseEntailment { root_domains: &roots, sources: &[&missing], conclusion: &[] },
            InferenceCheckConfig::default()
        ),
        ClauseCheckOutcome::Uncovered(InferenceUncoveredReason::MissingRootDomain { var: VarId(9) })
    ));
    let duplicated_roots = [roots[0].clone(), roots[0].clone()];
    assert!(matches!(
        check_clause_entailment(
            &ClauseEntailment { root_domains: &duplicated_roots, sources: &[&positive], conclusion: &[] },
            InferenceCheckConfig::default()
        ),
        ClauseCheckOutcome::Uncovered(InferenceUncoveredReason::InvalidDomain { var: VarId(0) })
    ));
}

#[test]
fn caps_and_interruption_report_uncovered_without_partial_clause_proofs() {
    let roots = [range(0, 0, 4), range(1, 0, 1)];
    let source = [eq(0, 2), eq(1, 0)];
    let inference = ClauseEntailment { root_domains: &roots, sources: &[&source], conclusion: &[] };
    for config in [
        InferenceCheckConfig { max_assignments: 0, max_work: 1000 },
        InferenceCheckConfig { max_assignments: 1, max_work: 1000 },
        InferenceCheckConfig { max_assignments: 1000, max_work: 0 },
        InferenceCheckConfig { max_assignments: 1000, max_work: 5 },
    ] {
        assert!(matches!(check_clause_entailment(&inference, config), ClauseCheckOutcome::Uncovered(reason) if reason.is_resource_limit()));
    }
    for limit in [0, 3, 12] {
        let visits = Cell::new(0);
        let stop = || {
            visits.set(visits.get() + 1);
            visits.get() > limit
        };
        assert!(matches!(
            check_clause_entailment_until(&inference, InferenceCheckConfig::default(), &stop),
            ClauseCheckOutcome::Uncovered(InferenceUncoveredReason::Interrupted)
        ));
    }
    let huge = vec![eq(0, 2); 1000];
    assert!(matches!(
        check_clause_entailment(
            &ClauseEntailment { root_domains: &roots, sources: &[&huge], conclusion: &[] },
            InferenceCheckConfig { max_assignments: 1000, max_work: 10 }
        ),
        ClauseCheckOutcome::Uncovered(InferenceUncoveredReason::WorkLimit)
    ));
    assert!(matches!(
        check_clause_entailment(&inference, InferenceCheckConfig { max_assignments: usize::MAX, max_work: usize::MAX }),
        ClauseCheckOutcome::Rejected(_)
    ));
}
