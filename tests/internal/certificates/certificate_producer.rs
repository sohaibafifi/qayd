use std::cell::Cell;

use crate::certificate::model::{from_package, Domain, ProofConstraint, ProofModel, Work};
use crate::certificate::produce::{generate, GeneratedCertificate};
use crate::certificate::{CertificateClaim, CertificateError, CertificateLimits};
use crate::model::{Constraint, IntExpr, IntGlobalConstraint, IntVarRef, Model, ModelPackage, Objective, Relation};
use crate::orchestrator::{SemanticAssumption, SemanticAssumptionOp, SolveRequest};

const HASH: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn normalized(model: Model) -> ProofModel {
    from_package(&ModelPackage::new(model), &SolveRequest::default(), &CertificateLimits::default(), &|| false).unwrap()
}

fn proof(model: &ProofModel, claim: CertificateClaim, witness: Option<&[i32]>) -> GeneratedCertificate {
    generate(model, claim, witness, HASH, &CertificateLimits::default(), &|| false).unwrap()
}

fn linear(model: &mut Model, terms: Vec<(i64, IntVarRef)>, relation: Relation, rhs: i64) {
    model.add_constraint(Constraint::Linear { terms, relation, rhs });
}

fn table(model: &mut Model, variables: Vec<IntVarRef>, rows: Vec<Vec<i32>>) {
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Table { variables, tuples: rows.into(), positive: true }));
}

fn parity() -> ProofModel {
    let mut model = Model::new();
    let variables = (0..4).map(|_| model.bool_var()).collect::<Vec<_>>();
    linear(&mut model, variables.into_iter().map(|v| (2, v)).collect(), Relation::Eq, 5);
    normalized(model)
}

#[test]
fn export_compacts_sparse_roots_preserves_ids_and_combines_repeated_terms() {
    let mut model = Model::new();
    model.int_range(i32::MIN, i32::MAX);
    let x = model.int_set(vec![9, 2, 1, 2, -4, -3]);
    let y = model.bool_var();
    linear(&mut model, vec![(2, x), (-1, x), (0, y)], Relation::Gt, -9);
    let model = normalized(model);
    assert_eq!(model.domains[1].ranges, [(-4, -3), (1, 2), (9, 9)]);
    let text = model.encode(&CertificateLimits::default(), &|| false).unwrap();
    assert!(text.contains("var 0 ranges 1 -2147483648 2147483647\n"));
    assert!(text.contains("linear 0 gt -9 1 1 1\n"));
    assert!(text.ends_with("objective none\nend-model\n"));
    assert_eq!(proof(&model, CertificateClaim::Satisfiable, Some(&[i32::MIN, -4, 1])).stats.steps, 0);
}

#[test]
fn range_intersection_matches_an_independent_finite_predicate_oracle() {
    let limits = CertificateLimits::default();
    let mut work = Work::new(&limits, &|| false);
    let original = Domain { ranges: vec![(-4, -2), (0, 0), (2, 5)] };
    for relation in [Relation::Eq, Relation::Ne, Relation::Le, Relation::Lt, Relation::Ge, Relation::Gt] {
        for bound in [i32::MIN, -5, -2, -1, 0, 2, 5, 6, i32::MAX] {
            let mut domain = original.clone();
            domain.apply(relation, bound, &mut work).unwrap();
            for value in -6..=7 {
                let expected = original.contains(value)
                    && match relation {
                        Relation::Eq => value == bound,
                        Relation::Ne => value != bound,
                        Relation::Le => value <= bound,
                        Relation::Lt => value < bound,
                        Relation::Ge => value >= bound,
                        Relation::Gt => value > bound,
                    };
                assert_eq!(domain.contains(value), expected, "{relation:?} {bound} at {value}");
            }
        }
    }
    for (relation, bound) in [(Relation::Lt, i32::MIN), (Relation::Gt, i32::MAX)] {
        let mut domain = Domain { ranges: vec![(i32::MIN, i32::MAX)] };
        domain.apply(relation, bound, &mut work).unwrap();
        assert!(domain.ranges.is_empty());
    }
}

#[test]
fn positive_tables_keep_wildcards_and_repeated_position_consistency() {
    let mut model = Model::new();
    let x = model.int_range(i32::MIN, i32::MIN);
    table(&mut model, vec![x, x], vec![vec![i32::MIN, i32::MIN]]);
    let model = normalized(model);
    assert!(model.encode(&CertificateLimits::default(), &|| false).unwrap().contains("row * *\n"));
    assert!(proof(&model, CertificateClaim::Satisfiable, Some(&[i32::MIN])).text.ends_with("begin\nfinish-sat\n"));

    let mut model = Model::new();
    let x = model.bool_var();
    table(&mut model, vec![x, x], vec![vec![0, 1], vec![1, 0]]);
    let generated = proof(&normalized(model), CertificateClaim::Unsatisfiable, None);
    assert_eq!(generated.stats.nodes, 1);
    assert!(generated.text.contains("conflict 1 0 0\nlearn 2 1 0\nfinish 2\n"));
}

#[test]
fn empty_and_zero_arity_tables_have_their_exact_semantics() {
    let mut model = Model::new();
    table(&mut model, vec![], vec![]);
    assert_eq!(proof(&normalized(model), CertificateClaim::Unsatisfiable, None).stats.conflicts, 1);
    let mut model = Model::new();
    table(&mut model, vec![], vec![vec![]]);
    assert!(proof(&normalized(model), CertificateClaim::Satisfiable, Some(&[])).text.ends_with("finish-sat\n"));
}

#[test]
fn unsafe_native_original_activity_is_rejected_even_when_terms_cancel() {
    let cases = [(vec![i64::MAX, -i64::MAX], 0, 2), (vec![i64::MAX, i64::MAX], 0, 1), (vec![i64::MIN, -1], 0, 1), (vec![i64::MIN], -1, -1)];
    for (coefficients, lo, hi) in cases {
        let mut model = Model::new();
        let x = model.int_range(lo, hi);
        linear(&mut model, coefficients.into_iter().map(|c| (c, x)).collect(), Relation::Eq, 0);
        assert!(matches!(
            from_package(&ModelPackage::new(model), &SolveRequest::default(), &CertificateLimits::default(), &|| false),
            Err(CertificateError::Unsupported(_))
        ));
    }
    let mut model = Model::new();
    let x = model.int_range(0, 1);
    linear(&mut model, vec![(i64::MAX, x), (-i64::MAX, x)], Relation::Eq, 0);
    assert!(matches!(&normalized(model).constraints[0], ProofConstraint::Linear { terms, .. } if terms.is_empty()));
}

#[test]
fn assumptions_are_explicit_and_root_contradictions_have_domain_proofs() {
    let mut model = Model::new();
    model.int_range(i32::MIN, i32::MAX);
    let request = SolveRequest {
        assumptions: vec![
            SemanticAssumption { variable: 0, operation: SemanticAssumptionOp::Ge, value: 7 },
            SemanticAssumption { variable: 0, operation: SemanticAssumptionOp::Lt, value: 7 },
        ],
        ..SolveRequest::default()
    };
    let model = from_package(&ModelPackage::new(model), &request, &CertificateLimits::default(), &|| false).unwrap();
    assert!(model.encode(&CertificateLimits::default(), &|| false).unwrap().contains("assumptions 2\nassume 0 ge 7\nassume 0 lt 7\n"));
    assert!(proof(&model, CertificateClaim::Unsatisfiable, None).text.contains("conflict 1 D 0\nlearn 2 1 0\nfinish 2\n"));
    assert!(matches!(
        generate(&model, CertificateClaim::Satisfiable, Some(&[7]), HASH, &CertificateLimits::default(), &|| false),
        Err(CertificateError::ClaimDisagreement(_))
    ));
}

#[test]
fn parity_refutation_closes_both_children_and_resolves_the_root() {
    let generated = proof(&parity(), CertificateClaim::Unsatisfiable, None);
    assert!(generated.stats.splits > 0);
    assert_eq!(generated.stats.nodes, 2 * generated.stats.splits + 1);
    assert_eq!(generated.stats.conflicts, generated.stats.splits + 1);
    assert_eq!(generated.stats.conflicts, generated.stats.learned_clauses);
    assert!(generated.stats.deductions > 0);
    assert!(generated.text.contains("\nright "));
    assert!(generated.text.contains("\nresolve "));
    let mut next = 1_u64;
    for line in generated.text.lines().skip(4) {
        let words = line.split(' ').collect::<Vec<_>>();
        if words[0] == "finish" {
            assert_eq!(words[1].parse::<u64>().unwrap(), next - 1);
        } else {
            assert_eq!(words[1].parse::<u64>().unwrap(), next);
            next += 1;
        }
    }
    assert_eq!(generated.stats.steps, next - 1);
    assert_eq!(generated.stats.proof_bytes as usize, generated.text.len());
}

#[test]
fn native_claim_and_witness_are_never_trusted() {
    let mut model = Model::new();
    let x = model.int_set(vec![0, 2]);
    linear(&mut model, vec![(1, x)], Relation::Eq, 2);
    let model = normalized(model);
    for witness in [vec![], vec![0], vec![1], vec![2, 2]] {
        assert!(matches!(
            generate(&model, CertificateClaim::Satisfiable, Some(&witness), HASH, &CertificateLimits::default(), &|| false),
            Err(CertificateError::ClaimDisagreement(_))
        ));
    }
    assert!(matches!(
        generate(&model, CertificateClaim::Unsatisfiable, None, HASH, &CertificateLimits::default(), &|| false),
        Err(CertificateError::ClaimDisagreement(_))
    ));
    assert!(matches!(
        generate(&model, CertificateClaim::Unsatisfiable, Some(&[2]), HASH, &CertificateLimits::default(), &|| false),
        Err(CertificateError::ClaimDisagreement(_))
    ));
    assert!(matches!(
        generate(&model, CertificateClaim::Optimal, Some(&[2]), HASH, &CertificateLimits::default(), &|| false),
        Err(CertificateError::ClaimDisagreement(_))
    ));
}

#[test]
fn minimization_and_maximization_use_explicit_strict_cut_facts() {
    for minimize in [true, false] {
        let mut model = Model::new();
        let x = model.int_range(-2, 3);
        model.add_objective(Objective::IntExpr { minimize, expr: IntExpr::Variable(x) });
        let model = normalized(model);
        let optimum = if minimize { -2 } else { 3 };
        let generated = proof(&model, CertificateClaim::Optimal, Some(&[optimum]));
        assert_eq!(generated.stats.nodes, 1);
        assert!(generated.text.contains("conflict 1 C 0\nlearn 2 1 0\nfinish 2\n"));
        assert!(matches!(
            generate(&model, CertificateClaim::Optimal, Some(&[0]), HASH, &CertificateLimits::default(), &|| false),
            Err(CertificateError::ClaimDisagreement(_))
        ));
    }
    let mut model = Model::new();
    let x = model.int_range(0, 3);
    let y = model.int_range(0, 3);
    linear(&mut model, vec![(1, x), (1, y)], Relation::Ge, 3);
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(x) });
    let generated = proof(&normalized(model), CertificateClaim::Optimal, Some(&[0, 3]));
    assert!(generated.text.contains(" C "));
}

#[test]
fn strict_objective_constants_at_both_i64_endpoints_need_no_sentinel() {
    for value in [i64::MIN, 0, i64::MAX] {
        for minimize in [true, false] {
            let mut model = Model::new();
            model.add_objective(Objective::IntExpr { minimize, expr: IntExpr::Constant(value) });
            let model = normalized(model);
            let generated = proof(&model, CertificateClaim::Optimal, Some(&[]));
            let sense = if minimize { "min" } else { "max" };
            assert!(generated.text.contains(&format!("bound {sense} {value}\n")));
            assert!(generated.text.ends_with("conflict 1 C 0\nlearn 2 1 0\nfinish 2\n"));
        }
    }
}

#[test]
fn all_six_linear_relations_work_at_exact_integer_extremes() {
    for (relation, rhs, satisfiable) in [
        (Relation::Eq, i64::from(i32::MAX), true),
        (Relation::Ne, i64::from(i32::MAX), false),
        (Relation::Lt, i64::from(i32::MAX), false),
        (Relation::Le, i64::from(i32::MAX) + 1, true),
        (Relation::Ge, i64::from(i32::MAX) + 1, false),
        (Relation::Gt, i64::from(i32::MIN), true),
    ] {
        let mut model = Model::new();
        let x = model.int_range(i32::MAX, i32::MAX);
        linear(&mut model, vec![(1, x)], relation, rhs);
        let model = normalized(model);
        if satisfiable {
            proof(&model, CertificateClaim::Satisfiable, Some(&[i32::MAX]));
        } else {
            assert_eq!(proof(&model, CertificateClaim::Unsatisfiable, None).stats.nodes, 1);
        }
    }
}

#[test]
fn supported_table_holes_create_checked_ne_facts_before_branching() {
    let mut model = Model::new();
    let x = model.int_range(0, 2);
    table(&mut model, vec![x], vec![vec![0], vec![2]]);
    linear(&mut model, vec![(1, x)], Relation::Eq, 1);
    let generated = proof(&normalized(model), CertificateClaim::Unsatisfiable, None);
    assert!(generated.text.contains("infer 1 0 0 ne 1 0\n"));
    assert_eq!(generated.stats.splits, 0);
}

#[test]
fn wide_sparse_domains_do_not_require_assignment_enumeration() {
    let mut model = Model::new();
    let x = model.int_range(i32::MIN, i32::MAX);
    table(&mut model, vec![x], vec![vec![i32::MAX]]);
    linear(&mut model, vec![(1, x)], Relation::Lt, i64::from(i32::MAX));
    let generated = proof(&normalized(model), CertificateClaim::Unsatisfiable, None);
    assert_eq!(generated.stats.nodes, 1);
    assert!(generated.stats.work < 10_000);
    assert!(generated.text.contains(" ge 2147483647 "));
}

#[test]
fn unsupported_shapes_and_malformed_inputs_are_rejected_before_generation() {
    let limits = CertificateLimits::default();
    let request = SolveRequest::default();
    let reject = |model: Model| from_package(&ModelPackage::new(model), &request, &limits, &|| false);
    let mut model = Model::new();
    model.add_constraint(Constraint::Clause(vec![]));
    assert!(matches!(reject(model), Err(CertificateError::Unsupported(_))));
    let mut model = Model::new();
    let x = model.bool_var();
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Table {
        variables: vec![x],
        tuples: vec![vec![0]].into(),
        positive: false,
    }));
    assert!(matches!(reject(model), Err(CertificateError::Unsupported(_))));
    let mut model = Model::new();
    model.interval(0, 1, 1);
    assert!(matches!(reject(model), Err(CertificateError::Unsupported(_))));
    let mut model = Model::new();
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Add(vec![]) });
    assert!(matches!(reject(model), Err(CertificateError::Unsupported(_))));
    let mut model = Model::new();
    model.int_range(2, 1);
    assert!(matches!(reject(model), Err(CertificateError::InvalidInput(_))));
    let mut model = Model::new();
    model.int_set(vec![]);
    assert!(matches!(reject(model), Err(CertificateError::InvalidInput(_))));
    let mut model = Model::new();
    linear(&mut model, vec![(0, IntVarRef(0))], Relation::Eq, 0);
    assert!(matches!(reject(model), Err(CertificateError::InvalidInput(_))));
    let mut model = Model::new();
    let x = model.bool_var();
    table(&mut model, vec![x], vec![vec![]]);
    assert!(matches!(reject(model), Err(CertificateError::InvalidInput(_))));
    let request =
        SolveRequest { assumptions: vec![SemanticAssumption { variable: 0, operation: SemanticAssumptionOp::Eq, value: 0 }], ..request };
    assert!(matches!(from_package(&ModelPackage::new(Model::new()), &request, &limits, &|| false), Err(CertificateError::InvalidInput(_))));
}

#[test]
fn every_producer_limit_returns_an_error_instead_of_a_terminal_artifact() {
    let model = parity();
    let configurations = [
        CertificateLimits { max_nodes: 0, ..CertificateLimits::default() },
        CertificateLimits { max_steps: 0, ..CertificateLimits::default() },
        CertificateLimits { max_work: 0, ..CertificateLimits::default() },
        CertificateLimits { max_depth: 0, ..CertificateLimits::default() },
        CertificateLimits { max_proof_bytes: 10, ..CertificateLimits::default() },
    ];
    for limits in configurations {
        assert!(matches!(
            generate(&model, CertificateClaim::Unsatisfiable, None, HASH, &limits, &|| false),
            Err(CertificateError::Limit(_))
        ));
    }
    assert!(matches!(
        model.encode(&CertificateLimits { max_model_bytes: 10, ..CertificateLimits::default() }, &|| false),
        Err(CertificateError::Limit(_))
    ));
    let mut original = Model::new();
    original.int_set(vec![1, 2, 3]);
    let package = ModelPackage::new(original);
    for limits in [
        CertificateLimits { max_variables: 0, ..CertificateLimits::default() },
        CertificateLimits { max_input_entries: 2, ..CertificateLimits::default() },
        CertificateLimits { max_work: 1, ..CertificateLimits::default() },
    ] {
        assert!(matches!(from_package(&package, &SolveRequest::default(), &limits, &|| false), Err(CertificateError::Limit(_))));
    }
}

#[test]
fn cancellation_is_observed_in_export_encoding_and_deep_proof_work() {
    let mut original = Model::new();
    original.int_set((0..1000).collect());
    let calls = Cell::new(0);
    let stop = || {
        calls.set(calls.get() + 1);
        calls.get() > 20
    };
    assert!(matches!(
        from_package(&ModelPackage::new(original), &SolveRequest::default(), &CertificateLimits::default(), &stop),
        Err(CertificateError::Interrupted)
    ));
    let model = parity();
    assert!(matches!(model.encode(&CertificateLimits::default(), &|| true), Err(CertificateError::Interrupted)));
    for threshold in [0, 10, 100, 500] {
        let calls = Cell::new(0);
        let stop = || {
            calls.set(calls.get() + 1);
            calls.get() > threshold
        };
        assert!(matches!(
            generate(&model, CertificateClaim::Unsatisfiable, None, HASH, &CertificateLimits::default(), &stop),
            Err(CertificateError::Interrupted)
        ));
    }
}

#[test]
fn invalid_hash_cannot_inject_records_into_the_proof_header() {
    let model = normalized(Model::new());
    for hash in ["", "x", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"] {
        assert!(matches!(
            generate(&model, CertificateClaim::Satisfiable, Some(&[]), hash, &CertificateLimits::default(), &|| false),
            Err(CertificateError::InvalidInput(_))
        ));
    }
}

#[test]
fn unsafe_native_normalization_residual_and_division_are_rejected_in_export() {
    for (coefficient, lo, hi, relation, rhs) in [
        (1, 0, 2, Relation::Lt, i64::MIN),
        (1, 0, 2, Relation::Gt, i64::MAX),
        (1, 0, 2, Relation::Ge, i64::MIN),
        (i64::MIN, 0, 0, Relation::Ge, 0),
        (-1, 0, 2, Relation::Ne, i64::MIN),
        (-1, 0, 2, Relation::Le, i64::MAX),
        (1, 0, 2, Relation::Eq, i64::MIN + 1),
    ] {
        let mut model = Model::new();
        let x = model.int_range(lo, hi);
        linear(&mut model, vec![(coefficient, x)], relation, rhs);
        assert!(
            matches!(
                from_package(&ModelPackage::new(model), &SolveRequest::default(), &CertificateLimits::default(), &|| false),
                Err(CertificateError::Unsupported(_))
            ),
            "{coefficient} * x {relation:?} {rhs}"
        );
    }
}
