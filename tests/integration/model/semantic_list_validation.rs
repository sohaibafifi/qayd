use std::sync::Arc;

use qayd::model::list::{
    Constraint as ReductionConstraint, Expr, ExprArena, ExprId, GlobalConstraint, Iterable, MaxTerm, Op, ReduceOp, Reduction,
};
use qayd::model::{Constraint, ListVarRef, Model, Objective};

fn model_with_lists() -> Model {
    let mut model = Model::new();
    let universe = vec![1, 2, 3];
    let lists = vec![model.list(universe.clone()), model.list(universe.clone())];
    model.add_constraint(Constraint::ListPartition { lists, items: universe });
    model
}

fn constant_reduction(list: usize, value: i64) -> Reduction {
    let mut arena = ExprArena::default();
    let body = arena.constant(value);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 }
}

fn validation_errors(model: &Model) -> String {
    model.validate().expect_err("the malformed semantic model must be rejected").join("\n")
}

#[test]
fn canonical_validation_accepts_nested_list_objectives_constraints_and_globals() {
    let mut model = model_with_lists();
    let nested = constant_reduction(1, 7);
    model.add_objective(Objective::ListTerms {
        minimize: true,
        terms: vec![constant_reduction(0, 1)],
        max_terms: Some(vec![MaxTerm { groups: vec![vec![nested]], coeff: 2 }]),
    });
    model.add_constraint(Constraint::ListReduction(ReductionConstraint { reduction: constant_reduction(1, 3), op: Op::Le, rhs: 10 }));
    model.add_constraint(Constraint::CollectionGlobal(GlobalConstraint::AllSameList { items: Arc::new(vec![1, 3]) }));

    model.validate().unwrap();
}

#[test]
fn canonical_validation_descends_into_nested_max_objective_terms() {
    let mut model = model_with_lists();
    model.add_objective(Objective::ListTerms {
        minimize: true,
        terms: Vec::new(),
        max_terms: Some(vec![MaxTerm { groups: vec![vec![constant_reduction(9, 1)]], coeff: 1 }]),
    });

    let errors = validation_errors(&model);
    assert!(errors.contains("objective 0, max term 0, group 0, term 0"), "{errors}");
    assert!(errors.contains("unknown list variable 9"), "{errors}");
}

#[test]
fn canonical_validation_rejects_dangling_reduction_roots_and_arena_children() {
    let mut model = model_with_lists();
    let missing_body =
        Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(0), arena: ExprArena::default(), body: ExprId(4), coeff: 1 };
    model.add_objective(Objective::ListTerms { minimize: true, terms: vec![missing_body], max_terms: None });

    let dangling_child = Reduction {
        op: ReduceOp::Sum,
        iterable: Iterable::Items(0),
        arena: ExprArena { exprs: vec![Expr::Abs(ExprId(8))] },
        body: ExprId(0),
        coeff: 1,
    };
    model.add_constraint(Constraint::ListReduction(ReductionConstraint { reduction: dangling_child, op: Op::Eq, rhs: 0 }));

    let errors = validation_errors(&model);
    assert!(errors.contains("reduction body references expression 4 outside an arena of length 0"), "{errors}");
    assert!(errors.contains("expression 0 child 0 references expression 8 outside an arena of length 1"), "{errors}");
}

#[test]
fn canonical_validation_rejects_dangling_scan_and_window_roots() {
    let mut model = model_with_lists();
    let mut scan_arena = ExprArena::default();
    let scan_body = scan_arena.constant(0);
    model.add_objective(Objective::ListTerms {
        minimize: true,
        terms: vec![Reduction {
            op: ReduceOp::Sum,
            iterable: Iterable::Scan { list: 0, init: 0, boundary: 0, step: ExprId(12), end: None },
            arena: scan_arena,
            body: scan_body,
            coeff: 1,
        }],
        max_terms: None,
    });

    let mut window_arena = ExprArena::default();
    let window_body = window_arena.constant(0);
    model.add_constraint(Constraint::ListReduction(ReductionConstraint {
        reduction: Reduction {
            op: ReduceOp::Count,
            iterable: Iterable::Windows { list: 1, size: 2, inner: ExprId(11) },
            arena: window_arena,
            body: window_body,
            coeff: 1,
        },
        op: Op::Le,
        rhs: 1,
    }));

    let errors = validation_errors(&model);
    assert!(errors.contains("scan step references expression 12 outside an arena of length 1"), "{errors}");
    assert!(errors.contains("window inner expression references expression 11 outside an arena of length 1"), "{errors}");
}

#[test]
fn canonical_validation_rejects_unknown_global_items_and_list_references() {
    let mut model = model_with_lists();
    model.add_constraint(Constraint::CollectionGlobal(GlobalConstraint::DifferentList { a: 1, b: 99 }));
    model.add_constraint(Constraint::ListLength { list: ListVarRef(17), min: 0, max: 1 });

    let errors = validation_errors(&model);
    assert!(errors.contains("collection-global item 99 is outside list variable 0 universe"), "{errors}");
    assert!(errors.contains("unknown list variable 17"), "{errors}");
}

#[test]
fn canonical_validation_rejects_expression_cycles_and_unbound_lambda_arguments() {
    let mut model = model_with_lists();
    model.add_objective(Objective::ListTerms {
        minimize: true,
        terms: vec![Reduction {
            op: ReduceOp::Sum,
            iterable: Iterable::Items(0),
            arena: ExprArena { exprs: vec![Expr::Abs(ExprId(0)), Expr::Arg(1)] },
            body: ExprId(1),
            coeff: 1,
        }],
        max_terms: None,
    });

    let errors = validation_errors(&model);
    assert!(errors.contains("expression arena contains a cycle through expression 0"), "{errors}");
    assert!(errors.contains("uses lambda argument 1, but its iterable binds only 1"), "{errors}");
}
