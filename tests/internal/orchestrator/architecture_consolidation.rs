use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;

use crate::engines::ls::schedule_ir::PrecedenceDag;
use crate::model::{AffineForm, Constraint, IntExpr, IntVarRef, Relation};

#[test]
fn semantic_integer_scope_is_recursive_sorted_and_unique() {
    let constraint = Constraint::Selected {
        selector: IntVarRef(2),
        constraint: Box::new(Constraint::Linear {
            terms: vec![(1, IntVarRef(3)), (-1, IntVarRef(1)), (2, IntVarRef(3))],
            relation: Relation::Eq,
            rhs: 0,
        }),
    };
    assert_eq!(constraint.int_scope(), vec![IntVarRef(1), IntVarRef(2), IntVarRef(3)]);
}

#[test]
fn affine_analysis_normalizes_expression_and_linear_forms_identically() {
    let stop = AtomicBool::new(false);
    let expression = IntExpr::Sub(
        Box::new(IntExpr::Add(vec![IntExpr::Mul(vec![IntExpr::Constant(2), IntExpr::Variable(IntVarRef(0))]), IntExpr::Constant(7)])),
        Box::new(IntExpr::Variable(IntVarRef(1))),
    );
    let expression = AffineForm::expression(&expression, &stop).expect("affine expression");
    let linear = AffineForm::linear(&[(2, IntVarRef(0)), (-1, IntVarRef(1))], -7).expect("affine linear form");

    assert_eq!(expression.constant, 7);
    assert_eq!(expression.coefficients, BTreeMap::from([(IntVarRef(0), 2), (IntVarRef(1), -1)]));
    assert_eq!(expression, linear);
}

#[test]
fn shared_precedence_dag_validates_and_evaluates_paths() {
    let stop = AtomicBool::new(false);
    let graph = PrecedenceDag::compile(vec![vec![2], vec![2], vec![3], vec![]], &stop).expect("valid DAG");

    assert_eq!(graph.topological(), [0, 1, 2, 3]);
    assert_eq!(graph.predecessors(2), [0, 1]);
    assert_eq!(graph.terminals().collect::<Vec<_>>(), [3]);
    assert_eq!(graph.remaining_paths(&[2, 5, 3, 7], &stop), Some(vec![12, 15, 10, 7]));

    assert!(PrecedenceDag::compile(vec![vec![1], vec![0]], &stop).is_none());
    assert!(PrecedenceDag::compile(vec![vec![1, 1], vec![]], &stop).is_none());
    assert!(PrecedenceDag::compile(vec![vec![2], vec![]], &stop).is_none());
}
