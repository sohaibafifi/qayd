use std::cell::Cell;
use std::collections::BTreeSet;
use std::sync::atomic::AtomicBool;

use crate::constraints::prefix_set::prefix_set_exclusion_interruptible;
use crate::lcg::lit::{Lit, LitOrConst};
use crate::lcg::trail::{Cdcl, Conflict, Reason};
use crate::lcg::view::Tri;
use crate::model::{CompiledCp, Constraint, IntExpr, IntVarRef, Model};
use crate::{count_solutions, Solver};

fn lit(value: LitOrConst) -> Lit {
    match value {
        LitOrConst::Lit(lit) => lit,
        other => panic!("expected an atom, got {other:?}"),
    }
}

fn equality(variable: IntVarRef, value: i64, constant_first: bool) -> IntExpr {
    if constant_first {
        IntExpr::Eq(Box::new(IntExpr::Constant(value)), Box::new(IntExpr::Variable(variable)))
    } else {
        IntExpr::Eq(Box::new(IntExpr::Variable(variable)), Box::new(IntExpr::Constant(value)))
    }
}

fn prefix_exclusion(index: IntVarRef, threshold: i64, slot: IntVarRef, forbidden: &[(i64, bool)]) -> Constraint {
    Constraint::Intension(IntExpr::Imp(
        Box::new(IntExpr::Gt(Box::new(IntExpr::Variable(index)), Box::new(IntExpr::Constant(threshold)))),
        Box::new(IntExpr::Not(Box::new(IntExpr::Or(forbidden.iter().map(|&(value, reverse)| equality(slot, value, reverse)).collect())))),
    ))
}

#[test]
fn compact_propagator_matches_a_holey_domain_oracle() {
    let index_values = [-2, 0, 3, 5];
    let slot_values = [[-1, 2, 4], [0, 2, 7], [-1, 3, 7]];
    let forbidden = [-1, 2, 7];
    let entries_by_slot = [(-1, 0usize), (0, 1), (3, 2), (3, 2)];
    let mut expected = 0u64;
    for index in index_values {
        for first in slot_values[0] {
            for second in slot_values[1] {
                for third in slot_values[2] {
                    let slots = [first, second, third];
                    if entries_by_slot.iter().all(|&(threshold, slot)| index <= threshold || !forbidden.contains(&slots[slot])) {
                        expected += 1;
                    }
                }
            }
        }
    }

    let mut solver = Solver::new();
    let index = solver.new_var_set(&index_values);
    let slots = slot_values.map(|values| solver.new_var_set(&values));
    let entries = entries_by_slot.map(|(threshold, slot)| (threshold, slots[slot]));
    assert!(prefix_set_exclusion_interruptible(&mut solver, index, &entries, &forbidden, &AtomicBool::new(false),));
    assert_eq!(count_solutions(&mut solver, &[index, slots[0], slots[1], slots[2]]), expected);
}

#[test]
fn propagation_stops_while_building_a_large_domain_reason() {
    let root_values = (0..4_096).collect::<Vec<_>>();
    let forbidden = (0..4_096).step_by(2).collect::<Vec<_>>();
    let mut solver = Solver::new();
    let index = solver.new_var_set(&[0, 1]);
    let slot = solver.new_var_set(&root_values);
    assert!(prefix_set_exclusion_interruptible(&mut solver, index, &[(0, slot)], &forbidden, &AtomicBool::new(false),));
    for value in (1..4_096).step_by(2) {
        solver.store.remove(slot, value).unwrap();
    }
    solver.store.set_explain(true);
    let polls = Cell::new(0usize);

    solver
        .propagate_until(|| {
            let next = polls.get() + 1;
            polls.set(next);
            next >= 7_000
        })
        .expect("cooperative interruption is not an inconsistency");

    assert!(polls.get() >= 7_000);
    assert_eq!(solver.store.max(index), 1, "an interrupted reason must not be used for filtering");
    assert!(solver.fd_at_fixpoint(), "the interrupted propagation queue must be cleared");
}

#[test]
fn index_bound_removals_have_a_single_bound_premise() {
    let mut solver = Solver::new();
    let index = solver.new_var_set(&[0, 2]);
    let slot = solver.new_var_set(&[1, 3, 5]);
    assert!(prefix_set_exclusion_interruptible(&mut solver, index, &[(0, slot)], &[1, 5], &AtomicBool::new(false),));
    let mut cdcl = Cdcl::new(&mut solver, &[index, slot]);
    assert!(cdcl.init());
    cdcl.decide(lit(cdcl.atoms.ge(index, 1))).unwrap();
    cdcl.propagate().unwrap();

    for removed_value in [1, 5] {
        assert!(!cdcl.solver.store.contains(slot, removed_value));
        let removed = lit(cdcl.atoms.eq(slot, removed_value)).negate();
        let Reason::Generic(reason) = cdcl.reason_of(removed.atom()) else {
            panic!("removal of {removed_value} did not retain a propagator reason");
        };
        assert!(!reason.is_empty());
        assert!(reason.iter().all(|&premise| cdcl.tvalue(premise) == Tri::True));
        assert!(reason.iter().all(|&premise| cdcl.atoms.var_of(premise.atom()) == index));
    }
}

#[test]
fn forbidden_fixed_slot_explains_the_index_upper_bound() {
    let mut solver = Solver::new();
    let index = solver.new_var_set(&[0, 2, 4]);
    let slot = solver.new_var_set(&[3, 7]);
    assert!(prefix_set_exclusion_interruptible(&mut solver, index, &[(2, slot)], &[7], &AtomicBool::new(false),));
    let mut cdcl = Cdcl::new(&mut solver, &[index, slot]);
    assert!(cdcl.init());
    cdcl.decide(lit(cdcl.atoms.eq(slot, 7))).unwrap();
    cdcl.propagate().unwrap();

    assert_eq!(cdcl.solver.store.max(index), 2);
    let removed_upper_tail = lit(cdcl.atoms.ge(index, 4)).negate();
    let Reason::Generic(reason) = cdcl.reason_of(removed_upper_tail.atom()) else {
        panic!("the inferred upper bound did not retain a propagator reason");
    };
    assert!(!reason.is_empty());
    assert!(reason.iter().all(|&premise| cdcl.tvalue(premise) == Tri::True));
    assert!(reason.iter().all(|&premise| cdcl.atoms.var_of(premise.atom()) == slot));
}

#[test]
fn a_domain_containing_only_forbidden_values_forces_the_prefix() {
    let mut solver = Solver::new();
    let index = solver.new_var_set(&[0, 2, 4]);
    let slot = solver.new_var_set(&[1, 3, 5]);
    assert!(prefix_set_exclusion_interruptible(&mut solver, index, &[(0, slot)], &[1, 5], &AtomicBool::new(false),));
    solver.enqueue_all();
    solver.propagate().unwrap();
    assert_eq!(solver.store.max(index), 4);

    solver.store.remove(slot, 3).unwrap();
    solver.propagate().unwrap();
    assert_eq!(solver.store.max(index), 0);
}

#[test]
fn incompatible_index_and_slot_facts_form_the_explicit_conflict() {
    let mut solver = Solver::new();
    let index = solver.new_var_set(&[0, 4]);
    let slot = solver.new_var_set(&[3, 7]);
    assert!(prefix_set_exclusion_interruptible(&mut solver, index, &[(2, slot)], &[7], &AtomicBool::new(false),));
    let mut cdcl = Cdcl::new(&mut solver, &[index, slot]);
    assert!(cdcl.init());
    cdcl.decide(lit(cdcl.atoms.ge(index, 4))).unwrap();
    cdcl.decide(lit(cdcl.atoms.eq(slot, 7))).unwrap();
    let Conflict::Generic(reason) = cdcl.propagate().expect_err("incompatible prefix facts were accepted") else {
        panic!("expected a propagator conflict");
    };
    assert!(reason.iter().all(|&premise| cdcl.tvalue(premise) == Tri::True));
    let variables = reason.iter().map(|premise| cdcl.atoms.var_of(premise.atom())).collect::<BTreeSet<_>>();
    assert_eq!(variables, BTreeSet::from([index, slot]));
}

#[test]
fn compiler_groups_permuted_orientations_and_duplicate_entries() {
    let mut model = Model::new();
    let index = model.int_set(vec![0, 2, 4]);
    let first = model.int_set(vec![0, 1, 3]);
    let second = model.int_set(vec![1, 2, 3]);
    let third = model.int_set(vec![0, 2, 4]);
    model.add_constraint(prefix_exclusion(index, 0, first, &[(1, false), (3, true)]));
    model.add_constraint(prefix_exclusion(index, 2, second, &[(3, false), (1, true)]));
    model.add_constraint(prefix_exclusion(index, 3, third, &[(1, true), (3, false), (1, false)]));
    model.add_constraint(prefix_exclusion(index, 2, second, &[(1, false), (3, false)]));

    let compiled = CompiledCp::compile_interruptible(&model, &AtomicBool::new(false)).unwrap().unwrap();
    assert_eq!(compiled.problem().solver.num_propagators(), 1);
    let expected = [0, 2, 4]
        .into_iter()
        .flat_map(|index_value| {
            [0, 1, 3].into_iter().flat_map(move |first_value| {
                [1, 2, 3].into_iter().flat_map(move |second_value| {
                    [0, 2, 4].into_iter().map(move |third_value| (index_value, first_value, second_value, third_value))
                })
            })
        })
        .filter(|&(index_value, first_value, second_value, third_value)| {
            let allowed = |threshold, value| index_value <= threshold || ![1, 3].contains(&value);
            allowed(0, first_value) && allowed(2, second_value) && allowed(3, third_value)
        })
        .count() as u64;
    let mut solver = compiled.problem().solver.clone();
    assert_eq!(count_solutions(&mut solver, compiled.int_variables()), expected);
}

fn generic_near_match_count(kind: usize) -> usize {
    let mut model = Model::new();
    let index = model.int_range(0, 4);
    let first = model.int_range(0, 4);
    let second = model.int_range(0, 4);
    model.add_constraint(prefix_exclusion(index, 0, first, &[(1, false), (3, false)]));
    model.add_constraint(prefix_exclusion(index, 1, second, &[(3, true), (1, true)]));
    let near_match = match kind {
        0 => Constraint::Intension(IntExpr::Imp(
            Box::new(IntExpr::Gt(Box::new(IntExpr::Variable(index)), Box::new(IntExpr::Constant(2)))),
            Box::new(IntExpr::Not(Box::new(IntExpr::Or(vec![equality(first, 1, false), equality(second, 3, false)])))),
        )),
        1 => Constraint::Intension(IntExpr::Imp(
            Box::new(IntExpr::Ge(Box::new(IntExpr::Variable(index)), Box::new(IntExpr::Constant(2)))),
            Box::new(IntExpr::Not(Box::new(IntExpr::Or(vec![equality(first, 1, false)])))),
        )),
        2 => Constraint::Intension(IntExpr::Imp(
            Box::new(IntExpr::Gt(Box::new(IntExpr::Variable(index)), Box::new(IntExpr::Constant(2)))),
            Box::new(IntExpr::Not(Box::new(IntExpr::Or(Vec::new())))),
        )),
        3 => prefix_exclusion(index, i64::from(i32::MAX), first, &[(1, false)]),
        4 => {
            let selector = model.bool_var();
            Constraint::Selected { selector, constraint: Box::new(prefix_exclusion(index, 2, first, &[(1, false), (3, false)])) }
        }
        _ => unreachable!(),
    };
    model.add_constraint(near_match);
    CompiledCp::compile_interruptible(&model, &AtomicBool::new(false)).unwrap().unwrap().problem().solver.num_propagators()
}

#[test]
fn compiler_leaves_every_quasi_match_on_the_generic_path() {
    for kind in 0..5 {
        assert_eq!(generic_near_match_count(kind), 2, "near-match kind {kind} was consumed by the compact group");
    }
}

#[test]
fn compact_groups_reduce_the_preflight_estimate() {
    fn model_with(strict: bool) -> Model {
        let mut model = Model::new();
        let index = model.int_range(0, 89);
        let slots = (0..89).map(|_| model.int_range(0, 15)).collect::<Vec<_>>();
        for (threshold, &slot) in slots.iter().enumerate() {
            let constraint = if strict {
                prefix_exclusion(
                    index,
                    threshold as i64,
                    slot,
                    &[(1, false), (3, true), (5, false), (7, true), (9, false), (11, true), (13, false), (15, true)],
                )
            } else {
                Constraint::Intension(IntExpr::Imp(
                    Box::new(IntExpr::Ge(Box::new(IntExpr::Variable(index)), Box::new(IntExpr::Constant(threshold as i64)))),
                    Box::new(IntExpr::Not(Box::new(IntExpr::Or(vec![equality(slot, 1, false)])))),
                ))
            };
            model.add_constraint(constraint);
        }
        model
    }

    let stop = AtomicBool::new(false);
    let compact = CompiledCp::estimate_semantic_bytes_interruptible(&model_with(true), &stop).unwrap();
    let generic = CompiledCp::estimate_semantic_bytes_interruptible(&model_with(false), &stop).unwrap();
    assert!(compact < generic, "compact={compact}, generic={generic}");
}

#[test]
fn posting_honors_an_already_requested_interruption() {
    let mut solver = Solver::new();
    let index = solver.new_var_range(0, 2);
    let slot = solver.new_var_range(0, 2);
    assert!(!prefix_set_exclusion_interruptible(&mut solver, index, &[(0, slot)], &[1], &AtomicBool::new(true),));
    assert_eq!(solver.num_propagators(), 0);
}
