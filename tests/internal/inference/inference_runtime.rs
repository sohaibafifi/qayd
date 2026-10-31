use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::ids::{PropId, VarId};
use crate::inference::{InferenceCheckConfig, InferenceConstraint, InferenceRelation};
use crate::propagator::{Event, Inconsistency, Propagator};
use crate::store::{Premise, Solver, Store};

#[derive(Clone, Copy)]
enum Operation {
    Remove(i32),
    Fix(i32),
    Lower(i32),
    Upper(i32),
    Conflict,
}

#[derive(Clone)]
struct Probe {
    x: VarId,
    y: Option<VarId>,
    descriptor: Option<InferenceConstraint>,
    operation: Operation,
    reason: Option<Vec<Premise>>,
}

impl Propagator for Probe {
    fn inference_constraint(&self) -> Option<InferenceConstraint> {
        self.descriptor.clone()
    }

    fn register(&mut self, store: &mut Store, me: PropId) {
        store.subscribe(self.x, me, Event::DomainChange);
        if let Some(y) = self.y {
            store.subscribe(y, me, Event::DomainChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        match (self.operation, self.reason.clone()) {
            (Operation::Remove(value), Some(reason)) => store.remove_because(self.x, value, reason),
            (Operation::Fix(value), Some(reason)) => store.fix_because(self.x, value, reason),
            (Operation::Lower(value), Some(reason)) => store.remove_below_because(self.x, value, reason),
            (Operation::Upper(value), Some(reason)) => store.remove_above_because(self.x, value, reason),
            (Operation::Conflict, Some(reason)) => Err(store.fail_because(reason)),
            (Operation::Remove(value), None) => store.remove(self.x, value),
            (Operation::Fix(value), None) => store.fix(self.x, value),
            (Operation::Lower(value), None) => store.remove_below(self.x, value),
            (Operation::Upper(value), None) => store.remove_above(self.x, value),
            (Operation::Conflict, None) => Err(Inconsistency),
        }
    }
}

fn row(terms: Vec<(VarId, i128)>, relation: InferenceRelation, rhs: i128) -> Option<InferenceConstraint> {
    Some(InferenceConstraint::Linear { terms, relation, rhs })
}

fn enable(solver: &mut Solver) {
    solver.set_inference_checking(Some(InferenceCheckConfig::default()));
}

fn panic_text(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(text) => *text,
        Err(payload) => (*payload.downcast::<&'static str>().expect("string panic")).to_owned(),
    }
}

#[test]
fn inference_runtime_rejects_each_bad_pruning_before_mutation() {
    for operation in
        [Operation::Remove(0), Operation::Fix(0), Operation::Lower(1), Operation::Upper(0), Operation::Lower(2), Operation::Upper(-1)]
    {
        let mut solver = Solver::new();
        let x = solver.new_var_range(0, 1);
        enable(&mut solver);
        solver.post(Box::new(Probe {
            x,
            y: None,
            descriptor: row(vec![(x, 0)], InferenceRelation::Eq, 0),
            operation,
            reason: Some(vec![]),
        }));
        let failure = catch_unwind(AssertUnwindSafe(|| solver.propagate())).expect_err("invalid inference must panic");
        let text = panic_text(failure);
        assert!(text.contains("runtime inference rejected"));
        assert!(text.contains("counterexample"));
        assert_eq!(solver.store.values(x).collect::<Vec<_>>(), [0, 1]);
        assert!(catch_unwind(AssertUnwindSafe(|| solver.propagate())).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| solver.set_inference_checking(None))).is_err());
    }
}

#[test]
fn inference_runtime_rejects_false_and_insufficient_tight_premises() {
    for false_premise in [false, true] {
        let mut solver = Solver::new();
        let x = solver.new_var_range(0, 1);
        let y = solver.new_var_range(0, 1);
        enable(&mut solver);
        if !false_premise {
            solver.store.push_level();
            solver.store.fix(y, 0).unwrap();
        }
        let reason = if false_premise { vec![Premise::Eq { var: y, val: 0 }] } else { vec![] };
        solver.post(Box::new(Probe {
            x,
            y: Some(y),
            descriptor: row(vec![(x, 1), (y, -1)], InferenceRelation::Eq, 0),
            operation: Operation::Fix(0),
            reason: Some(reason),
        }));
        assert!(catch_unwind(AssertUnwindSafe(|| solver.propagate())).is_err());
        assert_eq!(solver.store.size(x), 2);
    }
}

#[test]
fn inference_runtime_checks_scope_fallback_and_bare_conflicts() {
    for operation in [Operation::Fix(0), Operation::Conflict] {
        let mut solver = Solver::new();
        let x = solver.new_var_range(0, 1);
        enable(&mut solver);
        solver.post(Box::new(Probe { x, y: None, descriptor: row(vec![(x, 0)], InferenceRelation::Eq, 0), operation, reason: None }));
        assert!(catch_unwind(AssertUnwindSafe(|| solver.propagate_until(|| false))).is_err());
        assert_eq!(solver.store.size(x), 2);
    }
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    let y = solver.new_var_range(0, 1);
    enable(&mut solver);
    solver.post(Box::new(Probe {
        x,
        y: Some(y),
        descriptor: row(vec![(x, 1), (y, -1)], InferenceRelation::Eq, 0),
        operation: Operation::Fix(0),
        reason: None,
    }));
    solver.store.push_level();
    solver.store.fix(y, 0).unwrap();
    solver.propagate().unwrap();
    assert_eq!(solver.store.value(x), 0);
    assert_eq!(solver.inference_check_stats().checked_inferences, 1);
    solver.store.pop_level();
    solver.store.push_level();
    solver.store.fix(y, 1).unwrap();
    solver.enqueue_all();
    assert!(catch_unwind(AssertUnwindSafe(|| solver.propagate())).is_err(), "the frozen root must not inherit the first branch's x = 0");
    assert_eq!(solver.store.size(x), 2);
}

#[test]
fn inference_runtime_checks_failed_fix_and_wipeout_before_empty_domains() {
    for operation in [Operation::Fix(1), Operation::Lower(1), Operation::Remove(0)] {
        let mut solver = Solver::new();
        let x = solver.new_var_range(0, 1);
        enable(&mut solver);
        solver.post(Box::new(Probe {
            x,
            y: None,
            descriptor: row(vec![(x, 1)], InferenceRelation::Eq, 1),
            operation,
            reason: Some(vec![]),
        }));
        solver.store.push_level();
        solver.store.fix(x, 0).unwrap();
        assert_eq!(solver.propagate(), Err(Inconsistency));
        let stats = solver.inference_check_stats();
        assert_eq!(stats.checked_inferences, 1);
        assert_eq!(stats.checked_conflicts, 1);
        assert_eq!(stats.uncovered_conflicts, 0);
    }
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    enable(&mut solver);
    solver.post(Box::new(Probe {
        x,
        y: None,
        descriptor: row(vec![(x, 0)], InferenceRelation::Eq, 0),
        operation: Operation::Fix(1),
        reason: Some(vec![]),
    }));
    solver.store.push_level();
    solver.store.fix(x, 0).unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| solver.propagate())).is_err());
    assert_eq!(solver.store.value(x), 0);
}

#[test]
fn inference_runtime_reports_resource_caps_and_unsupported_guards() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    solver.set_inference_checking(Some(InferenceCheckConfig { max_assignments: 0, max_work: 0 }));
    solver.post(Box::new(Probe {
        x,
        y: None,
        descriptor: row(vec![(x, 1)], InferenceRelation::Eq, 0),
        operation: Operation::Fix(0),
        reason: Some(vec![]),
    }));
    solver.propagate().unwrap();
    let stats = solver.inference_check_stats();
    assert_eq!(stats.checked_inferences, 0);
    assert_eq!(stats.uncovered_inferences, 1);
    assert_eq!(stats.resource_limited, 1);

    let mut solver = Solver::new();
    let selector = solver.new_var_range(0, 1);
    let x = solver.new_var_range(0, 1);
    enable(&mut solver);
    solver.set_selector(Some(selector));
    solver.post(Box::new(Probe {
        x,
        y: None,
        descriptor: row(vec![(x, 1)], InferenceRelation::Eq, 0),
        operation: Operation::Fix(0),
        reason: Some(vec![]),
    }));
    solver.store.fix(selector, 1).unwrap();
    solver.propagate().unwrap();
    let stats = solver.inference_check_stats();
    assert_eq!(stats.checked_inferences, 0);
    assert_eq!(stats.unsupported_calls, 1);
    assert_eq!(stats.uncovered_inferences, 1);
    let families = solver.inference_check_handle().unwrap().uncovered_families();
    assert_eq!(families.len(), 1);
    assert!(families[0].0.contains("Selected"));
}

#[test]
fn inference_runtime_frozen_baseline_is_compact_and_captures_new_variables() {
    let mut solver = Solver::new();
    let wide = solver.new_var_range(i32::MIN, i32::MAX);
    solver.store.remove(wide, 0).unwrap();
    enable(&mut solver);
    assert!(solver.inference_check_stats().peak_snapshot_bytes < 1024);
    let x = solver.new_var_set(&[-100_000, 10]);
    solver.post(Box::new(Probe {
        x,
        y: None,
        descriptor: row(vec![(x, 1)], InferenceRelation::Le, -1000),
        operation: Operation::Fix(-100_000),
        reason: Some(vec![]),
    }));
    solver.propagate().unwrap();
    assert_eq!(solver.inference_check_stats().checked_inferences, 1);
    assert!(solver.inference_check_stats().peak_snapshot_bytes < 4096);
}

#[test]
fn inference_runtime_clones_share_stats_and_rejection_poison() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    enable(&mut solver);
    solver.post(Box::new(Probe {
        x,
        y: None,
        descriptor: row(vec![(x, 1)], InferenceRelation::Eq, 0),
        operation: Operation::Fix(0),
        reason: Some(vec![]),
    }));
    let handle = solver.inference_check_handle().unwrap();
    let mut clone = solver.clone();
    solver.propagate().unwrap();
    clone.propagate().unwrap();
    assert_eq!(handle.stats().checked_inferences, 2);

    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    enable(&mut solver);
    solver.post(Box::new(Probe {
        x,
        y: None,
        descriptor: row(vec![(x, 0)], InferenceRelation::Eq, 0),
        operation: Operation::Fix(0),
        reason: Some(vec![]),
    }));
    let mut clone = solver.clone();
    assert!(catch_unwind(AssertUnwindSafe(|| solver.propagate())).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| clone.propagate())).is_err());
    assert_eq!(clone.store.size(x), 2);
}

#[test]
fn inference_runtime_requires_level_zero_configuration() {
    let mut solver = Solver::new();
    solver.new_var_range(0, 1);
    solver.store.push_level();
    assert!(catch_unwind(AssertUnwindSafe(|| enable(&mut solver))).is_err());
}

#[derive(Clone)]
struct ReplacedConflict(VarId);

impl Propagator for ReplacedConflict {
    fn inference_constraint(&self) -> Option<InferenceConstraint> {
        row(vec![(self.0, 1)], InferenceRelation::Eq, 1)
    }

    fn register(&mut self, store: &mut Store, me: PropId) {
        store.subscribe(self.0, me, Event::DomainChange);
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        assert!(store.remove_below_because(self.0, 1, vec![]).is_err());
        // A valid failed deduction must not conceal a different, false reason.
        Err(store.fail_because(vec![Premise::Eq { var: self.0, val: 1 }]))
    }
}

#[test]
fn inference_runtime_rechecks_replaced_conflict_against_pre_wipeout_domain() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    enable(&mut solver);
    solver.post(Box::new(ReplacedConflict(x)));
    solver.store.push_level();
    solver.store.fix(x, 0).unwrap();
    let failure = catch_unwind(AssertUnwindSafe(|| solver.propagate())).expect_err("replacement conflict reason must be rejected");
    assert!(panic_text(failure).contains("cited premise is false"));
    assert_eq!(solver.inference_check_stats().checked_conflicts, 1);
}

#[derive(Clone)]
struct UnsupportedWideScope(Vec<VarId>);

impl Propagator for UnsupportedWideScope {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &var in &self.0 {
            store.subscribe(var, me, Event::DomainChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        assert!(!store.explaining(), "unsupported diagnostics must not request costly production reasons");
        store.fix(self.0[0], 0)
    }
}

#[test]
fn inference_runtime_zero_budget_skips_large_unsupported_snapshot_preparation() {
    let mut solver = Solver::new();
    let variables = (0..2048).map(|_| solver.new_var_range(0, 1)).collect();
    solver.set_inference_checking(Some(InferenceCheckConfig { max_assignments: 100, max_work: 0 }));
    solver.post(Box::new(UnsupportedWideScope(variables)));
    solver.propagate().unwrap();
    let stats = solver.inference_check_stats();
    assert_eq!(stats.peak_snapshot_bytes, 0);
    assert_eq!(stats.checked_inferences, 0);
    assert_eq!(stats.uncovered_inferences, 1);
    assert_eq!(stats.unsupported_calls, 1);
    assert_eq!(stats.resource_limited, 0);
}

#[derive(Clone)]
struct ExpensiveDescription(VarId);

impl Propagator for ExpensiveDescription {
    fn inference_description_work(&self) -> usize {
        usize::MAX
    }

    fn inference_constraint(&self) -> Option<InferenceConstraint> {
        panic!("descriptor construction exceeded its declared preparation budget")
    }

    fn register(&mut self, store: &mut Store, me: PropId) {
        store.subscribe(self.0, me, Event::DomainChange);
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        assert!(!store.explaining());
        store.fix(self.0, 0)
    }
}

#[test]
fn inference_runtime_caps_descriptor_preparation_before_calling_provider() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    enable(&mut solver);
    solver.post(Box::new(ExpensiveDescription(x)));
    solver.propagate().unwrap();
    let stats = solver.inference_check_stats();
    assert_eq!(stats.checked_inferences, 0);
    assert_eq!(stats.uncovered_inferences, 1);
    assert_eq!(stats.resource_limited, 1);
    assert_eq!(stats.unsupported_calls, 0);
}

#[derive(Clone)]
struct ManyInferences(VarId);

impl Propagator for ManyInferences {
    fn inference_constraint(&self) -> Option<InferenceConstraint> {
        row(vec![(self.0, 1)], InferenceRelation::Eq, 1000)
    }

    fn inference_description_work(&self) -> usize {
        1
    }

    fn register(&mut self, store: &mut Store, me: PropId) {
        store.subscribe(self.0, me, Event::DomainChange);
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        for value in 0..1000 {
            store.remove_because(self.0, value, vec![])?;
        }
        Ok(())
    }
}

#[test]
fn inference_runtime_budget_is_shared_across_one_invocation() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1000);
    solver.set_inference_checking(Some(InferenceCheckConfig { max_assignments: 100, max_work: 1000 }));
    solver.post(Box::new(ManyInferences(x)));
    solver.propagate().unwrap();
    let stats = solver.inference_check_stats();
    assert!(stats.checked_inferences > 0);
    assert!(stats.uncovered_inferences > 0);
    assert_eq!(stats.checked_inferences + stats.uncovered_inferences, 1000);
    assert!(stats.work <= 1000);
}
