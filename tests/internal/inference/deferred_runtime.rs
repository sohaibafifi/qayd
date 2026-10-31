use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::explanation::DeferredReasonSource;
use crate::ids::{PropId, VarId};
use crate::inference::{InferenceCheckConfig, InferenceConstraint, InferenceRelation};
use crate::lcg::clause::ClauseRef;
use crate::lcg::lit::{Lit, LitOrConst};
use crate::lcg::trail::{Cdcl, Reason};
use crate::lcg::view::Tri;
use crate::propagator::{Event, Inconsistency, Propagator};
use crate::store::{Premise, Solver, Store};

struct FrozenReason {
    variable: VarId,
    value: i32,
    calls: Arc<AtomicUsize>,
}

impl DeferredReasonSource for FrozenReason {
    fn premises(&self) -> Vec<Premise> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        vec![Premise::Eq { var: self.variable, val: self.value }]
    }

    fn max_premises(&self) -> usize {
        1
    }

    fn estimated_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

#[derive(Clone)]
struct HistoricalEquality {
    x: VarId,
    y: VarId,
    calls: Arc<AtomicUsize>,
    wrong_history: bool,
    extra_scope: Vec<VarId>,
    chain: Option<VarId>,
}

impl Propagator for HistoricalEquality {
    fn inference_constraint(&self) -> Option<InferenceConstraint> {
        Some(InferenceConstraint::Linear { terms: vec![(self.x, 1), (self.y, -1)], relation: InferenceRelation::Eq, rhs: 0 })
    }

    fn inference_description_work(&self) -> usize {
        2
    }

    fn register(&mut self, store: &mut Store, me: PropId) {
        for var in [self.x, self.y].into_iter().chain(self.chain).chain(self.extra_scope.iter().copied()) {
            store.subscribe(var, me, Event::DomainChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        if !store.is_fixed(self.y) {
            return Ok(());
        }
        let value = store.value(self.y);
        let history = if self.wrong_history { 1 - value } else { value };
        if store.deferred_explanations_enabled() {
            store.fix_deferred(
                self.x,
                value,
                Arc::new(FrozenReason { variable: self.y, value: history, calls: Arc::clone(&self.calls) }),
            )?;
            if let Some(z) = self.chain {
                store.fix_deferred(z, value, Arc::new(FrozenReason { variable: self.x, value, calls: Arc::clone(&self.calls) }))?;
            }
        } else {
            store.fix_because(self.x, value, vec![Premise::Eq { var: self.y, val: history }])?;
        }
        Ok(())
    }
}

fn literal(value: LitOrConst) -> Lit {
    let LitOrConst::Lit(lit) = value else { panic!("expected a nonconstant atom") };
    lit
}

fn model() -> (Solver, VarId, VarId, VarId, Arc<AtomicUsize>) {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    let y = solver.new_var_range(0, 1);
    let z = solver.new_var_range(0, 1);
    solver.set_deferred_explanations(true);
    solver.set_explanation_metrics(true);
    let calls = Arc::new(AtomicUsize::new(0));
    solver.post(Box::new(HistoricalEquality { x, y, calls: Arc::clone(&calls), wrong_history: false, extra_scope: vec![], chain: None }));
    (solver, x, y, z, calls)
}

#[test]
fn deferred_runtime_drain_does_not_expand_and_demand_is_cached() {
    let (mut solver, x, y, z, calls) = model();
    let handle = solver.explanation_handle().unwrap();
    {
        let mut cdcl = Cdcl::new(&mut solver, &[x, y, z]);
        assert!(cdcl.init());
        let y_one = literal(cdcl.atoms.ge(y, 1));
        let x_one = literal(cdcl.atoms.eq(x, 1));
        let z_one = literal(cdcl.atoms.eq(z, 1));
        cdcl.decide(y_one).unwrap();
        cdcl.propagate().unwrap();
        assert!(matches!(cdcl.reason_of(x_one.atom()), Reason::Deferred(_)));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(handle.stats().created_handles, 1);
        assert_eq!(handle.stats().lcg_first_expansions, 0);
        cdcl.decide(z_one).unwrap();
        assert_eq!(cdcl.reason_lits(x_one), [y_one.negate()]);
        assert_eq!(cdcl.reason_lits(x_one), [y_one.negate()]);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(handle.stats().lcg_requests, 2);
        assert_eq!(handle.stats().lcg_cache_hits, 1);
        assert!(handle.stats().retained_raw_bytes > 0);
        assert!(handle.stats().retained_cache_bytes > 0);
    }
    assert_eq!(handle.stats().retained_raw_bytes, 0);
    assert_eq!(handle.stats().retained_fallback_bytes, 0);
    assert_eq!(handle.stats().retained_cache_bytes, 0);
}

#[test]
fn deferred_runtime_survives_backjump_and_rederived_reason_gets_new_cache() {
    let (mut solver, x, y, z, calls) = model();
    let handle = solver.explanation_handle().unwrap();
    let mut cdcl = Cdcl::new(&mut solver, &[x, y, z]);
    assert!(cdcl.init());
    let y_one = literal(cdcl.atoms.ge(y, 1));
    let x_one = literal(cdcl.atoms.eq(x, 1));
    let z_one = literal(cdcl.atoms.eq(z, 1));
    cdcl.decide(y_one).unwrap();
    cdcl.propagate().unwrap();
    assert_eq!(cdcl.reason_lits(x_one), [y_one.negate()]);
    cdcl.decide(z_one).unwrap();
    cdcl.backjump_to(1);
    assert_eq!(cdcl.reason_lits(x_one), [y_one.negate()]);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    cdcl.backjump_to(0);
    assert!(matches!(cdcl.reason_of(x_one.atom()), Reason::Unset));
    assert_eq!(handle.stats().retained_raw_bytes, 0);
    cdcl.decide(y_one.negate()).unwrap();
    cdcl.propagate().unwrap();
    let x_zero = literal(cdcl.atoms.eq(x, 0));
    assert!(matches!(cdcl.reason_of(x_zero.atom()), Reason::Deferred(_)));
    assert_eq!(cdcl.reason_lits(x_zero), [y_one]);
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_eq!(handle.stats().created_handles, 2);
}

#[test]
fn deferred_runtime_assumption_core_demands_the_complete_reason() {
    let (mut solver, x, y, z, calls) = model();
    let mut cdcl = Cdcl::new(&mut solver, &[x, y, z]);
    assert!(cdcl.init());
    let y_one = literal(cdcl.atoms.ge(y, 1));
    let x_zero = literal(cdcl.atoms.eq(x, 0));
    cdcl.decide(y_one).unwrap();
    cdcl.propagate().unwrap();
    let core = cdcl.analyze_final(x_zero);
    assert_eq!(core.len(), 2);
    assert!(core.contains(&x_zero));
    assert!(core.contains(&y_one));
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn deferred_runtime_worker_clones_do_not_share_translated_caches() {
    let (solver, x, y, z, calls) = model();
    let mut first = solver.clone();
    let mut second = solver;
    let handle = first.explanation_handle().unwrap();
    for (index, worker) in [&mut first, &mut second].into_iter().enumerate() {
        let mut cdcl = Cdcl::new(worker, &[x, y, z]);
        assert!(cdcl.init());
        let value = index as i32;
        let y_one = literal(cdcl.atoms.ge(y, 1));
        let decision = if value == 1 { y_one } else { y_one.negate() };
        let implied = literal(cdcl.atoms.eq(x, value));
        cdcl.decide(decision).unwrap();
        cdcl.propagate().unwrap();
        assert_eq!(handle.stats().lcg_first_expansions, index as u64);
        assert_eq!(cdcl.reason_lits(implied), [decision.negate()]);
    }
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_eq!(handle.stats().lcg_first_expansions, 2);
}

#[test]
fn deferred_runtime_rejects_bad_history_before_domain_mutation_under_i2() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    let y = solver.new_var_range(0, 1);
    let calls = Arc::new(AtomicUsize::new(0));
    solver.set_deferred_explanations(true);
    solver.set_explanation_metrics(true);
    solver.set_inference_checking(Some(InferenceCheckConfig::default()));
    solver.post(Box::new(HistoricalEquality { x, y, calls: Arc::clone(&calls), wrong_history: true, extra_scope: vec![], chain: None }));
    let handle = solver.explanation_handle().unwrap();
    {
        let mut cdcl = Cdcl::new(&mut solver, &[x, y]);
        assert!(cdcl.init());
        let decision = literal(cdcl.atoms.ge(y, 1));
        cdcl.decide(decision).unwrap();
        assert!(catch_unwind(AssertUnwindSafe(|| cdcl.propagate())).is_err());
        assert_eq!(cdcl.solver.store.size(x), 2);
    }
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(handle.stats().checker_expansions, 1);
    assert_eq!(handle.stats().lcg_first_expansions, 0);
    assert!(catch_unwind(AssertUnwindSafe(|| solver.propagate())).is_err());
}

#[test]
fn deferred_runtime_scope_finesse_and_selected_paths_remain_eager() {
    for mode in 0..3 {
        let mut solver = Solver::new();
        let x = solver.new_var_range(0, 1);
        let y = solver.new_var_range(0, 1);
        let selector = solver.new_var_range(0, 1);
        solver.set_deferred_explanations(true);
        solver.set_explanation_metrics(true);
        if mode == 0 {
            solver.set_force_scope_reasons(true);
        }
        if mode == 1 {
            solver.store.enable_finesse();
        }
        if mode == 2 {
            solver.set_selector(Some(selector));
            solver.store.fix(selector, 1).unwrap();
        }
        let calls = Arc::new(AtomicUsize::new(0));
        solver.post(Box::new(HistoricalEquality {
            x,
            y,
            calls: Arc::clone(&calls),
            wrong_history: false,
            extra_scope: vec![],
            chain: None,
        }));
        let handle = solver.explanation_handle().unwrap();
        let mut cdcl = Cdcl::new(&mut solver, &[x, y, selector]);
        assert!(cdcl.init());
        let decision = literal(cdcl.atoms.ge(y, 1));
        let implied = literal(cdcl.atoms.eq(x, 1));
        cdcl.decide(decision).unwrap();
        cdcl.propagate().unwrap();
        assert!(matches!(cdcl.reason_of(implied.atom()), Reason::Generic(_)));
        assert_eq!(handle.stats().created_handles, 0);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert!(handle.stats().eager_premise_buffers > 0);
    }
}

#[test]
fn deferred_runtime_intermediate_antecedent_uses_full_entry_fallback() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    let y = solver.new_var_range(0, 1);
    let z = solver.new_var_range(0, 1);
    solver.set_deferred_explanations(true);
    solver.set_explanation_metrics(true);
    let calls = Arc::new(AtomicUsize::new(0));
    solver.post(Box::new(HistoricalEquality {
        x,
        y,
        calls: Arc::clone(&calls),
        wrong_history: false,
        extra_scope: vec![],
        chain: Some(z),
    }));
    let handle = solver.explanation_handle().unwrap();
    let mut cdcl = Cdcl::new(&mut solver, &[x, y, z]);
    assert!(cdcl.init());
    let decision = literal(cdcl.atoms.ge(y, 1));
    let implied = literal(cdcl.atoms.eq(z, 1));
    cdcl.decide(decision).unwrap();
    cdcl.propagate().unwrap();
    assert_eq!(handle.stats().created_handles, 2);
    assert_eq!(handle.stats().retained_fallback_bytes, std::mem::size_of::<Lit>() as u64, "one shared invocation fallback");
    // x is true now, but its event was not recorded at invocation entry.
    assert_eq!(cdcl.reason_lits(implied), [decision.negate()]);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn deferred_runtime_oversized_fallback_materializes_eagerly_and_noop_creates_no_handle() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    let y = solver.new_var_range(0, 1);
    let extra_scope: Vec<_> = (0..260).map(|_| solver.new_var_range(0, 1)).collect();
    solver.set_deferred_explanations(true);
    solver.set_explanation_metrics(true);
    let calls = Arc::new(AtomicUsize::new(0));
    solver.post(Box::new(HistoricalEquality {
        x,
        y,
        calls: Arc::clone(&calls),
        wrong_history: false,
        extra_scope: extra_scope.clone(),
        chain: None,
    }));
    let handle = solver.explanation_handle().unwrap();
    let mut variables = vec![x, y];
    variables.extend(extra_scope);
    let mut cdcl = Cdcl::new(&mut solver, &variables);
    assert!(cdcl.init());
    let decision = literal(cdcl.atoms.ge(y, 1));
    let implied = literal(cdcl.atoms.eq(x, 1));
    cdcl.decide(decision).unwrap();
    cdcl.propagate().unwrap();
    assert!(matches!(cdcl.reason_of(implied.atom()), Reason::Generic(_)));
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    let created = handle.stats().created_handles;
    cdcl.solver.enqueue_all();
    cdcl.propagate().unwrap();
    assert_eq!(handle.stats().created_handles, created);
}

#[test]
fn deferred_runtime_stop_does_not_make_a_partial_reason_and_probe_stays_sound() {
    let (mut solver, x, y, z, _) = model();
    let stop = AtomicBool::new(false);
    let mut cdcl = Cdcl::new(&mut solver, &[x, y, z]);
    cdcl.set_stop(&stop);
    assert!(cdcl.init());
    assert!(cdcl.root_probe(&[y]));
    let decision = literal(cdcl.atoms.ge(y, 1));
    let implied = literal(cdcl.atoms.eq(x, 1));
    cdcl.decide(decision).unwrap();
    cdcl.propagate().unwrap();
    assert_eq!(cdcl.value(implied), Tri::True);
    stop.store(true, Ordering::Relaxed);
    assert!(!cdcl.reason_lits(implied).is_empty());
}

#[test]
fn deferred_runtime_reason_enum_preserves_eager_side_array_layout() {
    #[allow(dead_code)]
    enum EagerReasonLayout {
        Unset,
        Fact,
        Decision,
        Clause(ClauseRef),
        Generic(Arc<[Lit]>),
    }
    assert_eq!(std::mem::size_of::<Reason>(), std::mem::size_of::<EagerReasonLayout>());
}

#[derive(Clone)]
struct RejectOne(VarId);

impl Propagator for RejectOne {
    fn register(&mut self, store: &mut Store, me: PropId) {
        store.subscribe(self.0, me, Event::Fix);
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        if store.is_fixed(self.0) && store.value(self.0) == 1 {
            Err(store.fail_because(vec![Premise::Eq { var: self.0, val: 1 }]))
        } else {
            Ok(())
        }
    }
}

#[test]
fn deferred_runtime_mus_footprint_expands_unguarded_background_reason() {
    let (mut solver, x, y, selector, calls) = model();
    solver.set_selector(Some(selector));
    solver.post(Box::new(RejectOne(x)));
    let handle = solver.explanation_handle().unwrap();
    let mut cdcl = Cdcl::new(&mut solver, &[x, y, selector]);
    assert!(cdcl.init());
    let assumption = literal(cdcl.atoms.ge(y, 1));
    let selected = literal(cdcl.atoms.ge(selector, 1));
    let footprint = cdcl.explain_by_propagation(&[assumption, selected]).expect("guarded rejection must refute x = y = 1");
    assert!(footprint.iter().any(|(variable, _)| *variable == selector));
    assert!(footprint.iter().any(|(variable, _)| *variable == y));
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(handle.stats().created_handles, 1);
    assert_eq!(handle.stats().lcg_first_expansions, 1);
}

#[derive(Clone)]
struct FalseRootHistory {
    x: VarId,
    y: VarId,
    root_zero: VarId,
    calls: Arc<AtomicUsize>,
}

impl Propagator for FalseRootHistory {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for variable in [self.x, self.y, self.root_zero] {
            store.subscribe(variable, me, Event::DomainChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        if store.is_fixed(self.y) {
            store.fix_deferred(
                self.x,
                store.value(self.y),
                Arc::new(FrozenReason { variable: self.root_zero, value: 1, calls: Arc::clone(&self.calls) }),
            )?;
        }
        Ok(())
    }
}

#[test]
fn deferred_runtime_false_root_premise_uses_nonempty_fallback() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    let y = solver.new_var_range(0, 1);
    let root_zero = solver.new_var_range(0, 0);
    solver.set_deferred_explanations(true);
    solver.set_explanation_metrics(true);
    let calls = Arc::new(AtomicUsize::new(0));
    // The intended relation is x = y. Its pruning is valid under the decision,
    // but this test deliberately supplies a historical premise false at root.
    solver.post(Box::new(FalseRootHistory { x, y, root_zero, calls: Arc::clone(&calls) }));
    let mut cdcl = Cdcl::new(&mut solver, &[x, y, root_zero]);
    assert!(cdcl.init());
    let decision = literal(cdcl.atoms.ge(y, 1));
    let inferred = literal(cdcl.atoms.eq(x, 1));
    cdcl.decide(decision).unwrap();
    cdcl.propagate().unwrap();
    assert!(matches!(cdcl.reason_of(inferred.atom()), Reason::Deferred(_)));
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(cdcl.reason_lits(inferred), [decision.negate()]);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[derive(Clone, Copy)]
enum BatchReason {
    Scope,
    Tight,
    Intermediate,
}

#[derive(Clone)]
struct BatchEquality {
    trigger: VarId,
    outputs: Vec<VarId>,
    context: Vec<VarId>,
    reason: BatchReason,
}

impl Propagator for BatchEquality {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for variable in std::iter::once(self.trigger).chain(self.outputs.iter().copied()).chain(self.context.iter().copied()) {
            store.subscribe(variable, me, Event::DomainChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        if !store.is_fixed(self.trigger) {
            return Ok(());
        }
        let value = store.value(self.trigger);
        let mut previous = self.trigger;
        for &output in &self.outputs {
            match self.reason {
                BatchReason::Scope => store.fix(output, value)?,
                BatchReason::Tight => store.fix_because(output, value, vec![Premise::Eq { var: self.trigger, val: value }])?,
                BatchReason::Intermediate => store.fix_because(output, value, vec![Premise::Eq { var: previous, val: value }])?,
            }
            previous = output;
        }
        Ok(())
    }
}

fn eager_antecedents(cdcl: &Cdcl<'_>, variable: VarId, value: i32) -> Arc<[Lit]> {
    let implied = literal(cdcl.atoms.eq(variable, value));
    let Reason::Generic(antecedents) = cdcl.reason_of(implied.atom()) else { panic!("expected eager inference") };
    Arc::clone(antecedents)
}

#[test]
fn eager_scope_fallbacks_share_one_immutable_payload_per_invocation_and_not_across_backjumps() {
    let mut solver = Solver::new();
    let trigger = solver.new_var_range(0, 1);
    let outputs: Vec<_> = (0..64).map(|_| solver.new_var_range(0, 1)).collect();
    let context: Vec<_> = (0..32).map(|_| solver.new_var_range(0, 1)).collect();
    solver.post(Box::new(BatchEquality { trigger, outputs: outputs.clone(), context: context.clone(), reason: BatchReason::Scope }));
    let variables: Vec<_> = std::iter::once(trigger).chain(outputs.iter().copied()).chain(context.iter().copied()).collect();
    let mut cdcl = Cdcl::new(&mut solver, &variables);
    assert!(cdcl.init());
    let mut expected = Vec::new();
    for &variable in &context {
        let decision = literal(cdcl.atoms.ge(variable, 1));
        cdcl.decide(decision).unwrap();
        cdcl.propagate().unwrap();
        expected.push(decision);
    }
    let trigger_one = literal(cdcl.atoms.ge(trigger, 1));
    cdcl.decide(trigger_one).unwrap();
    cdcl.propagate().unwrap();
    let first = eager_antecedents(&cdcl, outputs[0], 1);
    let mut complete_scope = vec![trigger_one];
    complete_scope.extend(expected.iter().copied());
    assert_eq!(first.as_ref(), complete_scope, "sharing must preserve every entry-scope premise and its order");
    for &output in &outputs {
        assert!(Arc::ptr_eq(&first, &eager_antecedents(&cdcl, output, 1)), "one retained scope payload for all 64 events");
    }
    let old_history = first.to_vec();
    cdcl.backjump_to(0);
    for &variable in &context {
        cdcl.decide(literal(cdcl.atoms.ge(variable, 1)).negate()).unwrap();
        cdcl.propagate().unwrap();
    }
    cdcl.decide(trigger_one.negate()).unwrap();
    cdcl.propagate().unwrap();
    let next = eager_antecedents(&cdcl, outputs[0], 0);
    assert!(!Arc::ptr_eq(&first, &next), "a new invocation must not inherit the previous branch's snapshot");
    assert_eq!(next.as_ref(), complete_scope.iter().map(|literal| literal.negate()).collect::<Vec<_>>());
    assert_eq!(first.as_ref(), old_history, "old immutable reasons remain valid while retained by a caller");
    for &output in &outputs {
        assert!(Arc::ptr_eq(&next, &eager_antecedents(&cdcl, output, 0)));
    }
}

#[test]
fn eager_scope_cache_preserves_tight_reasons_and_demotes_intermediate_facts_at_the_entry_cutoff() {
    for reason in [BatchReason::Tight, BatchReason::Intermediate] {
        let mut solver = Solver::new();
        let trigger = solver.new_var_range(0, 1);
        let context = solver.new_var_range(0, 1);
        let outputs: Vec<_> = (0..3).map(|_| solver.new_var_range(0, 1)).collect();
        solver.post(Box::new(BatchEquality { trigger, outputs: outputs.clone(), context: vec![context], reason }));
        let variables: Vec<_> = [trigger, context].into_iter().chain(outputs.iter().copied()).collect();
        let mut cdcl = Cdcl::new(&mut solver, &variables);
        assert!(cdcl.init());
        let context_one = literal(cdcl.atoms.ge(context, 1));
        cdcl.decide(context_one).unwrap();
        cdcl.propagate().unwrap();
        let trigger_one = literal(cdcl.atoms.ge(trigger, 1));
        cdcl.decide(trigger_one).unwrap();
        cdcl.propagate().unwrap();
        assert_eq!(eager_antecedents(&cdcl, outputs[0], 1).as_ref(), [trigger_one]);
        let second = eager_antecedents(&cdcl, outputs[1], 1);
        let third = eager_antecedents(&cdcl, outputs[2], 1);
        match reason {
            BatchReason::Tight => {
                assert_eq!(second.as_ref(), [trigger_one]);
                assert_eq!(third.as_ref(), [trigger_one]);
            }
            BatchReason::Intermediate => {
                assert_eq!(second.as_ref(), [trigger_one, context_one]);
                assert_eq!(third.as_ref(), [trigger_one, context_one]);
                assert!(Arc::ptr_eq(&second, &third));
            }
            BatchReason::Scope => unreachable!(),
        }
    }
}

#[test]
fn eager_scope_cache_preserves_all_solutions_across_conflicts_and_backjumps() {
    use std::collections::BTreeSet;
    for reason in [BatchReason::Scope, BatchReason::Tight, BatchReason::Intermediate] {
        let mut solver = Solver::new();
        let trigger = solver.new_var_range(0, 1);
        let outputs: Vec<_> = (0..3).map(|_| solver.new_var_range(0, 1)).collect();
        let context = solver.new_var_range(0, 1);
        solver.post(Box::new(BatchEquality { trigger, outputs: outputs.clone(), context: vec![context], reason }));
        let variables: Vec<_> = std::iter::once(trigger).chain(outputs.iter().copied()).chain(std::iter::once(context)).collect();
        let mut actual = BTreeSet::new();
        crate::search::solve(&mut solver, &variables, |state| {
            actual.insert(variables.iter().map(|&variable| state.store.value(variable)).collect::<Vec<_>>());
            crate::search::SearchControl::Continue
        });
        let expected = (0..=1).flat_map(|trigger| (0..=1).map(move |context| vec![trigger, trigger, trigger, trigger, context])).collect();
        assert_eq!(actual, expected, "independent equality predicate across both branches and both context values");
    }
}
