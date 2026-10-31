use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::ids::VarId;
use crate::inference::{InferenceCheckConfig, InferenceConclusion, InferenceLiteral};
use crate::lcg::clause::{ClauseSharing, SharedClausePool};
use crate::lcg::lit::{AtomKind, AtomTable, Lit, LitOrConst};
use crate::lcg::nogood::{self, NogoodStats, Plan, MAX_NOGOOD_CLAUSE_LEN, MAX_NOGOOD_DOMAIN_VALUES};
use crate::lcg::trail::{Cdcl, Conflict, Reason};
use crate::lcg::view::Tri;
use crate::store::Premise;
use crate::Solver;

fn literal(value: LitOrConst) -> Lit {
    match value {
        LitOrConst::Lit(literal) => literal,
        other => panic!("expected a nonconstant literal, got {other:?}"),
    }
}

fn holds(atoms: &AtomTable, literal: Lit, values: &[i32]) -> bool {
    let atom = match atoms.decode(literal.atom()) {
        AtomKind::Ge { var, k } => values[var.index()] >= k,
        AtomKind::Eq { var, v } => values[var.index()] == v,
    };
    atom == literal.is_positive()
}

fn domain(engine: &Cdcl<'_>, var: VarId) -> Vec<i32> {
    let mut values = engine.solver.store.values(var).collect::<Vec<_>>();
    values.sort_unstable();
    values
}

#[test]
fn bounded_semantic_plans_match_exhaustive_sparse_integer_truth_tables() {
    let mut solver = Solver::new();
    let x = solver.new_var_set(&[-2, 0, 3]);
    let y = solver.new_var_range(0, 1);
    let atoms = AtomTable::build(2, |var| if var == x { (-2, 3) } else { (0, 1) });
    let mut pool = [-2, 0, 3].into_iter().map(|value| literal(atoms.eq(x, value))).collect::<Vec<_>>();
    pool.extend([-1, 1, 3].into_iter().map(|bound| literal(atoms.ge(x, bound))));
    pool.push(literal(atoms.ge(y, 1)));
    pool.extend(pool.clone().into_iter().map(Lit::negate));
    let stop = AtomicBool::new(false);
    for mask in 1..8 {
        solver.store.push_level();
        solver.store.fix(y, 1).unwrap();
        let support = [-2, 0, 3]
            .into_iter()
            .enumerate()
            .filter_map(|(index, value)| {
                if mask & (1 << index) == 0 {
                    solver.store.remove(x, value).unwrap();
                    None
                } else {
                    Some(value)
                }
            })
            .collect::<Vec<_>>();
        for &first in &pool {
            for &second in &pool {
                for &third in &pool {
                    let clause = [first, second, third];
                    let rejected = support
                        .iter()
                        .copied()
                        .filter(|&value| !clause.iter().any(|&literal| holds(&atoms, literal, &[value, 1])))
                        .collect::<Vec<_>>();
                    let mut stats = NogoodStats::default();
                    match nogood::plan(&solver.store, &atoms, &clause, &stop, &mut stats) {
                        Plan::Inactive => assert!(rejected.is_empty()),
                        Plan::Conflict => assert_eq!(rejected, support),
                        Plan::Remove { var, values, antecedents } => {
                            assert_eq!(var, x);
                            assert_eq!(values, rejected);
                            assert!(antecedents.iter().all(|&lit| atoms.var_of(lit.atom()) == y && holds(&atoms, lit, &[0, 1])));
                            // Every emitted deletion follows from the clause
                            // and the outside-variable antecedents alone.
                            for value in values {
                                for guard in [0, 1] {
                                    if antecedents.iter().all(|&lit| holds(&atoms, lit, &[value, guard])) {
                                        assert!(!clause.iter().any(|&lit| holds(&atoms, lit, &[value, guard])));
                                    }
                                }
                            }
                        }
                        Plan::Interrupted => panic!("no stop was requested"),
                    }
                }
            }
        }
        solver.store.pop_level();
    }
}

#[test]
fn extended_nogoods_remove_a_strict_interior_block_and_retrigger_after_backjump() {
    for enabled in [false, true] {
        let mut solver = Solver::new();
        solver.set_extended_nogoods(enabled);
        let x = solver.new_var_range(0, 10);
        let y = solver.new_var_range(0, 1);
        solver.set_inference_checking(Some(InferenceCheckConfig::default()));
        let mut engine = Cdcl::new(&mut solver, &[x, y]);
        assert!(engine.init());
        let guard = literal(engine.atoms.ge(y, 1));
        let lower = literal(engine.atoms.ge(x, 3));
        let upper = literal(engine.atoms.ge(x, 8)).negate();
        engine.add_clause(vec![guard.negate(), lower.negate(), upper.negate()], false, 0);
        for _ in 0..2 {
            engine.decide(guard).unwrap();
            assert!(engine.propagate_and_learn());
            assert_eq!(domain(&engine, x), if enabled { vec![0, 1, 2, 8, 9, 10] } else { (0..=10).collect() });
            if enabled {
                for value in 3..=7 {
                    let removed = literal(engine.atoms.eq(x, value)).negate();
                    assert_eq!(engine.reason_lits(removed), [guard.negate()]);
                    assert_eq!(engine.level_of(removed.atom()), 1);
                }
            }
            engine.backjump_to(0);
            assert_eq!(domain(&engine, x), (0..=10).collect::<Vec<_>>());
        }
        if enabled {
            assert_eq!(engine.nogood_stats().removed_values, 10);
            assert_eq!(engine.solver.inference_check_stats().lcg_checked_inferences, 10);
        } else {
            assert_eq!(engine.nogood_stats(), NogoodStats::default());
        }
    }
}

#[test]
fn cpip_stops_at_two_predicates_on_one_variable_and_asserts_without_recurring_scan() {
    for enabled in [false, true] {
        let mut solver = Solver::new();
        solver.set_cpip(enabled);
        let x = solver.new_var_range(0, 10);
        let y = solver.new_var_range(0, 1);
        let z = solver.new_var_range(0, 1);
        solver.set_inference_checking(Some(InferenceCheckConfig::default()));
        let mut engine = Cdcl::new(&mut solver, &[x, y, z]);
        assert!(engine.init());
        let guard = literal(engine.atoms.ge(y, 1));
        let cause = literal(engine.atoms.ge(z, 1));
        let lower = literal(engine.atoms.ge(x, 3));
        let upper = literal(engine.atoms.ge(x, 8)).negate();
        engine.decide(guard).unwrap();
        engine.decide(cause).unwrap();
        // An explicit implication graph is the unit under test. These two
        // reason clauses and the conflict clause are supplied axioms; the
        // independent checker verifies their conditional resolution, not
        // their provenance against an external CP model.
        engine.assign(lower, Reason::Generic(Arc::from([cause]))).unwrap();
        engine.assign(upper, Reason::Generic(Arc::from([cause]))).unwrap();
        assert!(engine.resolve_conflict(Conflict::Generic(vec![guard, lower, upper])));
        assert_eq!(engine.decision_level(), 1);
        if enabled {
            assert_eq!(domain(&engine, x), [0, 1, 2, 8, 9, 10]);
            assert_eq!(engine.value(cause), Tri::Unknown);
            assert_eq!(engine.nogood_stats().extended_scans, 0);
            assert_eq!(engine.nogood_stats().cpip_multi_predicate, 1);
            assert_eq!(engine.nogood_stats().cpip_asserting_predicates, 2);
            assert_eq!(engine.solver.inference_check_stats().lcg_checked_derivations, 1);
            assert_eq!(engine.solver.inference_check_stats().lcg_checked_inferences, 5);
        } else {
            assert_eq!(domain(&engine, x), (0..=10).collect::<Vec<_>>());
            assert_eq!(engine.value(cause), Tri::False);
            assert_eq!(engine.nogood_stats(), NogoodStats::default());
        }
    }
}

#[test]
fn cpip_backjump_excludes_earlier_predicates_on_the_asserting_variable() {
    let mut solver = Solver::new();
    solver.set_cpip(true);
    let x = solver.new_var_range(0, 10);
    let y = solver.new_var_range(0, 1);
    let z = solver.new_var_range(0, 1);
    solver.set_inference_checking(Some(InferenceCheckConfig::default()));
    let mut engine = Cdcl::new(&mut solver, &[x, y, z]);
    assert!(engine.init());
    let guard = literal(engine.atoms.ge(y, 1));
    let cause = literal(engine.atoms.ge(z, 1));
    let lower = literal(engine.atoms.ge(x, 2));
    let upper = literal(engine.atoms.ge(x, 8)).negate();
    let hole = literal(engine.atoms.eq(x, 5)).negate();
    engine.decide(guard).unwrap();
    engine.decide(lower).unwrap();
    engine.decide(cause).unwrap();
    engine.assign(upper, Reason::Generic(Arc::from([cause]))).unwrap();
    engine.assign(hole, Reason::Generic(Arc::from([cause]))).unwrap();
    assert!(engine.resolve_conflict(Conflict::Generic(vec![guard, lower, upper, hole])));
    assert_eq!(engine.decision_level(), 1, "the older x>=2 predicate must not limit the backjump");
    assert_eq!(domain(&engine, x), [0, 1, 5, 8, 9, 10]);
    assert_eq!(engine.nogood_stats().cpip_asserting_predicates, 2);
    assert_eq!(engine.solver.inference_check_stats().lcg_checked_derivations, 1);
}

#[test]
fn cpip_resolution_checks_dropped_root_units_and_earliest_single_variable_block() {
    let mut solver = Solver::new();
    solver.set_cpip(true);
    let x = solver.new_var_range(0, 10);
    let root = solver.new_var_range(0, 1);
    let y = solver.new_var_range(0, 1);
    let z = solver.new_var_range(0, 1);
    solver.set_inference_checking(Some(InferenceCheckConfig::default()));
    let mut engine = Cdcl::new(&mut solver, &[x, root, y, z]);
    assert!(engine.init());
    let root_fact = literal(engine.atoms.ge(root, 1));
    assert!(engine.assert_root_lit(root_fact));
    let guard = literal(engine.atoms.ge(y, 1));
    let cause = literal(engine.atoms.ge(z, 1));
    let lower = literal(engine.atoms.ge(x, 3));
    let upper = literal(engine.atoms.ge(x, 8)).negate();
    engine.decide(guard).unwrap();
    engine.decide(cause).unwrap();
    engine.assign(lower, Reason::Generic(Arc::from([cause, root_fact]))).unwrap();
    engine.assign(upper, Reason::Generic(Arc::from([cause]))).unwrap();
    // The latest x predicate resolves first, leaving z and lower(x), then
    // lower(x) resolves to z. The resulting one-variable block is Boolean.
    assert!(engine.resolve_conflict(Conflict::Generic(vec![guard, cause, lower, upper, root_fact])));
    assert_eq!(engine.value(cause), Tri::False);
    assert_eq!(engine.nogood_stats().cpip_resolution_steps, 2);
    assert_eq!(engine.solver.inference_check_stats().lcg_checked_derivations, 1);
}

#[test]
fn cpip_falls_back_when_root_domain_exceeds_cap_even_if_current_domain_is_small() {
    let mut solver = Solver::new();
    solver.set_cpip(true);
    let x = solver.new_var_range(0, MAX_NOGOOD_DOMAIN_VALUES as i32);
    let y = solver.new_var_range(0, 1);
    let z = solver.new_var_range(0, 1);
    let mut engine = Cdcl::new(&mut solver, &[x, y, z]);
    assert!(engine.init());
    let guard = literal(engine.atoms.ge(y, 1));
    let cause = literal(engine.atoms.ge(z, 1));
    let lower = literal(engine.atoms.ge(x, 3));
    let upper = literal(engine.atoms.ge(x, 5)).negate();
    engine.decide(guard).unwrap();
    engine.decide(cause).unwrap();
    engine.assign(lower, Reason::Generic(Arc::from([cause]))).unwrap();
    engine.assign(upper, Reason::Generic(Arc::from([cause]))).unwrap();
    assert_eq!(engine.solver.store.size(x), 2);
    assert!(engine.resolve_conflict(Conflict::Generic(vec![guard, lower, upper])));
    assert_eq!(engine.value(cause), Tri::False);
    assert_eq!(engine.nogood_stats().skipped_domain_limit, 1);
    assert_eq!(engine.nogood_stats().cpip_fallbacks, 1);
    assert_eq!(engine.nogood_stats().cpip_learned, 0);
}

#[test]
fn bounded_plans_skip_large_shapes_and_stop_without_partial_mutation() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, MAX_NOGOOD_DOMAIN_VALUES as i32);
    let y = solver.new_var_range(0, 1);
    let atoms = AtomTable::build(2, |var| if var == x { (0, MAX_NOGOOD_DOMAIN_VALUES as i32) } else { (0, 1) });
    let low = literal(atoms.ge(x, 3)).negate();
    let high = literal(atoms.ge(x, 8));
    let guard = literal(atoms.ge(y, 1));
    let stop = AtomicBool::new(false);
    let mut stats = NogoodStats::default();
    assert!(matches!(nogood::plan(&solver.store, &atoms, &[low, high], &stop, &mut stats), Plan::Inactive));
    assert_eq!(stats.skipped_domain_limit, 1);
    assert!(matches!(nogood::plan(&solver.store, &atoms, &vec![low; MAX_NOGOOD_CLAUSE_LEN + 1], &stop, &mut stats), Plan::Inactive));
    assert_eq!(stats.skipped_clause_limit, 1);
    // Two unknown variables never authorize a generalized unit reduction.
    assert!(matches!(nogood::plan(&solver.store, &atoms, &[low, guard], &stop, &mut stats), Plan::Inactive));
    stop.store(true, Ordering::Relaxed);
    assert!(matches!(nogood::plan(&solver.store, &atoms, &[low, high], &stop, &mut stats), Plan::Interrupted));
    assert_eq!(stats.interrupted, 1);
    assert_eq!(solver.store.size(x), MAX_NOGOOD_DOMAIN_VALUES + 1);
}

#[test]
fn proof_logger_suppresses_extended_propagation_and_cpip_learning() {
    let mut solver = Solver::new();
    solver.set_extended_nogoods(true);
    solver.set_cpip(true);
    let x = solver.new_var_range(0, 10);
    let y = solver.new_var_range(0, 1);
    let z = solver.new_var_range(0, 1);
    let mut proof = |_: &[Lit]| {};
    let mut engine = Cdcl::new(&mut solver, &[x, y, z]);
    engine.set_proof_logger(&mut proof);
    assert!(engine.init());
    let guard = literal(engine.atoms.ge(y, 1));
    let cause = literal(engine.atoms.ge(z, 1));
    let lower = literal(engine.atoms.ge(x, 3));
    let upper = literal(engine.atoms.ge(x, 8)).negate();
    engine.add_clause(vec![guard.negate(), lower.negate(), upper.negate()], false, 0);
    engine.decide(guard).unwrap();
    assert!(engine.propagate_and_learn());
    assert_eq!(domain(&engine, x), (0..=10).collect::<Vec<_>>());
    engine.decide(cause).unwrap();
    engine.assign(lower, Reason::Generic(Arc::from([cause]))).unwrap();
    engine.assign(upper, Reason::Generic(Arc::from([cause]))).unwrap();
    assert!(engine.resolve_conflict(Conflict::Generic(vec![guard, lower, upper])));
    assert_eq!(engine.value(cause), Tri::False);
    assert_eq!(engine.nogood_stats().cpip_attempts, 0);
    assert_eq!(engine.nogood_stats().removed_values, 0);
    assert!(engine.nogood_stats().skipped_proof_logging > 0);
}

#[test]
fn interrupted_cpip_is_not_reported_as_a_root_contradiction() {
    for conflict_len in [1, MAX_NOGOOD_CLAUSE_LEN + 1] {
        let mut solver = Solver::new();
        solver.set_cpip(true);
        let x = solver.new_var_range(0, 1);
        let stop = AtomicBool::new(false);
        let mut engine = Cdcl::new(&mut solver, &[x]);
        engine.set_stop(&stop);
        assert!(engine.init());
        let decision = literal(engine.atoms.ge(x, 1));
        engine.decide(decision).unwrap();
        stop.store(true, Ordering::Relaxed);
        assert!(engine.resolve_conflict(Conflict::Generic(vec![decision; conflict_len])));
        assert_eq!(engine.decision_level(), 1);
        assert_eq!(engine.nogood_stats().interrupted, 1);
        assert_eq!(engine.nogood_stats().cpip_learned, 0);
        assert_eq!(engine.nogood_stats().cpip_fallbacks, 0);
        assert_eq!(engine.nogood_stats().skipped_clause_limit, 0);
    }
}

#[test]
fn imported_clause_uses_extended_propagation_with_worker_local_reasons() {
    let mut solver = Solver::new();
    solver.set_extended_nogoods(true);
    let x = solver.new_var_range(0, 10);
    let y = solver.new_var_range(0, 1);
    let pool = Arc::new(SharedClausePool::default());
    let publisher = ClauseSharing::new(Arc::clone(&pool), 0);
    let mut engine = Cdcl::new_with_registry(&mut solver, &[x, y], pool.lazy_atoms());
    engine.set_clause_sharing(ClauseSharing::new(Arc::clone(&pool), 1));
    assert!(engine.init());
    let guard = literal(engine.atoms.ge(y, 1));
    let low = literal(engine.atoms.ge(x, 3)).negate();
    let high = literal(engine.atoms.ge(x, 8));
    engine.decide(guard).unwrap();
    publisher.publish(Arc::from([guard.negate(), low, high]), 2);
    assert!(engine.sync_shared_clauses());
    assert_eq!(pool.imported(), 1);
    assert_eq!(domain(&engine, x), [0, 1, 2, 8, 9, 10]);
    let removed = literal(engine.atoms.eq(x, 5)).negate();
    assert_eq!(engine.reason_lits(removed), [guard.negate()]);
}

#[test]
fn root_semantic_reasons_survive_clause_deletion_and_slot_reuse() {
    let mut solver = Solver::new();
    solver.set_extended_nogoods(true);
    let x = solver.new_var_range(0, 10);
    let y = solver.new_var_range(0, 1);
    let mut engine = Cdcl::new(&mut solver, &[x, y]);
    assert!(engine.init());
    let low = literal(engine.atoms.ge(x, 3)).negate();
    let high = literal(engine.atoms.ge(x, 8));
    let five = literal(engine.atoms.eq(x, 5));
    let source = engine.add_clause(vec![low, high, five], true, 100);
    assert!(engine.propagate_and_learn());
    let removed = literal(engine.atoms.eq(x, 4)).negate();
    assert!(matches!(engine.reason_of(removed.atom()), Reason::Generic(_)));
    assert!(engine.reason_lits(removed).is_empty());
    let guard = literal(engine.atoms.ge(y, 1));
    // The standard soft database cap is 2000; low-LBD clauses are protected.
    for _ in 0..2000 {
        engine.add_clause(vec![guard, guard.negate(), low], true, 1);
    }
    engine.maybe_reduce_db();
    assert!(engine.clauses.get(source).lits.is_empty());
    let recycled = engine.add_clause(vec![guard, guard.negate(), high], true, 1);
    assert_eq!(source, recycled);
    assert!(engine.reason_lits(removed).is_empty());
    assert!(engine.propagate_and_learn());
    assert_eq!(domain(&engine, x), [0, 1, 2, 5, 8, 9, 10]);
}

#[test]
fn inference_hooks_reject_before_mutation_and_poison_solver_clones() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 2);
    solver.set_inference_checking(Some(InferenceCheckConfig::default()));
    let mut sibling = solver.clone();
    let clause = [InferenceLiteral { predicate: Premise::Eq { var: x, val: 0 }, positive: true }];
    assert!(catch_unwind(AssertUnwindSafe(|| {
        solver.store.check_lcg_inference(&clause, &[], InferenceConclusion::Predicate(Premise::Ne { var: x, val: 0 }));
    }))
    .is_err());
    assert!(solver.store.contains(x, 0));
    assert!(catch_unwind(AssertUnwindSafe(|| sibling.propagate())).is_err());
}

#[test]
fn derivation_hook_does_not_borrow_current_nonroot_facts() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    let y = solver.new_var_range(0, 1);
    solver.set_inference_checking(Some(InferenceCheckConfig::default()));
    solver.store.push_level();
    solver.store.fix(x, 1).unwrap();
    let x_true = InferenceLiteral { predicate: Premise::Eq { var: x, val: 1 }, positive: true };
    let y_true = InferenceLiteral { predicate: Premise::Eq { var: y, val: 1 }, positive: true };
    let source = [InferenceLiteral { positive: false, ..x_true }, y_true];
    assert!(catch_unwind(AssertUnwindSafe(|| solver.store.check_lcg_clause_entailment(&[&source], &[y_true]))).is_err());
    assert!(!solver.store.is_fixed(y));
}

#[test]
fn bounded_inference_preparation_reports_uncovered_complete_checks() {
    let mut solver = Solver::new();
    solver.set_cpip(true);
    let x = solver.new_var_range(0, 10);
    let y = solver.new_var_range(0, 1);
    let z = solver.new_var_range(0, 1);
    solver.set_inference_checking(Some(InferenceCheckConfig { max_work: 0, ..InferenceCheckConfig::default() }));
    let mut engine = Cdcl::new(&mut solver, &[x, y, z]);
    assert!(engine.init());
    let guard = literal(engine.atoms.ge(y, 1));
    let cause = literal(engine.atoms.ge(z, 1));
    let lower = literal(engine.atoms.ge(x, 3));
    let upper = literal(engine.atoms.ge(x, 8)).negate();
    engine.decide(guard).unwrap();
    engine.decide(cause).unwrap();
    engine.assign(lower, Reason::Generic(Arc::from([cause]))).unwrap();
    engine.assign(upper, Reason::Generic(Arc::from([cause]))).unwrap();
    assert!(engine.resolve_conflict(Conflict::Generic(vec![guard, lower, upper])));
    let stats = engine.solver.inference_check_stats();
    assert_eq!(stats.lcg_checked_derivations, 0);
    assert_eq!(stats.lcg_uncovered_derivations, 1);
    assert_eq!(stats.lcg_checked_inferences, 0);
    assert_eq!(stats.lcg_uncovered_inferences, 5);
    assert_eq!(domain(&engine, x), [0, 1, 2, 8, 9, 10]);
}

#[test]
fn late_lazy_predicate_remains_sound_across_cpip_backjump() {
    let mut solver = Solver::new();
    solver.set_cpip(true);
    let x = solver.new_var_range(0, 1000);
    let a = solver.new_var_range(0, 1);
    let b = solver.new_var_range(0, 1);
    solver.set_inference_checking(Some(InferenceCheckConfig::default()));
    let mut engine = Cdcl::new(&mut solver, &[x, a, b]);
    assert!(engine.init());
    let lower = literal(engine.atoms.ge(x, 100));
    let a_one = literal(engine.atoms.ge(a, 1));
    let b_one = literal(engine.atoms.ge(b, 1));
    engine.decide(lower).unwrap();
    engine.decide(a_one.negate()).unwrap();
    engine.decide(b_one.negate()).unwrap();
    // Shared imports can first materialize a known consequence at a later
    // level. Its mathematical truth survives a jump which removes its record.
    let fifty = literal(engine.atoms.eq(x, 50));
    engine.assign(fifty.negate(), Reason::Generic(Arc::from([lower]))).unwrap();
    assert_eq!(engine.level_of(fifty.atom()), 3);
    let conflict = engine.add_clause(vec![fifty, a_one], false, 0);
    assert!(engine.resolve_conflict(Conflict::Clause(conflict)));
    assert!(engine.propagate_and_learn());
    assert_eq!(engine.decision_level(), 1);
    assert_eq!(engine.value(a_one), Tri::True);
    engine.backjump_to(0);
    assert!(engine.solver.store.contains(x, 50));
    assert!(engine.solver.store.contains(a, 0));
    engine.decide(fifty).unwrap();
    engine.decide(a_one.negate()).unwrap();
    assert!(engine.propagate_and_learn());
    assert_eq!(engine.solver.store.value(x), 50);
    assert_eq!(engine.solver.store.value(a), 0);
}

#[test]
fn extended_scan_resynchronizes_late_lazy_outside_antecedents() {
    let mut solver = Solver::new();
    solver.set_extended_nogoods(true);
    let x = solver.new_var_range(0, 1000);
    let y = solver.new_var_range(0, 10);
    let b = solver.new_var_range(0, 1);
    let mut engine = Cdcl::new(&mut solver, &[x, y, b]);
    assert!(engine.init());
    let lower = literal(engine.atoms.ge(x, 100));
    let branch = literal(engine.atoms.ge(b, 1));
    engine.decide(lower).unwrap();
    engine.decide(branch).unwrap();
    let fifty = literal(engine.atoms.eq(x, 50));
    engine.assign(fifty.negate(), Reason::Generic(Arc::from([lower]))).unwrap();
    let y_low = literal(engine.atoms.ge(y, 3)).negate();
    let y_high = literal(engine.atoms.ge(y, 8));
    engine.add_clause(vec![fifty, y_low, y_high], false, 0);
    engine.backjump_to(1);
    assert_eq!(engine.tvalue(fifty), Tri::Unknown);
    assert_eq!(engine.value(fifty), Tri::False);
    assert!(engine.propagate_and_learn());
    assert_eq!(domain(&engine, y), [0, 1, 2, 8, 9, 10]);
    engine.backjump_to(0);
    assert_eq!(domain(&engine, y), (0..=10).collect::<Vec<_>>());
    engine.decide(fifty).unwrap();
    assert!(engine.propagate_and_learn());
    assert_eq!(domain(&engine, y), (0..=10).collect::<Vec<_>>());
}

#[test]
fn synchronization_without_domain_change_drains_new_boolean_work_before_fixpoint() {
    let mut solver = Solver::new();
    solver.set_extended_nogoods(true);
    let x = solver.new_var_range(0, 1000);
    let a = solver.new_var_range(0, 1);
    let b = solver.new_var_range(0, 1);
    let y = solver.new_var_range(0, 1);
    let padding = (0..MAX_NOGOOD_CLAUSE_LEN - 1).map(|_| solver.new_var_range(0, 1)).collect::<Vec<_>>();
    let mut variables = vec![x, a, b, y];
    variables.extend_from_slice(&padding);
    let mut engine = Cdcl::new(&mut solver, &variables);
    assert!(engine.init());
    let mut root_false = Vec::new();
    for var in padding {
        let positive = literal(engine.atoms.ge(var, 1));
        assert!(engine.assert_root_lit(positive.negate()));
        root_false.push(positive);
    }
    let lower = literal(engine.atoms.ge(x, 100));
    let branch = literal(engine.atoms.ge(b, 1));
    let a_one = literal(engine.atoms.ge(a, 1));
    let y_one = literal(engine.atoms.ge(y, 1));
    engine.decide(lower).unwrap();
    engine.decide(branch).unwrap();
    let fifty = literal(engine.atoms.eq(x, 50));
    engine.assign(fifty.negate(), Reason::Generic(Arc::from([lower]))).unwrap();
    let mut long_clause = vec![fifty, a_one];
    long_clause.extend(root_false);
    assert_eq!(long_clause.len(), MAX_NOGOOD_CLAUSE_LEN + 1);
    engine.add_clause(long_clause, false, 0);
    // This final tautology causes synchronization but no semantic deletion.
    // The long clause is skipped by semantic scanning and must wake through
    // its ordinary SAT watch when the restored x!=50 record is propagated.
    engine.add_clause(vec![fifty, y_one, y_one.negate()], false, 0);
    engine.backjump_to(1);
    assert!(engine.propagate_and_learn());
    assert_eq!(engine.value(a_one), Tri::True);
    assert_eq!(engine.nogood_stats().removed_values, 0);
    engine.backjump_to(0);
    engine.decide(fifty).unwrap();
    engine.decide(a_one.negate()).unwrap();
    assert!(engine.propagate_and_learn());
}
