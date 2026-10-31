use std::cell::Cell;
use std::collections::BTreeSet;
use std::sync::atomic::AtomicBool;

use crate::constraints::divisibility::shared_divisibility_exclusion_interruptible;
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

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

#[test]
fn aggregated_divisibility_matches_the_assignment_oracle() {
    const VALUES: [i32; 11] = [-6, -5, -3, -2, -1, 0, 1, 2, 3, 5, 6];
    const CANDIDATE_DIVISORS: [i64; 4] = [2, 3, 5, 7];
    let mut seed = 0x05ee_dc0d_ed15_ca11_u64;

    for case in 0..128 {
        let mut left_domain = VALUES.into_iter().filter(|_| next(&mut seed) & 1 != 0).collect::<Vec<_>>();
        let mut right_domain = VALUES.into_iter().filter(|_| next(&mut seed) & 1 != 0).collect::<Vec<_>>();
        let mut divisors = CANDIDATE_DIVISORS.into_iter().filter(|_| next(&mut seed) & 1 != 0).collect::<Vec<_>>();
        if left_domain.is_empty() {
            left_domain.push(VALUES[(next(&mut seed) as usize) % VALUES.len()]);
        }
        if right_domain.is_empty() {
            right_domain.push(VALUES[(next(&mut seed) as usize) % VALUES.len()]);
        }
        if divisors.is_empty() {
            divisors.push(CANDIDATE_DIVISORS[(next(&mut seed) as usize) % CANDIDATE_DIVISORS.len()]);
        }

        let expected = left_domain
            .iter()
            .flat_map(|&left| right_domain.iter().map(move |&right| (left, right)))
            .filter(|&(left, right)| divisors.iter().all(|&divisor| i64::from(left) % divisor != 0 || i64::from(right) % divisor != 0))
            .count() as u64;

        let mut solver = Solver::new();
        let left = solver.new_var_set(&left_domain);
        let right = solver.new_var_set(&right_domain);
        assert!(shared_divisibility_exclusion_interruptible(&mut solver, left, right, &divisors, &AtomicBool::new(false),));

        assert_eq!(count_solutions(&mut solver, &[left, right]), expected, "oracle mismatch in generated case {case}");
    }
}

#[test]
fn propagation_stops_while_snapshotting_a_large_target_domain() {
    let values = (0..4_096).collect::<Vec<_>>();
    let mut solver = Solver::new();
    let source = solver.new_var_set(&[2]);
    let target = solver.new_var_set(&values);
    assert!(shared_divisibility_exclusion_interruptible(&mut solver, source, target, &[2], &AtomicBool::new(false),));
    let polls = Cell::new(0usize);

    solver
        .propagate_until(|| {
            let next = polls.get() + 1;
            polls.set(next);
            next >= 64
        })
        .expect("cooperative interruption is not an inconsistency");

    assert!(polls.get() >= 64);
    assert_eq!(solver.store.size(target), values.len(), "no filtering may follow an interrupted snapshot");
    assert!(solver.fd_at_fixpoint(), "the interrupted propagation queue must be cleared");
}

#[test]
fn lcg_removals_are_explained_only_by_the_fixed_source() {
    let mut solver = Solver::new();
    let source = solver.new_var_set(&[5, 6]);
    let target = solver.new_var_set(&[7, 10, 11, 12]);
    assert!(shared_divisibility_exclusion_interruptible(&mut solver, source, target, &[2, 3, 5], &AtomicBool::new(false),));
    let mut cdcl = Cdcl::new(&mut solver, &[source, target]);
    assert!(cdcl.init());

    cdcl.decide(lit(cdcl.atoms.eq(source, 6))).unwrap();
    cdcl.propagate().unwrap();

    for removed_value in [10, 12] {
        assert!(!cdcl.solver.store.contains(target, removed_value));
        let removed = lit(cdcl.atoms.eq(target, removed_value)).negate();
        let Reason::Generic(reason) = cdcl.reason_of(removed.atom()) else {
            panic!("removal of {removed_value} did not retain a propagator reason");
        };
        assert!(!reason.is_empty());
        assert!(reason.iter().all(|&premise| cdcl.tvalue(premise) == Tri::True));
        assert!(reason.iter().all(|&premise| cdcl.atoms.var_of(premise.atom()) == source));
    }
}

#[test]
fn lcg_conflict_cites_both_incompatible_fixed_values() {
    let mut solver = Solver::new();
    let left = solver.new_var_set(&[5, 6]);
    let right = solver.new_var_set(&[11, 12]);
    assert!(shared_divisibility_exclusion_interruptible(&mut solver, left, right, &[2, 3], &AtomicBool::new(false),));
    let mut cdcl = Cdcl::new(&mut solver, &[left, right]);
    assert!(cdcl.init());

    cdcl.decide(lit(cdcl.atoms.eq(left, 6))).unwrap();
    cdcl.decide(lit(cdcl.atoms.eq(right, 12))).unwrap();
    let Conflict::Generic(reason) = cdcl.propagate().expect_err("incompatible fixed values were accepted") else {
        panic!("expected a propagator conflict");
    };

    assert!(reason.iter().all(|&premise| cdcl.tvalue(premise) == Tri::True));
    let variables = reason.iter().map(|premise| cdcl.atoms.var_of(premise.atom())).collect::<BTreeSet<_>>();
    assert_eq!(variables, BTreeSet::from([left, right]));
}

fn mod_not_zero(variable: IntVarRef, divisor: i64, zero_first: bool) -> IntExpr {
    let modulo = IntExpr::Mod(Box::new(IntExpr::Variable(variable)), Box::new(IntExpr::Constant(divisor)));
    if zero_first {
        IntExpr::Ne(Box::new(IntExpr::Constant(0)), Box::new(modulo))
    } else {
        IntExpr::Ne(Box::new(modulo), Box::new(IntExpr::Constant(0)))
    }
}

fn exclusion(left: IntVarRef, left_divisor: i64, right: IntVarRef, right_divisor: i64, reverse_ne: bool) -> Constraint {
    Constraint::Intension(IntExpr::Or(vec![mod_not_zero(left, left_divisor, reverse_ne), mod_not_zero(right, right_divisor, reverse_ne)]))
}

#[test]
fn cp_compiler_groups_modular_disjunctions_and_leaves_near_matches_generic() {
    let mut model = Model::new();
    let a = model.int_range(1, 8);
    let b = model.int_range(1, 8);
    let c = model.int_range(1, 8);

    for divisor in [2, 3, 5] {
        model.add_constraint(exclusion(a, divisor, b, divisor, false));
    }
    // Reverse both the disjunction branches and the operands of `ne`.
    model.add_constraint(Constraint::Intension(IntExpr::Or(vec![mod_not_zero(b, 7, true), mod_not_zero(a, 7, true)])));
    for divisor in [2, 3] {
        model.add_constraint(exclusion(c, divisor, b, divisor, false));
    }
    // This near-match shares an already grouped pair, but its two modulo
    // operations use different divisors and must remain a generic intension.
    model.add_constraint(exclusion(a, 2, b, 3, false));

    let compiled = CompiledCp::compile_interruptible(&model, &AtomicBool::new(false)).unwrap().unwrap();

    assert_eq!(compiled.problem().solver.num_propagators(), 3);
    let expected = (1..=8)
        .flat_map(|av| (1..=8).flat_map(move |bv| (1..=8).map(move |cv| (av, bv, cv))))
        .filter(|&(av, bv, cv)| {
            [2, 3, 5, 7].into_iter().all(|divisor| av % divisor != 0 || bv % divisor != 0)
                && [2, 3].into_iter().all(|divisor| bv % divisor != 0 || cv % divisor != 0)
                && (av % 2 != 0 || bv % 3 != 0)
        })
        .count() as u64;
    let mut solver = compiled.problem().solver.clone();
    assert_eq!(count_solutions(&mut solver, compiled.int_variables()), expected);
}
