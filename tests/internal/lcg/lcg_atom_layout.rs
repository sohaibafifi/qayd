use std::sync::Arc;

use crate::constraints::linear::{linear, Relation};
use crate::ids::VarId;
use crate::lcg::lit::{AtomKind, AtomTable, LazyAtomRegistry, Lit, LitOrConst};
use crate::lcg::trail::{Cdcl, Reason};
use crate::lcg::view::Tri;
use crate::{count_solutions, Solver};

fn lit(value: LitOrConst) -> Lit {
    match value {
        LitOrConst::Lit(lit) => lit,
        other => panic!("expected an atom, got {other:?}"),
    }
}

#[test]
fn dense_explicit_gaps_keep_root_support_for_atom_layout() {
    let mut solver = Solver::new();
    let x = solver.new_var_set(&[6, 0, 4, 2, 2]);

    // The compact set deliberately uses dense physical arrays.
    assert_eq!(solver.store.num_sparse_domains(), 0);
    let support = solver.store.sparse_values(x).expect("explicit root support was lost");
    assert_eq!(support.as_ref(), &[0, 2, 4, 6]);

    let atoms = AtomTable::build_active_sparse(
        solver.store.num_vars(),
        |_| true,
        |_| false,
        |var| solver.store.sparse_values(var),
        |var| (solver.store.min(var), solver.store.max(var)),
    );

    assert_eq!(atoms.num_atoms(), 7);
    assert_eq!(atoms.eq(x, 1), LitOrConst::False);
    assert_eq!(atoms.eq(x, 3), LitOrConst::False);
    assert_eq!(atoms.eq(x, 5), LitOrConst::False);
    assert_eq!(atoms.ge(x, 1), atoms.ge(x, 2));
    let ge_two = lit(atoms.ge(x, 1));
    assert_eq!(atoms.decode(ge_two.atom()), AtomKind::Ge { var: x, k: 2 });
}

#[test]
fn contiguous_range_switches_to_lazy_above_255_atoms() {
    let eager = AtomTable::build(1, |_| (0, 127));
    assert!(!eager.is_lazy(VarId(0)));
    assert_eq!(eager.num_atoms(), 255);

    let lazy = AtomTable::build(1, |_| (0, 128));
    assert!(lazy.is_lazy(VarId(0)));
    assert_eq!(lazy.num_atoms(), 0);

    let eq = lit(lazy.eq(VarId(0), 64));
    let ge = lit(lazy.ge(VarId(0), 64));
    assert_eq!(lazy.num_atoms(), 2);
    assert_eq!(lazy.decode(eq.atom()), AtomKind::Eq { var: VarId(0), v: 64 });
    assert_eq!(lazy.decode(ge.atom()), AtomKind::Ge { var: VarId(0), k: 64 });
}

#[test]
fn contiguous_explicit_enumeration_uses_the_range_atom_budget() {
    let mut solver = Solver::new();
    let values = (0..=128).collect::<Vec<_>>();
    let x = solver.new_var_set(&values);

    assert!(solver.store.sparse_values(x).is_none());
    let atoms = AtomTable::build_active_sparse(
        solver.store.num_vars(),
        |_| true,
        |_| false,
        |var| solver.store.sparse_values(var),
        |var| (solver.store.min(var), solver.store.max(var)),
    );
    assert!(atoms.is_lazy(x));
    assert_eq!(atoms.num_atoms(), 0);
}

#[test]
fn large_explicit_support_is_lazy_without_materializing_numeric_gaps() {
    let support: Arc<[i32]> = Arc::from((0..=256).step_by(2).collect::<Vec<_>>());
    let atoms = AtomTable::build_active_sparse(1, |_| true, |_| false, |_| Some(Arc::clone(&support)), |_| (0, 256));

    assert!(atoms.is_lazy(VarId(0)));
    assert_eq!(atoms.num_atoms(), 0);
    assert_eq!(atoms.ge_i64_existing(VarId(0), 2), None);
    assert_eq!(atoms.eq_existing(VarId(0), 2), None);
    assert_eq!(atoms.eq(VarId(0), 1), LitOrConst::False);
    assert_eq!(atoms.ge(VarId(0), 1), atoms.ge(VarId(0), 2));
    let ge_two = lit(atoms.ge(VarId(0), 1));
    let eq_two = lit(atoms.eq(VarId(0), 2));
    assert_eq!(atoms.num_atoms(), 2);
    assert_eq!(atoms.decode(ge_two.atom()), AtomKind::Ge { var: VarId(0), k: 2 });
    assert_eq!(atoms.decode(eq_two.atom()), AtomKind::Eq { var: VarId(0), v: 2 });
    assert_eq!(atoms.ge_i64_existing(VarId(0), 2), Some(LitOrConst::Lit(ge_two)));
    assert_eq!(atoms.eq_existing(VarId(0), 2), Some(LitOrConst::Lit(eq_two)));
}

#[test]
fn sign_and_inactive_domains_do_not_consume_range_budget() {
    let sign_support: Arc<[i32]> = Arc::from([-1, 1]);
    let atoms = AtomTable::build_active_sparse(
        3,
        |var| var != VarId(0),
        |var| var == VarId(1),
        |var| (var == VarId(1)).then(|| Arc::clone(&sign_support)),
        |var| match var {
            VarId(0) => (i32::MIN, i32::MAX),
            VarId(1) => (-1, 1),
            VarId(2) => (0, 127),
            _ => unreachable!(),
        },
    );

    assert!(!atoms.is_lazy(VarId(0)));
    assert!(atoms.is_sign(VarId(1)));
    assert!(!atoms.is_lazy(VarId(2)));
    assert_eq!(atoms.num_atoms(), 1 + 255);
    assert_eq!(atoms.eq(VarId(1), 0), LitOrConst::False);
}

#[test]
fn cumulative_range_budget_has_a_stable_boundary() {
    const EAGER_RANGES: usize = 1_960;
    let support: Arc<[i32]> = Arc::from((0..=126).step_by(2).collect::<Vec<_>>());
    let num_vars = EAGER_RANGES + 2;
    let build = || {
        AtomTable::build_active_sparse(
            num_vars,
            |_| true,
            |_| false,
            |var| (var == VarId(0)).then(|| Arc::clone(&support)),
            |var| if var == VarId(0) { (0, 126) } else { (0, 127) },
        )
    };

    let atoms = build();
    // The explicit set contributes 127 atoms to the shared 500,000 atom budget.
    // Each following 128-value range contributes 255.
    assert!(!atoms.is_lazy(VarId(EAGER_RANGES as u32)));
    assert!(atoms.is_lazy(VarId(EAGER_RANGES as u32 + 1)));
    assert_eq!(atoms.num_atoms(), 127 + EAGER_RANGES * 255);
    assert_eq!(atoms.eq(VarId(0), 1), LitOrConst::False);
    drop(atoms);

    let rebuilt = build();
    assert!(!rebuilt.is_lazy(VarId(EAGER_RANGES as u32)));
    assert!(rebuilt.is_lazy(VarId(EAGER_RANGES as u32 + 1)));
    assert_eq!(rebuilt.num_atoms(), 127 + EAGER_RANGES * 255);
}

#[test]
fn cumulative_sparse_budget_has_a_stable_boundary() {
    const EAGER_SPARSE: usize = 3_937;
    let support: Arc<[i32]> = Arc::from((0..=126).step_by(2).collect::<Vec<_>>());
    let atoms = AtomTable::build_active_sparse(EAGER_SPARSE + 1, |_| true, |_| false, |_| Some(Arc::clone(&support)), |_| (0, 126));

    assert!(!atoms.is_lazy(VarId(EAGER_SPARSE as u32 - 1)));
    assert!(atoms.is_lazy(VarId(EAGER_SPARSE as u32)));
    assert_eq!(atoms.num_atoms(), 499_999);
}

#[test]
fn shared_registry_keeps_lazy_atom_ids_compatible() {
    let registry = LazyAtomRegistry::new();
    let build = || {
        AtomTable::build_active_sparse_with_registry(
            2,
            |_| true,
            |_| false,
            |_| None,
            |var| if var == VarId(0) { (0, 2) } else { (0, 128) },
            Arc::clone(&registry),
        )
    };
    let first = build();
    let second = build();

    assert_eq!(first.num_atoms(), 5);
    assert_eq!(second.num_atoms(), 5);
    let first_eq = lit(first.eq(VarId(1), 64));
    let second_eq = lit(second.eq(VarId(1), 64));
    let second_ge = lit(second.ge(VarId(1), 65));
    let first_ge = lit(first.ge(VarId(1), 65));
    assert_eq!(first_eq, second_eq);
    assert_eq!(first_ge, second_ge);
    assert_eq!(first.decode(first_eq.atom()), second.decode(second_eq.atom()));
    assert_eq!(first.decode(first_ge.atom()), second.decode(second_ge.atom()));
}

#[test]
fn dense_gapped_atoms_restore_with_sound_secondary_reasons() {
    let mut solver = Solver::new();
    let x = solver.new_var_set(&[0, 2, 4, 6]);
    let mut cdcl = Cdcl::new(&mut solver, &[x]);
    assert!(cdcl.init());
    assert_eq!(cdcl.atoms.num_atoms(), 7);

    let decision = lit(cdcl.atoms.eq(x, 2));
    cdcl.decide(decision).unwrap();
    assert_eq!(cdcl.solver.store.value(x), 2);

    let mut atoms = Vec::new();
    cdcl.atoms.append_atoms(x, &mut atoms);
    assert_eq!(atoms.len(), 7);
    for atom in atoms.iter().copied() {
        assert!(cdcl.is_assigned(atom));
        if atom == decision.atom() {
            assert!(matches!(cdcl.reason_of(atom), Reason::Decision));
        } else {
            let Reason::Generic(reason) = cdcl.reason_of(atom) else {
                panic!("secondary atom {:?} did not retain a generic reason", cdcl.atoms.decode(atom));
            };
            assert_eq!(reason.as_ref(), &[decision]);
            assert!(reason.iter().all(|&premise| cdcl.tvalue(premise) == Tri::True));
        }
    }

    cdcl.backjump_to(0);
    let mut values = cdcl.solver.store.values(x).collect::<Vec<_>>();
    values.sort_unstable();
    assert_eq!(values, [0, 2, 4, 6]);
    assert!(atoms.iter().all(|&atom| !cdcl.is_assigned(atom)));
}

#[test]
fn gapped_and_lazy_domains_solve_and_restore_propagator_reasons() {
    let mut counted = Solver::new();
    let counted_x = counted.new_var_set(&[0, 2, 4]);
    let counted_y = counted.new_var_range(0, 128);
    linear(&mut counted, &[1, 1], &[counted_x, counted_y], Relation::Eq, 64);
    assert_eq!(count_solutions(&mut counted, &[counted_x, counted_y]), 3);

    let mut solver = Solver::new();
    let x = solver.new_var_set(&[0, 2, 4]);
    let y = solver.new_var_range(0, 128);
    linear(&mut solver, &[1, 1], &[x, y], Relation::Eq, 64);
    let mut cdcl = Cdcl::new(&mut solver, &[x, y]);
    assert!(cdcl.init());
    assert!(cdcl.atoms.is_lazy(y));
    assert_eq!((cdcl.solver.store.min(y), cdcl.solver.store.max(y)), (60, 64));

    let decision = lit(cdcl.atoms.eq(x, 2));
    cdcl.decide(decision).unwrap();
    cdcl.propagate().unwrap();
    assert_eq!(cdcl.solver.store.value(y), 62);

    let lower = lit(cdcl.atoms.ge(y, 62));
    let upper = lit(cdcl.atoms.ge(y, 63)).negate();
    for inferred in [lower, upper] {
        assert_eq!(cdcl.tvalue(inferred), Tri::True);
        let Reason::Generic(reason) = cdcl.reason_of(inferred.atom()) else {
            panic!("lazy bound {:?} did not retain its propagator reason", cdcl.atoms.decode(inferred.atom()));
        };
        assert!(!reason.is_empty());
        assert!(reason.iter().all(|&premise| cdcl.tvalue(premise) == Tri::True));
    }

    cdcl.backjump_to(0);
    assert_eq!((cdcl.solver.store.min(x), cdcl.solver.store.max(x)), (0, 4));
    assert_eq!((cdcl.solver.store.min(y), cdcl.solver.store.max(y)), (60, 64));
    assert_eq!(cdcl.tvalue(lower), Tri::Unknown);
    assert_eq!(cdcl.tvalue(upper), Tri::Unknown);
}

#[test]
fn lazy_holes_created_before_cdcl_are_seeded_as_root_facts() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 128);
    let y = solver.new_var_range(0, 1);
    linear(&mut solver, &[1, 1], &[x, y], Relation::Le, 100);

    // Reproduce portfolio root preparation: propagation can leave a hole while
    // explanations are disabled, then the LCG engine is built from that state.
    solver.store.remove(x, 64).unwrap();
    assert!(!solver.store.contains(x, 64));

    let mut cdcl = Cdcl::new(&mut solver, &[x, y]);
    assert!(cdcl.atoms.is_lazy(x));
    assert!(cdcl.init());

    // Asking for the hole after init must find an atom that seed_facts already
    // recorded. The root propagation above also exercises fallback snapshots
    // containing this literal.
    let removed = lit(cdcl.atoms.eq(x, 64)).negate();
    assert_eq!(cdcl.tvalue(removed), Tri::True);
    assert_eq!(cdcl.level_of(removed.atom()), 0);
    assert!(matches!(cdcl.reason_of(removed.atom()), Reason::Fact));
}

#[test]
fn lazy_sparse_root_pruning_never_creates_untrailed_facts() {
    let mut solver = Solver::new();
    let support = (0..=256).step_by(2).collect::<Vec<_>>();
    let x = solver.new_var_set(&support);
    let y = solver.new_var_range(0, 1);
    linear(&mut solver, &[1, 1], &[x, y], Relation::Le, 200);
    solver.store.remove(x, 0).unwrap();
    solver.store.remove(x, 64).unwrap();

    let mut cdcl = Cdcl::new(&mut solver, &[x, y]);
    assert!(cdcl.atoms.is_lazy(x));
    assert!(cdcl.init());

    assert_eq!(cdcl.atoms.ge(x, 2), LitOrConst::True);
    assert_eq!(cdcl.atoms.eq(x, 0), LitOrConst::False);
    let removed = lit(cdcl.atoms.eq(x, 64)).negate();
    assert_eq!(cdcl.tvalue(removed), Tri::True);
    assert_eq!(cdcl.level_of(removed.atom()), 0);
    assert!(matches!(cdcl.reason_of(removed.atom()), Reason::Fact));
}

#[test]
fn lazy_variable_fixed_before_cdcl_seeds_its_equality_fact() {
    let mut solver = Solver::new();
    let support = (0..=256).step_by(2).collect::<Vec<_>>();
    let x = solver.new_var_set(&support);
    solver.store.fix(x, 64).unwrap();

    let mut cdcl = Cdcl::new(&mut solver, &[x]);
    assert!(cdcl.atoms.is_lazy(x));
    assert!(cdcl.init());

    let fixed = lit(cdcl.atoms.eq(x, 64));
    assert_eq!(cdcl.tvalue(fixed), Tri::True);
    assert_eq!(cdcl.level_of(fixed.atom()), 0);
    assert!(matches!(cdcl.reason_of(fixed.atom()), Reason::Fact));
}
