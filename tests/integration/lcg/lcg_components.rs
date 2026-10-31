use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use qayd::constraints::linear::{linear, Relation};
use qayd::lcg::lit::{AtomKind, AtomTable, Lit, LitOrConst};
use qayd::lcg::view::{apply, atom_value, lit_value, Tri};
use qayd::{count_solutions, first_solution, first_solution_assuming, optimize_var_assuming, Assumption, Solver, Store};

static NEVER_STOP: AtomicBool = AtomicBool::new(false);

fn atoms_for(store: &Store) -> AtomTable {
    AtomTable::build(store.num_vars(), |var| (store.min(var), store.max(var)))
}

fn lit(loc: LitOrConst) -> Lit {
    match loc {
        LitOrConst::Lit(lit) => lit,
        _ => panic!("expected a real literal, got {loc:?}"),
    }
}

#[test]
fn lit_packing_roundtrips() {
    for atom in [0u32, 1, 7, 1000] {
        let positive = Lit::positive(atom);
        let negative = Lit::negative(atom);
        assert_eq!(positive.atom(), atom);
        assert_eq!(negative.atom(), atom);
        assert!(positive.is_positive());
        assert!(!negative.is_positive());
        assert_eq!(positive.negate(), negative);
        assert_eq!(negative.negate(), positive);
        assert_eq!(Lit::from_code(positive.code()), positive);
    }
}

#[test]
fn atom_table_lookups_roundtrip_and_fold_constants() {
    let mut store = Store::new();
    let x = store.new_var_range(0, 2);
    let y = store.new_var_range(5, 7);
    let fixed = store.new_var_range(4, 4);
    let atoms = atoms_for(&store);

    assert_eq!(atoms.ge(x, 0), LitOrConst::True);
    assert_eq!(atoms.ge(x, 3), LitOrConst::False);
    assert_eq!(atoms.eq(x, -1), LitOrConst::False);
    assert_eq!(atoms.eq(x, 3), LitOrConst::False);
    assert_eq!(atoms.ge(fixed, 4), LitOrConst::True);
    assert_eq!(atoms.ge(fixed, 5), LitOrConst::False);

    for (var, lo, hi) in [(x, 0, 2), (y, 5, 7)] {
        for k in (lo + 1)..=hi {
            let order = lit(atoms.ge(var, k));
            assert_eq!(atoms.decode(order.atom()), AtomKind::Ge { var, k });
        }
        for v in lo..=hi {
            let equality = lit(atoms.eq(var, v));
            assert_eq!(atoms.decode(equality.atom()), AtomKind::Eq { var, v });
        }
    }
}

#[test]
fn atom_values_track_domain_mutations() {
    let mut store = Store::new();
    let x = store.new_var_range(0, 3);
    let ge2 = AtomKind::Ge { var: x, k: 2 };
    let eq2 = AtomKind::Eq { var: x, v: 2 };

    assert_eq!(atom_value(&store, ge2), Tri::Unknown);
    assert_eq!(atom_value(&store, eq2), Tri::Unknown);
    store.remove_below(x, 2).unwrap();
    assert_eq!(atom_value(&store, ge2), Tri::True);
    store.remove(x, 3).unwrap();
    assert_eq!(atom_value(&store, eq2), Tri::True);
}

#[test]
fn apply_and_signed_lit_value_roundtrip() {
    let mut store = Store::new();
    let x = store.new_var_range(0, 9);
    let atoms = atoms_for(&store);

    let ge4 = lit(atoms.ge(x, 4));
    apply(&mut store, &atoms, ge4).unwrap();
    assert_eq!(store.min(x), 4);
    assert_eq!(lit_value(&store, &atoms, ge4), Tri::True);
    assert_eq!(lit_value(&store, &atoms, ge4.negate()), Tri::False);

    apply(&mut store, &atoms, lit(atoms.ge(x, 8)).negate()).unwrap();
    assert_eq!(store.max(x), 7);
    apply(&mut store, &atoms, lit(atoms.eq(x, 5)).negate()).unwrap();
    assert!(!store.contains(x, 5));
    apply(&mut store, &atoms, lit(atoms.eq(x, 6))).unwrap();
    assert_eq!(store.value(x), 6);
}

#[test]
fn sparse_atom_table_canonicalizes_numeric_gaps() {
    let mut store = Store::new();
    let support: Arc<[i32]> = Arc::from([i32::MIN, 0, i32::MAX]);
    let x = store.new_var_set(&support);
    let atoms = AtomTable::build_active_sparse(
        store.num_vars(),
        |_| true,
        |_| false,
        |var| (var == x).then(|| Arc::clone(&support)),
        |var| (store.min(var), store.max(var)),
    );

    assert_eq!(atoms.num_atoms(), 5);
    assert_eq!(atoms.ge(x, i32::MIN), LitOrConst::True);
    assert_eq!(atoms.ge(x, 1), atoms.ge(x, i32::MAX));
    assert_eq!(atoms.ge_i64(x, i64::from(i32::MAX) + 1), LitOrConst::False);
    assert_eq!(atoms.eq(x, 1), LitOrConst::False);
    let ge_zero = lit(atoms.ge(x, -1));
    assert_eq!(atoms.decode(ge_zero.atom()), AtomKind::Ge { var: x, k: 0 });
}

#[test]
fn lcg_enumerates_sparse_domain_without_span_expansion() {
    let mut solver = Solver::new();
    let x = solver.new_var_set(&[i32::MIN, 0, i32::MAX]);
    assert_eq!(count_solutions(&mut solver, &[x]), 3);
}

#[test]
fn lcg_solves_active_wide_range_with_lazy_atoms() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 20_000_001);
    assert_eq!(first_solution(&mut solver, &[x]), Some(vec![0]));
}

#[test]
fn first_solution_respects_assumptions() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    let y = solver.new_var_range(0, 1);
    linear(&mut solver, &[1, 1], &[x, y], Relation::Le, 1);

    let (solution, _, complete) = first_solution_assuming(&mut solver, &[x, y], &[Assumption::eq(x, 1)]);

    assert!(complete);
    assert_eq!(solution, Some(vec![1, 0]));
}

#[test]
fn contradictory_assumptions_are_unsat() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);

    let (solution, _, complete) = first_solution_assuming(&mut solver, &[x], &[Assumption::eq(x, 0), Assumption::eq(x, 1)]);

    assert!(complete);
    assert!(solution.is_none());
}

#[test]
fn optimization_respects_assumptions() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 5);
    let y = solver.new_var_range(0, 5);
    linear(&mut solver, &[1, 1], &[x, y], Relation::Ge, 5);

    let (best, _, complete) = optimize_var_assuming(&mut solver, &[x, y], &[Assumption::eq(x, 2)], y, true, &NEVER_STOP, |_, _| {});

    assert!(complete);
    assert_eq!(best, Some((vec![2, 3], 3)));
}
