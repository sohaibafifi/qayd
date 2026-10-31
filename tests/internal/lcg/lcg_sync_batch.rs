use std::sync::Arc;

use crate::constraints::linear::{linear, Relation};
use crate::ids::VarId;
use crate::lcg::lit::{AtomKind, AtomTable, Lit, LitOrConst};
use crate::lcg::trail::{Cdcl, Reason};
use crate::lcg::view::{self, Tri};
use crate::Solver;

fn lit(value: LitOrConst) -> Lit {
    match value {
        LitOrConst::Lit(value) => value,
        other => panic!("expected a real atom, got {other:?}"),
    }
}

/// The pre-optimization synchronization algorithm, kept outside production.
fn reference_sync(cdcl: &mut Cdcl<'_>, var: VarId) {
    if cdcl.atoms.is_sign(var) {
        let value = lit(cdcl.atoms.ge(var, 1));
        if !cdcl.is_assigned(value.atom()) {
            assert_eq!(view::lit_value(&cdcl.solver.store, &cdcl.atoms, value), Tri::Unknown);
        }
        return;
    }
    if cdcl.atoms.is_lazy(var) {
        let _ = cdcl.atoms.ge(var, cdcl.solver.store.min(var));
        let _ = cdcl.atoms.ge_i64(var, i64::from(cdcl.solver.store.max(var)) + 1);
    }
    let mut atoms = Vec::new();
    cdcl.atoms.append_atoms(var, &mut atoms);
    let mut todo = Vec::new();
    for &atom in &atoms {
        if !cdcl.is_assigned(atom) {
            let value = Lit::positive(atom);
            match view::lit_value(&cdcl.solver.store, &cdcl.atoms, value) {
                Tri::True => todo.push(value),
                Tri::False => todo.push(value.negate()),
                Tri::Unknown => {}
            }
        }
    }
    if todo.is_empty() {
        return;
    }
    let mut current_atoms = Vec::new();
    cdcl.atoms.append_atoms(var, &mut current_atoms);
    let reason: Arc<[Lit]> = Arc::from(
        current_atoms
            .into_iter()
            .filter_map(|atom| match cdcl.tvalue(Lit::positive(atom)) {
                Tri::True => Some(Lit::positive(atom)),
                Tri::False => Some(Lit::negative(atom)),
                Tri::Unknown => None,
            })
            .collect::<Vec<_>>(),
    );
    if let Some(handle) = cdcl.solver.explanation_handle() {
        handle.eager_lcg(reason.len());
    }
    for value in todo {
        cdcl.audit_record_sync(value, Reason::Generic(Arc::clone(&reason)));
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Why {
    Unset,
    Fact,
    Decision,
    Clause(String),
    Generic(Vec<Lit>),
}

fn snapshot(cdcl: &Cdcl<'_>) -> Vec<(u32, AtomKind, Tri, u32, Why)> {
    (0..cdcl.atoms.num_atoms() as u32)
        .map(|atom| {
            let truth = cdcl.tvalue(Lit::positive(atom));
            let reason = if cdcl.is_assigned(atom) {
                match cdcl.reason_of(atom) {
                    Reason::Unset => Why::Unset,
                    Reason::Fact => Why::Fact,
                    Reason::Decision => Why::Decision,
                    Reason::Clause(reference) => Why::Clause(format!("{reference:?}")),
                    Reason::Generic(reason) => Why::Generic(reason.to_vec()),
                    Reason::Deferred(_) => panic!("this fixture does not enable deferred explanations"),
                }
            } else {
                Why::Unset
            };
            (atom, cdcl.atoms.decode(atom), truth, cdcl.level_of(atom), reason)
        })
        .collect()
}

fn assert_same(left: &Cdcl<'_>, right: &Cdcl<'_>, vars: &[VarId]) {
    assert_eq!(left.audit_sync_trail(), right.audit_sync_trail(), "literal order must remain identical");
    assert_eq!(snapshot(left), snapshot(right), "truth, level and exact antecedent order must remain identical");
    for &var in vars {
        let mut a = left.solver.store.values(var).collect::<Vec<_>>();
        let mut b = right.solver.store.values(var).collect::<Vec<_>>();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
    }
}

fn fixture() -> (Solver, Vec<VarId>) {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 15);
    let y = solver.new_var_range(0, 5000);
    solver.store.remove(y, 42).unwrap();
    let sparse = (0..=300).map(|value| 10 * value).collect::<Vec<_>>();
    let z = solver.new_var_set(&sparse);
    let sign = solver.new_var_set(&[-1, 1]);
    linear(&mut solver, &[1, 1], &[x, y], Relation::Le, 5000);
    (solver, vec![x, y, z, sign])
}

fn allocate_atoms(cdcl: &Cdcl<'_>, vars: &[VarId]) {
    for value in [1, 50, 100, 1000, 2500, 4990, 5000] {
        let _ = cdcl.atoms.ge(vars[1], value);
        let _ = cdcl.atoms.eq(vars[1], value);
    }
    for value in [10, 20, 100, 1000, 2000, 3000] {
        let _ = cdcl.atoms.ge(vars[2], value);
        let _ = cdcl.atoms.eq(vars[2], value);
    }
}

#[test]
fn batched_sync_matches_reference_across_bounds_holes_new_atoms_and_backjumps() {
    let (mut left_solver, vars) = fixture();
    let (mut right_solver, right_vars) = fixture();
    assert_eq!(vars, right_vars);
    let mut left = Cdcl::new(&mut left_solver, &vars);
    let mut right = Cdcl::new(&mut right_solver, &vars);
    right.audit_set_sync_reference(reference_sync);
    allocate_atoms(&left, &vars);
    allocate_atoms(&right, &vars);
    assert!(left.init());
    assert!(right.init());
    assert_same(&left, &right, &vars);

    for (var, value, positive, equality) in
        [(vars[1], 1000, true, false), (vars[1], 2500, false, true), (vars[2], 1000, true, false), (vars[2], 2000, false, true)]
    {
        let literal = |cdcl: &Cdcl<'_>| {
            let atom = lit(if equality { cdcl.atoms.eq(var, value) } else { cdcl.atoms.ge(var, value) });
            if positive {
                atom
            } else {
                atom.negate()
            }
        };
        left.decide(literal(&left)).unwrap();
        right.decide(literal(&right)).unwrap();
        assert_eq!(left.propagate_and_learn(), right.propagate_and_learn());
        assert_same(&left, &right, &vars);
    }

    left.backjump_to(1);
    right.backjump_to(1);
    assert_same(&left, &right, &vars);
    // These atoms were not present during the first snapshot. Their IDs and
    // determination levels must still agree after the next real update.
    for cdcl in [&left, &right] {
        let _ = cdcl.atoms.ge(vars[1], 400);
        let _ = cdcl.atoms.eq(vars[1], 4999);
        let _ = cdcl.atoms.ge(vars[2], 700);
    }
    for (var, value, equality) in [(vars[1], 4990, false), (vars[0], 3, true), (vars[3], 1, false)] {
        let literal = |cdcl: &Cdcl<'_>| lit(if equality { cdcl.atoms.eq(var, value) } else { cdcl.atoms.ge(var, value) });
        left.decide(literal(&left)).unwrap();
        right.decide(literal(&right)).unwrap();
        assert_eq!(left.propagate_and_learn(), right.propagate_and_learn());
        assert_same(&left, &right, &vars);
    }
    left.backjump_to(0);
    right.backjump_to(0);
    assert_same(&left, &right, &vars);
}

#[test]
fn batched_sync_preserves_conflict_learning_and_backjump_reasons() {
    fn constrained() -> (Solver, Vec<VarId>) {
        let mut solver = Solver::new();
        let x = solver.new_var_range(0, 300);
        let y = solver.new_var_range(0, 300);
        linear(&mut solver, &[1, 1], &[x, y], Relation::Le, 300);
        (solver, vec![x, y])
    }
    let (mut left_solver, vars) = constrained();
    let (mut right_solver, _) = constrained();
    let mut left = Cdcl::new(&mut left_solver, &vars);
    let mut right = Cdcl::new(&mut right_solver, &vars);
    right.audit_set_sync_reference(reference_sync);
    for cdcl in [&left, &right] {
        for &var in &vars {
            for bound in [50, 100, 150, 200, 250] {
                let _ = cdcl.atoms.ge(var, bound);
                let _ = cdcl.atoms.eq(var, bound);
            }
        }
    }
    assert!(left.init());
    assert!(right.init());
    // Queue both decisions before FD propagation, forcing an actual conflict
    // and learned backjump rather than an immediate rejected domain update.
    for &var in &vars {
        left.decide(lit(left.atoms.ge(var, 200))).unwrap();
        right.decide(lit(right.atoms.ge(var, 200))).unwrap();
    }
    assert!(left.propagate_and_learn());
    assert!(right.propagate_and_learn());
    assert!(left.conflicts > 0);
    assert_eq!(left.conflicts, right.conflicts);
    assert_eq!(left.learned_lits, right.learned_lits);
    assert_same(&left, &right, &vars);
    left.backjump_to(0);
    right.backjump_to(0);
    left.decide(lit(left.atoms.eq(vars[0], 125))).unwrap();
    right.decide(lit(right.atoms.eq(vars[0], 125))).unwrap();
    assert_eq!(left.propagate_and_learn(), right.propagate_and_learn());
    assert_same(&left, &right, &vars);
}

#[test]
fn batched_atom_kinds_preserve_order_for_every_layout_and_later_interning() {
    let (mut solver, vars) = fixture();
    let eager_sparse = solver.new_var_set(&[0, 2, 8, 20]);
    let mut all = vars.clone();
    all.push(eager_sparse);
    let atoms = AtomTable::build_active_sparse(
        solver.store.num_vars(),
        |_| true,
        |var| var == vars[3],
        |var| solver.store.sparse_values(var),
        |var| (solver.store.min(var), solver.store.max(var)),
    );
    for round in 0..2 {
        for &var in &all {
            let _ = atoms.ge(var, solver.store.min(var) + 1);
            let _ = atoms.eq(var, solver.store.max(var));
            let mut ids = Vec::new();
            atoms.append_atoms(var, &mut ids);
            let expected = ids.into_iter().map(|id| (id, atoms.decode(id))).collect::<Vec<_>>();
            let mut batch = Vec::new();
            atoms.append_atom_kinds(var, &mut batch);
            assert_eq!(batch, expected, "round {round}, variable {var:?}");
        }
        let _ = atoms.ge(vars[1], 2500);
        let _ = atoms.eq(vars[2], 1500);
    }
}

#[test]
fn cached_domain_view_matches_original_for_holes_fixed_empty_and_restored_domains() {
    fn check(solver: &Solver, var: VarId) {
        let cached = view::VariableDomainView::new(&solver.store, var);
        for value in [-1, 0, 1, 2, 7, 8, 10, 20, 50, 1000, 3000, 5000, 5001] {
            for kind in [AtomKind::Eq { var, v: value }, AtomKind::Ge { var, k: value }] {
                assert_eq!(cached.atom_value(kind), view::atom_value(&solver.store, kind), "{kind:?}");
            }
        }
    }
    let (mut solver, vars) = fixture();
    for var in vars {
        check(&solver, var);
        solver.store.push_level();
        solver.store.remove(var, 7).unwrap();
        check(&solver, var);
        let value = solver.store.min(var);
        solver.store.fix(var, value).unwrap();
        check(&solver, var);
        assert!(solver.store.remove(var, value).is_err());
        check(&solver, var);
        solver.store.pop_level();
        check(&solver, var);
    }
}
