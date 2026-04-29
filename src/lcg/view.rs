//! Bridge between Boolean atoms and the integer domain.
//!
//! Atom truth is not stored — it is a pure function of the trailed domain, so
//! backtracking the trail restores every atom's truth for free. [`apply`] is
//! the inverse: it pushes a literal into the domain.

use crate::lcg::lit::{AtomKind, AtomTable, Lit};
use crate::propagator::Inconsistency;
use crate::store::Store;

/// A three-valued truth: assigned true/false, or still open.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tri {
    True,
    False,
    Unknown,
}

impl Tri {
    /// Swap `True`/`False`; `Unknown` is its own negation.
    #[inline]
    pub fn negate(self) -> Tri {
        match self {
            Tri::True => Tri::False,
            Tri::False => Tri::True,
            Tri::Unknown => Tri::Unknown,
        }
    }
}

/// The truth of `kind` under the current domain.
#[inline]
pub fn atom_value(store: &Store, kind: AtomKind) -> Tri {
    match kind {
        AtomKind::Ge { var, k } => {
            if store.min(var) >= k {
                Tri::True
            } else if store.max(var) < k {
                Tri::False
            } else {
                Tri::Unknown
            }
        }
        AtomKind::Eq { var, v } => {
            if !store.contains(var, v) {
                Tri::False
            } else if store.is_fixed(var) {
                Tri::True
            } else {
                Tri::Unknown
            }
        }
    }
}

/// The truth of `lit` under the current domain (the atom's value, signed).
#[inline]
pub fn lit_value(store: &Store, atoms: &AtomTable, lit: Lit) -> Tri {
    let t = atom_value(store, atoms.decode(lit.atom()));
    if lit.is_positive() {
        t
    } else {
        t.negate()
    }
}

/// Enforce `lit` on the domain (inverse of [`atom_value`]).
/// `Err(Inconsistency)` if the mutation empties the domain.
#[inline]
pub fn apply(store: &mut Store, atoms: &AtomTable, lit: Lit) -> Result<(), Inconsistency> {
    match atoms.decode(lit.atom()) {
        AtomKind::Ge { var, k } => {
            if lit.is_positive() {
                store.remove_below(var, k)
            } else {
                store.remove_above(var, k - 1)
            }
        }
        AtomKind::Eq { var, v } => {
            if lit.is_positive() {
                store.fix(var, v)
            } else {
                store.remove(var, v)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lcg::lit::LitOrConst;
    use crate::store::Store;

    fn atoms_for(store: &Store) -> AtomTable {
        AtomTable::build(store.num_vars(), |v| (store.min(v), store.max(v)))
    }

    fn lit(loc: LitOrConst) -> Lit {
        match loc {
            LitOrConst::Lit(l) => l,
            _ => panic!("expected a real literal, got {loc:?}"),
        }
    }

    #[test]
    fn order_atom_tracks_bounds() {
        let mut store = Store::new();
        let x = store.new_var_range(0, 9);
        let ge5 = AtomKind::Ge { var: x, k: 5 };

        assert_eq!(atom_value(&store, ge5), Tri::Unknown);
        store.remove_below(x, 5).unwrap(); // x >= 5
        assert_eq!(atom_value(&store, ge5), Tri::True);
        // Pushing the max below k makes [x >= k] false.
        store.remove_above(x, 6).unwrap(); // x in [5, 6]
        assert_eq!(
            atom_value(&store, AtomKind::Ge { var: x, k: 7 }),
            Tri::False
        );
    }

    #[test]
    fn equality_atom_true_only_when_fixed() {
        let mut store = Store::new();
        let x = store.new_var_range(0, 3);
        let eq2 = AtomKind::Eq { var: x, v: 2 };

        assert_eq!(atom_value(&store, eq2), Tri::Unknown);
        store.remove(x, 0).unwrap();
        store.remove(x, 1).unwrap();
        assert_eq!(atom_value(&store, eq2), Tri::Unknown); // {2,3} left
        store.remove(x, 3).unwrap();
        assert_eq!(atom_value(&store, eq2), Tri::True); // fixed to 2
    }

    #[test]
    fn equality_atom_false_when_removed() {
        let mut store = Store::new();
        let x = store.new_var_range(0, 3);
        store.remove(x, 2).unwrap();
        assert_eq!(
            atom_value(&store, AtomKind::Eq { var: x, v: 2 }),
            Tri::False
        );
    }

    #[test]
    fn apply_pushes_literal_into_domain() {
        let mut store = Store::new();
        let x = store.new_var_range(0, 9);
        let atoms = atoms_for(&store);

        // [x >= 4]
        apply(&mut store, &atoms, lit(atoms.ge(x, 4))).unwrap();
        assert_eq!(store.min(x), 4);
        // ¬[x >= 8]  =>  x <= 7
        apply(&mut store, &atoms, lit(atoms.ge(x, 8)).negate()).unwrap();
        assert_eq!(store.max(x), 7);
        // ¬[x = 5]  =>  remove 5
        apply(&mut store, &atoms, lit(atoms.eq(x, 5)).negate()).unwrap();
        assert!(!store.contains(x, 5));
        // [x = 6]  =>  fix
        apply(&mut store, &atoms, lit(atoms.eq(x, 6))).unwrap();
        assert!(store.is_fixed(x));
        assert_eq!(store.value(x), 6);
    }

    #[test]
    fn lit_value_is_signed_atom_value() {
        let mut store = Store::new();
        let x = store.new_var_range(0, 9);
        let atoms = atoms_for(&store);
        store.remove_below(x, 5).unwrap();
        let ge5 = lit(atoms.ge(x, 5));
        assert_eq!(lit_value(&store, &atoms, ge5), Tri::True);
        assert_eq!(lit_value(&store, &atoms, ge5.negate()), Tri::False);

        let ge7 = lit(atoms.ge(x, 7)); // unknown (domain [5,9])
        assert_eq!(lit_value(&store, &atoms, ge7), Tri::Unknown);
        assert_eq!(lit_value(&store, &atoms, ge7.negate()), Tri::Unknown);
    }
}
