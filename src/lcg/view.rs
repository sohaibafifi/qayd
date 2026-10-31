//! Bridge between Boolean atoms and the integer domain.
//!
//! Atom truth is not stored; it is a pure function of the trailed domain, so
//! backtracking the trail restores every atom's truth for free. [`apply`] is
//! the inverse: it pushes a literal into the domain.

use crate::ids::VarId;
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

/// One immutable domain view for a batch of atoms belonging to one variable.
/// Keeping the store borrowed prevents these cached bounds from outliving a
/// domain mutation. Interior equality holes still consult the actual support.
pub(crate) struct VariableDomainView<'a> {
    store: &'a Store,
    var: VarId,
    min: i32,
    max: i32,
    size: usize,
}

impl<'a> VariableDomainView<'a> {
    pub(crate) fn new(store: &'a Store, var: VarId) -> Self {
        Self { store, var, min: store.min(var), max: store.max(var), size: store.size(var) }
    }

    #[inline]
    pub(crate) fn atom_value(&self, kind: AtomKind) -> Tri {
        match kind {
            AtomKind::Ge { var, k } => {
                debug_assert_eq!(var, self.var);
                if self.min >= k {
                    Tri::True
                } else if self.max < k {
                    Tri::False
                } else {
                    Tri::Unknown
                }
            }
            AtomKind::Eq { var, v } => {
                debug_assert_eq!(var, self.var);
                if self.size == 0 || v < self.min || v > self.max {
                    Tri::False
                } else if self.size == 1 {
                    Tri::True
                } else if self.store.contains(var, v) {
                    Tri::Unknown
                } else {
                    Tri::False
                }
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
