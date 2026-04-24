//! Boolean atoms and literals over integer variables (the LCG literal layer).
//!
//! Each integer variable `x` with initial domain `[l, u]` is given two eager
//! families of Boolean atoms:
//!
//! - **order atoms** `[x ≥ k]` for `k ∈ (l, u]` — bounds,
//! - **equality atoms** `[x = v]` for `v ∈ [l, u]` — membership (holes).
//!
//! The endpoints fold to constants: `[x ≥ l] ≡ ⊤` and `[x ≥ u+1] ≡ ⊥`, and an
//! equality outside `[l, u]` is `⊥`. Atoms are never *stored* as truth values —
//! their truth is a pure view over the trailed [`Domain`](crate::domain::Domain)
//! (see [`view`](crate::lcg::view)) — this module only owns the *naming*: a flat,
//! contiguous numbering of atoms and the maps in both directions.
//!
//! A [`Lit`] is an atom plus a sign, packed into a `u32` (atom in the high bits,
//! sign in bit 0) so it can index watch lists directly.

use crate::ids::VarId;

/// A Boolean atom, identified by a flat index assigned by [`AtomTable`].
pub type Atom = u32;

/// A Boolean literal: an [`Atom`] with a sign. Positive means "the atom holds".
///
/// Packed as `atom << 1 | sign`, with `sign == 0` for positive. The packing
/// keeps `lit` and `¬lit` adjacent, so `code()` is a dense key for watch arrays.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Lit(u32);

impl Lit {
    /// The literal asserting `atom` with the given polarity.
    #[inline]
    pub fn new(atom: Atom, positive: bool) -> Lit {
        Lit((atom << 1) | (!positive as u32))
    }
    /// The positive literal of `atom` (the atom is true).
    #[inline]
    pub fn positive(atom: Atom) -> Lit {
        Lit::new(atom, true)
    }
    /// The negative literal of `atom` (the atom is false).
    #[inline]
    pub fn negative(atom: Atom) -> Lit {
        Lit::new(atom, false)
    }
    /// The underlying atom.
    #[inline]
    pub fn atom(self) -> Atom {
        self.0 >> 1
    }
    /// Whether this is the positive literal.
    #[inline]
    pub fn is_positive(self) -> bool {
        self.0 & 1 == 0
    }
    /// The opposite literal (same atom, flipped sign).
    #[inline]
    pub fn negate(self) -> Lit {
        Lit(self.0 ^ 1)
    }
    /// The packed code, a dense `u32` key (e.g. for watch-list indexing).
    #[inline]
    pub fn code(self) -> u32 {
        self.0
    }
    /// Rebuild a literal from its [`code`](Lit::code).
    #[inline]
    pub fn from_code(code: u32) -> Lit {
        Lit(code)
    }
}

/// What an [`Atom`] means, recovered from its index.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AtomKind {
    /// `[var ≥ k]`.
    Ge { var: VarId, k: i32 },
    /// `[var = v]`.
    Eq { var: VarId, v: i32 },
}

/// The result of asking for a literal that may fold to a constant.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LitOrConst {
    /// A real literal.
    Lit(Lit),
    /// Folded to `⊤` (e.g. `[x ≥ l]`).
    True,
    /// Folded to `⊥` (e.g. `[x ≥ u+1]`, or an out-of-range equality).
    False,
}

/// Per-variable atom ranges (contiguous in the flat numbering).
struct VarAtoms {
    lo: i32,
    hi: i32,
    /// Atom of `[x ≥ lo+1]`; the order atoms run `ge_base .. ge_base+(hi-lo)`.
    ge_base: Atom,
    /// Atom of `[x = lo]`; the equality atoms run `eq_base .. eq_base+(hi-lo+1)`.
    eq_base: Atom,
}

/// Owns the atom numbering: var → atom ranges, and atom → [`AtomKind`].
///
/// Built once from the store's *initial* domains, before search. Atoms are
/// allocated over the full integer span `[min, max]` of each variable (holes in
/// a `new_var_set` domain still get equality atoms — permanently false from the
/// start, which the view reports correctly via `contains`).
pub struct AtomTable {
    vars: Vec<VarAtoms>,
    decode: Vec<AtomKind>,
}

impl AtomTable {
    /// Allocate atoms for every variable, reading initial bounds from `bounds`
    /// (`bounds(var) == (min, max)` of the variable's starting domain).
    pub fn build(num_vars: usize, bounds: impl Fn(VarId) -> (i32, i32)) -> AtomTable {
        let mut vars = Vec::with_capacity(num_vars);
        let mut decode = Vec::new();
        for i in 0..num_vars {
            let var = VarId(i as u32);
            let (lo, hi) = bounds(var);
            debug_assert!(lo <= hi, "variable {var:?} has an empty initial domain");
            let ge_base = decode.len() as Atom;
            for k in (lo + 1)..=hi {
                decode.push(AtomKind::Ge { var, k });
            }
            let eq_base = decode.len() as Atom;
            for v in lo..=hi {
                decode.push(AtomKind::Eq { var, v });
            }
            vars.push(VarAtoms {
                lo,
                hi,
                ge_base,
                eq_base,
            });
        }
        AtomTable { vars, decode }
    }

    /// Total number of atoms allocated.
    #[inline]
    pub fn num_atoms(&self) -> usize {
        self.decode.len()
    }

    /// What `atom` means.
    #[inline]
    pub fn decode(&self, atom: Atom) -> AtomKind {
        self.decode[atom as usize]
    }

    /// The initial integer span `[lo, hi]` a variable's atoms cover.
    #[inline]
    pub fn var_span(&self, x: VarId) -> (i32, i32) {
        let a = &self.vars[x.index()];
        (a.lo, a.hi)
    }

    /// The variable an atom constrains.
    #[inline]
    pub fn var_of(&self, atom: Atom) -> VarId {
        match self.decode[atom as usize] {
            AtomKind::Ge { var, .. } | AtomKind::Eq { var, .. } => var,
        }
    }

    /// The literal `[x ≥ k]`, folding the constant endpoints.
    pub fn ge(&self, x: VarId, k: i32) -> LitOrConst {
        let a = &self.vars[x.index()];
        if k <= a.lo {
            LitOrConst::True
        } else if k > a.hi {
            LitOrConst::False
        } else {
            LitOrConst::Lit(Lit::positive(a.ge_base + (k - (a.lo + 1)) as u32))
        }
    }

    /// The literal `[x = v]`, folding to `⊥` outside the initial span.
    pub fn eq(&self, x: VarId, v: i32) -> LitOrConst {
        let a = &self.vars[x.index()];
        if v < a.lo || v > a.hi {
            LitOrConst::False
        } else {
            LitOrConst::Lit(Lit::positive(a.eq_base + (v - a.lo) as u32))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;

    fn atoms_for(store: &Store) -> AtomTable {
        AtomTable::build(store.num_vars(), |v| (store.min(v), store.max(v)))
    }

    #[test]
    fn lit_packing_roundtrips() {
        for atom in [0u32, 1, 7, 1000] {
            let p = Lit::positive(atom);
            let n = Lit::negative(atom);
            assert_eq!(p.atom(), atom);
            assert_eq!(n.atom(), atom);
            assert!(p.is_positive());
            assert!(!n.is_positive());
            assert_eq!(p.negate(), n);
            assert_eq!(n.negate(), p);
            assert_eq!(Lit::from_code(p.code()), p);
        }
    }

    #[test]
    fn order_atoms_fold_at_endpoints() {
        let mut store = Store::new();
        let x = store.new_var_range(2, 6); // [2, 6]
        let atoms = atoms_for(&store);

        assert_eq!(atoms.ge(x, 2), LitOrConst::True); // [x >= 2] always holds
        assert_eq!(atoms.ge(x, 1), LitOrConst::True);
        assert_eq!(atoms.ge(x, 7), LitOrConst::False); // [x >= 7] impossible
        assert_eq!(atoms.ge(x, 100), LitOrConst::False);
        // Interior order atoms are real and distinct.
        for k in 3..=6 {
            assert!(matches!(atoms.ge(x, k), LitOrConst::Lit(_)));
        }
    }

    #[test]
    fn equality_atoms_fold_outside_span() {
        let mut store = Store::new();
        let x = store.new_var_range(0, 3);
        let atoms = atoms_for(&store);
        assert_eq!(atoms.eq(x, -1), LitOrConst::False);
        assert_eq!(atoms.eq(x, 4), LitOrConst::False);
        for v in 0..=3 {
            assert!(matches!(atoms.eq(x, v), LitOrConst::Lit(_)));
        }
    }

    #[test]
    fn decode_inverts_lookup() {
        let mut store = Store::new();
        let x = store.new_var_range(0, 2);
        let y = store.new_var_range(5, 7);
        let atoms = atoms_for(&store);

        // Every real (var, kind) round-trips through its atom.
        for (var, lo, hi) in [(x, 0, 2), (y, 5, 7)] {
            for k in (lo + 1)..=hi {
                let LitOrConst::Lit(l) = atoms.ge(var, k) else {
                    panic!("expected literal")
                };
                assert_eq!(atoms.decode(l.atom()), AtomKind::Ge { var, k });
            }
            for v in lo..=hi {
                let LitOrConst::Lit(l) = atoms.eq(var, v) else {
                    panic!("expected literal")
                };
                assert_eq!(atoms.decode(l.atom()), AtomKind::Eq { var, v });
            }
        }
    }

    #[test]
    fn fixed_variable_has_no_order_atoms() {
        let mut store = Store::new();
        let x = store.new_var_range(4, 4); // already fixed
        let atoms = atoms_for(&store);
        assert_eq!(atoms.ge(x, 4), LitOrConst::True);
        assert_eq!(atoms.ge(x, 5), LitOrConst::False);
        assert!(matches!(atoms.eq(x, 4), LitOrConst::Lit(_)));
        assert_eq!(atoms.num_atoms(), 1); // exactly [x = 4]
    }
}
