//! Boolean atom numbering and literal packing for the LCG layer.
//!
//! Per variable `x` over `[l, u]`: order atoms `[x ≥ k]` (`k ∈ (l, u]`) and
//! equality atoms `[x = v]` (`v ∈ [l, u]`). Endpoints fold to constants. Atom
//! truth is a view over the domain (see [`view`](crate::lcg::view)); this module
//! owns only the naming.

use crate::ids::VarId;

/// A Boolean atom, identified by a flat index assigned by [`AtomTable`].
pub type Atom = u32;

/// A Boolean literal: an [`Atom`] with a sign, packed as `atom << 1 | sign`
/// (`sign == 0` positive). `lit` and `¬lit` are adjacent, so `code()` keys watch arrays.
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
/// Built once from initial domains over the full span `[min, max]`; holes still
/// get (permanently false) equality atoms.
pub struct AtomTable {
    vars: Vec<VarAtoms>,
    decode: Vec<AtomKind>,
}

impl AtomTable {
    /// Allocate atoms for every variable; `bounds(var) == (min, max)` of its
    /// starting domain.
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
