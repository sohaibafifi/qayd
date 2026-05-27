//! Boolean atom numbering and literal packing for the LCG layer.
//!
//! Per variable `x` over `[l, u]`: order atoms `[x ≥ k]` (`k ∈ (l, u]`) and
//! equality atoms `[x = v]` (`v ∈ [l, u]`). Sparse domains allocate atoms only
//! for root-supported values. Wide domains allocate atoms on demand. Endpoints
//! fold to constants. Atom truth is a view over the domain (see
//! [`view`](crate::lcg::view)); this module owns only the naming.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::ids::VarId;

const MAX_EAGER_ATOMS: usize = 20_000_000;

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
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
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
#[derive(Clone, Copy, PartialEq, Eq)]
enum VarLayout {
    Span,
    Sparse,
    Sign,
    Lazy,
}

struct VarAtoms {
    lo: i32,
    hi: i32,
    active: bool,
    layout: VarLayout,
    /// Atom of `[x ≥ lo+1]`; the order atoms run `ge_base .. ge_base+(hi-lo)`.
    ge_base: Atom,
    /// Atom of `[x = lo]`; the equality atoms run `eq_base .. eq_base+(hi-lo+1)`.
    eq_base: Atom,
    /// First atom after this variable's contiguous atom range.
    end: Atom,
    /// Sorted root support for [`VarLayout::Sparse`].
    sparse_values: Option<Arc<[i32]>>,
}

/// Owns the atom numbering: var → atom ranges, and atom → [`AtomKind`].
///
/// Compact ranges use their full span, sparse domains use root support, and
/// wide ranges use shared on-demand atoms.
pub struct AtomTable {
    vars: Vec<VarAtoms>,
    decode: Vec<AtomKind>,
    lazy: Arc<LazyAtomRegistry>,
}

#[derive(Default)]
struct LazyAtoms {
    base: Option<Atom>,
    ids: HashMap<AtomKind, Atom>,
    decode: Vec<AtomKind>,
    by_var: HashMap<VarId, Vec<Atom>>,
}

/// Shared numbering for atoms created on demand by wide contiguous domains.
#[derive(Default)]
pub(crate) struct LazyAtomRegistry {
    inner: Mutex<LazyAtoms>,
}

impl LazyAtomRegistry {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn set_base(&self, base: Atom) {
        let mut lazy = self.inner.lock().unwrap();
        match lazy.base {
            Some(old) => assert_eq!(old, base, "inconsistent eager LCG atom layout"),
            None => lazy.base = Some(base),
        }
    }

    fn intern(&self, kind: AtomKind) -> Atom {
        let mut lazy = self.inner.lock().unwrap();
        if let Some(&atom) = lazy.ids.get(&kind) {
            return atom;
        }
        let base = lazy.base.expect("lazy LCG atom registry has no eager base");
        let atom = base
            .checked_add(Atom::try_from(lazy.decode.len()).expect("LCG atom count overflow"))
            .expect("LCG atom count overflow");
        assert!(atom <= u32::MAX >> 1, "LCG literal count overflow");
        lazy.ids.insert(kind, atom);
        lazy.decode.push(kind);
        lazy.by_var.entry(kind.var()).or_default().push(atom);
        atom
    }

    fn decode(&self, atom: Atom) -> AtomKind {
        let lazy = self.inner.lock().unwrap();
        let base = lazy.base.expect("lazy LCG atom registry has no eager base");
        lazy.decode[(atom - base) as usize]
    }

    fn append_atoms(&self, var: VarId, out: &mut Vec<Atom>) {
        if let Some(atoms) = self.inner.lock().unwrap().by_var.get(&var) {
            out.extend_from_slice(atoms);
        }
    }

    fn len(&self) -> usize {
        self.inner.lock().unwrap().decode.len()
    }
}

impl AtomKind {
    fn var(self) -> VarId {
        match self {
            AtomKind::Ge { var, .. } | AtomKind::Eq { var, .. } => var,
        }
    }
}

impl AtomTable {
    /// Allocate atoms for every variable; `bounds(var) == (min, max)` of its
    /// starting domain.
    pub fn build(num_vars: usize, bounds: impl Fn(VarId) -> (i32, i32)) -> AtomTable {
        Self::build_active(num_vars, |_| true, |_| false, bounds)
    }

    /// Allocate atoms only for selected variables.
    pub fn build_active(
        num_vars: usize,
        active: impl Fn(VarId) -> bool,
        sign: impl Fn(VarId) -> bool,
        bounds: impl Fn(VarId) -> (i32, i32),
    ) -> AtomTable {
        Self::build_active_sparse(num_vars, active, sign, |_| None, bounds)
    }

    /// Allocate atoms only for selected variables, using root support for sparse
    /// domains. Sparse order literals are canonicalized to their next supported
    /// threshold.
    pub fn build_active_sparse(
        num_vars: usize,
        active: impl Fn(VarId) -> bool,
        sign: impl Fn(VarId) -> bool,
        sparse_values: impl Fn(VarId) -> Option<Arc<[i32]>>,
        bounds: impl Fn(VarId) -> (i32, i32),
    ) -> AtomTable {
        Self::build_active_sparse_with_registry(
            num_vars,
            active,
            sign,
            sparse_values,
            bounds,
            LazyAtomRegistry::new(),
        )
    }

    pub(crate) fn build_active_sparse_with_registry(
        num_vars: usize,
        active: impl Fn(VarId) -> bool,
        sign: impl Fn(VarId) -> bool,
        sparse_values: impl Fn(VarId) -> Option<Arc<[i32]>>,
        bounds: impl Fn(VarId) -> (i32, i32),
        lazy: Arc<LazyAtomRegistry>,
    ) -> AtomTable {
        let mut vars = Vec::with_capacity(num_vars);
        let mut decode = Vec::new();
        for i in 0..num_vars {
            let var = VarId(i as u32);
            let (lo, hi) = bounds(var);
            debug_assert!(lo <= hi, "variable {var:?} has an empty initial domain");
            let ge_base = decode.len() as Atom;
            let active = active(var);
            let sparse_values = sparse_values(var);
            let layout = if sign(var) {
                VarLayout::Sign
            } else if sparse_values.is_some() {
                VarLayout::Sparse
            } else if active
                && decode
                    .len()
                    .saturating_add(atom_count(VarLayout::Span, lo, hi, &None))
                    > MAX_EAGER_ATOMS
            {
                VarLayout::Lazy
            } else {
                VarLayout::Span
            };
            if active {
                reserve_atoms(&mut decode, var, atom_count(layout, lo, hi, &sparse_values));
                match layout {
                    VarLayout::Span => {
                        for k in (i64::from(lo) + 1)..=i64::from(hi) {
                            decode.push(AtomKind::Ge { var, k: k as i32 });
                        }
                    }
                    VarLayout::Sparse => {
                        for &k in sparse_values.as_deref().unwrap().iter().skip(1) {
                            decode.push(AtomKind::Ge { var, k });
                        }
                    }
                    VarLayout::Sign => decode.push(AtomKind::Ge { var, k: 1 }),
                    VarLayout::Lazy => {}
                }
            }
            let eq_base = decode.len() as Atom;
            if active {
                match layout {
                    VarLayout::Span => {
                        for v in i64::from(lo)..=i64::from(hi) {
                            decode.push(AtomKind::Eq { var, v: v as i32 });
                        }
                    }
                    VarLayout::Sparse => {
                        for &v in sparse_values.as_deref().unwrap() {
                            decode.push(AtomKind::Eq { var, v });
                        }
                    }
                    VarLayout::Sign => {}
                    VarLayout::Lazy => {}
                }
            }
            vars.push(VarAtoms {
                lo,
                hi,
                active,
                layout,
                ge_base,
                eq_base,
                end: decode.len() as Atom,
                sparse_values,
            });
        }
        lazy.set_base(decode.len() as Atom);
        AtomTable { vars, decode, lazy }
    }

    /// Total number of atoms allocated.
    #[inline]
    pub fn num_atoms(&self) -> usize {
        self.decode.len() + self.lazy.len()
    }

    /// What `atom` means.
    #[inline]
    pub fn decode(&self, atom: Atom) -> AtomKind {
        self.decode
            .get(atom as usize)
            .copied()
            .unwrap_or_else(|| self.lazy.decode(atom))
    }

    /// The initial integer span `[lo, hi]` a variable's atoms cover.
    #[inline]
    pub fn var_span(&self, x: VarId) -> (i32, i32) {
        let a = &self.vars[x.index()];
        (a.lo, a.hi)
    }

    /// Append the atoms allocated for `x`.
    #[inline]
    pub(crate) fn append_atoms(&self, x: VarId, out: &mut Vec<Atom>) {
        let a = &self.vars[x.index()];
        if a.layout == VarLayout::Lazy {
            self.lazy.append_atoms(x, out);
        } else {
            out.extend(a.ge_base..a.end);
        }
    }

    /// Whether `x ∈ {-1, 1}` uses one signed order atom.
    #[inline]
    pub fn is_sign(&self, x: VarId) -> bool {
        self.vars[x.index()].layout == VarLayout::Sign
    }

    /// Whether `x` creates atoms on demand.
    #[inline]
    pub(crate) fn is_lazy(&self, x: VarId) -> bool {
        self.vars[x.index()].layout == VarLayout::Lazy
    }

    /// The variable an atom constrains.
    #[inline]
    pub fn var_of(&self, atom: Atom) -> VarId {
        self.decode(atom).var()
    }

    /// The literal `[x ≥ k]`, folding the constant endpoints.
    pub fn ge(&self, x: VarId, k: i32) -> LitOrConst {
        self.ge_i64(x, i64::from(k))
    }

    /// The literal `[x ≥ k]`, accepting one-past-`i32` bounds for constant
    /// folding.
    pub fn ge_i64(&self, x: VarId, k: i64) -> LitOrConst {
        let a = &self.vars[x.index()];
        assert!(a.active, "LCG literal requested for inactive {x:?}");
        match a.layout {
            VarLayout::Sign => {
                if k <= -1 {
                    LitOrConst::True
                } else if k > 1 {
                    LitOrConst::False
                } else {
                    LitOrConst::Lit(Lit::positive(a.ge_base))
                }
            }
            VarLayout::Span => {
                if k <= i64::from(a.lo) {
                    LitOrConst::True
                } else if k > i64::from(a.hi) {
                    LitOrConst::False
                } else {
                    let offset = k - (i64::from(a.lo) + 1);
                    LitOrConst::Lit(Lit::positive(a.ge_base + offset as u32))
                }
            }
            VarLayout::Sparse => {
                let values = a.sparse_values.as_deref().unwrap();
                let i = values.partition_point(|&v| i64::from(v) < k);
                if i == 0 {
                    LitOrConst::True
                } else if i == values.len() {
                    LitOrConst::False
                } else {
                    LitOrConst::Lit(Lit::positive(a.ge_base + (i - 1) as u32))
                }
            }
            VarLayout::Lazy => {
                if k <= i64::from(a.lo) {
                    LitOrConst::True
                } else if k > i64::from(a.hi) {
                    LitOrConst::False
                } else {
                    LitOrConst::Lit(Lit::positive(self.lazy.intern(AtomKind::Ge {
                        var: x,
                        k: k as i32,
                    })))
                }
            }
        }
    }

    /// The literal `[x = v]`, folding to `⊥` outside the initial support.
    pub fn eq(&self, x: VarId, v: i32) -> LitOrConst {
        let a = &self.vars[x.index()];
        assert!(a.active, "LCG literal requested for inactive {x:?}");
        match a.layout {
            VarLayout::Sign => match v {
                -1 => LitOrConst::Lit(Lit::negative(a.ge_base)),
                1 => LitOrConst::Lit(Lit::positive(a.ge_base)),
                _ => LitOrConst::False,
            },
            VarLayout::Span => {
                if v < a.lo || v > a.hi {
                    LitOrConst::False
                } else {
                    let offset = i64::from(v) - i64::from(a.lo);
                    LitOrConst::Lit(Lit::positive(a.eq_base + offset as u32))
                }
            }
            VarLayout::Sparse => match a.sparse_values.as_deref().unwrap().binary_search(&v) {
                Ok(i) => LitOrConst::Lit(Lit::positive(a.eq_base + i as u32)),
                Err(_) => LitOrConst::False,
            },
            VarLayout::Lazy => {
                if v < a.lo || v > a.hi {
                    LitOrConst::False
                } else {
                    LitOrConst::Lit(Lit::positive(self.lazy.intern(AtomKind::Eq { var: x, v })))
                }
            }
        }
    }
}

fn atom_count(layout: VarLayout, lo: i32, hi: i32, sparse_values: &Option<Arc<[i32]>>) -> usize {
    match layout {
        VarLayout::Span => span_len(lo, hi)
            .checked_mul(2)
            .and_then(|n| n.checked_sub(1))
            .expect("LCG atom count overflow"),
        VarLayout::Sparse => sparse_values
            .as_deref()
            .unwrap()
            .len()
            .checked_mul(2)
            .and_then(|n| n.checked_sub(1))
            .expect("LCG atom count overflow"),
        VarLayout::Sign => 1,
        VarLayout::Lazy => 0,
    }
}

fn reserve_atoms(decode: &mut Vec<AtomKind>, var: VarId, additional: usize) {
    let total = decode
        .len()
        .checked_add(additional)
        .expect("LCG atom count overflow");
    assert!(
        total <= MAX_EAGER_ATOMS,
        "eager LCG allocation exceeds {MAX_EAGER_ATOMS} atoms at {var:?}; wide contiguous ranges require lazy atoms"
    );
    decode.reserve(additional);
}

fn span_len(lo: i32, hi: i32) -> usize {
    (i64::from(hi) - i64::from(lo) + 1) as usize
}
