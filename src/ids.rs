//! Integer-id newtypes.
//!
//! The solver refers to variables, propagators, and value positions by id,
//! never by reference: central ownership, no cross-references. The ids are
//! plain `u32` wrappers, so the layout stays flat and cache-friendly.

/// Identifies a variable owned by the [`Store`](crate::store::Store).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct VarId(pub u32);

/// Identifies a propagator owned by the [`Solver`](crate::store::Solver).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct PropId(pub u32);

/// Identifies a value position (used by value-based globals from Phase 3 on).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct ValId(pub u32);

impl VarId {
    /// The id as a `usize` index into the solver's variable-indexed arrays.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl PropId {
    /// The id as a `usize` index into the solver's propagator-indexed arrays.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl ValId {
    /// The id as a `usize` index.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}
