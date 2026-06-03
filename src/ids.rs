//! Integer-id newtypes: `u32` wrappers for variables and propagators.

/// Identifies a variable owned by the [`Store`](crate::store::Store).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct VarId(pub u32);

/// Identifies a propagator owned by the [`Solver`](crate::store::Solver).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct PropId(pub u32);

impl VarId {
    /// The id as a `usize` index.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl PropId {
    /// The id as a `usize` index.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}
