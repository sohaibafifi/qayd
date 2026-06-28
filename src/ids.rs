//! Integer-id newtypes: `u32` wrappers for solver arenas.

/// Identifies a variable owned by the [`Store`](crate::store::Store).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct VarId(pub u32);

/// Identifies a list domain owned by the [`Store`](crate::store::Store).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct ListId(pub u32);

/// Identifies a interval domain owned by the [`Store`](crate::store::Store).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct IntervalId(pub u32);

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

impl ListId {
    /// The id as a `usize` index.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl IntervalId {
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
