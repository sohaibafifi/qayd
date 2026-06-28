//! Integer variable declarations.

/// Reference to an integer variable declaration inside [`Model`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct IntVarRef(pub usize);

/// Integer variable domain declaration.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum IntDomain {
    /// Contiguous integer range.
    Range { lo: i32, hi: i32 },
    /// Explicit finite set.
    Set(Vec<i32>),
}
