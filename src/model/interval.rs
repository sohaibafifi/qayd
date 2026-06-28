//! Interval variable declarations.

/// Reference to a structured interval declaration inside [`Model`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct IntervalVarRef(pub usize);

/// Structured interval declaration.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct IntervalDecl {
    /// Earliest start.
    pub start_min: i64,
    /// Latest start.
    pub start_max: i64,
    /// Fixed duration.
    pub duration: i64,
    /// Alternative execution modes. Empty means fixed duration and no machine.
    pub modes: Vec<IntervalMode>,
    /// Whether the interval may be absent.
    pub optional: bool,
}

/// Alternative interval execution mode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IntervalMode {
    /// Machine index.
    pub machine: usize,
    /// Mode duration.
    pub duration: i64,
}
