//! List domain over a fixed item universe.
//!
//! Membership is integer-backed: each item has a boolean variable (`1` = in the
//! list, `0` = excluded, unfixed = undecided), so the fact `item in list` is a
//! first-class atom the learning engine reasons over. The `Store` owns the
//! variables and the logic; this is the handle plus the trailed length bounds.

use crate::ids::VarId;

/// List domain event kinds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ListEvent {
    /// Possible membership changed (an item was excluded).
    PossibleChange,
    /// Required membership changed (an item was required).
    RequiredChange,
    /// Length bounds changed.
    LengthChange,
    /// Successor or predecessor arcs changed.
    ArcChange,
    /// Position information changed.
    PositionChange,
}

/// List domain over a fixed item universe, fully integer-backed: a boolean
/// membership variable per item, plus a length variable. The coupling
/// `sum(members) == length` is enforced by the `ListCardinality` propagator, not
/// by the `Store`, so a learning decision on any of these variables routes back
/// through normal explained propagation.
#[derive(Clone)]
pub struct ListDomain {
    pub(crate) universe: Vec<i32>,
    /// `members[i]` is the boolean membership variable of `universe[i]`.
    pub(crate) members: Vec<VarId>,
    /// The list length variable, with domain `[0, universe.len()]`.
    pub(crate) length: VarId,
}

impl ListDomain {
    /// Immutable item universe.
    pub fn universe(&self) -> &[i32] {
        &self.universe
    }

    /// Position of `item` in the universe, if present.
    pub(crate) fn index_of(&self, item: i32) -> Option<usize> {
        self.universe.iter().position(|&v| v == item)
    }
}
