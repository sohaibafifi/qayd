//! Fixed-duration interval domain.
//!
//! A handle backed by integer variables (start + presence), so an interval's
//! facts are first-class atoms the learning engine reasons over. The `Store`
//! owns the variables and the logic.

use crate::ids::VarId;

/// Structured interval domain event kinds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IntervalEvent {
    /// Start bounds changed.
    StartBoundChange,
    /// End bounds changed.
    EndBoundChange,
    /// Presence changed.
    PresenceChange,
    /// Execution mode changed.
    ModeChange,
}

/// Presence state of an interval domain.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IntervalPresence {
    /// Presence is not decided yet.
    Optional,
    /// The interval is absent.
    Absent,
    /// The interval is present.
    Present,
}

/// A fixed-duration interval, backed by integer variables so that its facts
/// (start bound, presence) are first-class atoms the learning engine reasons
/// over. The `Store` owns the variables and all the logic; this is just the
/// handle that records which variables back the interval.
#[derive(Clone, Copy)]
pub struct IntervalDomain {
    /// Start-time variable; its domain is the start window `[start_min, start_max]`.
    pub(crate) start: VarId,
    pub(crate) duration: i32,
    /// Presence boolean variable (`1` present, `0` absent); `None` for a
    /// mandatory interval (always present).
    pub(crate) presence: Option<VarId>,
}
