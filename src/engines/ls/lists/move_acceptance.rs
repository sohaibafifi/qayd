/// Acceptance policy for local-search states whose objective is minimized.
///
/// This captures only the makespan comparison shared by the JSSP and
/// resource-scheduling states. State transitions and rollback remain owned by
/// those engines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MinimizingMoveAcceptance {
    Improving,
    NonWorsening,
    Always,
}

impl MinimizingMoveAcceptance {
    pub(crate) fn accepts(self, current: i64, candidate: i64) -> bool {
        match self {
            Self::Improving => candidate < current,
            Self::NonWorsening => candidate <= current,
            Self::Always => true,
        }
    }
}
