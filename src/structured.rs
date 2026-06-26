//! Structured list and interval domains.
//!
//! These are kernel-owned, trailed domains. They are intentionally independent
//! from the collection local-search IR: this module stores exact mutable state,
//! while collection search can later act as an incumbent source.

use crate::propagator::Inconsistency;
use crate::trail::{ReversibleInt, Trail};

const FALSE: i32 = 0;
const TRUE: i32 = 1;

/// Structured list domain event kinds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ListEvent {
    /// Possible membership changed.
    PossibleChange,
    /// Required membership changed.
    RequiredChange,
    /// Length bounds changed.
    LengthChange,
    /// Successor or predecessor arcs changed.
    ArcChange,
    /// Position information changed.
    PositionChange,
}

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

impl IntervalPresence {
    fn encode(self) -> i32 {
        match self {
            Self::Optional => -1,
            Self::Absent => 0,
            Self::Present => 1,
        }
    }

    fn decode(v: i32) -> Self {
        match v {
            0 => Self::Absent,
            1 => Self::Present,
            _ => Self::Optional,
        }
    }
}

/// Structured list domain over a fixed item universe.
#[derive(Clone)]
pub struct ListDomain {
    universe: Vec<i32>,
    possible: Vec<ReversibleInt>,
    required: Vec<ReversibleInt>,
    len_min: ReversibleInt,
    len_max: ReversibleInt,
}

impl ListDomain {
    /// Create a list where every item starts possible and no item is required.
    pub fn new(universe: Vec<i32>, trail: &mut Trail) -> Self {
        let n = universe.len();
        assert!(i32::try_from(n).is_ok(), "structured list universe is too large");
        let mut sorted = universe.clone();
        sorted.sort_unstable();
        assert!(sorted.windows(2).all(|w| w[0] != w[1]), "structured list universe contains duplicate items");
        let possible = (0..n).map(|_| trail.new_int(TRUE)).collect();
        let required = (0..n).map(|_| trail.new_int(FALSE)).collect();
        let len_min = trail.new_int(0);
        let len_max = trail.new_int(n as i32);
        Self { universe, possible, required, len_min, len_max }
    }

    /// Immutable item universe.
    pub fn universe(&self) -> &[i32] {
        &self.universe
    }

    /// Current length lower bound.
    pub fn len_min(&self, trail: &Trail) -> usize {
        trail.get(self.len_min) as usize
    }

    /// Current length upper bound.
    pub fn len_max(&self, trail: &Trail) -> usize {
        trail.get(self.len_max) as usize
    }

    /// Whether `item` can still appear in the list.
    pub fn is_possible(&self, item: i32, trail: &Trail) -> bool {
        self.index_of(item).is_some_and(|i| trail.get(self.possible[i]) == TRUE)
    }

    /// Whether `item` must appear in the list.
    pub fn is_required(&self, item: i32, trail: &Trail) -> bool {
        self.index_of(item).is_some_and(|i| trail.get(self.required[i]) == TRUE)
    }

    /// Current possible item count.
    pub fn possible_count(&self, trail: &Trail) -> usize {
        self.possible.iter().filter(|&&r| trail.get(r) == TRUE).count()
    }

    /// Current required item count.
    pub fn required_count(&self, trail: &Trail) -> usize {
        self.required.iter().filter(|&&r| trail.get(r) == TRUE).count()
    }

    /// Require `item` in the list.
    pub fn require_item(&mut self, item: i32, trail: &mut Trail) -> Result<bool, Inconsistency> {
        let Some(i) = self.index_of(item) else {
            return Err(Inconsistency);
        };
        if trail.get(self.possible[i]) != TRUE {
            return Err(Inconsistency);
        }
        if trail.get(self.required[i]) == TRUE {
            return Ok(false);
        }
        trail.set(self.required[i], TRUE);
        self.close_cardinality(trail)?;
        Ok(true)
    }

    /// Forbid `item` from the list.
    pub fn forbid_item(&mut self, item: i32, trail: &mut Trail) -> Result<bool, Inconsistency> {
        let Some(i) = self.index_of(item) else {
            return Ok(false);
        };
        if trail.get(self.required[i]) == TRUE {
            return Err(Inconsistency);
        }
        if trail.get(self.possible[i]) != TRUE {
            return Ok(false);
        }
        trail.set(self.possible[i], FALSE);
        self.close_cardinality(trail)?;
        Ok(true)
    }

    /// Raise the length lower bound.
    pub fn set_len_min(&mut self, min: usize, trail: &mut Trail) -> Result<bool, Inconsistency> {
        if min > self.len_max(trail) || min > self.possible_count(trail) {
            return Err(Inconsistency);
        }
        if min <= self.len_min(trail) {
            return Ok(false);
        }
        trail.set(self.len_min, min as i32);
        self.close_cardinality(trail)?;
        Ok(true)
    }

    /// Lower the length upper bound.
    pub fn set_len_max(&mut self, max: usize, trail: &mut Trail) -> Result<bool, Inconsistency> {
        if max < self.len_min(trail) || max < self.required_count(trail) {
            return Err(Inconsistency);
        }
        if max >= self.len_max(trail) {
            return Ok(false);
        }
        trail.set(self.len_max, max as i32);
        self.close_cardinality(trail)?;
        Ok(true)
    }

    fn index_of(&self, item: i32) -> Option<usize> {
        self.universe.iter().position(|&v| v == item)
    }

    fn close_cardinality(&mut self, trail: &mut Trail) -> Result<bool, Inconsistency> {
        let mut changed = false;
        loop {
            let required = self.required_count(trail);
            let possible = self.possible_count(trail);
            let len_min = self.len_min(trail);
            let len_max = self.len_max(trail);
            if required > len_max || possible < len_min {
                return Err(Inconsistency);
            }

            let mut pass_changed = false;
            if required > len_min {
                trail.set(self.len_min, required as i32);
                pass_changed = true;
            }
            if possible < len_max {
                trail.set(self.len_max, possible as i32);
                pass_changed = true;
            }

            let required = self.required_count(trail);
            let possible = self.possible_count(trail);
            let len_min = self.len_min(trail);
            let len_max = self.len_max(trail);

            if required == len_max {
                for i in 0..self.universe.len() {
                    if trail.get(self.possible[i]) == TRUE && trail.get(self.required[i]) != TRUE {
                        trail.set(self.possible[i], FALSE);
                        pass_changed = true;
                    }
                }
            }

            if possible == len_min {
                for i in 0..self.universe.len() {
                    if trail.get(self.possible[i]) == TRUE && trail.get(self.required[i]) != TRUE {
                        trail.set(self.required[i], TRUE);
                        pass_changed = true;
                    }
                }
            }

            if !pass_changed {
                return Ok(changed);
            }
            changed = true;
        }
    }
}

/// Structured fixed-duration interval domain.
#[derive(Clone)]
pub struct IntervalDomain {
    start_min: ReversibleInt,
    start_max: ReversibleInt,
    duration: i32,
    presence: ReversibleInt,
}

impl IntervalDomain {
    /// Create an interval. Use `presence` to choose mandatory or optional.
    pub fn new(start_min: i32, start_max: i32, duration: i32, presence: IntervalPresence, trail: &mut Trail) -> Self {
        assert!(start_min <= start_max, "interval start_min must be <= start_max");
        assert!(duration >= 0, "interval duration must be non-negative");
        Self {
            start_min: trail.new_int(start_min),
            start_max: trail.new_int(start_max),
            duration,
            presence: trail.new_int(presence.encode()),
        }
    }

    /// Start lower bound.
    pub fn start_min(&self, trail: &Trail) -> i32 {
        trail.get(self.start_min)
    }

    /// Start upper bound.
    pub fn start_max(&self, trail: &Trail) -> i32 {
        trail.get(self.start_max)
    }

    /// Fixed duration.
    pub fn duration(&self) -> i32 {
        self.duration
    }

    /// End lower bound.
    pub fn end_min(&self, trail: &Trail) -> i32 {
        self.start_min(trail).saturating_add(self.duration)
    }

    /// End upper bound.
    pub fn end_max(&self, trail: &Trail) -> i32 {
        self.start_max(trail).saturating_add(self.duration)
    }

    /// Current presence state.
    pub fn presence(&self, trail: &Trail) -> IntervalPresence {
        IntervalPresence::decode(trail.get(self.presence))
    }

    /// Raise the start lower bound.
    pub fn set_start_min(&mut self, min: i32, trail: &mut Trail) -> Result<bool, Inconsistency> {
        if min > self.start_max(trail) {
            return Err(Inconsistency);
        }
        if min <= self.start_min(trail) {
            return Ok(false);
        }
        trail.set(self.start_min, min);
        Ok(true)
    }

    /// Lower the start upper bound.
    pub fn set_start_max(&mut self, max: i32, trail: &mut Trail) -> Result<bool, Inconsistency> {
        if max < self.start_min(trail) {
            return Err(Inconsistency);
        }
        if max >= self.start_max(trail) {
            return Ok(false);
        }
        trail.set(self.start_max, max);
        Ok(true)
    }

    /// Require the interval to be present.
    pub fn require_presence(&mut self, trail: &mut Trail) -> Result<bool, Inconsistency> {
        self.set_presence(IntervalPresence::Present, trail)
    }

    /// Mark the interval absent.
    pub fn forbid_presence(&mut self, trail: &mut Trail) -> Result<bool, Inconsistency> {
        self.set_presence(IntervalPresence::Absent, trail)
    }

    fn set_presence(&mut self, next: IntervalPresence, trail: &mut Trail) -> Result<bool, Inconsistency> {
        let current = self.presence(trail);
        if current == next {
            return Ok(false);
        }
        if current != IntervalPresence::Optional {
            return Err(Inconsistency);
        }
        trail.set(self.presence, next.encode());
        Ok(true)
    }
}
