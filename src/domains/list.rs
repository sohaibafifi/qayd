//! Structured list domain over a fixed item universe.
//!
//! Kernel-owned, trailed membership and length state. Independent of any
//! list local-search IR.

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
