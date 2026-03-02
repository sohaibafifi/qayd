//! Trail-based state manager.
//!
//! A flat arena of reversible integers plus an undo log and per-level marks.
//! Every piece of mutable solver state that must be restored on backtrack is a
//! [`ReversibleInt`] living here. No `unsafe`, no boxed closures: undoing a
//! level just replays `(index, old_value)` entries in reverse.

/// Handle to a reversible integer stored in a [`Trail`].
///
/// Cheap to copy; it is only an index into [`Trail::values`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ReversibleInt(u32);

impl ReversibleInt {
    /// The underlying arena index. Exposed for debugging/assertions only.
    #[inline]
    pub fn raw(self) -> u32 {
        self.0
    }
}

/// State manager: reversible ints + undo log + level marks.
#[derive(Default)]
pub struct Trail {
    /// Current value of every reversible int, indexed by `ReversibleInt`.
    values: Vec<i32>,
    /// `(index, old_value)` entries to replay on `pop_level`.
    log: Vec<(u32, i32)>,
    /// `log` length captured at each `push_level`.
    levels: Vec<usize>,
}

impl Trail {
    /// An empty trail at the root level.
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate a new reversible int initialised to `v`. Allocation is *not*
    /// reversible — do it at the root, before search begins.
    pub fn new_int(&mut self, v: i32) -> ReversibleInt {
        let i = self.values.len() as u32;
        self.values.push(v);
        ReversibleInt(i)
    }

    /// Read the current value.
    #[inline]
    pub fn get(&self, r: ReversibleInt) -> i32 {
        self.values[r.0 as usize]
    }

    /// Write a value, logging the previous one so it can be undone. A write
    /// that does not change the value is not logged (keeps the log tight).
    #[inline]
    pub fn set(&mut self, r: ReversibleInt, v: i32) {
        let i = r.0 as usize;
        if self.values[i] != v {
            self.log.push((r.0, self.values[i]));
            self.values[i] = v;
        }
    }

    /// Open a new level. Pair every `push_level` with exactly one `pop_level`.
    pub fn push_level(&mut self) {
        self.levels.push(self.log.len());
    }

    /// Undo every change made since the matching `push_level`.
    pub fn pop_level(&mut self) {
        let mark = self
            .levels
            .pop()
            .expect("pop_level without a matching push_level");
        while self.log.len() > mark {
            let (i, old) = self.log.pop().unwrap();
            self.values[i as usize] = old;
        }
    }

    /// Current search depth (number of open levels). Used by trail-integrity
    /// assertions in search and tests.
    #[inline]
    pub fn level(&self) -> usize {
        self.levels.len()
    }

    /// Number of reversible ints allocated.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether no reversible ints have been allocated yet.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}
