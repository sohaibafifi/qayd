//! Trail-based state manager: reversible ints + undo log + per-level marks.
//!
//! Mutable solver state that must survive backtrack lives here as a
//! [`ReversibleInt`]; `pop_level` replays `(index, old_value)` entries.

/// Handle to a reversible integer in a [`Trail`] (an index into `values`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ReversibleInt(u32);

impl ReversibleInt {
    /// The underlying arena index (debugging/assertions only).
    #[inline]
    pub fn raw(self) -> u32 {
        self.0
    }
}

/// State manager: reversible ints + undo log + level marks.
#[derive(Clone, Default)]
pub struct Trail {
    values: Vec<i32>,
    /// Epoch of each reversible's last logged write (parallel to `values`).
    /// A write is only logged when this differs from `cur_epoch`, so N writes
    /// within one level incarnation produce a single undo entry.
    epochs: Vec<u64>,
    /// `(index, old_value)` entries to replay on `pop_level`.
    log: Vec<(u32, i32)>,
    /// `(log length, epoch)` captured at each `push_level`.
    levels: Vec<(usize, u64)>,
    /// Epoch of the current (deepest) level; `0` at root.
    cur_epoch: u64,
    /// Monotonic epoch source; never reused, so a stamp from a popped level
    /// can never collide with a current or future `cur_epoch`.
    next_epoch: u64,
}

impl Trail {
    /// An empty trail at the root level.
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate a reversible int set to `v`. Not reversible; do it at the
    /// root, before search.
    pub fn new_int(&mut self, v: i32) -> ReversibleInt {
        let i = self.values.len() as u32;
        self.values.push(v);
        self.epochs.push(0);
        ReversibleInt(i)
    }

    /// Read the current value.
    #[inline]
    pub fn get(&self, r: ReversibleInt) -> i32 {
        self.values[r.0 as usize]
    }

    /// Write a value, logging the old one. No-op writes are not logged.
    #[inline]
    pub fn set(&mut self, r: ReversibleInt, v: i32) {
        let i = r.0 as usize;
        if self.values[i] != v {
            if self.epochs[i] != self.cur_epoch {
                self.log.push((r.0, self.values[i]));
                self.epochs[i] = self.cur_epoch;
            }
            self.values[i] = v;
        }
    }

    /// Open a new level. Pair every `push_level` with exactly one `pop_level`.
    pub fn push_level(&mut self) {
        self.next_epoch += 1;
        self.cur_epoch = self.next_epoch;
        self.levels.push((self.log.len(), self.cur_epoch));
    }

    /// Undo every change made since the matching `push_level`.
    pub fn pop_level(&mut self) {
        let (mark, _) = self.levels.pop().expect("pop_level without a matching push_level");
        while self.log.len() > mark {
            let (i, old) = self.log.pop().unwrap();
            self.values[i as usize] = old;
        }
        self.cur_epoch = self.levels.last().map_or(0, |&(_, e)| e);
    }

    /// Current search depth (number of open levels).
    #[inline]
    pub fn level(&self) -> usize {
        self.levels.len()
    }

    /// Number of reversible ints allocated.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Number of undo entries currently on the trail (test/stat accessor).
    #[inline]
    pub fn log_len(&self) -> usize {
        self.log.len()
    }

    /// Whether no reversible ints have been allocated yet.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}
