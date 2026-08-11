//! The clause database: an arena of disjunctive [`Lit`] clauses (axiom, learned,
//! constraint). Literal truth lives in the domain view, not here.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::lcg::lit::{LazyAtomRegistry, Lit};

const MAX_SHARED_LEN: usize = 8;
const MAX_SHARED_LBD: u32 = 4;

/// Ring capacity for the shared clause pool (power of two). Bounds memory on
/// long runs; a lagging reader that falls further behind than this drops the
/// oldest shared clauses, which is sound (sharing is best-effort pruning).
const SHARED_POOL_CAP: usize = 1 << 14;

/// A handle to a clause in the [`ClauseDb`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ClauseRef(pub u32);

impl ClauseRef {
    /// The arena index.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A disjunctive clause.
pub struct Clause {
    /// Immutable literals. Empty means a deleted tombstone.
    pub lits: Arc<[Lit]>,
    /// Per-engine watch positions into [`lits`](Self::lits).
    pub watch: [usize; 2],
    /// Eligible for deletion during DB reduction (only 1-UIP learned clauses).
    pub deletable: bool,
    /// Literal Block Distance at learn time; lower is better. Meaningful only when `deletable`.
    pub lbd: u32,
}

/// An arena of clauses.
#[derive(Default)]
pub struct ClauseDb {
    clauses: Vec<Clause>,
    /// Tombstoned slots available for reuse, so the arena tracks peak live
    /// clauses instead of every clause ever created.
    free: Vec<u32>,
}

impl ClauseDb {
    /// An empty database.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of arena slots (live plus tombstoned).
    #[inline]
    pub fn len(&self) -> usize {
        self.clauses.len()
    }

    /// Number of live (non-tombstone) clauses. Bounded across learn/delete
    /// cycles because deleted slots are reused rather than left dead.
    #[inline]
    pub fn live_len(&self) -> usize {
        self.clauses.len() - self.free.len()
    }

    /// Whether the database is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.clauses.is_empty()
    }

    /// Borrow a clause.
    #[inline]
    pub fn get(&self, c: ClauseRef) -> &Clause {
        &self.clauses[c.index()]
    }

    /// Mutably borrow a clause to update its watches.
    #[inline]
    pub fn get_mut(&mut self, c: ClauseRef) -> &mut Clause {
        &mut self.clauses[c.index()]
    }

    /// Add a clause, returning its handle. Reuses a tombstoned slot when one is
    /// free, so a deleted slot's index is recycled with a fresh clause.
    pub fn add(&mut self, lits: Arc<[Lit]>, watch: [usize; 2], deletable: bool, lbd: u32) -> ClauseRef {
        let clause = Clause { lits, watch, deletable, lbd };
        if let Some(i) = self.free.pop() {
            self.clauses[i as usize] = clause;
            ClauseRef(i)
        } else {
            let r = ClauseRef(self.clauses.len() as u32);
            self.clauses.push(clause);
            r
        }
    }

    /// Tombstone a slot and mark it for reuse. The index stays valid (empty
    /// literals) until `add` recycles it. Sound only when no live watch,
    /// binary-implication, or dereferenced reason still points at `c`.
    pub fn remove(&mut self, c: ClauseRef) {
        self.clauses[c.index()].lits = Arc::from([]);
        self.free.push(c.0);
    }
}

/// One learned clause published by a portfolio worker.
#[derive(Clone)]
pub(crate) struct SharedClause {
    pub lits: Arc<[Lit]>,
    pub lbd: u32,
    source: usize,
}

/// Fixed-capacity ring of shared clauses with a monotonic write index.
struct SharedRing {
    slots: Vec<Option<SharedClause>>,
    /// `capacity - 1`; capacity is a power of two so this is the index mask.
    mask: usize,
    /// Total clauses ever written; the live window is `[write - capacity, write)`.
    write: u64,
}

/// Bounded shared clause pool. Workers keep private watch state and read
/// cursors; the ring caps memory and drops old clauses under a lagging reader.
pub(crate) struct SharedClausePool {
    ring: Mutex<SharedRing>,
    imported: AtomicU64,
    lazy_atoms: Arc<LazyAtomRegistry>,
}

impl Default for SharedClausePool {
    fn default() -> Self {
        Self::with_capacity(SHARED_POOL_CAP)
    }
}

impl SharedClausePool {
    /// Build a ring with an explicit requested capacity, rounded down to a power
    /// of two. A smaller ring only drops more optional shared clauses.
    pub(crate) fn with_capacity(requested: usize) -> Self {
        let requested = requested.max(1);
        let cap = 1usize << (usize::BITS - 1 - requested.leading_zeros());
        Self {
            ring: Mutex::new(SharedRing { slots: vec![None; cap], mask: cap - 1, write: 0 }),
            imported: AtomicU64::new(0),
            lazy_atoms: LazyAtomRegistry::new(),
        }
    }
    /// Total clauses published to the pool (not the live window size).
    pub fn len(&self) -> usize {
        self.ring.lock().unwrap().write as usize
    }

    pub fn imported(&self) -> u64 {
        self.imported.load(Ordering::Relaxed)
    }

    pub fn lazy_atoms(&self) -> Arc<LazyAtomRegistry> {
        Arc::clone(&self.lazy_atoms)
    }

    /// Snapshot the live shared clauses in chronological order. `limit` keeps
    /// only the newest clauses when non-zero.
    pub fn snapshot(&self, limit: usize) -> Vec<(u32, Arc<[Lit]>)> {
        let ring = self.ring.lock().unwrap();
        let write = ring.write;
        let cap = ring.mask as u64 + 1;
        let mut start = write.saturating_sub(cap);
        if limit > 0 {
            start = start.max(write.saturating_sub(limit as u64));
        }
        let mut out = Vec::with_capacity((write - start) as usize);
        for i in start..write {
            if let Some(clause) = &ring.slots[(i as usize) & ring.mask] {
                out.push((clause.lbd, Arc::clone(&clause.lits)));
            }
        }
        out
    }
}

/// Per-worker view of the shared clause pool.
pub(crate) struct ClauseSharing {
    pool: Arc<SharedClausePool>,
    worker: usize,
    cursor: u64,
}

impl ClauseSharing {
    pub fn new(pool: Arc<SharedClausePool>, worker: usize) -> Self {
        Self { pool, worker, cursor: 0 }
    }

    pub fn publish(&self, lits: Arc<[Lit]>, lbd: u32) {
        if lits.is_empty() || lits.len() > MAX_SHARED_LEN || lbd > MAX_SHARED_LBD {
            return;
        }
        let mut ring = self.pool.ring.lock().unwrap();
        let idx = (ring.write as usize) & ring.mask;
        ring.slots[idx] = Some(SharedClause { lits, lbd, source: self.worker });
        ring.write += 1;
    }

    pub fn copy_new_into(&mut self, out: &mut Vec<SharedClause>) {
        out.clear();
        let ring = self.pool.ring.lock().unwrap();
        let write = ring.write;
        let cap = ring.mask as u64 + 1;
        // A reader lagging more than the ring holds skips forward to the oldest
        // still-present clause; the gap's clauses are lost (sound: less pruning).
        let start = self.cursor.max(write.saturating_sub(cap));
        for i in start..write {
            if let Some(c) = &ring.slots[(i as usize) & ring.mask] {
                out.push(c.clone());
            }
        }
        self.cursor = write;
    }

    pub fn is_own(&self, clause: &SharedClause) -> bool {
        clause.source == self.worker
    }

    pub fn record_import(&self) {
        self.pool.imported.fetch_add(1, Ordering::Relaxed);
    }

    pub fn lazy_atoms(&self) -> Arc<LazyAtomRegistry> {
        Arc::clone(&self.pool.lazy_atoms)
    }
}
