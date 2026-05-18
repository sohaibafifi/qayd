//! The clause database: an arena of disjunctive [`Lit`] clauses (axiom, learned,
//! constraint). Literal truth lives in the domain view, not here.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::lcg::lit::Lit;

const MAX_SHARED_LEN: usize = 8;
const MAX_SHARED_LBD: u32 = 4;

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
}

impl ClauseDb {
    /// An empty database.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of clauses.
    #[inline]
    pub fn len(&self) -> usize {
        self.clauses.len()
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

    /// Add a clause, returning its handle.
    pub fn add(
        &mut self,
        lits: Arc<[Lit]>,
        watch: [usize; 2],
        deletable: bool,
        lbd: u32,
    ) -> ClauseRef {
        let r = ClauseRef(self.clauses.len() as u32);
        self.clauses.push(Clause {
            lits,
            watch,
            deletable,
            lbd,
        });
        r
    }
}

/// One learned clause published by a portfolio worker.
#[derive(Clone)]
pub(crate) struct SharedClause {
    pub lits: Arc<[Lit]>,
    pub lbd: u32,
    source: usize,
}

/// Append-only shared clause pool. Workers keep private watch state.
#[derive(Default)]
pub(crate) struct SharedClausePool {
    clauses: Mutex<Vec<SharedClause>>,
    imported: AtomicU64,
}

impl SharedClausePool {
    pub fn len(&self) -> usize {
        self.clauses.lock().unwrap().len()
    }

    pub fn imported(&self) -> u64 {
        self.imported.load(Ordering::Relaxed)
    }
}

/// Per-worker view of the shared clause pool.
pub(crate) struct ClauseSharing {
    pool: Arc<SharedClausePool>,
    worker: usize,
    cursor: usize,
}

impl ClauseSharing {
    pub fn new(pool: Arc<SharedClausePool>, worker: usize) -> Self {
        Self {
            pool,
            worker,
            cursor: 0,
        }
    }

    pub fn publish(&self, lits: Arc<[Lit]>, lbd: u32) {
        if lits.is_empty() || lits.len() > MAX_SHARED_LEN || lbd > MAX_SHARED_LBD {
            return;
        }
        self.pool.clauses.lock().unwrap().push(SharedClause {
            lits,
            lbd,
            source: self.worker,
        });
    }

    pub fn copy_new_into(&mut self, out: &mut Vec<SharedClause>) {
        let clauses = self.pool.clauses.lock().unwrap();
        out.clear();
        out.extend_from_slice(&clauses[self.cursor..]);
        self.cursor = clauses.len();
    }

    pub fn is_own(&self, clause: &SharedClause) -> bool {
        clause.source == self.worker
    }

    pub fn record_import(&self) {
        self.pool.imported.fetch_add(1, Ordering::Relaxed);
    }
}
