//! The clause database.
//!
//! Clauses are disjunctions of [`Lit`]s. The database holds three kinds, all in
//! one arena:
//!
//! - **axiom** clauses — the order-encoding consistency clauses that channel a
//!   variable's order and equality atoms (posted once, up front);
//! - **learned** clauses — produced by conflict analysis;
//! - **constraint** clauses — blocking clauses (enumeration) and objective
//!   bounds (optimization).
//!
//! Literal *truth* lives in the domain view, not here, so the database stores
//! only the literals. This milestone evaluates clauses by scanning (simple and
//! obviously correct); a watched-literal scheme is a later, behaviour-preserving
//! performance change.

use crate::lcg::lit::Lit;

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
    /// The literals; the clause is satisfied iff at least one is true. A
    /// *deleted* clause has an empty `lits` (a tombstone — its slot stays so
    /// existing [`ClauseRef`]s remain valid).
    pub lits: Vec<Lit>,
    /// Whether this is a learned clause eligible for deletion during database
    /// reduction. Axiom (channeling), blocking, and objective-bound clauses are
    /// not deletable — only 1-UIP learned clauses are.
    pub deletable: bool,
    /// Literal Block Distance at learn time — the number of distinct decision
    /// levels among its literals. Lower is better (LBD ≤ 2 = "glue"); used to
    /// pick which deletable clauses to drop. Meaningful only when `deletable`.
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

    /// Mutably borrow a clause (to reorder its watched literals).
    #[inline]
    pub fn get_mut(&mut self, c: ClauseRef) -> &mut Clause {
        &mut self.clauses[c.index()]
    }

    /// Add a clause, returning its handle.
    pub fn add(&mut self, lits: Vec<Lit>, deletable: bool, lbd: u32) -> ClauseRef {
        let r = ClauseRef(self.clauses.len() as u32);
        self.clauses.push(Clause {
            lits,
            deletable,
            lbd,
        });
        r
    }
}
