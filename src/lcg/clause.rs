//! The clause database: an arena of disjunctive [`Lit`] clauses (axiom, learned,
//! constraint). Literal truth lives in the domain view, not here.

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
    /// Literals (satisfied iff one is true). Empty = deleted tombstone, kept so
    /// existing [`ClauseRef`]s stay valid.
    pub lits: Vec<Lit>,
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
