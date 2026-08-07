//! Free-list reuse in the learned-clause arena: repeated learn/delete cycles
//! must recycle tombstoned slots instead of growing the arena forever.

use qayd::lcg::clause::{ClauseDb, ClauseRef};
use qayd::lcg::lit::Lit;

fn lits(n: u32) -> std::sync::Arc<[Lit]> {
    (0..n).map(Lit::positive).collect()
}

#[test]
fn learn_delete_cycles_do_not_grow_the_arena() {
    let mut db = ClauseDb::new();
    // A base of never-deleted clauses (like axioms / root units).
    for _ in 0..10 {
        db.add(lits(3), [0, 1], false, 0);
    }
    let base = db.len();
    assert_eq!(db.live_len(), base);

    // Many rounds of learning a batch of clauses then deleting them all. With a
    // free list the arena tracks peak live clauses, not the cumulative total.
    let batch = 50;
    for _ in 0..1000 {
        let refs: Vec<ClauseRef> = (0..batch).map(|_| db.add(lits(4), [0, 1], true, 3)).collect();
        for c in refs {
            db.remove(c);
        }
    }

    // Peak live was base + batch; the arena must not exceed that despite the
    // 50_000 clauses added over the run.
    assert_eq!(db.len(), base + batch, "arena grew past peak live: {}", db.len());
    assert_eq!(db.live_len(), base, "base clauses lost");
}

#[test]
fn reused_slot_returns_the_new_clause() {
    let mut db = ClauseDb::new();
    let a = db.add(lits(2), [0, 1], true, 2);
    db.remove(a);
    let b = db.add(lits(5), [0, 1], true, 4);
    // Slot recycled; the handle now resolves to the fresh clause, not a tombstone.
    assert_eq!(a, b);
    assert_eq!(db.get(b).lits.len(), 5);
    assert_eq!(db.live_len(), 1);
}
