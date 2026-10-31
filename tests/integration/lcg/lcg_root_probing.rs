use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use qayd::propagator::{Event, Inconsistency, Propagator};
use qayd::{count_solutions, PropId, Store, VarId};

#[derive(Clone)]
struct BranchesForceY {
    x: VarId,
    y: VarId,
    saw_root_y: Arc<AtomicBool>,
}

impl Propagator for BranchesForceY {
    fn register(&mut self, store: &mut Store, me: PropId) {
        store.subscribe(self.x, me, Event::DomainChange);
        store.subscribe(self.y, me, Event::Fix);
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        if store.level() == 0 && store.is_fixed(self.y) && store.value(self.y) == 1 {
            self.saw_root_y.store(true, Ordering::Relaxed);
        }
        if store.is_fixed(self.x) || !store.contains(self.x, 0) {
            store.fix(self.y, 1)?;
        }
        Ok(())
    }
}

#[test]
fn root_probing_asserts_common_branch_implication() {
    let saw_root_y = Arc::new(AtomicBool::new(false));
    let mut solver = qayd::Solver::new();
    let x = solver.new_var_range(0, 2);
    let y = solver.new_var_range(0, 1);
    solver.post(Box::new(BranchesForceY { x, y, saw_root_y: Arc::clone(&saw_root_y) }));

    assert_eq!(count_solutions(&mut solver, &[x, y]), 3);
    assert!(saw_root_y.load(Ordering::Relaxed));
}
