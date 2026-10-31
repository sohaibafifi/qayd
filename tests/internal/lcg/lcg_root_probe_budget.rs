use crate::lcg::trail::Cdcl;
use crate::propagator::{Event, Inconsistency, Propagator};
use crate::{PropId, Solver, Store, VarId};

#[derive(Clone)]
struct BranchesForceValue {
    decision: VarId,
    implied: VarId,
}

impl Propagator for BranchesForceValue {
    fn register(&mut self, store: &mut Store, me: PropId) {
        store.subscribe(self.decision, me, Event::DomainChange);
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        if store.is_fixed(self.decision) || !store.contains(self.decision, 0) {
            store.fix(self.implied, 1)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
struct Dormant;

impl Propagator for Dormant {
    fn register(&mut self, _store: &mut Store, _me: PropId) {}

    fn propagate(&mut self, _store: &mut Store) -> Result<(), Inconsistency> {
        Ok(())
    }
}

#[derive(Clone)]
struct WideDormant {
    variables: Vec<VarId>,
}

impl Propagator for WideDormant {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &variable in &self.variables {
            store.subscribe(variable, me, Event::DomainChange);
        }
    }

    fn propagate(&mut self, _store: &mut Store) -> Result<(), Inconsistency> {
        Ok(())
    }
}

fn probing_fixes_the_common_value(extra_variables: usize, dormant_propagators: usize) -> bool {
    let mut solver = Solver::new();
    let decision = solver.new_var_range(0, 2);
    let implied = solver.new_var_range(0, 1);
    for _ in 0..extra_variables {
        solver.new_var_range(0, 0);
    }
    solver.post(Box::new(BranchesForceValue { decision, implied }));
    for _ in 0..dormant_propagators {
        solver.post(Box::new(Dormant));
    }

    let mut cdcl = Cdcl::new(&mut solver, &[decision, implied]);
    assert!(cdcl.init());
    assert!(!cdcl.solver.store.is_fixed(implied));
    assert!(cdcl.root_probe(&[decision, implied]));
    cdcl.solver.store.is_fixed(implied)
}

#[test]
fn root_probing_runs_below_both_work_guards() {
    assert!(probing_fixes_the_common_value(0, 0));
}

#[test]
fn root_probing_is_skipped_for_a_large_physical_decomposition() {
    assert!(!probing_fixes_the_common_value(0, 8_192));
}

#[test]
fn root_probing_is_skipped_when_the_physical_model_is_dense() {
    let mut solver = Solver::new();
    let decision = solver.new_var_range(0, 2);
    let implied = solver.new_var_range(0, 1);
    let mut variables = vec![decision, implied];
    variables.extend((0..500).map(|_| solver.new_var_range(0, 0)));
    solver.post(Box::new(BranchesForceValue { decision, implied }));
    for _ in 0..500 {
        solver.post(Box::new(WideDormant { variables: variables.clone() }));
    }

    let mut cdcl = Cdcl::new(&mut solver, &[decision, implied]);
    assert!(cdcl.init());
    assert!(cdcl.root_probe(&[decision, implied]));
    assert!(!cdcl.solver.store.is_fixed(implied));
}
