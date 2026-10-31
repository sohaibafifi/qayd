use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

use qayd::engines::list_exact::{self, Status};
use qayd::engines::CollectionCompileContext;
use qayd::model::list::CollectionModel;
use qayd::model::{CompiledCollection, Model};
use qayd::orchestrator::SolveRequest;

fn collection(lists: usize, item_count: usize) -> CollectionModel {
    CollectionModel {
        items: (0..item_count).map(|item| i32::try_from(item).expect("test item fits i32")).collect(),
        lists,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    }
}

fn lower(model: &CollectionModel) -> (CompiledCollection, list_exact::CompiledListExact) {
    let semantic = Model::from_collection(model);
    let compiled = CompiledCollection::compile(&semantic).expect("semantic list model compiles");
    let request = SolveRequest::default();
    let context = CollectionCompileContext::new(&semantic, &request, &compiled);
    let blueprint = list_exact::lower(&context, &AtomicBool::new(false)).expect("list exact lowering accepts the model");
    (compiled, blueprint)
}

fn assert_interrupted(outcome: &list_exact::Outcome) {
    assert_eq!(outcome.status, Status::Unknown);
    assert!(outcome.objectives.is_empty());
    assert!(outcome.solution.is_none());
    assert_eq!(outcome.stats, Default::default());
}

#[test]
fn prearmed_stop_returns_unknown_without_a_partial_list_candidate() {
    let model = collection(2, 4);
    let (domain, compiled) = lower(&model);
    let stop = AtomicBool::new(true);

    let compiled_outcome = list_exact::solve_compiled(&compiled, domain.as_model(), &stop, |_| {});
    assert_interrupted(&compiled_outcome);

    let direct_outcome = list_exact::solve(&model, &[], &stop, |_| {})
        .expect("cancellation is not an engine error")
        .expect("a supported interrupted model is not reported as unsupported");
    assert_interrupted(&direct_outcome);
}

#[test]
fn stop_during_a_large_physical_build_returns_unknown_without_searching() {
    let (domain, compiled) = lower(&collection(64, 4_096));
    let stop = AtomicBool::new(false);
    let polls = Cell::new(0usize);
    const STOP_AFTER_POLLS: usize = 5_000;
    let should_stop = || {
        let next = polls.get() + 1;
        polls.set(next);
        if next == STOP_AFTER_POLLS {
            stop.store(true, Ordering::Release);
        }
        stop.load(Ordering::Acquire)
    };

    let outcome = list_exact::solve_compiled_with_build_stop_for_test(&compiled, domain.as_model(), &stop, &should_stop);

    assert!(polls.get() >= STOP_AFTER_POLLS, "the stop must fire after physical construction has begun");
    assert!(stop.load(Ordering::Acquire));
    assert_interrupted(&outcome);
}
