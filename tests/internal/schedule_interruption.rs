use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

use qayd::constraints::interval::{no_overlap, no_overlap_until};
use qayd::constraints::scheduling::no_overlap as no_overlap_ints;
use qayd::engines::schedule::{self, Options, Status};
use qayd::engines::CollectionCompileContext;
use qayd::model::list::{CollectionModel, IntervalVar, Mode, Resource, Schedule};
use qayd::model::{CompiledCollection, Model};
use qayd::orchestrator::SolveRequest;
use qayd::Solver;

fn collection_with_schedule(schedule: Schedule) -> CollectionModel {
    CollectionModel {
        items: Vec::new(),
        lists: 0,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: Some(schedule),
    }
}

fn fixed_schedule() -> Schedule {
    Schedule {
        intervals: (0..8).map(|_| IntervalVar { duration: 1, horizon: 8, modes: Vec::new(), optional: false }).collect(),
        precedences: vec![(0, 1), (2, 3)],
        resources: vec![Resource::NoOverlap((0..8).collect())],
        minimize_makespan: true,
    }
}

fn moded_schedule() -> Schedule {
    Schedule {
        intervals: vec![IntervalVar {
            duration: 0,
            horizon: 4,
            modes: vec![Mode { reference: Some(0), machine: 0, duration: 1, start_window: (0, 3) }],
            optional: false,
        }],
        precedences: Vec::new(),
        resources: vec![Resource::MachineNoOverlap],
        minimize_makespan: true,
    }
}

fn lower(schedule: Schedule) -> schedule::CompiledSchedule {
    let semantic = Model::from_collection(&collection_with_schedule(schedule));
    let collection = CompiledCollection::compile(&semantic).expect("semantic schedule compiles");
    let request = SolveRequest::default();
    let context = CollectionCompileContext::new(&semantic, &request, &collection);
    schedule::lower(&context, &AtomicBool::new(false)).expect("physical lowering succeeds").expect("schedule backend accepts model")
}

fn assert_unknown(outcome: &schedule::Outcome) {
    assert_eq!(outcome.status, Status::Unknown);
    assert_eq!(outcome.objective, None);
    assert!(outcome.starts.is_empty());
    assert!(outcome.presences.is_empty());
    assert!(outcome.machines.is_empty());
    assert!(outcome.modes.is_empty());
}

#[test]
fn immediate_stop_returns_unknown_for_both_physical_schedule_builders() {
    for compiled in [lower(fixed_schedule()), lower(moded_schedule())] {
        let outcome = schedule::solve_compiled(&compiled, &AtomicBool::new(true), Options::default(), |_| {})
            .expect("an interrupted physical build is not an engine error");
        assert_unknown(&outcome);
        assert_eq!(outcome.stats, Default::default());
    }
}

#[test]
fn direct_schedule_solve_distinguishes_cancellation_from_unsupported() {
    let outcome = schedule::solve(&fixed_schedule(), &AtomicBool::new(true), Options::default(), |_| {})
        .expect("cancellation is not an engine error")
        .expect("a supported schedule stopped during lowering returns UNKNOWN");
    assert_unknown(&outcome);
}

#[test]
fn no_overlap_pair_registration_stops_cooperatively() {
    let mut solver = Solver::new();
    let intervals = (0..64).map(|_| solver.store.new_interval(0, 63, 1)).collect::<Vec<_>>();
    let polls = Cell::new(0usize);
    let should_stop = || {
        let next = polls.get() + 1;
        polls.set(next);
        next >= 8
    };

    let posted = no_overlap_until(&mut solver, intervals, &should_stop);

    assert!(posted.is_none());
    assert!(polls.get() >= 8);
    assert!(solver.store.disjunctive_pair_count() < 64 * 63 / 2);
}

#[test]
fn no_overlap_propagation_stops_inside_one_expensive_call() {
    let mut solver = Solver::new();
    let intervals = (0..64).map(|_| solver.store.new_interval(0, 63, 1)).collect::<Vec<_>>();
    no_overlap(&mut solver, &intervals);
    let polls = Cell::new(0usize);

    solver
        .propagate_until(|| {
            let next = polls.get() + 1;
            polls.set(next);
            next >= 8
        })
        .expect("cooperative interruption is not an inconsistency");

    assert!(polls.get() >= 8);
    assert!(solver.fd_at_fixpoint(), "the interrupted propagation queue must be cleared");
}

#[test]
fn integer_no_overlap_global_propagation_stops_inside_one_expensive_call() {
    let mut solver = Solver::new();
    let mut starts = vec![solver.new_var_range(0, 0)];
    starts.extend((1..33).map(|_| solver.new_var_range(0, 64)));
    let mut durations = vec![10];
    durations.extend([1; 32]);
    no_overlap_ints(&mut solver, &starts, &durations);
    assert_eq!(solver.num_propagators(), 1, "the test must exercise the global fallback");
    let polls = Cell::new(0usize);

    solver
        .propagate_until(|| {
            let next = polls.get() + 1;
            polls.set(next);
            next >= 5
        })
        .expect("cooperative interruption is not an inconsistency");

    assert!(polls.get() >= 5);
    assert_eq!(solver.store.min(starts[1]), 10, "the first sound inference may survive cancellation");
    assert_eq!(solver.store.min(starts[2]), 0, "propagation must stop before the next pair");
    assert!(solver.fd_at_fixpoint(), "the interrupted propagation queue must be cleared");
}

#[test]
fn cdcl_replay_drops_an_incumbent_when_the_stop_flag_fires() {
    let stop = AtomicBool::new(false);
    let mut improved = false;
    let outcome = schedule::solve(&moded_schedule(), &stop, Options { seed: 7, optional_modes_cdcl: true }, |_| {
        improved = true;
        stop.store(true, Ordering::Release);
    })
    .expect("cancellation during replay is not an engine error")
    .expect("the moded schedule is supported");

    assert!(improved, "the stop must fire after CDCL has found an incumbent");
    assert!(stop.load(Ordering::Acquire));
    assert_unknown(&outcome);
}
