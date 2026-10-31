use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

use qayd::constraints::interval::{no_overlap, no_overlap_until};
use qayd::constraints::scheduling::no_overlap as no_overlap_ints;
use qayd::engines::schedule::{self, Options, Status};
use qayd::engines::CollectionCompileContext;
use qayd::model::list::{verify_collection_solution, CollectionModel, CollectionSolution, IntervalVar, Mode, Resource, Schedule};
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

fn assert_replayed(schedule: &Schedule, outcome: &schedule::Outcome) {
    let solution = CollectionSolution {
        lists: Vec::new(),
        objectives: outcome.objective.into_iter().collect(),
        feasible: true,
        starts: outcome.starts.clone(),
        presences: outcome.presences.clone(),
        machines: outcome.machines.clone(),
        modes: outcome.modes.clone(),
        bound: None,
    };
    assert_eq!(verify_collection_solution(&collection_with_schedule(schedule.clone()), &solution).unwrap(), solution.objectives);
}

#[test]
fn immediate_stop_returns_unknown_for_both_physical_schedule_builders() {
    for compiled in [lower(fixed_schedule()), lower(moded_schedule())] {
        let mut execution = schedule::Execution::default();
        let outcome = schedule::solve_compiled(&compiled, &AtomicBool::new(true), Options::default(), &mut execution, |_| {})
            .expect("an interrupted physical build is not an engine error");
        assert_unknown(&outcome);
        assert_eq!(outcome.stats, Default::default());
        assert!(matches!(execution.encoding, Some("fixed" | "moded")));
        assert_eq!(execution.search_kernel, None, "a prearmed stop must not claim a search call");
    }
}

#[test]
fn runtime_trace_records_the_invoked_kernel_for_each_exact_encoding() {
    for (schedule, cdcl, encoding, kernel, objective) in [
        (fixed_schedule(), false, "fixed", "cdcl", 8),
        (fixed_schedule(), true, "fixed", "cdcl", 8),
        (moded_schedule(), false, "moded", "chronological-dfs", 1),
        (moded_schedule(), true, "moded", "cdcl", 1),
    ] {
        let compiled = lower(schedule);
        let mut execution = schedule::Execution::default();
        let outcome = schedule::solve_compiled(
            &compiled,
            &AtomicBool::new(false),
            Options { seed: 3, optional_modes_cdcl: cdcl },
            &mut execution,
            |_| {},
        )
        .expect("supported exact schedule runs");
        assert_eq!(outcome.status, Status::Optimal);
        assert_eq!(outcome.objective, Some(objective));
        assert_eq!(execution.encoding, Some(encoding));
        assert_eq!(execution.search_kernel, Some(kernel));
        if kernel == "cdcl" {
            assert!(outcome.stats.lcg_atoms > 0, "initialization is observed independently of the invocation marker");
        }
    }
}

#[test]
fn runtime_trace_keeps_a_cached_cdcl_incumbent_when_search_is_interrupted() {
    let mut flexible = moded_schedule();
    let mut alternative = flexible.intervals[0].modes[0];
    alternative.reference = Some(1);
    flexible.intervals[0].modes.push(alternative);
    let compiled = lower(flexible.clone());
    let stop = AtomicBool::new(false);
    let mut execution = schedule::Execution::default();
    let outcome = schedule::solve_compiled(&compiled, &stop, Options { seed: 0, optional_modes_cdcl: true }, &mut execution, |_| {
        stop.store(true, Ordering::Release)
    })
    .expect("search interruption is not an engine error");
    assert_eq!(outcome.status, Status::Satisfiable);
    assert_eq!(outcome.objective, Some(1));
    assert_replayed(&flexible, &outcome);
    assert_eq!(execution.encoding, Some("moded"));
    assert_eq!(execution.search_kernel, Some("cdcl"));
    assert!(outcome.stats.lcg_atoms > 0, "this flexible model must find its incumbent through CDCL");
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
    assert_eq!(solver.store.min(starts[8]), 0, "propagation must stop within a bounded number of pairs");
    assert!(solver.fd_at_fixpoint(), "the interrupted propagation queue must be cleared");
}

#[test]
fn cdcl_returns_an_already_constructed_incumbent_when_the_stop_flag_fires() {
    let stop = AtomicBool::new(false);
    let mut improved = false;
    let outcome = schedule::solve(&moded_schedule(), &stop, Options { seed: 7, optional_modes_cdcl: true }, |_| {
        improved = true;
        stop.store(true, Ordering::Release);
    })
    .expect("cancellation after caching is not an engine error")
    .expect("the moded schedule is supported");

    assert!(improved, "the stop must fire after a complete incumbent is cached");
    assert!(stop.load(Ordering::Acquire));
    assert_eq!(outcome.status, Status::Satisfiable);
    assert_eq!(outcome.objective, Some(1));
    assert_replayed(&moded_schedule(), &outcome);
}

#[test]
fn constructive_incumbents_are_verified_before_cdcl_and_survive_a_callback_stop() {
    for (source, encoding) in [(fixed_schedule(), "fixed"), (moded_schedule(), "moded")] {
        let compiled = lower(source.clone());
        let stop = AtomicBool::new(false);
        let mut execution = schedule::Execution::default();
        let mut announced = Vec::new();
        let outcome =
            schedule::solve_compiled(&compiled, &stop, Options { seed: 0, optional_modes_cdcl: true }, &mut execution, |objective| {
                announced.push(objective);
                stop.store(true, Ordering::Release);
            })
            .unwrap();
        assert_eq!(outcome.status, Status::Satisfiable, "an amorce is not an optimality proof");
        assert_eq!(announced, outcome.objective.into_iter().collect::<Vec<_>>());
        assert_eq!(execution.encoding, Some(encoding));
        assert_eq!(execution.search_kernel, None, "stopping after construction must not claim a search invocation");
        assert_eq!(outcome.stats, Default::default());
        assert_replayed(&source, &outcome);
    }
}

fn snapshot_solution(objective: i64, snapshot: schedule::ScheduleIncumbent<'_>) -> CollectionSolution {
    CollectionSolution {
        lists: Vec::new(),
        objectives: vec![objective],
        feasible: true,
        starts: snapshot.starts.to_vec(),
        presences: snapshot.presences.to_vec(),
        machines: snapshot.machines.to_vec(),
        modes: snapshot.modes.to_vec(),
        bound: None,
    }
}

fn assert_snapshot_matches_outcome(snapshot: &CollectionSolution, outcome: &schedule::Outcome) {
    assert_eq!(snapshot.objectives, outcome.objective.into_iter().collect::<Vec<_>>());
    assert_eq!(snapshot.starts, outcome.starts);
    assert_eq!(snapshot.presences, outcome.presences);
    assert_eq!(snapshot.machines, outcome.machines);
    assert_eq!(snapshot.modes, outcome.modes);
}

#[test]
fn constructive_snapshots_are_complete_before_an_observer_cancels_search() {
    let mut native = moded_schedule();
    native.intervals[0].modes[0].machine = 13;
    for source in [fixed_schedule(), native] {
        let compiled = lower(source.clone());
        let stop = AtomicBool::new(false);
        let mut execution = schedule::Execution::default();
        let mut saved = None;
        let outcome = schedule::solve_compiled_with_observer(
            &compiled,
            &stop,
            Options { seed: 0, optional_modes_cdcl: true },
            None,
            &mut execution,
            |objective, snapshot| {
                let snapshot = snapshot.expect("construction supplies a complete physical incumbent");
                let count = source.intervals.len();
                assert_eq!(
                    (snapshot.starts.len(), snapshot.presences.len(), snapshot.machines.len(), snapshot.modes.len()),
                    (count, count, count, count)
                );
                let solution = snapshot_solution(objective, snapshot);
                assert_eq!(verify_collection_solution(&collection_with_schedule(source.clone()), &solution).unwrap(), vec![objective]);
                saved = Some(solution);
                stop.store(true, Ordering::Release);
            },
        )
        .unwrap();
        assert_eq!(outcome.status, Status::Satisfiable, "an interrupted construction cannot claim a proof");
        assert_eq!(execution.search_kernel, None);
        assert_snapshot_matches_outcome(saved.as_ref().unwrap(), &outcome);
        if !source.intervals[0].modes.is_empty() {
            assert_eq!(outcome.machines, vec![13]);
            assert_eq!(outcome.modes, vec![Some(0)]);
        }
    }
}

#[test]
fn a_cdcl_discovered_mode_snapshot_survives_observer_cancellation_with_its_identity() {
    let mut source = moded_schedule();
    source.intervals[0].modes[0].machine = 5;
    let mut alternative = source.intervals[0].modes[0];
    alternative.reference = Some(1);
    alternative.machine = 13;
    source.intervals[0].modes.push(alternative);
    let compiled = lower(source.clone());
    let stop = AtomicBool::new(false);
    let mut execution = schedule::Execution::default();
    let mut saved = None;
    let outcome = schedule::solve_compiled_with_observer(
        &compiled,
        &stop,
        Options { seed: 0, optional_modes_cdcl: true },
        None,
        &mut execution,
        |objective, snapshot| {
            saved = Some(snapshot_solution(objective, snapshot.expect("the CDCL improvement is reconstructed before notification")));
            stop.store(true, Ordering::Release);
        },
    )
    .unwrap();
    assert_eq!(execution.search_kernel, Some("cdcl"));
    assert!(outcome.stats.lcg_atoms > 0);
    assert_eq!(outcome.status, Status::Satisfiable);
    assert!(matches!((outcome.machines.as_slice(), outcome.modes.as_slice()), ([5], [Some(0)]) | ([13], [Some(1)])));
    assert_snapshot_matches_outcome(saved.as_ref().unwrap(), &outcome);
    assert_replayed(&source, &outcome);
}

#[test]
fn unobserved_general_search_progress_does_not_claim_a_reconstructed_snapshot() {
    let mut cumulative = fixed_schedule();
    cumulative.intervals.truncate(2);
    cumulative.precedences.clear();
    cumulative.resources = vec![Resource::Cumulative { demands: vec![(0, 1), (1, 1)], capacity: 1 }];
    for (source, cdcl, kernel) in [(cumulative, true, "cdcl"), (moded_schedule(), false, "chronological-dfs")] {
        let compiled = lower(source.clone());
        let mut execution = schedule::Execution::default();
        let mut events = 0;
        let outcome = schedule::solve_compiled_with_observer(
            &compiled,
            &AtomicBool::new(false),
            Options { seed: 0, optional_modes_cdcl: cdcl },
            None,
            &mut execution,
            |_, snapshot| {
                assert!(snapshot.is_none());
                events += 1;
            },
        )
        .unwrap();
        assert!(events > 0);
        assert_eq!(execution.search_kernel, Some(kernel));
        assert_eq!(outcome.status, Status::Optimal);
        assert_replayed(&source, &outcome);
    }
}

#[test]
fn failed_greedy_window_order_leaves_exact_search_free_to_find_another_order() {
    let fixed = Schedule {
        intervals: vec![
            IntervalVar { duration: 1, horizon: 4, modes: Vec::new(), optional: false },
            IntervalVar { duration: 2, horizon: 2, modes: Vec::new(), optional: false },
        ],
        precedences: Vec::new(),
        resources: vec![Resource::NoOverlap(vec![0, 1])],
        minimize_makespan: true,
    };
    let mut moded = fixed.clone();
    for (index, interval) in moded.intervals.iter_mut().enumerate() {
        interval.modes.push(Mode {
            reference: Some(index),
            machine: 0,
            duration: interval.duration,
            start_window: (0, interval.horizon - interval.duration),
        });
        interval.duration = 0;
    }
    moded.resources = vec![Resource::MachineNoOverlap];
    // Earliest finish chooses operation 0 first and blocks operation 1's
    // fixed start. The exact feasible order is operation 1, then operation 0.
    for source in [fixed, moded] {
        let compiled = lower(source.clone());
        let mut execution = schedule::Execution::default();
        let mut announcements = Vec::new();
        let outcome = schedule::solve_compiled(
            &compiled,
            &AtomicBool::new(false),
            Options { seed: 0, optional_modes_cdcl: true },
            &mut execution,
            |objective| announcements.push(objective),
        )
        .unwrap();
        assert_eq!(outcome.status, Status::Optimal);
        assert_eq!(outcome.objective, Some(3));
        assert_eq!(execution.search_kernel, Some("cdcl"));
        assert!(outcome.stats.lcg_atoms > 0);
        assert_eq!(announcements.last(), Some(&3));
        assert_replayed(&source, &outcome);
    }
}

#[test]
fn constructive_upper_bound_and_order_phases_allow_improvement_and_a_complete_proof() {
    let fixed = Schedule {
        intervals: [3, 1, 1, 10]
            .into_iter()
            .map(|duration| IntervalVar { duration, horizon: 20, modes: Vec::new(), optional: false })
            .collect(),
        precedences: vec![(1, 2), (0, 3)],
        resources: vec![Resource::NoOverlap(vec![0, 2])],
        minimize_makespan: true,
    };
    let mut moded = fixed.clone();
    for (index, interval) in moded.intervals.iter_mut().enumerate() {
        interval.modes.push(Mode {
            reference: Some(index),
            machine: if index == 2 { 0 } else { index },
            duration: interval.duration,
            start_window: (0, interval.horizon - interval.duration),
        });
        interval.duration = 0;
    }
    moded.resources = vec![Resource::MachineNoOverlap];
    // The short chain 1->2 delays operation 0 in the greedy construction,
    // yielding 15. Reversing its machine order gives the optimum 13.
    for source in [fixed, moded] {
        for seed in 0..4 {
            let mut announcements = Vec::new();
            let outcome = schedule::solve(&source, &AtomicBool::new(false), Options { seed, optional_modes_cdcl: true }, |objective| {
                announcements.push(objective)
            })
            .unwrap()
            .unwrap();
            assert_eq!(announcements, vec![15, 13]);
            assert_eq!(outcome.objective, Some(13));
            assert_eq!(outcome.status, Status::Optimal);
            assert!(outcome.stats.lcg_atoms > 0);
            assert_replayed(&source, &outcome);
        }
    }
}

#[test]
fn cyclic_or_repeated_unary_operations_never_publish_a_constructive_incumbent() {
    let mut cycle = fixed_schedule();
    cycle.precedences = vec![(0, 1), (1, 0)];
    let mut repeated = fixed_schedule();
    repeated.resources = vec![Resource::NoOverlap(vec![0, 0])];
    for source in [cycle, repeated] {
        let mut announcements = 0;
        let outcome = schedule::solve(&source, &AtomicBool::new(false), Options::default(), |_| announcements += 1)
            .expect("a failed construction must not become an engine error")
            .unwrap();
        assert_eq!(announcements, 0);
        assert_eq!(outcome.status, Status::Unsatisfiable);
        assert_eq!(outcome.objective, None);
    }
}

#[test]
fn certified_makespan_lower_bounds_preserve_incumbents_and_optimality_proofs() {
    for source in [fixed_schedule(), moded_schedule()] {
        let compiled = lower(source.clone());
        let expected = if source.intervals[0].modes.is_empty() { 8 } else { 1 };
        // The valid bound closes at the optimum. An incompatible supplied
        // bound is ignored once the verified construction disproves it.
        for bound in [expected, i64::MAX] {
            let mut execution = schedule::Execution::default();
            let outcome = schedule::solve_compiled_with_lower_bound(
                &compiled,
                &AtomicBool::new(false),
                Options { seed: 0, optional_modes_cdcl: true },
                Some(bound),
                &mut execution,
                |_| {},
            )
            .unwrap();
            assert_eq!(outcome.status, Status::Optimal);
            assert_eq!(outcome.objective, Some(expected));
            assert_eq!(execution.search_kernel, Some("cdcl"));
            assert_replayed(&source, &outcome);
        }
    }
}

#[test]
fn a_certified_load_beyond_the_horizon_proves_infeasibility_without_wrapping() {
    let fixed = Schedule {
        intervals: (0..2).map(|_| IntervalVar { duration: 3, horizon: 5, modes: Vec::new(), optional: false }).collect(),
        precedences: Vec::new(),
        resources: vec![Resource::NoOverlap(vec![0, 1])],
        minimize_makespan: true,
    };
    let mut moded = fixed.clone();
    for (index, interval) in moded.intervals.iter_mut().enumerate() {
        interval.modes.push(Mode { reference: Some(index), machine: 0, duration: 3, start_window: (0, 2) });
        interval.duration = 0;
    }
    moded.resources = vec![Resource::MachineNoOverlap];
    for source in [fixed, moded] {
        let compiled = lower(source);
        for (bound, stopped) in [(6, false), (i64::MAX, false), (6, true)] {
            let mut execution = schedule::Execution::default();
            let outcome = schedule::solve_compiled_with_lower_bound(
                &compiled,
                &AtomicBool::new(stopped),
                Options { seed: 0, optional_modes_cdcl: true },
                Some(bound),
                &mut execution,
                |_| panic!("this horizon admits no incumbent"),
            )
            .unwrap();
            assert_eq!(outcome.status, if stopped { Status::Unknown } else { Status::Unsatisfiable });
            assert_eq!(outcome.objective, None);
            assert_eq!(execution.search_kernel, None);
            assert_eq!(outcome.stats, Default::default());
        }
    }
}

#[test]
fn structural_lower_bounds_are_ignored_when_mode_dates_can_be_negative() {
    let source = Schedule {
        intervals: vec![IntervalVar {
            duration: 0,
            horizon: 3,
            modes: (0..2).map(|index| Mode { reference: Some(index), machine: index, duration: 2, start_window: (-1, 1) }).collect(),
            optional: false,
        }],
        precedences: Vec::new(),
        resources: vec![Resource::MachineNoOverlap],
        minimize_makespan: true,
    };
    let compiled = lower(source.clone());
    let mut execution = schedule::Execution::default();
    let outcome = schedule::solve_compiled_with_lower_bound(
        &compiled,
        &AtomicBool::new(false),
        Options { seed: 0, optional_modes_cdcl: true },
        Some(i64::MAX),
        &mut execution,
        |_| {},
    )
    .unwrap();
    assert_eq!(outcome.status, Status::Optimal);
    assert_eq!(outcome.objective, Some(1));
    assert_eq!(execution.search_kernel, Some("cdcl"));
    assert_replayed(&source, &outcome);
}

#[test]
fn cdcl_caches_moded_windows_and_zero_duration_cycles_before_reporting() {
    let windowed = Schedule {
        intervals: vec![
            IntervalVar {
                duration: 0,
                horizon: 12,
                modes: vec![
                    Mode { reference: Some(0), machine: 0, duration: 3, start_window: (5, 8) },
                    Mode { reference: Some(1), machine: 1, duration: 2, start_window: (0, 2) },
                ],
                optional: false,
            },
            IntervalVar {
                duration: 0,
                horizon: 12,
                modes: vec![Mode { reference: Some(2), machine: 0, duration: 4, start_window: (0, 8) }],
                optional: false,
            },
        ],
        precedences: vec![(0, 1)],
        resources: vec![Resource::MachineNoOverlap],
        minimize_makespan: true,
    };
    let zero_cycle = Schedule {
        intervals: (0..2)
            .map(|index| IntervalVar {
                duration: 0,
                horizon: 4,
                modes: vec![Mode {
                    reference: Some(index),
                    machine: index,
                    duration: 0,
                    start_window: if index == 0 { (2, 2) } else { (0, 4) },
                }],
                optional: false,
            })
            .collect(),
        precedences: vec![(0, 1), (1, 0)],
        resources: vec![Resource::MachineNoOverlap],
        minimize_makespan: true,
    };
    for schedule in [windowed, zero_cycle] {
        let stop = AtomicBool::new(false);
        let mut callbacks = 0;
        let outcome = schedule::solve(&schedule, &stop, Options { seed: 0, optional_modes_cdcl: true }, |_| {
            callbacks += 1;
            stop.store(true, Ordering::Release);
        })
        .unwrap()
        .unwrap();
        assert_eq!(callbacks, 1);
        assert_eq!(outcome.status, Status::Satisfiable);
        assert!(outcome.stats.lcg_atoms > 0);
        assert_replayed(&schedule, &outcome);
    }
}

#[test]
fn fixed_cdcl_searches_orders_without_enumerating_avoidable_waiting_time() {
    let schedule = Schedule {
        intervals: [2, 3, 2, 3]
            .into_iter()
            .map(|duration| IntervalVar { duration, horizon: 10, modes: Vec::new(), optional: false })
            .collect(),
        precedences: vec![(0, 1), (2, 3)],
        resources: vec![Resource::NoOverlap(vec![0, 3]), Resource::NoOverlap(vec![1, 2])],
        minimize_makespan: true,
    };
    // All feasible machine-order combinations have an earliest makespan of
    // either 5 (parallel jobs) or 10 (serial jobs). Intermediate objectives
    // merely enumerate waiting time within the same orders.
    for seed in 0..8 {
        let mut objectives = Vec::new();
        let outcome = schedule::solve(&schedule, &AtomicBool::new(false), Options { seed, optional_modes_cdcl: false }, |objective| {
            objectives.push(objective)
        })
        .unwrap()
        .unwrap();
        assert_eq!(outcome.status, Status::Optimal);
        assert_eq!(outcome.objective, Some(5));
        assert!(
            objectives.iter().all(|objective| matches!(objective, 5 | 10)),
            "unary order search must not enumerate slack: {objectives:?}"
        );
        assert_replayed(&schedule, &outcome);
    }
    let mut extra_precedence = schedule.clone();
    extra_precedence.precedences.push((0, 2));
    let outcome = schedule::solve(&extra_precedence, &AtomicBool::new(false), Options::default(), |_| {}).unwrap().unwrap();
    assert_eq!(outcome.status, Status::Optimal);
    assert_eq!(outcome.objective, Some(7));
    assert_replayed(&extra_precedence, &outcome);

    let mut overlapping_resource = schedule;
    overlapping_resource.resources.push(Resource::NoOverlap(vec![0, 1, 2, 3]));
    let outcome = schedule::solve(&overlapping_resource, &AtomicBool::new(false), Options::default(), |_| {}).unwrap().unwrap();
    assert_eq!(outcome.status, Status::Optimal);
    assert_eq!(outcome.objective, Some(10));
    assert_replayed(&overlapping_resource, &outcome);
}

#[test]
fn fixed_order_reconstruction_preserves_optional_presence_and_general_resource_fallbacks() {
    let optional = Schedule {
        intervals: vec![
            IntervalVar { duration: 2, horizon: 4, modes: Vec::new(), optional: true },
            IntervalVar { duration: 2, horizon: 4, modes: Vec::new(), optional: false },
        ],
        precedences: vec![(0, 1)],
        resources: vec![Resource::NoOverlap(vec![0, 1])],
        minimize_makespan: true,
    };
    let cumulative = Schedule {
        intervals: [2, 3].into_iter().map(|duration| IntervalVar { duration, horizon: 5, modes: Vec::new(), optional: false }).collect(),
        precedences: Vec::new(),
        resources: vec![Resource::Cumulative { demands: vec![(0, 1), (1, 1)], capacity: 1 }],
        minimize_makespan: true,
    };
    let zero_cycle = Schedule {
        intervals: (0..2).map(|_| IntervalVar { duration: 0, horizon: 1, modes: Vec::new(), optional: false }).collect(),
        precedences: vec![(0, 1), (1, 0)],
        resources: vec![Resource::NoOverlap(vec![0, 1])],
        minimize_makespan: true,
    };
    for (schedule, objective) in [(optional, 2), (cumulative, 5), (zero_cycle, 0)] {
        let outcome = schedule::solve(&schedule, &AtomicBool::new(false), Options::default(), |_| {}).unwrap().unwrap();
        assert_eq!(outcome.status, Status::Optimal);
        assert_eq!(outcome.objective, Some(objective));
        assert_replayed(&schedule, &outcome);
    }
}

#[test]
fn order_primary_scope_completes_makespan_without_any_resource_pair_decision() {
    let fixed = Schedule {
        intervals: vec![IntervalVar { duration: 2, horizon: 4, modes: Vec::new(), optional: false }],
        precedences: Vec::new(),
        resources: Vec::new(),
        minimize_makespan: true,
    };
    for source in [fixed, moded_schedule()] {
        let expected = if source.intervals[0].modes.is_empty() { 2 } else { 1 };
        let compiled = lower(source.clone());
        let mut execution = schedule::Execution::default();
        let mut announcements = Vec::new();
        let outcome = schedule::solve_compiled(
            &compiled,
            &AtomicBool::new(false),
            Options { seed: 0, optional_modes_cdcl: true },
            &mut execution,
            |objective| announcements.push(objective),
        )
        .unwrap();
        assert_eq!(announcements, vec![expected]);
        assert_eq!(outcome.objective, Some(expected));
        assert_eq!(outcome.status, Status::Optimal);
        assert_eq!(execution.search_kernel, Some("cdcl"));
        assert_replayed(&source, &outcome);
    }
}
