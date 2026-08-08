use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use qayd::engines::ls::lists::solve_schedule;
use qayd::model::list::{verify_collection_solution, CollectionModel, CollectionSolution, IntervalVar, Mode, Resource, Schedule};

fn assert_unknown(solution: &qayd::model::list::CollectionSolution) {
    assert!(!solution.feasible);
    assert!(solution.lists.is_empty());
    assert!(solution.objectives.is_empty());
    assert!(solution.starts.is_empty());
    assert!(solution.presences.is_empty());
    assert!(solution.machines.is_empty());
    assert!(solution.modes.is_empty());
    assert!(solution.bound.is_none());
}

fn fixed_no_overlap(count: usize) -> Schedule {
    Schedule {
        intervals: (0..count).map(|_| IntervalVar { duration: 1, horizon: count as i64, modes: Vec::new(), optional: false }).collect(),
        precedences: Vec::new(),
        resources: vec![Resource::NoOverlap((0..count).collect())],
        minimize_makespan: true,
    }
}

fn optional_machine_schedule(count: usize) -> Schedule {
    Schedule {
        intervals: (0..count)
            .map(|index| IntervalVar {
                duration: 0,
                horizon: 1,
                modes: vec![Mode { reference: Some(index), machine: 0, duration: 1, start_window: (0, 0) }],
                optional: true,
            })
            .collect(),
        precedences: Vec::new(),
        resources: vec![Resource::MachineNoOverlap],
        minimize_makespan: true,
    }
}

#[test]
fn schedule_ls_prearmed_stop_returns_unknown_before_construction() {
    let schedule = fixed_no_overlap(100_000);
    let stop = AtomicBool::new(true);
    let mut reports = 0usize;
    let started = Instant::now();

    let (solution, metrics) = solve_schedule(&schedule, 7, &stop, &mut |_| reports += 1);

    assert_unknown(&solution);
    assert_eq!(reports, 0);
    assert_eq!(metrics.candidates, 0);
    assert_eq!(metrics.first_feasible, None);
    assert!(started.elapsed() < Duration::from_millis(250));
}

#[test]
fn schedule_ls_stops_inside_large_resource_construction() {
    let schedule = fixed_no_overlap(1_500);
    let stop = AtomicBool::new(false);
    let started = Instant::now();

    let (solution, metrics) = std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(Duration::from_millis(5));
            stop.store(true, Ordering::Release);
        });
        solve_schedule(&schedule, 11, &stop, &mut |_| {})
    });

    assert!(stop.load(Ordering::Acquire));
    assert_unknown(&solution);
    assert!(metrics.candidates > 0, "the constructor must begin before cancellation");
    assert_eq!(metrics.first_feasible, None);
    assert!(started.elapsed() < Duration::from_secs(1));
}

#[test]
fn schedule_ls_stops_inside_pairwise_initial_scoring() {
    let schedule = optional_machine_schedule(25_000);
    let stop = AtomicBool::new(false);
    let started = Instant::now();

    let (solution, metrics) = std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(Duration::from_millis(50));
            stop.store(true, Ordering::Release);
        });
        solve_schedule(&schedule, 13, &stop, &mut |_| {})
    });

    assert!(stop.load(Ordering::Acquire));
    assert_unknown(&solution);
    assert_eq!(metrics.first_feasible, None);
    assert!(started.elapsed() < Duration::from_secs(1));
}

#[test]
fn schedule_ls_keeps_only_the_complete_incumbent_when_a_trial_is_interrupted() {
    let demand_entries = 500_000;
    let schedule = Schedule {
        intervals: vec![IntervalVar { duration: 1, horizon: 1, modes: Vec::new(), optional: true }],
        precedences: Vec::new(),
        resources: vec![Resource::Cumulative { demands: vec![(0, 1); demand_entries], capacity: demand_entries as i64 }],
        minimize_makespan: true,
    };
    let stop = AtomicBool::new(false);
    let incumbent_ready = AtomicBool::new(false);
    let started = Instant::now();

    let (solution, metrics) = std::thread::scope(|scope| {
        scope.spawn(|| {
            while !incumbent_ready.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            std::thread::sleep(Duration::from_millis(2));
            stop.store(true, Ordering::Release);
        });
        solve_schedule(&schedule, 17, &stop, &mut |_| {
            incumbent_ready.store(true, Ordering::Release);
        })
    });

    assert!(stop.load(Ordering::Acquire));
    assert!(solution.feasible);
    assert_eq!(solution.objectives, vec![0]);
    assert_eq!(solution.starts, vec![0]);
    assert_eq!(solution.presences, vec![false]);
    assert_eq!(solution.machines, vec![-1]);
    assert_eq!(solution.modes, vec![None]);
    assert!(metrics.first_feasible.is_some());
    assert!(started.elapsed() < Duration::from_secs(1));
}

#[test]
fn schedule_ls_keeps_a_complete_cumulative_incumbent_when_reporting_stops_search() {
    let schedule = Schedule {
        intervals: (0..4).map(|_| IntervalVar { duration: 2, horizon: 8, modes: Vec::new(), optional: false }).collect(),
        precedences: Vec::new(),
        resources: vec![Resource::Cumulative { demands: (0..4).map(|index| (index, 1)).collect(), capacity: 2 }],
        minimize_makespan: true,
    };
    let stop = AtomicBool::new(false);

    let (solution, metrics) = solve_schedule(&schedule, 19, &stop, &mut |_| stop.store(true, Ordering::Release));

    assert!(stop.load(Ordering::Acquire));
    assert!(solution.feasible);
    assert_eq!(solution.objectives, vec![4]);
    assert_eq!(solution.starts, vec![0, 0, 2, 2]);
    assert_eq!(solution.presences, vec![true; 4]);
    assert!(metrics.first_feasible.is_some());
}

#[test]
fn cumulative_usage_above_i64_max_never_wraps_to_feasible() {
    let schedule = Schedule {
        intervals: (0..2).map(|_| IntervalVar { duration: 1, horizon: 2, modes: Vec::new(), optional: false }).collect(),
        precedences: Vec::new(),
        resources: vec![Resource::Cumulative { demands: vec![(0, i64::MAX), (1, i64::MAX)], capacity: i64::MAX }],
        minimize_makespan: true,
    };
    let model = CollectionModel {
        items: Vec::new(),
        lists: 0,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: Some(schedule.clone()),
    };
    let overlapping = CollectionSolution {
        lists: Vec::new(),
        objectives: vec![1],
        feasible: true,
        starts: vec![0, 0],
        presences: vec![true, true],
        machines: vec![-1, -1],
        modes: vec![None, None],
        bound: None,
    };

    let error = verify_collection_solution(&model, &overlapping).expect_err("two maximum demands exceed one maximum capacity");
    assert!(error.contains("18446744073709551614"), "canonical usage must be accumulated exactly: {error}");

    let stop = AtomicBool::new(false);
    let (solution, _) = solve_schedule(&schedule, 29, &stop, &mut |_| stop.store(true, Ordering::Release));
    assert!(solution.feasible);
    assert_eq!(solution.objectives, vec![2]);
    assert_ne!(solution.starts[0], solution.starts[1]);
    assert_eq!(verify_collection_solution(&model, &solution).unwrap(), vec![2]);
}
