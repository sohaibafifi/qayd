use std::cell::Cell;

use crate::constraints::resource_profile::{EnergeticAnalysis, EnergeticConflict, EnergeticWorkspace, FixedCumulativeTask};

#[derive(Clone, Copy, Debug)]
struct TaskSpec {
    earliest: i32,
    latest: i32,
    duration: i64,
    demand: i64,
}

impl TaskSpec {
    fn kernel(self) -> FixedCumulativeTask {
        FixedCumulativeTask::new(self.earliest, self.latest, self.duration, self.demand)
    }
}

fn feasible_assignments(tasks: &[TaskSpec], capacity: i128) -> Vec<Vec<i32>> {
    fn visit(tasks: &[TaskSpec], capacity: i128, index: usize, starts: &mut Vec<i32>, out: &mut Vec<Vec<i32>>) {
        if index != tasks.len() {
            for start in tasks[index].earliest..=tasks[index].latest {
                starts.push(start);
                visit(tasks, capacity, index + 1, starts, out);
                starts.pop();
            }
            return;
        }

        let lower = starts.iter().copied().min().unwrap_or(0);
        let upper = tasks
            .iter()
            .zip(starts.iter().copied())
            .map(|(task, start)| i128::from(start) + i128::from(task.duration))
            .max()
            .unwrap_or(i128::from(lower));
        let feasible = (i128::from(lower)..upper).all(|time| {
            tasks
                .iter()
                .zip(starts.iter().copied())
                .filter(|(task, start)| {
                    let start = i128::from(*start);
                    start <= time && time < start + i128::from(task.duration)
                })
                .map(|(task, _)| i128::from(task.demand))
                .sum::<i128>()
                <= capacity
        });
        if feasible {
            out.push(starts.clone());
        }
    }

    let mut out = Vec::new();
    visit(tasks, capacity, 0, &mut Vec::new(), &mut out);
    out
}

#[test]
fn energetic_conflict_exposes_the_exact_overloaded_subset() {
    let tasks = vec![FixedCumulativeTask::new(0, 0, 1, 1), FixedCumulativeTask::new(0, 0, 1, 1), FixedCumulativeTask::new(0, 0, 1, 1)];
    let mut workspace = EnergeticWorkspace::default();
    let conflict = workspace.analyse(&tasks, 2).expect_err("three units cannot fit in capacity two");

    let EnergeticConflict::Overload(conflict) = conflict else {
        panic!("expected an energetic overload witness");
    };
    assert_eq!((conflict.window_start, conflict.window_end), (0, 1));
    assert_eq!((conflict.energy, conflict.available_energy), (3, 2));
    assert_eq!(workspace.conflict_tasks(), &[0, 1, 2]);
}

#[test]
fn energetic_kernel_keeps_i32_boundary_endpoints_in_i128() {
    let tasks = [FixedCumulativeTask::new(i32::MAX, i32::MAX, 1, 1)];
    let mut workspace = EnergeticWorkspace::default();

    workspace.analyse(&tasks, 1).expect("one task ending at i32::MAX + 1 is feasible");
    assert_eq!(workspace.lower_bounds(), &[i128::from(i32::MAX)]);
}

#[test]
fn edge_bound_beyond_latest_start_is_an_explicit_conflict() {
    // On a unary resource the long task cannot overlap the unit task fixed at
    // i32::MAX. It would have to start at i32::MAX + 1, beyond both its latest
    // start and the Store's representable range. This must not be clamped back
    // to i32::MAX and mistaken for a no-op.
    let tasks = [FixedCumulativeTask::new(i32::MAX - 1, i32::MAX, 2, 1), FixedCumulativeTask::new(i32::MAX, i32::MAX, 1, 1)];
    let mut workspace = EnergeticWorkspace::default();

    let conflict = workspace.analyse(&tasks, 1).expect_err("the derived edge bound is outside the task window");
    let EnergeticConflict::EdgeBound(conflict) = conflict else {
        panic!("expected an edge-bound conflict, got {conflict:?}");
    };
    assert_eq!(conflict.task, 0);
    assert_eq!(conflict.lower_bound, i128::from(i32::MAX) + 1);
    assert_eq!(conflict.latest_start, i128::from(i32::MAX));
}

#[test]
fn energetic_and_edge_results_are_sound_against_an_exhaustive_oracle() {
    for capacity in 1..=3i64 {
        let shapes: Vec<TaskSpec> = (0..=1)
            .flat_map(|earliest| {
                (earliest..=2).flat_map(move |latest| {
                    (1..=2).flat_map(move |duration| (1..=capacity).map(move |demand| TaskSpec { earliest, latest, duration, demand }))
                })
            })
            .collect();

        for first in 0..shapes.len() {
            for second in 0..shapes.len() {
                for third in 0..shapes.len() {
                    let specs = [shapes[first], shapes[second], shapes[third]];
                    let tasks = specs.iter().copied().map(TaskSpec::kernel).collect::<Vec<_>>();
                    let feasible = feasible_assignments(&specs, i128::from(capacity));
                    let mut workspace = EnergeticWorkspace::default();
                    match workspace.analyse(&tasks, i128::from(capacity)) {
                        Err(conflict) => assert!(
                            feasible.is_empty(),
                            "false conflict {conflict:?} for capacity={capacity}, tasks={specs:?}, feasible={feasible:?}"
                        ),
                        Ok(()) => {
                            for (task, &lower_bound) in workspace.lower_bounds().iter().enumerate() {
                                assert!(
                                    feasible.iter().all(|starts| i128::from(starts[task]) >= lower_bound),
                                    "unsound lower bound {lower_bound} for task {task}, capacity={capacity}, tasks={specs:?}, feasible={feasible:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn energetic_workspace_can_be_reused_across_different_task_counts() {
    let infeasible = vec![FixedCumulativeTask::new(0, 0, 1, 1); 3];
    let feasible = [FixedCumulativeTask::new(4, 4, 2, 2)];
    let mut workspace = EnergeticWorkspace::default();

    assert!(workspace.analyse(&infeasible, 2).is_err());
    workspace.analyse(&feasible, 2).unwrap();
    assert_eq!(workspace.lower_bounds(), &[4]);
    assert!(workspace.analyse(&infeasible, 2).is_err());
}

#[test]
fn energetic_analysis_honors_a_prearmed_stop_without_reporting_a_conflict() {
    let infeasible = vec![FixedCumulativeTask::new(0, 0, 1, 1); 3];
    let mut workspace = EnergeticWorkspace::default();

    let outcome = workspace.analyse_until(&infeasible, 2, &|| true).expect("cancellation is not an energetic conflict");

    assert_eq!(outcome, EnergeticAnalysis::Interrupted);
    assert!(workspace.conflict_tasks().is_empty());

    let feasible = [FixedCumulativeTask::new(4, 4, 2, 2)];
    workspace.analyse(&feasible, 2).expect("the workspace remains reusable after a prearmed stop");
    assert_eq!(workspace.lower_bounds(), &[4]);
}

#[test]
fn energetic_analysis_stops_inside_quadratic_work_and_can_be_reused() {
    let tasks = vec![FixedCumulativeTask::new(0, 1_000, 1, 1); 128];
    let mut workspace = EnergeticWorkspace::default();
    let polls = Cell::new(0usize);
    let should_stop = || {
        let next = polls.get() + 1;
        polls.set(next);
        next >= 5
    };

    let outcome = workspace.analyse_until(&tasks, 128, &should_stop).expect("cancellation inside energetic reasoning is not a conflict");

    assert_eq!(outcome, EnergeticAnalysis::Interrupted);
    assert_eq!(polls.get(), 5, "the fifth poll occurs inside the first quadratic scan");
    assert!(workspace.conflict_tasks().is_empty());

    workspace.analyse(&tasks, 128).expect("an interrupted workspace must support a complete retry");
    assert_eq!(workspace.lower_bounds(), vec![0; tasks.len()]);
}
