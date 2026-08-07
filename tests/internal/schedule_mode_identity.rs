use std::sync::atomic::AtomicBool;

use qayd::engines::schedule::{self, Options, Status};
use qayd::model::list::{verify_collection_solution, CollectionSolution, Resource};
use qayd::model::{CompiledCollection, Constraint, IntervalModeRef, Model, ModelPackage, Objective};
use qayd::orchestrator::{solve_model_silent, SolveMode, SolveRequest, SolveStatus};

fn compiled_schedule() -> (CompiledCollection, IntervalModeRef, IntervalModeRef, IntervalModeRef) {
    let mut model = Model::new();
    let first = model.interval(0, 10, 0);
    let second = model.interval(0, 10, 0);

    // Add modes in an interleaved order. The fast mode of `first` therefore has
    // stable arena identity 2, while its local position within the interval is 1.
    let first_slow = model.add_interval_mode(first, 3, 8, Some((0, 0))).unwrap();
    let second_mode = model.add_interval_mode(second, 4, 1, Some((0, 0))).unwrap();
    let first_fast = model.add_interval_mode(first, 3, 1, Some((4, 4))).unwrap();
    model.add_constraint(Constraint::IntervalResource(Resource::MachineNoOverlap));
    model.add_objective(Objective::Makespan { minimize: true, intervals: vec![first, second] });
    model.validate().unwrap();

    (CompiledCollection::compile(&model).unwrap(), first_slow, second_mode, first_fast)
}

#[test]
fn exact_schedule_decodes_stable_mode_identity_and_window() {
    let (compiled, first_slow, second_mode, first_fast) = compiled_schedule();
    let physical = compiled.as_model();
    let schedule = physical.schedule.as_ref().unwrap();
    assert_eq!(schedule.intervals[0].modes[0].reference, Some(first_slow.0));
    assert_eq!(schedule.intervals[0].modes[1].reference, Some(first_fast.0));
    assert_eq!(schedule.intervals[0].modes[1].start_window, (4, 4));

    for optional_modes_cdcl in [false, true] {
        let outcome =
            schedule::solve(schedule, &AtomicBool::new(false), Options { seed: 17, optional_modes_cdcl }, |_| {}).unwrap().unwrap();
        assert_eq!(outcome.status, Status::Optimal);
        assert_eq!(outcome.objective, Some(5));
        assert_eq!(outcome.starts, vec![4, 0]);
        assert_eq!(outcome.machines, vec![3, 4]);
        assert_eq!(outcome.modes, vec![Some(first_fast.0), Some(second_mode.0)]);
    }
}

#[test]
fn canonical_schedule_replay_checks_mode_machine_duration_and_window() {
    let (compiled, first_slow, second_mode, first_fast) = compiled_schedule();
    let physical = compiled.as_model();
    let valid = CollectionSolution {
        lists: Vec::new(),
        objectives: vec![5],
        feasible: true,
        starts: vec![4, 0],
        presences: vec![true, true],
        machines: vec![3, 4],
        modes: vec![Some(first_fast.0), Some(second_mode.0)],
        bound: None,
    };

    // Both modes of interval 0 use machine 3. Its stable identity is therefore
    // what selects duration 1 and produces the canonical makespan 5.
    assert_eq!(verify_collection_solution(physical, &valid).unwrap(), vec![5]);

    let mut slow_mode = valid.clone();
    slow_mode.starts[0] = 0;
    slow_mode.modes[0] = Some(first_slow.0);
    slow_mode.objectives[0] = 8;
    assert_eq!(verify_collection_solution(physical, &slow_mode).unwrap(), vec![8]);

    let mut wrong_machine = valid.clone();
    wrong_machine.machines[0] = 4;
    assert!(verify_collection_solution(physical, &wrong_machine).unwrap_err().contains("uses machine 3"));

    let mut missing_mode = valid.clone();
    missing_mode.modes[0] = None;
    assert!(verify_collection_solution(physical, &missing_mode).unwrap_err().contains("omits its semantic mode identity"));

    let mut outside_window = valid;
    outside_window.starts[0] = 3;
    assert!(verify_collection_solution(physical, &outside_window).unwrap_err().contains("outside mode window 4..4"));
}

#[test]
fn local_search_solution_preserves_stable_mode_identity() {
    let mut model = Model::new();
    let first = model.interval(0, 10, 0);
    let dummies = (0..48).map(|_| model.interval(0, 0, 0)).collect::<Vec<_>>();
    model.add_interval_mode(first, 3, 8, Some((0, 0))).unwrap();
    for (index, &dummy) in dummies.iter().enumerate() {
        model.add_interval_mode(dummy, index + 10, 0, Some((0, 0))).unwrap();
    }
    let first_fast = model.add_interval_mode(first, 3, 1, Some((4, 4))).unwrap();
    let intervals = std::iter::once(first).chain(dummies).collect::<Vec<_>>();
    model.add_constraint(Constraint::IntervalResource(Resource::MachineNoOverlap));
    model.add_objective(Objective::Makespan { minimize: true, intervals });

    let request = SolveRequest { mode: SolveMode::LocalSearch, seed: 23, ..SolveRequest::default() };
    let result = solve_model_silent(&ModelPackage::new(model), &request).unwrap();
    assert_eq!(result.status(), SolveStatus::Satisfiable);
    let candidate = result.primal().unwrap();
    assert_eq!(candidate.objectives(), [5]);
    assert_eq!(candidate.assignment().intervals[0].start, Some(4));
    assert_eq!(candidate.assignment().intervals[0].machine, Some(3));
    assert_eq!(candidate.assignment().intervals[0].mode, Some(first_fast.0));
}
