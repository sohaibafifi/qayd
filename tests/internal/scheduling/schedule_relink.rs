use std::sync::atomic::AtomicBool;

use qayd::engines::ls::lists::move_acceptance::MinimizingMoveAcceptance;
use qayd::engines::ls::lists::schedule_elite::{ScheduleEliteArchive, ScheduleEliteError};
use qayd::engines::ls::lists::schedule_relink::{
    ScheduleRelinkGuideKind, ScheduleRelinkMetrics, ScheduleRelinkRequest, ScheduleRelinkWorkspace, RELINK_MACRO_CAPACITY,
    RELINK_MACRO_MIN_MOVES, RELINK_ORACLE_CAPACITY,
};
use qayd::engines::ls::lists::schedule_state::{DispatchRule, JobShopProblem, JobShopState, MoveOutcome, MoveRejection};
use qayd::model::list::{IntervalVar, Resource, Schedule};

fn diversified_job_shop() -> Schedule {
    let durations = [3, 2, 2, 2, 1, 4, 4, 3, 1, 2, 3, 3];
    let horizon = durations.iter().sum();
    Schedule {
        intervals: durations.into_iter().map(|duration| IntervalVar { duration, horizon, modes: Vec::new(), optional: false }).collect(),
        precedences: vec![(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (9, 10), (10, 11)],
        resources: vec![
            Resource::NoOverlap(vec![0, 3, 7, 11]),
            Resource::NoOverlap(vec![1, 5, 6, 10]),
            Resource::NoOverlap(vec![2, 4, 8, 9]),
        ],
        minimize_makespan: true,
    }
}

fn nine_block_macro_job_shop() -> (Schedule, Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let machine_count = 9usize;
    let operation_count = machine_count * 2;
    let schedule = Schedule {
        intervals: vec![
            IntervalVar { duration: 1, horizon: i64::try_from(operation_count).unwrap(), modes: Vec::new(), optional: false };
            operation_count
        ],
        precedences: (0..machine_count - 1).map(|machine| (machine * 2 + 1, (machine + 1) * 2)).collect(),
        resources: (0..machine_count).map(|machine| Resource::NoOverlap(vec![machine * 2, machine * 2 + 1])).collect(),
        minimize_makespan: true,
    };
    let current = (0..machine_count).map(|machine| vec![machine * 2, machine * 2 + 1]).collect();
    let guide = (0..machine_count).map(|machine| vec![machine * 2 + 1, machine * 2]).collect();
    (schedule, current, guide)
}

fn coordinated_unknown_macro_job_shop() -> (Schedule, Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let durations = [9, 5, 3, 2, 8, 5, 2, 6, 8, 9, 4, 5, 2, 9, 2, 3];
    let horizon = durations.iter().sum();
    let schedule = Schedule {
        intervals: durations.into_iter().map(|duration| IntervalVar { duration, horizon, modes: Vec::new(), optional: false }).collect(),
        precedences: vec![(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11), (12, 13), (13, 14), (14, 15)],
        resources: vec![
            Resource::NoOverlap(vec![2, 5, 9, 13]),
            Resource::NoOverlap(vec![0, 6, 10, 14]),
            Resource::NoOverlap(vec![3, 7, 11, 12]),
            Resource::NoOverlap(vec![1, 4, 8, 15]),
        ],
        minimize_makespan: true,
    };
    // Compact machine ids follow first occurrence in operation order, which
    // maps raw machines [1, 3, 0, 2] to compact ids [0, 1, 2, 3].
    let current = vec![vec![10, 0, 6, 14], vec![8, 4, 15, 1], vec![5, 9, 13, 2], vec![11, 12, 3, 7]];
    let guide = vec![vec![10, 0, 6, 14], vec![8, 1, 4, 15], vec![9, 13, 2, 5], vec![11, 12, 3, 7]];
    (schedule, current, guide)
}

fn state(problem: &JobShopProblem, seed: u64) -> JobShopState {
    JobShopState::giffler_thompson(problem, seed, DispatchRule::Randomized, &AtomicBool::new(false)).unwrap().unwrap()
}

fn find_relink_case() -> (JobShopProblem, Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&diversified_job_shop(), &stop).unwrap().unwrap();
    for current_seed in 0..64 {
        let current_order = state(&problem, current_seed).machine_sequences().to_vec();
        for guide_seed in 0..64 {
            let guide_state = state(&problem, guide_seed);
            if guide_state.machine_sequences() == current_order {
                continue;
            }
            let mut archive = ScheduleEliteArchive::new();
            archive.consider(&guide_state, &stop).unwrap();
            let mut current = JobShopState::from_machine_sequences(&problem, current_order.clone(), &stop).unwrap().unwrap();
            let mut workspace = ScheduleRelinkWorkspace::default();
            let mut metrics = ScheduleRelinkMetrics::default();
            workspace
                .prepare(
                    &mut current,
                    ScheduleRelinkRequest { guide: archive.best().unwrap(), kind: ScheduleRelinkGuideKind::Best },
                    &mut metrics,
                    &stop,
                )
                .unwrap();
            if !workspace.shortlist().is_empty() {
                return (problem, current_order, guide_state.machine_sequences().to_vec());
            }
        }
    }
    panic!("expected a bounded guide-directed insertion on the diversified JSSP")
}

fn archive_for_order(problem: &JobShopProblem, order: Vec<Vec<usize>>) -> ScheduleEliteArchive {
    let stop = AtomicBool::new(false);
    let guide = JobShopState::from_machine_sequences(problem, order, &stop).unwrap().unwrap();
    let mut archive = ScheduleEliteArchive::new();
    archive.consider(&guide, &stop).unwrap();
    archive
}

fn apply_move(order: &mut [Vec<usize>], movement: qayd::engines::ls::lists::schedule_state::ScheduleMove) {
    match movement {
        qayd::engines::ls::lists::schedule_state::ScheduleMove::AdjacentSwap { machine, first_position } => {
            order[machine].swap(first_position, first_position + 1);
        }
        qayd::engines::ls::lists::schedule_state::ScheduleMove::Insert { machine, from, to } => {
            let operation = order[machine].remove(from);
            order[machine].insert(to, operation);
        }
    }
}

fn guide_arc_count(order: &[Vec<usize>], guide: &[Vec<usize>]) -> usize {
    let guide_arcs = guide
        .iter()
        .enumerate()
        .flat_map(|(machine, sequence)| sequence.windows(2).map(move |pair| (machine, pair[0], pair[1])))
        .collect::<std::collections::BTreeSet<_>>();
    order
        .iter()
        .enumerate()
        .flat_map(|(machine, sequence)| sequence.windows(2).map(move |pair| (machine, pair[0], pair[1])))
        .filter(|arc| guide_arcs.contains(arc))
        .count()
}

#[test]
fn n8_relink_is_deterministic_bounded_and_oracle_transactional() {
    let stop = AtomicBool::new(false);
    let (problem, current_order, guide_order) = find_relink_case();
    let archive = archive_for_order(&problem, guide_order.clone());
    let request = ScheduleRelinkRequest { guide: archive.best().unwrap(), kind: ScheduleRelinkGuideKind::Diverse };
    let mut first_state = JobShopState::from_machine_sequences(&problem, current_order.clone(), &stop).unwrap().unwrap();
    let mut second_state = JobShopState::from_machine_sequences(&problem, current_order, &stop).unwrap().unwrap();
    let mut first = ScheduleRelinkWorkspace::default();
    let mut second = ScheduleRelinkWorkspace::default();
    let mut first_metrics = ScheduleRelinkMetrics::default();
    let mut second_metrics = ScheduleRelinkMetrics::default();

    first.prepare(&mut first_state, request, &mut first_metrics, &stop).unwrap();
    second.prepare(&mut second_state, request, &mut second_metrics, &stop).unwrap();

    assert_eq!(first.shortlist(), second.shortlist());
    assert!(!first.shortlist().is_empty());
    assert!(first.shortlist().len() <= RELINK_ORACLE_CAPACITY);
    assert!(first.shortlist().iter().all(|candidate| candidate.guide_arc_gain > 0));
    let before_guide_arcs = guide_arc_count(first_state.machine_sequences(), &guide_order);
    for candidate in first.shortlist() {
        let mut moved = first_state.machine_sequences().to_vec();
        apply_move(&mut moved, candidate.movement);
        assert_eq!(guide_arc_count(&moved, &guide_order).saturating_sub(before_guide_arcs), usize::from(candidate.guide_arc_gain));
    }
    assert!(first_metrics.candidates_retained <= 16);
    assert_eq!(first_metrics.candidates_shortlisted, u64::try_from(first.shortlist().len()).unwrap());
    assert_eq!(first_metrics.best_guides, 0);
    assert_eq!(first_metrics.diverse_guides, 1);
    assert!(first.workspace_heap_bound_is_large_ta_safe());

    let movement = first.shortlist()[0].movement;
    let before = first_state.machine_sequences().to_vec();
    let outcome = first_state.consider_move_full_oracle(movement, MinimizingMoveAcceptance::Always, &stop).unwrap();
    match outcome {
        MoveOutcome::Accepted { .. } => assert_ne!(first_state.machine_sequences(), before),
        MoveOutcome::Rejected(_) => assert_eq!(first_state.machine_sequences(), before),
    }
    assert!(first_state.matches_full_oracle(&stop).unwrap());
}

#[test]
fn identical_guide_has_no_path_move() {
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&diversified_job_shop(), &stop).unwrap().unwrap();
    let mut current = state(&problem, 7);
    let archive = archive_for_order(&problem, current.machine_sequences().to_vec());
    let mut workspace = ScheduleRelinkWorkspace::default();
    let mut metrics = ScheduleRelinkMetrics::default();

    workspace
        .prepare(
            &mut current,
            ScheduleRelinkRequest { guide: archive.best().unwrap(), kind: ScheduleRelinkGuideKind::Best },
            &mut metrics,
            &stop,
        )
        .unwrap();

    assert!(workspace.shortlist().is_empty());
    assert_eq!(metrics.no_move, 1);
    assert_eq!(metrics.oracle_attempts, 0);
}

#[test]
fn guide_load_fails_closed_on_incompatibility_and_interruption() {
    let stop = AtomicBool::new(false);
    let guide_problem = JobShopProblem::recognize(&diversified_job_shop(), &stop).unwrap().unwrap();
    let archive = archive_for_order(&guide_problem, state(&guide_problem, 3).machine_sequences().to_vec());
    let different = Schedule {
        intervals: vec![IntervalVar { duration: 1, horizon: 2, modes: Vec::new(), optional: false }; 2],
        precedences: Vec::new(),
        resources: vec![Resource::NoOverlap(vec![0, 1])],
        minimize_makespan: true,
    };
    let different_problem = JobShopProblem::recognize(&different, &stop).unwrap().unwrap();
    let mut different_state = state(&different_problem, 0);
    let different_before = different_state.machine_sequences().to_vec();
    let mut workspace = ScheduleRelinkWorkspace::default();
    let mut metrics = ScheduleRelinkMetrics::default();
    let request = ScheduleRelinkRequest { guide: archive.best().unwrap(), kind: ScheduleRelinkGuideKind::Best };

    assert_eq!(workspace.prepare(&mut different_state, request, &mut metrics, &stop), Err(ScheduleEliteError::IncompatibleProblem));
    assert_eq!(different_state.machine_sequences(), different_before);
    assert_eq!(metrics.guide_incompatible, 1);

    let mut matching = state(&guide_problem, 9);
    let matching_before = matching.machine_sequences().to_vec();
    let interrupted = AtomicBool::new(true);
    let mut interrupted_metrics = ScheduleRelinkMetrics::default();
    assert_eq!(workspace.prepare(&mut matching, request, &mut interrupted_metrics, &interrupted), Err(ScheduleEliteError::Interrupted));
    assert_eq!(matching.machine_sequences(), matching_before);
    assert_eq!(interrupted_metrics.guide_interruptions, 1);
}

#[test]
fn macro_relink_is_deterministic_capped_distinct_and_exact_without_applying() {
    let stop = AtomicBool::new(false);
    let (schedule, current_order, guide_order) = nine_block_macro_job_shop();
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let archive = archive_for_order(&problem, guide_order.clone());
    let request = ScheduleRelinkRequest { guide: archive.best().unwrap(), kind: ScheduleRelinkGuideKind::Best };
    let mut first_state = JobShopState::from_machine_sequences(&problem, current_order.clone(), &stop).unwrap().unwrap();
    let mut second_state = JobShopState::from_machine_sequences(&problem, current_order.clone(), &stop).unwrap().unwrap();
    first_state.retain_canonical_critical_blocks_only();
    second_state.retain_canonical_critical_blocks_only();
    assert!(first_state.critical_blocks().is_empty());
    assert_eq!(first_state.canonical_critical_blocks().len(), 9);
    let mut first = ScheduleRelinkWorkspace::default();
    let mut second = ScheduleRelinkWorkspace::default();
    let mut first_metrics = ScheduleRelinkMetrics::default();
    let mut second_metrics = ScheduleRelinkMetrics::default();

    first.prepare_macro(&mut first_state, request, &mut first_metrics, &stop).unwrap();
    second.prepare_macro(&mut second_state, request, &mut second_metrics, &stop).unwrap();

    assert_eq!(first.shortlist(), second.shortlist());
    assert_eq!(first.shortlist().len(), RELINK_MACRO_CAPACITY);
    assert!(first.shortlist().len() >= RELINK_MACRO_MIN_MOVES);
    assert!(first.shortlist().iter().all(|candidate| {
        matches!(candidate.movement, qayd::engines::ls::lists::schedule_state::ScheduleMove::Insert { .. }) && candidate.guide_arc_gain > 0
    }));
    let machines = first
        .shortlist()
        .iter()
        .map(|candidate| match candidate.movement {
            qayd::engines::ls::lists::schedule_state::ScheduleMove::Insert { machine, .. } => machine,
            qayd::engines::ls::lists::schedule_state::ScheduleMove::AdjacentSwap { .. } => unreachable!(),
        })
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(machines.len(), first.shortlist().len());
    assert_eq!(first_state.machine_sequences(), current_order);
    assert_eq!(second_state.machine_sequences(), current_order);

    let before_guide_arcs = guide_arc_count(&current_order, &guide_order);
    let mut moved = current_order.clone();
    for candidate in first.shortlist() {
        apply_move(&mut moved, candidate.movement);
    }
    let exact_gain = guide_arc_count(&moved, &guide_order).saturating_sub(before_guide_arcs);
    assert_eq!(u64::try_from(exact_gain).unwrap(), first.shortlist_guide_arc_gain());
    assert_eq!(first_metrics.guide_arc_gain_shortlisted, first.shortlist_guide_arc_gain());
    assert_eq!(first_metrics.candidates_shortlisted, u64::try_from(RELINK_MACRO_CAPACITY).unwrap());
    assert_eq!(first_metrics.no_move, 0);

    let movements = first.shortlist().iter().map(|candidate| candidate.movement).collect::<Vec<_>>();
    assert!(matches!(
        second_state.consider_move_batch_full_oracle(&movements, MinimizingMoveAcceptance::Always, &stop).unwrap(),
        MoveOutcome::Accepted { .. }
    ));
    assert!(second_state.matches_full_oracle(&stop).unwrap());
}

#[test]
fn macro_relink_identical_guide_publishes_no_partial_burst() {
    let stop = AtomicBool::new(false);
    let (schedule, current_order, _) = nine_block_macro_job_shop();
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let archive = archive_for_order(&problem, current_order.clone());
    let request = ScheduleRelinkRequest { guide: archive.best().unwrap(), kind: ScheduleRelinkGuideKind::Best };
    let mut current = JobShopState::from_machine_sequences(&problem, current_order.clone(), &stop).unwrap().unwrap();
    current.retain_canonical_critical_blocks_only();
    let mut workspace = ScheduleRelinkWorkspace::default();
    let mut metrics = ScheduleRelinkMetrics::default();

    workspace.prepare_macro(&mut current, request, &mut metrics, &stop).unwrap();

    assert!(workspace.shortlist().is_empty());
    assert_eq!(workspace.shortlist_guide_arc_gain(), 0);
    assert_eq!(metrics.no_move, 1);
    assert_eq!(current.machine_sequences(), current_order);
}

#[test]
fn macro_relink_requires_two_distinct_machine_components() {
    let stop = AtomicBool::new(false);
    let schedule = Schedule {
        intervals: vec![IntervalVar { duration: 1, horizon: 2, modes: Vec::new(), optional: false }; 2],
        precedences: Vec::new(),
        resources: vec![Resource::NoOverlap(vec![0, 1])],
        minimize_makespan: true,
    };
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let current_order = vec![vec![0, 1]];
    let archive = archive_for_order(&problem, vec![vec![1, 0]]);
    let request = ScheduleRelinkRequest { guide: archive.best().unwrap(), kind: ScheduleRelinkGuideKind::Best };
    let mut current = JobShopState::from_machine_sequences(&problem, current_order.clone(), &stop).unwrap().unwrap();
    current.retain_canonical_critical_blocks_only();
    let mut workspace = ScheduleRelinkWorkspace::default();
    let mut metrics = ScheduleRelinkMetrics::default();

    workspace.prepare_macro(&mut current, request, &mut metrics, &stop).unwrap();

    assert_eq!(metrics.candidates_retained, 1);
    assert!(workspace.shortlist().is_empty());
    assert_eq!(metrics.candidates_shortlisted, 0);
    assert_eq!(metrics.no_move, 1);
    assert_eq!(current.machine_sequences(), current_order);
}

#[test]
fn macro_relink_keeps_coordinated_unknowns_for_the_batch_oracle() {
    let stop = AtomicBool::new(false);
    let (schedule, current_order, guide_order) = coordinated_unknown_macro_job_shop();
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let archive = archive_for_order(&problem, guide_order.clone());
    let request = ScheduleRelinkRequest { guide: archive.best().unwrap(), kind: ScheduleRelinkGuideKind::Best };
    let mut current = JobShopState::from_machine_sequences(&problem, current_order.clone(), &stop).unwrap().unwrap();
    current.retain_canonical_critical_blocks_only();
    let mut workspace = ScheduleRelinkWorkspace::default();
    let mut metrics = ScheduleRelinkMetrics::default();

    workspace.prepare_macro(&mut current, request, &mut metrics, &stop).unwrap();

    assert_eq!(
        workspace.shortlist(),
        [
            qayd::engines::ls::lists::schedule_relink::ScheduleRelinkCandidate {
                movement: qayd::engines::ls::lists::schedule_state::ScheduleMove::Insert { machine: 1, from: 3, to: 1 },
                guide_arc_gain: 2,
            },
            qayd::engines::ls::lists::schedule_relink::ScheduleRelinkCandidate {
                movement: qayd::engines::ls::lists::schedule_state::ScheduleMove::Insert { machine: 2, from: 0, to: 3 },
                guide_arc_gain: 1,
            },
        ]
    );
    assert_eq!(metrics.acyclicity_certified, 0);
    assert!(metrics.acyclicity_unknown >= 2);
    assert_eq!(metrics.prefilter_rejections, 0);
    assert_eq!(workspace.shortlist_guide_arc_gain(), 3);
    assert_eq!(current.machine_sequences(), current_order);

    for candidate in workspace.shortlist() {
        let mut isolated = JobShopState::from_machine_sequences(&problem, current_order.clone(), &stop).unwrap().unwrap();
        assert_eq!(
            isolated.consider_move_full_oracle(candidate.movement, MinimizingMoveAcceptance::Always, &stop).unwrap(),
            MoveOutcome::Rejected(MoveRejection::Cycle)
        );
        assert_eq!(isolated.machine_sequences(), current_order);
    }

    let movements = workspace.shortlist().iter().map(|candidate| candidate.movement).collect::<Vec<_>>();
    assert!(matches!(
        current.consider_move_batch_full_oracle(&movements, MinimizingMoveAcceptance::Always, &stop).unwrap(),
        MoveOutcome::Accepted { .. }
    ));
    assert_eq!(current.machine_sequences(), guide_order);
    assert!(current.matches_full_oracle(&stop).unwrap());
}

#[test]
fn macro_relink_incompatibility_and_interruption_clear_the_published_plan() {
    let stop = AtomicBool::new(false);
    let (schedule, current_order, guide_order) = nine_block_macro_job_shop();
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let archive = archive_for_order(&problem, guide_order);
    let request = ScheduleRelinkRequest { guide: archive.best().unwrap(), kind: ScheduleRelinkGuideKind::Diverse };
    let mut matching = JobShopState::from_machine_sequences(&problem, current_order.clone(), &stop).unwrap().unwrap();
    matching.retain_canonical_critical_blocks_only();
    let mut workspace = ScheduleRelinkWorkspace::default();
    let mut metrics = ScheduleRelinkMetrics::default();
    workspace.prepare_macro(&mut matching, request, &mut metrics, &stop).unwrap();
    assert_eq!(workspace.shortlist().len(), RELINK_MACRO_CAPACITY);

    let different = Schedule {
        intervals: vec![IntervalVar { duration: 1, horizon: 2, modes: Vec::new(), optional: false }; 2],
        precedences: Vec::new(),
        resources: vec![Resource::NoOverlap(vec![0, 1])],
        minimize_makespan: true,
    };
    let different_problem = JobShopProblem::recognize(&different, &stop).unwrap().unwrap();
    let mut different_state = state(&different_problem, 0);
    let different_before = different_state.machine_sequences().to_vec();
    let mut incompatible_metrics = ScheduleRelinkMetrics::default();
    assert_eq!(
        workspace.prepare_macro(&mut different_state, request, &mut incompatible_metrics, &stop),
        Err(ScheduleEliteError::IncompatibleProblem)
    );
    assert!(workspace.shortlist().is_empty());
    assert_eq!(different_state.machine_sequences(), different_before);
    assert_eq!(incompatible_metrics.guide_incompatible, 1);

    let mut recovery_metrics = ScheduleRelinkMetrics::default();
    workspace.prepare_macro(&mut matching, request, &mut recovery_metrics, &stop).unwrap();
    assert_eq!(workspace.shortlist().len(), RELINK_MACRO_CAPACITY);
    let matching_before = matching.machine_sequences().to_vec();
    let interrupted = AtomicBool::new(true);
    let mut interrupted_metrics = ScheduleRelinkMetrics::default();
    assert_eq!(
        workspace.prepare_macro(&mut matching, request, &mut interrupted_metrics, &interrupted),
        Err(ScheduleEliteError::Interrupted)
    );
    assert!(workspace.shortlist().is_empty());
    assert_eq!(matching.machine_sequences(), matching_before);
    assert_eq!(interrupted_metrics.guide_interruptions, 1);
}

#[test]
fn interrupted_guide_replacement_invalidates_the_old_cache_before_reuse() {
    let stop = AtomicBool::new(false);
    let (problem, current_order, first_guide_order) = find_relink_case();
    let first_archive = archive_for_order(&problem, first_guide_order);
    let mut current = JobShopState::from_machine_sequences(&problem, current_order, &stop).unwrap().unwrap();
    let mut workspace = ScheduleRelinkWorkspace::default();
    let mut first_metrics = ScheduleRelinkMetrics::default();
    let first_request = ScheduleRelinkRequest { guide: first_archive.best().unwrap(), kind: ScheduleRelinkGuideKind::Best };
    workspace.prepare(&mut current, first_request, &mut first_metrics, &stop).unwrap();
    let expected = workspace.shortlist().to_vec();

    let second_archive = (0..64)
        .map(|seed| archive_for_order(&problem, state(&problem, seed).machine_sequences().to_vec()))
        .find(|archive| archive.best().unwrap().order_hash() != first_archive.best().unwrap().order_hash())
        .expect("a distinct second guide exists");
    let interrupted = AtomicBool::new(true);
    let mut interrupted_metrics = ScheduleRelinkMetrics::default();
    let second_request = ScheduleRelinkRequest { guide: second_archive.best().unwrap(), kind: ScheduleRelinkGuideKind::Diverse };
    assert_eq!(
        workspace.prepare(&mut current, second_request, &mut interrupted_metrics, &interrupted),
        Err(ScheduleEliteError::Interrupted)
    );

    let mut recovered_metrics = ScheduleRelinkMetrics::default();
    workspace.prepare(&mut current, first_request, &mut recovered_metrics, &stop).unwrap();
    assert_eq!(workspace.shortlist(), expected);
    assert_eq!(recovered_metrics.guide_loads, 1, "the old key must not survive a failed replacement");
}

#[test]
fn an_anchor_just_outside_a_critical_block_remains_a_valid_boundary_insertion() {
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&diversified_job_shop(), &stop).unwrap().unwrap();
    for current_seed in 0..64 {
        let current_order = state(&problem, current_seed).machine_sequences().to_vec();
        for guide_seed in 0..64 {
            let guide_order = state(&problem, guide_seed).machine_sequences().to_vec();
            let archive = archive_for_order(&problem, guide_order.clone());
            let mut current = JobShopState::from_machine_sequences(&problem, current_order.clone(), &stop).unwrap().unwrap();
            let mut workspace = ScheduleRelinkWorkspace::default();
            let mut metrics = ScheduleRelinkMetrics::default();
            workspace
                .prepare(
                    &mut current,
                    ScheduleRelinkRequest { guide: archive.best().unwrap(), kind: ScheduleRelinkGuideKind::Best },
                    &mut metrics,
                    &stop,
                )
                .unwrap();
            for candidate in workspace.shortlist() {
                let qayd::engines::ls::lists::schedule_state::ScheduleMove::Insert { machine, from, to } = candidate.movement else {
                    continue;
                };
                if current.critical_blocks().iter().any(|block| {
                    block.machine() == machine
                        && (block.first_position()..=block.last_position()).contains(&from)
                        && (block.first_position()..=block.last_position()).contains(&to)
                }) {
                    let mut moved = current_order.clone();
                    apply_move(&mut moved, candidate.movement);
                    assert!(guide_arc_count(&moved, &guide_order) > guide_arc_count(&current_order, &guide_order));
                    return;
                }
            }
        }
    }
    panic!("expected a boundary insertion whose post-removal index overlaps the old block")
}

trait LargeTaWorkspaceAudit {
    fn workspace_heap_bound_is_large_ta_safe(&self) -> bool;
}

impl LargeTaWorkspaceAudit for ScheduleRelinkWorkspace {
    fn workspace_heap_bound_is_large_ta_safe(&self) -> bool {
        self.heap_lower_bound_bytes() < 16 * 1024 * 1024
    }
}

#[test]
fn positive_duration_classical_chain_keeps_the_paper_n8_equality_case() {
    let stop = AtomicBool::new(false);
    let schedule = Schedule {
        intervals: vec![IntervalVar { duration: 1, horizon: 4, modes: Vec::new(), optional: false }; 4],
        precedences: vec![(0, 2), (2, 3)],
        resources: vec![Resource::NoOverlap(vec![0, 1]), Resource::NoOverlap(vec![2]), Resource::NoOverlap(vec![3])],
        minimize_makespan: true,
    };
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let mut current = JobShopState::from_machine_sequences(&problem, vec![vec![0, 1], vec![2], vec![3]], &stop).unwrap().unwrap();
    let movement = qayd::engines::ls::lists::schedule_state::ScheduleMove::Insert { machine: 0, from: 0, to: 1 };
    let rank_zero = current.topological_order().iter().position(|&operation| operation == 0).unwrap();
    let rank_one = current.topological_order().iter().position(|&operation| operation == 1).unwrap();

    assert!(rank_zero < rank_one, "the move's new 1->0 arc must be backward in the accepted topological order");
    assert!(current.certifies_insert_acyclicity(movement, &stop).unwrap());
    assert!(matches!(
        current.consider_move_full_oracle(movement, MinimizingMoveAcceptance::Always, &stop).unwrap(),
        MoveOutcome::Accepted { .. }
    ));
}

#[test]
fn zero_duration_equality_is_unknown_and_cannot_pass_the_n8_certificate() {
    let stop = AtomicBool::new(false);
    let schedule = Schedule {
        intervals: vec![IntervalVar { duration: 0, horizon: 0, modes: Vec::new(), optional: false }; 4],
        precedences: vec![(0, 1), (2, 3)],
        resources: vec![Resource::NoOverlap(vec![0, 3]), Resource::NoOverlap(vec![1, 2])],
        minimize_makespan: true,
    };
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let mut current = JobShopState::from_machine_sequences(&problem, vec![vec![0, 3], vec![1, 2]], &stop).unwrap().unwrap();
    let movement = qayd::engines::ls::lists::schedule_state::ScheduleMove::Insert { machine: 0, from: 0, to: 1 };
    let before = current.machine_sequences().to_vec();

    assert!(!current.certifies_insert_acyclicity(movement, &stop).unwrap());
    assert_eq!(
        current.consider_move_full_oracle(movement, MinimizingMoveAcceptance::Always, &stop).unwrap(),
        MoveOutcome::Rejected(MoveRejection::Cycle)
    );
    assert_eq!(current.machine_sequences(), before);
}

#[test]
fn positive_duration_machine_revisit_disables_the_n8_propositions() {
    let stop = AtomicBool::new(false);
    let schedule = Schedule {
        intervals: vec![IntervalVar { duration: 1, horizon: 12, modes: Vec::new(), optional: false }; 6],
        precedences: vec![(1, 2), (5, 1), (3, 4)],
        resources: vec![Resource::NoOverlap(vec![4, 1]), Resource::NoOverlap(vec![3, 5, 2, 0])],
        minimize_makespan: true,
    };
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    // Qayd compacts the conceptual M1 to index 0 because operation 0 uses it.
    // The represented orders remain M0 [4,1] / M1 [3,5,2,0].
    let current_order = vec![vec![3, 5, 2, 0], vec![4, 1]];
    let guide_order = vec![vec![5, 0, 3, 2], vec![1, 4]];
    let mut current = JobShopState::from_machine_sequences(&problem, current_order.clone(), &stop).unwrap().unwrap();
    let movement = qayd::engines::ls::lists::schedule_state::ScheduleMove::Insert { machine: 0, from: 1, to: 2 };

    assert!(!current.certifies_insert_acyclicity(movement, &stop).unwrap());
    let archive = archive_for_order(&problem, guide_order);
    let mut workspace = ScheduleRelinkWorkspace::default();
    let mut metrics = ScheduleRelinkMetrics::default();
    workspace
        .prepare(
            &mut current,
            ScheduleRelinkRequest { guide: archive.best().unwrap(), kind: ScheduleRelinkGuideKind::Best },
            &mut metrics,
            &stop,
        )
        .unwrap();
    assert!(!workspace.shortlist().iter().any(|candidate| candidate.movement == movement));

    assert_eq!(
        current.consider_move_full_oracle(movement, MinimizingMoveAcceptance::Always, &stop).unwrap(),
        MoveOutcome::Rejected(MoveRejection::Cycle)
    );
    assert_eq!(current.machine_sequences(), current_order);
}

#[test]
fn every_shortlisted_n8_move_is_certified_and_never_rejected_for_a_cycle() {
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&diversified_job_shop(), &stop).unwrap().unwrap();
    let mut certified = 0u64;
    let mut unknown = 0u64;
    for current_seed in 0..64 {
        for guide_seed in 0..64 {
            let current_order = state(&problem, current_seed).machine_sequences().to_vec();
            let archive = archive_for_order(&problem, state(&problem, guide_seed).machine_sequences().to_vec());
            let mut current = JobShopState::from_machine_sequences(&problem, current_order.clone(), &stop).unwrap().unwrap();
            let mut workspace = ScheduleRelinkWorkspace::default();
            let mut metrics = ScheduleRelinkMetrics::default();
            workspace
                .prepare(
                    &mut current,
                    ScheduleRelinkRequest { guide: archive.best().unwrap(), kind: ScheduleRelinkGuideKind::Best },
                    &mut metrics,
                    &stop,
                )
                .unwrap();
            assert_eq!(metrics.candidates_positive_gain, metrics.acyclicity_certified + metrics.acyclicity_unknown);
            assert_eq!(metrics.acyclicity_unknown, metrics.prefilter_rejections);
            certified = certified.saturating_add(metrics.acyclicity_certified);
            unknown = unknown.saturating_add(metrics.acyclicity_unknown);
            for candidate in workspace.shortlist().to_vec() {
                let mut trial = JobShopState::from_machine_sequences(&problem, current_order.clone(), &stop).unwrap().unwrap();
                assert!(trial.certifies_insert_acyclicity(candidate.movement, &stop).unwrap());
                let outcome = trial.consider_move_full_oracle(candidate.movement, MinimizingMoveAcceptance::Always, &stop).unwrap();
                assert!(!matches!(outcome, MoveOutcome::Rejected(MoveRejection::Cycle)));
                if matches!(outcome, MoveOutcome::Rejected(_)) {
                    assert_eq!(trial.machine_sequences(), current_order);
                    assert!(trial.matches_full_oracle(&stop).unwrap());
                }
            }
        }
    }
    assert!(certified > 0, "the sufficient N8 conditions must retain useful moves");
    assert!(unknown > 0, "unknown moves must exercise the fail-closed prefilter");
}
