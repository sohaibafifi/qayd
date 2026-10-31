use std::sync::atomic::{AtomicBool, Ordering};

use qayd::engines::ls::lists::move_acceptance::MinimizingMoveAcceptance;
use qayd::engines::ls::lists::schedule_state::{DispatchRule, JobShopProblem, JobShopState, MachineArc, MoveOutcome};
use qayd::engines::ls::lists::{
    audit_tsab_elite_restart_policy, audit_tsab_owner_mask, audit_tsab_phase_policy, solve_schedule_capped_persistent_hybrid_with_strategy,
    ScheduleConstructionMetrics, ScheduleJsspSearchStrategy, ScheduleSearchSession,
};
use qayd::model::list::{verify_collection_solution, CollectionModel, CollectionSolution, IntervalVar, Resource, Schedule};

fn job_shop(jobs: usize, machines: usize) -> Schedule {
    let operation_count = jobs * machines;
    let durations: Vec<i64> = (0..operation_count).map(|operation| 1 + ((operation * 7 + 9) % 11) as i64).collect();
    let horizon = durations.iter().sum();
    let mut groups = vec![Vec::new(); machines];
    for job in 0..jobs {
        for operation in 0..machines {
            groups[(job + machines - operation) % machines].push(job * machines + operation);
        }
    }
    Schedule {
        intervals: durations.into_iter().map(|duration| IntervalVar { duration, horizon, modes: Vec::new(), optional: false }).collect(),
        precedences: (0..jobs)
            .flat_map(|job| (0..machines - 1).map(move |operation| (job * machines + operation, job * machines + operation + 1)))
            .collect(),
        resources: groups.into_iter().map(Resource::NoOverlap).collect(),
        minimize_makespan: true,
    }
}

#[allow(clippy::too_many_arguments)]
fn adaptive_slice(
    schedule: &Schedule,
    seed: u64,
    work: u64,
    incumbent: Option<&CollectionSolution>,
    session: Option<ScheduleSearchSession>,
    search_stop: &AtomicBool,
    finalization_stop: &AtomicBool,
) -> (CollectionSolution, ScheduleConstructionMetrics, Option<ScheduleSearchSession>) {
    solve_schedule_capped_persistent_hybrid_with_strategy(
        schedule,
        seed,
        seed,
        0,
        1,
        search_stop,
        finalization_stop,
        work,
        incumbent,
        session,
        false,
        None,
        false,
        false,
        ScheduleJsspSearchStrategy::TsabAdaptive,
        None,
        &mut |_| {},
    )
}

fn replay(schedule: &Schedule, solution: &CollectionSolution) {
    assert!(solution.feasible);
    let model = CollectionModel {
        items: Vec::new(),
        lists: 0,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: Some(schedule.clone()),
    };
    assert_eq!(verify_collection_solution(&model, solution).expect("complete independent replay"), solution.objectives);
    let makespan = schedule.intervals.iter().enumerate().map(|(i, interval)| solution.starts[i] + interval.duration).max().unwrap_or(0);
    assert_eq!(solution.objectives, [makespan]);
}

fn two_job_endpoint_fixture() -> (Schedule, CollectionSolution, CollectionSolution) {
    let schedule = Schedule {
        intervals: [2, 2, 3, 2]
            .into_iter()
            .map(|duration| IntervalVar { duration, horizon: 9, modes: Vec::new(), optional: false })
            .collect(),
        precedences: vec![(0, 1), (2, 3)],
        resources: vec![Resource::NoOverlap(vec![0, 3]), Resource::NoOverlap(vec![1, 2])],
        minimize_makespan: true,
    };
    let make_solution = |starts, objective| CollectionSolution {
        lists: Vec::new(),
        starts,
        objectives: vec![objective],
        feasible: true,
        presences: vec![true; 4],
        machines: vec![-1; 4],
        modes: vec![None; 4],
        bound: None,
    };
    let initial = make_solution(vec![0, 2, 4, 7], 9);
    let optimum = make_solution(vec![0, 3, 0, 3], 5);
    replay(&schedule, &initial);
    replay(&schedule, &optimum);
    (schedule, initial, optimum)
}

fn blocked_n5_fixture() -> (Schedule, CollectionSolution, CollectionSolution, Vec<(MachineArc, u64)>) {
    // Two disjoint feasible N5 swaps add arcs with expiration pairs[2,9] and
    // [4,7]. Their first possible admission is therefore min(9,7)=7.
    let stop = AtomicBool::new(false);
    let schedule = Schedule {
        intervals: [3, 6, 4, 6, 1, 9, 9, 9, 1]
            .into_iter()
            .map(|duration| IntervalVar { duration, horizon: 57, modes: Vec::new(), optional: false })
            .collect(),
        precedences: vec![(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)],
        resources: vec![Resource::NoOverlap(vec![1, 5, 8]), Resource::NoOverlap(vec![3, 2, 7]), Resource::NoOverlap(vec![6, 0, 4])],
        minimize_makespan: true,
    };
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let mut sequences = vec![Vec::new(); 3];
    for sequence in [vec![1, 5, 8], vec![3, 2, 7], vec![6, 0, 4]] {
        let machine = problem.machine(sequence[0]);
        sequences[machine] = sequence;
    }
    let state = JobShopState::from_machine_sequences(&problem, sequences, &stop).unwrap().unwrap();
    let current = state.to_solution();
    assert_eq!(current.objectives, [32]);
    let mut trial = state.rebuilt_from_machine_sequences(state.machine_sequences().to_vec(), &stop).unwrap().unwrap();
    let movement = qayd::engines::ls::lists::schedule_state::ScheduleMove::AdjacentSwap { machine: problem.machine(6), first_position: 0 };
    assert!(matches!(
        trial.consider_move_full_oracle(movement, MinimizingMoveAcceptance::Always, &stop).unwrap(),
        MoveOutcome::Accepted { .. }
    ));
    let elite = trial.to_solution();
    assert_eq!(elite.objectives, [23]);
    let mut markers = Vec::new();
    let mut moves = 0;
    state
        .visit_strict_n5_canonical_moves(&stop, |_, arcs| {
            let machine = arcs.added.into_iter().flatten().next().unwrap().machine;
            let expirations = if machine == problem.machine(3) { [2, 9] } else { [4, 7] };
            let added: Vec<_> = arcs.added.into_iter().flatten().collect();
            assert_eq!(added.len(), 2);
            markers.extend(added.into_iter().zip(expirations));
            moves += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(moves, 2);
    replay(&schedule, &current);
    replay(&schedule, &elite);
    (schedule, current, elite, markers)
}

fn tied_n5_fixture() -> (Schedule, CollectionSolution) {
    let schedule = Schedule {
        intervals: vec![IntervalVar { duration: 1, horizon: 9, modes: Vec::new(), optional: false }; 9],
        precedences: vec![(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)],
        resources: vec![Resource::NoOverlap(vec![3, 2, 8]), Resource::NoOverlap(vec![0, 4, 7]), Resource::NoOverlap(vec![6, 1, 5])],
        minimize_makespan: true,
    };
    let current = CollectionSolution {
        lists: Vec::new(),
        starts: vec![0, 1, 2, 0, 1, 2, 0, 2, 3],
        objectives: vec![4],
        feasible: true,
        presences: vec![true; 9],
        machines: vec![-1; 9],
        modes: vec![None; 9],
        bound: None,
    };
    replay(&schedule, &current);
    (schedule, current)
}

#[test]
fn adaptive_tsab_owns_only_the_single_worker_and_preserves_pilot_policies() {
    for workers in [0, 1, 2, 6, 7, 8, 64] {
        assert_eq!(audit_tsab_owner_mask(ScheduleJsspSearchStrategy::TsabAdaptive, workers), u64::from(workers == 1));
    }
    assert_eq!(audit_tsab_owner_mask(ScheduleJsspSearchStrategy::TsabCandidate, 7), 1 << 6);
    assert_eq!(audit_tsab_owner_mask(ScheduleJsspSearchStrategy::TsabMultiCandidate, 7), 0x7f);
    assert_eq!(audit_tsab_owner_mask(ScheduleJsspSearchStrategy::TsabCandidate, 1), 0);
    assert_eq!(audit_tsab_phase_policy(), (512, 128, 8, 4));
    assert_eq!(audit_tsab_elite_restart_policy(), (16, 8, 4, 64));
}

#[test]
fn adaptive_tsab_activates_from_its_local_constructor_without_warmup_or_shared_incumbent() {
    let schedule = job_shop(6, 5);
    let stop = AtomicBool::new(false);
    let (initial, metrics, session) = adaptive_slice(&schedule, 13, 0, None, None, &stop, &stop);
    replay(&schedule, &initial);
    assert_eq!(metrics.work_steps, 0);
    assert_eq!(metrics.schedule_tsab.activations, 1);
    assert_eq!(metrics.schedule_tsab.owner_worker_mask, 1);
    assert_eq!(metrics.schedule_tsab.activation_boundary, Some(0));
    assert_eq!(metrics.schedule_tsab.legacy_warmup_work_steps, 0);
    assert_eq!(metrics.baseline_island_profile_mask, 0);
    assert_eq!(metrics.scored_island_profile_mask, 0);
    let session = session.expect("complete constructor owns a session");
    assert_eq!(session.test_tsab_requested_strategy(), ScheduleJsspSearchStrategy::TsabAdaptive);
    let (active, _, warmup, boundary, _, _, _, _, tenure) = session.test_tsab_phase_state();
    assert_eq!(active, ScheduleJsspSearchStrategy::TsabCandidate);
    assert_eq!((warmup, boundary, tenure), (0, Some(0), 5));
    let (solution, metrics, mut session) = adaptive_slice(&schedule, 13, 16, Some(&initial), Some(session), &stop, &stop);
    replay(&schedule, &solution);
    assert_eq!(metrics.schedule_tsab.activations, 0);
    assert_eq!(metrics.work_steps, 16);
    assert!(metrics.schedule_tsab.fast_attempts > 0);
    assert_eq!(metrics.schedule_tsab.fast_oracle_mismatches, 0);
    // The micro-kernel keeps semantic dates exact while other derived buffers
    // are refreshed only at its complete-oracle checkpoints.
    assert!(session.as_mut().expect("active trajectory").test_tsab_fast_dates_match_full_oracle());
    assert!(session.as_mut().expect("active trajectory").test_tsab_state_matches_full_oracle());
}

#[test]
fn adaptive_tsab_restarts_and_oracle_checkpoints_are_invariant_to_work_slicing() {
    let schedule = job_shop(8, 6);
    let stop = AtomicBool::new(false);
    let run = |slices: &[u64]| {
        let (mut solution, _, session) = adaptive_slice(&schedule, 19, 0, None, None, &stop, &stop);
        let mut session = session.expect("initial trajectory");
        session.test_force_tsab_restart_due();
        let mut session = Some(session);
        let mut work = 0;
        let mut restarts = 0;
        for &slice in slices {
            let (next, metrics, next_session) = adaptive_slice(&schedule, 19, slice, Some(&solution), session, &stop, &stop);
            replay(&schedule, &next);
            assert!(next.objectives[0] <= solution.objectives[0]);
            assert_eq!(metrics.schedule_tsab.fast_oracle_mismatches, 0);
            work += metrics.work_steps;
            restarts += metrics.schedule_tsab.restart_attempts;
            solution = next;
            session = next_session;
        }
        (solution, session.expect("retained trajectory"), work, restarts)
    };
    let (whole, mut whole_session, whole_work, whole_restarts) = run(&[128]);
    let (split, mut split_session, split_work, split_restarts) = run(&[1, 2, 5, 3, 32, 85]);
    assert_eq!((whole_work, split_work), (128, 128));
    assert!(whole_restarts > 0);
    assert_eq!(whole_restarts, split_restarts);
    assert_eq!(whole.objectives, split.objectives);
    assert_eq!(whole.starts, split.starts);
    assert!(whole_session.test_same_tabu_trajectory(&split_session));
    assert!(whole_session.test_tsab_fast_dates_match_full_oracle());
    assert!(split_session.test_tsab_fast_dates_match_full_oracle());
    assert!(whole_session.test_tsab_state_matches_full_oracle());
    assert!(split_session.test_tsab_state_matches_full_oracle());
}

#[test]
fn adaptive_tsab_empty_escape_preserves_its_current_trajectory() {
    let schedule = job_shop(1, 4);
    let stop = AtomicBool::new(false);
    let (solution, _, session) = adaptive_slice(&schedule, 4, 0, None, None, &stop, &stop);
    let mut session = session.expect("initial trajectory");
    let snapshot = session.test_tsab_trajectory_snapshot();
    session.test_force_adaptive_restart_without_rewinding();
    let (next, metrics, next_session) = adaptive_slice(&schedule, 4, 8, Some(&solution), Some(session), &stop, &stop);
    let mut next_session = next_session.expect("empty escape retains the current trajectory");
    assert_eq!(metrics.schedule_tsab.restart_attempts, 1);
    assert_eq!(metrics.schedule_tsab.restart_rejections, 1);
    assert_eq!(metrics.schedule_tsab.elite_restarts, 0);
    assert_eq!(metrics.schedule_tsab.restart_oracle_commits, 0);
    assert_eq!(next_session.test_tsab_trajectory_snapshot(), snapshot);
    assert_eq!(next_session.test_tsab_restart_state().0, 0);
    assert_eq!(next_session.test_tsab_restart_state().1, 0);
    assert!(next_session.test_tsab_state_matches_full_oracle());
    replay(&schedule, &next);
}

#[test]
fn adaptive_tsab_blocked_n6_credit_cooldown_and_forced_fallback_are_split_invariant() {
    let (schedule, current, elite, markers) = blocked_n5_fixture();
    let stop = AtomicBool::new(false);
    let run = |slices: &[u64], cancel_partial_credit: bool| {
        let (_, _, session) = adaptive_slice(&schedule, 0, 0, Some(&current), None, &stop, &stop);
        let mut session = session.unwrap();
        session.test_set_verified_elite(elite.clone());
        for &(arc, remaining) in &markers {
            session.test_mark_tabu_arc(arc, remaining);
        }
        let initial_policy = session.test_tsab_canonical_path();
        let initial_tenure = session.test_tsab_phase_state().8;
        let mut solution = elite.clone();
        let mut session = Some(session);
        let mut requests = 0;
        let mut attempts = 0;
        let mut skipped = 0;
        let mut work = 0;
        let mut probes = 0;
        let mut forced = 0;
        for (index, &slice) in slices.iter().enumerate() {
            let (next, metrics, next_session) = adaptive_slice(&schedule, 0, slice, Some(&solution), session, &stop, &stop);
            solution = next;
            session = next_session;
            requests += metrics.schedule_tsab.blocked_n6_requests;
            attempts += metrics.schedule_tsab.blocked_n6_attempts;
            skipped += metrics.schedule_tsab.blocked_n6_cooldown_skips;
            work += metrics.work_steps;
            probes += metrics.schedule_tsab.restart_delta_probes;
            forced += metrics.tabu_forced_moves;
            assert_eq!(metrics.schedule_tsab.fast_oracle_mismatches, 0);
            let state = session.as_ref().unwrap().test_adaptive_blocked_escape_state();
            assert_eq!(state.1, 7, "the earliest move needs all of its tabu arcs to expire");
            let current_session = session.as_ref().unwrap();
            let (clock, _, epoch, _, _, _) = current_session.test_tsab_restart_state();
            assert_eq!(clock, 1 + state.2, "a blocked opportunity preserves the real N5 stagnation clock");
            assert_eq!(epoch, 0);
            assert_eq!(current_session.test_tsab_adaptive_policy().1, 0);
            assert_eq!(current_session.test_tsab_phase_state().8, initial_tenure);
            assert_eq!(current_session.test_tsab_canonical_path().0, initial_policy.0);
            if state.2 == 0 {
                assert_eq!(current_session.test_tsab_canonical_path(), initial_policy);
            }
            if index == 0 && slices[0] == 1 {
                assert!(state.0);
                assert_eq!(state.2, 0);
                assert_eq!(requests, 1);
                assert_eq!(attempts, 0);
            }
            if cancel_partial_credit && index == 1 {
                let before = session.as_ref().unwrap().test_tsab_trajectory_snapshot();
                stop.store(true, Ordering::Release);
                let (_, cancelled, next_session) = adaptive_slice(&schedule, 0, 5, Some(&solution), session, &stop, &stop);
                assert_eq!(cancelled.work_steps, 0);
                assert_eq!(cancelled.schedule_tsab.blocked_n6_attempts, 0);
                session = next_session;
                assert_eq!(session.as_ref().unwrap().test_tsab_trajectory_snapshot(), before);
                stop.store(false, Ordering::Release);
            }
            replay(&schedule, &solution);
        }
        let mut session = session.unwrap();
        assert_eq!((requests, attempts, skipped, work, forced), (1, 1, 1, 10, 1));
        assert!(probes <= 8);
        assert_eq!(session.test_adaptive_blocked_escape_state(), (false, 7, 1));
        assert!(session.test_tsab_fast_dates_match_full_oracle());
        (solution, session)
    };
    let (whole, whole_session) = run(&[9, 1], false);
    let (split, split_session) = run(&[1, 3, 5, 1], true);
    assert_eq!(whole.objectives, split.objectives);
    assert_eq!(whole.starts, split.starts);
    assert!(whole_session.test_same_tabu_trajectory(&split_session));
}

#[test]
fn adaptive_tsab_blocked_opportunity_waits_a_complete_tenure() {
    let (schedule, current, elite, markers) = blocked_n5_fixture();
    let stop = AtomicBool::new(false);
    let (_, _, session) = adaptive_slice(&schedule, 0, 0, Some(&current), None, &stop, &stop);
    let mut session = session.unwrap();
    session.test_set_verified_elite(elite.clone());
    for (arc, _) in markers {
        session.test_mark_tabu_arc(arc, 1);
    }
    let policy = session.test_tsab_canonical_path();
    let (solution, metrics, session) = adaptive_slice(&schedule, 0, 9, Some(&elite), Some(session), &stop, &stop);
    let mut session = session.unwrap();
    assert_eq!(metrics.schedule_tsab.blocked_n6_requests, 1);
    assert_eq!(metrics.schedule_tsab.blocked_n6_attempts, 1);
    assert_eq!(
        session.test_adaptive_blocked_escape_state(),
        (false, 7, 0),
        "one-step tabu expirations cannot trigger an N6 opportunity every step"
    );
    assert_eq!(session.test_tsab_restart_state().0, 1);
    assert_eq!(session.test_tsab_restart_state().2, 0);
    assert_eq!(session.test_tsab_adaptive_policy().1, 0);
    assert_eq!(session.test_tsab_canonical_path(), policy);
    assert!(session.test_tsab_state_matches_full_oracle());
    replay(&schedule, &solution);
}

#[test]
fn adaptive_tsab_true_stagnation_takes_precedence_over_a_tabu_opportunity() {
    let (schedule, current, elite, markers) = blocked_n5_fixture();
    let stop = AtomicBool::new(false);
    let (_, _, session) = adaptive_slice(&schedule, 0, 0, Some(&current), None, &stop, &stop);
    let mut session = session.unwrap();
    session.test_set_verified_elite(elite.clone());
    for (arc, remaining) in markers {
        session.test_mark_tabu_arc(arc, remaining);
    }
    session.test_require_adaptive_stagnation_after_next_work_unit();
    let (solution, queued, session) = adaptive_slice(&schedule, 0, 1, Some(&elite), Some(session), &stop, &stop);
    let session = session.unwrap();
    assert_eq!(queued.schedule_tsab.blocked_n6_requests, 0);
    assert!(!session.test_adaptive_blocked_escape_state().0);
    assert_eq!(session.test_tsab_restart_state().0, session.test_tsab_adaptive_policy().0);
    let (solution, completed, session) = adaptive_slice(&schedule, 0, 8, Some(&solution), Some(session), &stop, &stop);
    let mut session = session.unwrap();
    assert_eq!(completed.schedule_tsab.restart_attempts, 1);
    assert_eq!(completed.schedule_tsab.blocked_n6_attempts, 0);
    assert_eq!(session.test_tsab_restart_state().0, 0);
    assert_eq!(session.test_tsab_restart_state().2, 1);
    assert_eq!(session.test_tsab_adaptive_policy().1, 1);
    assert!(session.test_tsab_state_matches_full_oracle());
    replay(&schedule, &solution);
}

#[test]
fn adaptive_tsab_primary_pruning_matches_full_oracle_for_tabu_masks_elites_and_ties() {
    use std::collections::BTreeSet;
    let stop = AtomicBool::new(false);
    let (inversion_schedule, inversion_current, _, _) = blocked_n5_fixture();
    let (tie_schedule, tie_current) = tied_n5_fixture();
    let mut fixtures = vec![(inversion_schedule, inversion_current, 0), (tie_schedule, tie_current, 0)];
    for (jobs, machines) in [(3, 3), (4, 3), (4, 4)] {
        let schedule = job_shop(jobs, machines);
        let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
        for seed in 0..4 {
            let state = JobShopState::giffler_thompson(&problem, seed, DispatchRule::EarliestStart, &stop).unwrap().unwrap();
            fixtures.push((schedule.clone(), state.to_solution(), seed));
        }
    }
    let mut saved = 0;
    let mut equality_prunes = 0;
    let mut no_admissible_cases = 0;
    let mut stop_checked = false;
    for (fixture, (schedule, current, seed)) in fixtures.into_iter().enumerate() {
        let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
        let (_, _, session) = adaptive_slice(&schedule, seed, 0, Some(&current), None, &stop, &stop);
        let ranks = session.unwrap().test_adaptive_ranked_moves();
        if ranks.is_empty() || ranks.len() > 6 {
            continue;
        }
        let mut sequences = vec![Vec::new(); problem.machine_count()];
        for operation in 0..problem.operation_count() {
            sequences[problem.machine(operation)].push(operation);
        }
        for sequence in &mut sequences {
            sequence.sort_by_key(|&operation| current.starts[operation]);
        }
        let state = JobShopState::from_machine_sequences(&problem, sequences, &stop).unwrap().unwrap();
        let mut oracle = Vec::new();
        let mut elites = vec![current.clone()];
        for &(movement, lower_bound) in &ranks {
            let arcs = state.move_arcs(movement).unwrap();
            let mut trial = state.rebuilt_from_machine_sequences(state.machine_sequences().to_vec(), &stop).unwrap().unwrap();
            let objective = match trial.consider_move_full_oracle(movement, MinimizingMoveAcceptance::Always, &stop).unwrap() {
                MoveOutcome::Accepted { current, .. } => {
                    assert!(lower_bound <= i128::from(current));
                    if current <= elites[0].objectives[0] {
                        elites.push(trial.to_solution());
                    }
                    Some(current)
                }
                MoveOutcome::Rejected(_) => None,
            };
            oracle.push((movement, arcs, objective));
        }
        if fixture == 0 {
            assert_eq!(ranks.iter().map(|(_, bound)| *bound).collect::<Vec<_>>(), [22, 23]);
            assert_eq!(oracle.iter().map(|candidate| candidate.2).collect::<Vec<_>>(), [Some(28), Some(23)]);
        } else if fixture == 1 {
            // Distinct critical blocks yield two equal exact objectives; the
            // later ranked move must lose the tie without another exact probe.
            assert_eq!(ranks.iter().map(|(_, bound)| *bound).collect::<Vec<_>>(), [5, 5]);
            assert_eq!(oracle.iter().map(|candidate| candidate.2).collect::<Vec<_>>(), [Some(5), Some(5)]);
        }
        elites.sort_by_key(|elite| elite.objectives[0]);
        elites.dedup_by_key(|elite| elite.objectives[0]);
        for elite in &elites {
            for mask in 0..1usize << ranks.len() {
                let marked: BTreeSet<_> = oracle
                    .iter()
                    .enumerate()
                    .filter(|(rank, _)| mask & (1 << rank) != 0)
                    .flat_map(|(_, candidate)| candidate.1.added.into_iter().flatten())
                    .collect();
                // Select the unchanged four best entries of each tabu class,
                // then evaluate each by an independent complete reconstruction.
                let mut counts = [0; 2];
                let shortlist: Vec<_> = oracle
                    .iter()
                    .enumerate()
                    .filter_map(|(rank, candidate)| {
                        let tabu = candidate.1.added.into_iter().flatten().any(|arc| marked.contains(&arc));
                        let group = usize::from(tabu);
                        counts[group] += 1;
                        (counts[group] <= 4).then_some((rank, tabu, candidate))
                    })
                    .collect();
                let reference = shortlist
                    .iter()
                    .filter_map(|&(rank, tabu, candidate)| {
                        let objective = candidate.2?;
                        let aspiration = tabu && objective < elite.objectives[0];
                        (!tabu || aspiration).then_some((objective, rank, candidate.0, aspiration))
                    })
                    .min_by_key(|candidate| (candidate.0, candidate.1));
                let expected = reference.map(|(_, _, movement, aspiration)| (movement, aspiration, false));
                let make_session = || {
                    let (_, _, session) = adaptive_slice(&schedule, seed, 0, Some(&current), None, &stop, &stop);
                    let mut session = session.unwrap();
                    session.test_set_verified_elite(elite.clone());
                    for &arc in &marked {
                        session.test_mark_tabu_arc(arc, 100);
                    }
                    session
                };
                let mut pruned = make_session();
                let mut unpruned = make_session();
                unpruned.test_disable_adaptive_primary_pruning();
                let (ok, selected, metrics, avoided) = pruned.test_select_adaptive_candidate(&stop);
                let (reference_ok, reference_selected, reference_metrics, reference_avoided) =
                    unpruned.test_select_adaptive_candidate(&stop);
                assert!(ok && reference_ok);
                assert_eq!(selected, expected);
                assert_eq!(selected, reference_selected);
                assert_eq!(reference_avoided, 0);
                assert_eq!(reference_metrics.delta_probes - metrics.delta_probes, avoided);
                assert!(pruned.test_same_tabu_decisions(&unpruned));
                if reference.is_none() {
                    no_admissible_cases += 1;
                    assert_eq!(avoided, 0, "all blocked candidates must be probed to preserve the cooldown forecast");
                }
                if avoided != 0 {
                    saved += avoided;
                    let first_skipped = metrics.delta_probes as usize;
                    let lower_bound = ranks[shortlist[first_skipped].0].1;
                    let best_exact = reference.unwrap().0;
                    assert!(lower_bound >= i128::from(best_exact));
                    equality_prunes += usize::from(lower_bound == i128::from(best_exact));
                    if !stop_checked {
                        let mut interrupted = make_session();
                        let control = make_session();
                        interrupted.test_stop_before_adaptive_primary_prune();
                        let (completed, choice, _, _) = interrupted.test_select_adaptive_candidate(&stop);
                        assert!(!completed, "an interrupted prune must not be reported as an empty neighborhood");
                        assert!(choice.is_none());
                        assert!(stop.load(Ordering::Acquire));
                        assert!(interrupted.test_same_tabu_decisions(&control));
                        stop.store(false, Ordering::Release);
                        stop_checked = true;
                    }
                }
            }
        }
    }
    assert!(
        saved > 0 && equality_prunes > 0 && no_admissible_cases > 0 && stop_checked,
        "saved={saved}, equality={equality_prunes}, no_admissible={no_admissible_cases}, stop={stop_checked}"
    );
}

#[test]
fn adaptive_tsab_primary_pruning_preserves_global_fixed_work_trajectories() {
    let stop = AtomicBool::new(false);
    let mut saved = 0;
    for (jobs, machines, seed) in [(6, 5, 0), (8, 6, 19)] {
        let schedule = job_shop(jobs, machines);
        let (mut solution, _, session) = adaptive_slice(&schedule, seed, 0, None, None, &stop, &stop);
        let (mut reference, _, reference_session) = adaptive_slice(&schedule, seed, 0, None, None, &stop, &stop);
        let mut reference_session = reference_session.unwrap();
        reference_session.test_disable_adaptive_primary_pruning();
        let mut session = session;
        let mut reference_session = Some(reference_session);
        for work in [1, 3, 5, 17, 63, 39, 128, 128, 128, 128] {
            let (next, metrics, next_session) = adaptive_slice(&schedule, seed, work, Some(&solution), session, &stop, &stop);
            let (next_reference, reference_metrics, next_reference_session) =
                adaptive_slice(&schedule, seed, work, Some(&reference), reference_session, &stop, &stop);
            assert_eq!(metrics.work_steps, reference_metrics.work_steps);
            assert_eq!(next.starts, next_reference.starts);
            assert_eq!(next.objectives, next_reference.objectives);
            assert!(next_session.as_ref().unwrap().test_same_tabu_decisions(next_reference_session.as_ref().unwrap()));
            assert_eq!(
                next_session.as_ref().unwrap().test_tsab_canonical_path(),
                next_reference_session.as_ref().unwrap().test_tsab_canonical_path()
            );
            assert!(metrics.schedule_tsab.delta_probes <= reference_metrics.schedule_tsab.delta_probes);
            saved += reference_metrics.schedule_tsab.delta_probes - metrics.schedule_tsab.delta_probes;
            replay(&schedule, &next);
            solution = next;
            reference = next_reference;
            session = next_session;
            reference_session = next_reference_session;
        }
    }
    assert!(saved > 0);
}

#[test]
fn adaptive_tsab_forward_probes_preserve_generic_probe_fixed_work_trajectories() {
    let stop = AtomicBool::new(false);
    let mut forward_attempts = 0;
    for (jobs, machines, seed) in [(6, 5, 0), (8, 6, 19), (20, 15, 3)] {
        let schedule = job_shop(jobs, machines);
        let (mut solution, _, session) = adaptive_slice(&schedule, seed, 0, None, None, &stop, &stop);
        let (mut reference, _, reference_session) = adaptive_slice(&schedule, seed, 0, None, None, &stop, &stop);
        let mut reference_session = reference_session.unwrap();
        reference_session.test_use_generic_adaptive_probes();
        let mut session = session;
        let mut reference_session = Some(reference_session);
        for work in [1, 3, 5, 17, 63, 39, 128, 128] {
            let (next, metrics, next_session) = adaptive_slice(&schedule, seed, work, Some(&solution), session, &stop, &stop);
            let (next_reference, reference_metrics, next_reference_session) =
                adaptive_slice(&schedule, seed, work, Some(&reference), reference_session, &stop, &stop);
            assert_eq!(metrics.work_steps, reference_metrics.work_steps);
            assert_eq!(metrics.tabu_steps, reference_metrics.tabu_steps);
            assert_eq!(metrics.schedule_tsab.delta_probes, reference_metrics.schedule_tsab.delta_probes);
            assert_eq!(metrics.schedule_tsab.fast_transitions, reference_metrics.schedule_tsab.fast_transitions);
            assert_eq!(next.starts, next_reference.starts);
            assert_eq!(next.objectives, next_reference.objectives);
            assert!(next_session.as_ref().unwrap().test_same_tabu_decisions(next_reference_session.as_ref().unwrap()));
            assert_eq!(
                next_session.as_ref().unwrap().test_tsab_canonical_path(),
                next_reference_session.as_ref().unwrap().test_tsab_canonical_path()
            );
            assert_eq!(reference_metrics.schedule_tsab.forward_probe_attempts, 0);
            forward_attempts += metrics.schedule_tsab.forward_probe_attempts;
            replay(&schedule, &next);
            solution = next;
            reference = next_reference;
            session = next_session;
            reference_session = next_reference_session;
        }
    }
    assert!(forward_attempts > 100, "both paths must exercise exact N5 selection");
}

#[test]
fn adaptive_tsab_forward_probe_fallback_disables_only_subsequent_forward_probes() {
    let stop = AtomicBool::new(false);
    let (schedule, initial, _) = two_job_endpoint_fixture();
    for cap in [0, 1] {
        let (_, _, session) = adaptive_slice(&schedule, 0, 0, Some(&initial), None, &stop, &stop);
        let (_, _, reference_session) = adaptive_slice(&schedule, 0, 0, Some(&initial), None, &stop, &stop);
        let mut session = session.unwrap();
        let mut reference_session = reference_session.unwrap();
        session.test_force_tsab_fast_work_cap(cap);
        reference_session.test_use_generic_adaptive_probes();
        let (ok, selected, metrics, _) = session.test_select_adaptive_candidate(&stop);
        let (reference_ok, reference_selected, _, _) = reference_session.test_select_adaptive_candidate(&stop);
        assert!(ok && reference_ok && selected.is_some());
        assert_eq!(selected, reference_selected);
        assert_eq!(metrics.forward_probe_attempts, 1);
        assert_eq!(metrics.forward_probe_fallbacks, 1);
        assert_eq!(metrics.forward_probe_disables, 1);
        assert!(session.test_adaptive_forward_probes_disabled());
        assert!(session.test_same_tabu_decisions(&reference_session));
        for _ in 0..3 {
            let (ok, selected, metrics, _) = session.test_select_adaptive_candidate(&stop);
            let (reference_ok, reference_selected, _, _) = reference_session.test_select_adaptive_candidate(&stop);
            assert!(ok && reference_ok);
            assert_eq!(selected, reference_selected);
            assert_eq!(metrics.delta_probes, 1);
            assert_eq!(metrics.forward_probe_attempts, 0);
            assert_eq!(metrics.forward_probe_fallbacks, 0);
            assert_eq!(metrics.forward_probe_disables, 0);
            assert!(session.test_adaptive_forward_probes_disabled());
            assert!(session.test_same_tabu_decisions(&reference_session));
        }
        assert!(session.test_tsab_state_matches_full_oracle());
    }
}

#[test]
fn adaptive_tsab_forward_probe_disable_preserves_work_and_survives_checkpoints_and_resume() {
    let stop = AtomicBool::new(false);
    let schedule = job_shop(6, 5);
    let (mut solution, _, session) = adaptive_slice(&schedule, 0, 0, None, None, &stop, &stop);
    let (mut reference, _, reference_session) = adaptive_slice(&schedule, 0, 0, None, None, &stop, &stop);
    let mut session = session.unwrap();
    let mut reference_session = reference_session.unwrap();
    session.test_force_tsab_fast_work_cap(0);
    reference_session.test_force_tsab_fast_work_cap(0);
    reference_session.test_use_generic_adaptive_probes();
    let (ok, selected, metrics, _) = session.test_select_adaptive_candidate(&stop);
    let (reference_ok, reference_selected, _, _) = reference_session.test_select_adaptive_candidate(&stop);
    assert!(ok && reference_ok && selected.is_some());
    assert_eq!(selected, reference_selected);
    assert_eq!(metrics.forward_probe_fallbacks, 1);
    assert_eq!(metrics.forward_probe_disables, 1);
    let mut session = Some(session);
    let mut reference_session = Some(reference_session);
    let mut validations = 0;
    for work in [1, 3, 5, 17, 63, 39, 128, 128] {
        let (next, metrics, next_session) = adaptive_slice(&schedule, 0, work, Some(&solution), session, &stop, &stop);
        let (next_reference, reference_metrics, next_reference_session) =
            adaptive_slice(&schedule, 0, work, Some(&reference), reference_session, &stop, &stop);
        assert_eq!(metrics.work_steps, reference_metrics.work_steps);
        assert_eq!(metrics.tabu_steps, reference_metrics.tabu_steps);
        assert_eq!(metrics.schedule_tsab.delta_probes, reference_metrics.schedule_tsab.delta_probes);
        assert_eq!(next.starts, next_reference.starts);
        assert_eq!(next.objectives, next_reference.objectives);
        assert_eq!(metrics.schedule_tsab.forward_probe_attempts, 0);
        assert_eq!(metrics.schedule_tsab.forward_probe_fallbacks, 0);
        assert_eq!(metrics.schedule_tsab.forward_probe_disables, 0);
        assert!(next_session.as_ref().unwrap().test_adaptive_forward_probes_disabled());
        assert!(next_session.as_ref().unwrap().test_same_tabu_decisions(next_reference_session.as_ref().unwrap()));
        validations += metrics.schedule_tsab.fast_full_validations;
        replay(&schedule, &next);
        solution = next;
        reference = next_reference;
        session = next_session;
        reference_session = next_reference_session;
    }
    assert!(validations > 0, "persistent disable must cross complete oracle checkpoints");
}

#[test]
fn adaptive_tsab_interrupted_forward_probe_disables_and_resumes_with_a_generic_probe() {
    let stop = AtomicBool::new(false);
    let (schedule, initial, _) = two_job_endpoint_fixture();
    let (_, _, session) = adaptive_slice(&schedule, 0, 0, Some(&initial), None, &stop, &stop);
    let mut session = session.unwrap();
    session.test_stop_adaptive_forward_probe_at_work_cap(1);
    let (ok, selected, metrics, _) = session.test_select_adaptive_candidate(&stop);
    assert!(!ok && selected.is_none());
    assert!(stop.load(Ordering::Acquire));
    assert_eq!(metrics.forward_probe_disables, 1);
    assert_eq!(metrics.forward_probe_attempts, 0, "attempt accounting includes completed forward calls only");
    assert!(session.test_adaptive_forward_probes_disabled());
    stop.store(false, Ordering::Release);
    assert!(session.test_tsab_state_matches_full_oracle());
    let (ok, selected, metrics, _) = session.test_select_adaptive_candidate(&stop);
    assert!(ok && selected.is_some());
    assert_eq!(metrics.delta_probes, 1);
    assert_eq!(metrics.forward_probe_attempts, 0);
    assert_eq!(metrics.forward_probe_fallbacks, 0);
    assert_eq!(metrics.forward_probe_disables, 0);
}

#[test]
fn adaptive_tsab_forward_probes_remain_enabled_without_a_fallback() {
    let stop = AtomicBool::new(false);
    let (schedule, initial, _) = two_job_endpoint_fixture();
    let (_, _, session) = adaptive_slice(&schedule, 0, 0, Some(&initial), None, &stop, &stop);
    let mut session = session.unwrap();
    for _ in 0..3 {
        let (ok, selected, metrics, _) = session.test_select_adaptive_candidate(&stop);
        assert!(ok && selected.is_some());
        assert_eq!(metrics.forward_probe_attempts, 1);
        assert_eq!(metrics.forward_probe_fallbacks, 0);
        assert_eq!(metrics.forward_probe_disables, 0);
        assert!(!session.test_adaptive_forward_probes_disabled());
    }
}

#[test]
fn adaptive_tsab_primary_pruning_is_disabled_for_zero_duration_shapes() {
    let mut schedule = job_shop(3, 3);
    schedule.intervals[0].duration = 0;
    let stop = AtomicBool::new(false);
    let (_, _, session) = adaptive_slice(&schedule, 0, 0, None, None, &stop, &stop);
    let mut session = session.unwrap();
    let (ok, selected, metrics, avoided) = session.test_select_adaptive_candidate(&stop);
    assert!(ok && selected.is_none());
    assert_eq!(metrics.delta_probes, 0);
    assert_eq!(avoided, 0);
}

#[test]
fn adaptive_tsab_blocked_n5_can_escape_to_an_admissible_broader_move() {
    let stop = AtomicBool::new(false);
    for (jobs, machines) in [(4, 3), (4, 4), (5, 4)] {
        let schedule = job_shop(jobs, machines);
        let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
        for seed in 0..16 {
            let state = JobShopState::giffler_thompson(&problem, seed, DispatchRule::EarliestStart, &stop).unwrap().unwrap();
            let current = state.to_solution();
            let mut moves = Vec::new();
            state
                .visit_strict_n5_canonical_moves(&stop, |movement, arcs| {
                    moves.push((movement, arcs));
                    Ok(())
                })
                .unwrap();
            if moves.is_empty() {
                continue;
            }
            let mut elite = current.clone();
            let mut feasible = 0;
            for &(movement, _) in &moves {
                let mut trial = state.rebuilt_from_machine_sequences(state.machine_sequences().to_vec(), &stop).unwrap().unwrap();
                if let MoveOutcome::Accepted { current, .. } =
                    trial.consider_move_full_oracle(movement, MinimizingMoveAcceptance::Always, &stop).unwrap()
                {
                    feasible += 1;
                    if current < elite.objectives[0] {
                        elite = trial.to_solution();
                    }
                }
            }
            if feasible == 0 {
                continue;
            }
            let (_, _, session) = adaptive_slice(&schedule, seed, 0, Some(&current), None, &stop, &stop);
            let mut session = session.unwrap();
            session.test_set_verified_elite(elite.clone());
            for (_, arcs) in moves {
                for arc in arcs.added.into_iter().flatten() {
                    session.test_mark_tabu_arc(arc, 100);
                }
            }
            let (solution, metrics, session) = adaptive_slice(&schedule, seed, 9, Some(&elite), Some(session), &stop, &stop);
            assert_eq!(metrics.schedule_tsab.blocked_n6_requests, 1);
            assert_eq!(metrics.schedule_tsab.blocked_n6_attempts, 1);
            assert!(metrics.schedule_tsab.restart_delta_probes <= 8);
            assert_eq!(metrics.tabu_forced_moves, 0, "blocked N6 accepts only admissible or aspirating moves");
            if metrics.schedule_tsab.restart_oracle_commits == 0 {
                continue;
            }
            let mut session = session.unwrap();
            assert_eq!(metrics.work_steps, 9);
            assert_eq!(session.test_adaptive_blocked_escape_state().2, 1);
            assert_eq!(
                session.test_tsab_restart_state().0,
                u64::from(solution.objectives[0] >= elite.objectives[0]),
                "only a genuine elite improvement resets the N5 clock"
            );
            assert_eq!(session.test_tsab_restart_state().2, 0);
            assert_eq!(session.test_tsab_adaptive_policy().1, 0);
            assert!(session.test_tsab_state_matches_full_oracle());
            replay(&schedule, &solution);
            return;
        }
    }
    panic!("small deterministic fixtures must expose an admissible broader move beyond a tabu-blocked N5 neighborhood");
}

#[test]
fn adaptive_tsab_escape_uses_current_schedule_and_preserves_unrelated_tabu() {
    let (schedule, initial, optimum) = two_job_endpoint_fixture();
    let stop = AtomicBool::new(false);
    let (_, _, session) = adaptive_slice(&schedule, 0, 0, Some(&initial), None, &stop, &stop);
    let mut session = session.expect("initial trajectory");
    assert_eq!(session.test_tsab_fast_checkpoint_state(), (0, false));
    session.test_set_verified_elite(optimum.clone());
    let unrelated = MachineArc { machine: 0, before: 0, after: 3 };
    session.test_mark_tabu_arc(unrelated, 100);
    session.test_force_adaptive_restart_without_rewinding();
    let (solution, metrics, session) = adaptive_slice(&schedule, 0, 8, Some(&optimum), Some(session), &stop, &stop);
    let mut session = session.expect("escape retains current trajectory");
    assert_eq!(metrics.work_steps, 8);
    assert_eq!(metrics.schedule_tsab.restart_best_base_objective, Some(9));
    assert_eq!(metrics.schedule_tsab.restart_oracle_commits, 1);
    assert_eq!(metrics.schedule_tsab.elite_restarts, 0);
    let snapshot = session.test_tsab_trajectory_snapshot();
    assert_eq!(snapshot.5, 5, "the escape starts from current9, rather than rewinding to elite5");
    assert_eq!(snapshot.2.get(&unrelated), Some(&100));
    assert!(session.test_tsab_state_matches_full_oracle());
    replay(&schedule, &solution);
    assert_eq!(solution.objectives, optimum.objectives);
}

#[test]
fn adaptive_tsab_exact_ranking_honors_tabu_aspiration_with_a_non_tabu_alternative() {
    let stop = AtomicBool::new(false);
    for (jobs, machines) in [(3, 3), (4, 3), (4, 4), (6, 5)] {
        let schedule = job_shop(jobs, machines);
        let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
        for seed in 0..32 {
            let state = JobShopState::giffler_thompson(&problem, seed, DispatchRule::EarliestStart, &stop).unwrap().unwrap();
            let mut moves = Vec::new();
            state
                .visit_strict_n5_canonical_moves(&stop, |movement, arcs| {
                    moves.push((movement, arcs));
                    Ok(())
                })
                .unwrap();
            if !(2..=4).contains(&moves.len()) {
                continue;
            }
            let mut oracle = Vec::new();
            for &(movement, arcs) in &moves {
                let mut trial = state.rebuilt_from_machine_sequences(state.machine_sequences().to_vec(), &stop).unwrap().unwrap();
                if let MoveOutcome::Accepted { current, .. } =
                    trial.consider_move_full_oracle(movement, MinimizingMoveAcceptance::Always, &stop).unwrap()
                {
                    oracle.push((current, movement, arcs));
                }
            }
            oracle.sort_unstable_by_key(|candidate| (candidate.0, candidate.1));
            let Some(&(best, _, arcs)) = oracle.first() else { continue };
            if best >= state.makespan() || oracle.iter().filter(|candidate| candidate.0 == best).count() != 1 {
                continue;
            }
            let Some(marker) = arcs
                .added
                .into_iter()
                .flatten()
                .find(|&arc| oracle.iter().any(|candidate| candidate.0 > best && !candidate.2.added.contains(&Some(arc))))
            else {
                continue;
            };
            let (initial, _, session) = adaptive_slice(&schedule, seed, 0, None, None, &stop, &stop);
            assert_eq!(initial.starts, state.starts());
            let mut session = session.unwrap();
            session.test_mark_tabu_arc(marker, 100);
            let (_, metrics, session) = adaptive_slice(&schedule, seed, 1, Some(&initial), Some(session), &stop, &stop);
            let session = session.expect("aspirating trajectory");
            assert_eq!(metrics.tabu_aspirations, 1);
            assert_eq!(metrics.schedule_tsab.aspirations, 1);
            assert_eq!(metrics.schedule_tsab.ranking_audits, 1);
            assert_eq!(session.test_tsab_pending_elite_objective(), Some(best));
            let finalization = AtomicBool::new(false);
            stop.store(true, Ordering::Release);
            let (solution, _, _) = adaptive_slice(&schedule, seed, 0, Some(&initial), Some(session), &stop, &finalization);
            replay(&schedule, &solution);
            assert_eq!(solution.objectives, [best]);
            return;
        }
    }
    panic!("small deterministic fixtures must contain a uniquely improving tabu move and a feasible non-tabu alternative");
}

#[test]
fn adaptive_tsab_escape_credit_cancellation_and_path_rotation_preserve_semantics() {
    let schedule = job_shop(1, 4);
    let stop = AtomicBool::new(false);
    let (mut solution, _, session) = adaptive_slice(&schedule, 7, 0, None, None, &stop, &stop);
    let mut session = session.unwrap();
    let mut policies = Vec::new();
    for _ in 0..7 {
        session.test_force_adaptive_restart_without_rewinding();
        let snapshot = session.test_tsab_trajectory_snapshot();
        let (next, partial, next_session) = adaptive_slice(&schedule, 7, 3, Some(&solution), Some(session), &stop, &stop);
        assert_eq!(partial.schedule_tsab.restart_attempts, 0);
        assert_eq!(partial.work_steps, 3);
        let session_before_cancel = next_session.unwrap();
        assert_eq!(session_before_cancel.test_tsab_trajectory_snapshot(), snapshot);
        stop.store(true, Ordering::Release);
        let (_, cancelled, next_session) = adaptive_slice(&schedule, 7, 5, Some(&next), Some(session_before_cancel), &stop, &stop);
        assert_eq!(cancelled.work_steps, 0);
        let next_session = next_session.unwrap();
        assert_eq!(next_session.test_tsab_trajectory_snapshot(), snapshot);
        stop.store(false, Ordering::Release);
        let (next, completed, next_session) = adaptive_slice(&schedule, 7, 5, Some(&next), Some(next_session), &stop, &stop);
        assert_eq!(completed.schedule_tsab.restart_attempts, 1);
        assert_eq!(completed.work_steps, 5);
        session = next_session.unwrap();
        assert!(session.test_tsab_state_matches_full_oracle());
        assert_eq!(session.test_tsab_trajectory_snapshot(), snapshot);
        let policy = session.test_tsab_canonical_path().0;
        assert!(!policies.contains(&policy));
        policies.push(policy);
        replay(&schedule, &next);
        solution = next;
    }
    assert_eq!(policies.len(), 7);
}

#[test]
fn adaptive_tsab_ft06_fixed_work_regression_reaches_55_for_all_five_seeds() {
    // Fisher-Thompson ft06, the formerly failing seed0 stopped at58.
    let jobs = [
        [(2, 1), (0, 3), (1, 6), (3, 7), (5, 3), (4, 6)],
        [(1, 8), (2, 5), (4, 10), (5, 10), (0, 10), (3, 4)],
        [(2, 5), (3, 4), (5, 8), (0, 9), (1, 1), (4, 7)],
        [(1, 5), (0, 5), (2, 5), (3, 3), (4, 8), (5, 9)],
        [(2, 9), (1, 3), (4, 5), (5, 4), (0, 3), (3, 1)],
        [(1, 3), (3, 3), (5, 9), (0, 10), (4, 4), (2, 1)],
    ];
    let horizon = jobs.iter().flatten().map(|pair| pair.1).sum();
    let mut groups = vec![Vec::new(); 6];
    let intervals = jobs
        .iter()
        .flatten()
        .enumerate()
        .map(|(operation, &(machine, duration))| {
            groups[machine].push(operation);
            IntervalVar { duration, horizon, modes: Vec::new(), optional: false }
        })
        .collect();
    let schedule = Schedule {
        intervals,
        precedences: (0..6).flat_map(|job| (0..5).map(move |operation| (job * 6 + operation, job * 6 + operation + 1))).collect(),
        resources: groups.into_iter().map(Resource::NoOverlap).collect(),
        minimize_makespan: true,
    };
    for seed in 0..5 {
        let stop = AtomicBool::new(false);
        let mut solution = None;
        let mut session = None;
        let mut work = 0;
        while work < 10_000 {
            let (next, metrics, next_session) =
                adaptive_slice(&schedule, seed, (10_000 - work).min(128), solution.as_ref(), session, &stop, &stop);
            assert!(metrics.work_steps > 0);
            assert_eq!(metrics.schedule_tsab.fast_oracle_mismatches, 0);
            work += metrics.work_steps;
            solution = Some(next);
            session = next_session;
        }
        stop.store(true, Ordering::Release);
        let finalization = AtomicBool::new(false);
        let (solution, metrics, _) = adaptive_slice(&schedule, seed, 0, solution.as_ref(), session, &stop, &finalization);
        assert_eq!(metrics.schedule_tsab.fast_oracle_mismatches, 0);
        replay(&schedule, &solution);
        assert_eq!(solution.objectives, [55], "ft06 seed{seed}");
    }
}

#[test]
fn adaptive_tsab_empty_n5_neighborhood_still_returns_a_complete_bounded_incumbent() {
    let stop = AtomicBool::new(false);
    for schedule in [job_shop(1, 1), job_shop(1, 4), job_shop(5, 1), job_shop(2, 2)] {
        let (solution, metrics, mut session) = adaptive_slice(&schedule, 7, 8, None, None, &stop, &stop);
        replay(&schedule, &solution);
        assert_eq!(metrics.work_steps, 8);
        assert_eq!(metrics.schedule_tsab.fast_oracle_mismatches, 0);
        assert!(session.as_mut().expect("recognized JSSP").test_tsab_state_matches_full_oracle());
    }
}

#[test]
fn adaptive_tsab_interruption_retains_only_a_verified_elite_and_can_resume_validation() {
    let schedule = job_shop(6, 5);
    let stop = AtomicBool::new(false);
    let (initial, _, session) = adaptive_slice(&schedule, 13, 0, None, None, &stop, &stop);
    let mut session = session.expect("initial trajectory");
    session.test_stop_after_fast_attempts(1);
    let (interrupted, metrics, session) = adaptive_slice(&schedule, 13, 8, Some(&initial), Some(session), &stop, &stop);
    assert!(stop.load(Ordering::Acquire));
    assert_eq!(metrics.schedule_tsab.fast_attempts, 1);
    replay(&schedule, &interrupted);
    assert_eq!(interrupted.objectives, initial.objectives);
    let session = session.expect("atomic private trajectory survives an interruption");
    assert!(session.test_tsab_fast_validation_due_on_resume());
    stop.store(false, Ordering::Release);
    let (resumed, metrics, mut session) = adaptive_slice(&schedule, 13, 0, Some(&interrupted), Some(session), &stop, &stop);
    replay(&schedule, &resumed);
    assert_eq!(metrics.work_steps, 0);
    assert_eq!(metrics.schedule_tsab.fast_full_validations, 2, "pending and current schedules each cross a complete oracle");
    assert!(!session.as_ref().expect("resumed session").test_tsab_fast_validation_due_on_resume());
    assert!(session.as_mut().expect("resumed session").test_tsab_state_matches_full_oracle());
}

#[test]
fn adaptive_tsab_pending_promotion_preserves_current_tabu_policy_and_shadow_identity() {
    use qayd::engines::ls::lists::schedule_elite::{ScheduleEliteArchive, ScheduleEliteSolveToken};
    let schedule = job_shop(6, 5);
    let stop = AtomicBool::new(false);
    for seed in 0..8 {
        let (mut solution, _, session) = adaptive_slice(&schedule, seed, 0, None, None, &stop, &stop);
        let mut session = session.unwrap();
        for _ in 0..96 {
            let (next, _, next_session) = adaptive_slice(&schedule, seed, 1, Some(&solution), Some(session), &stop, &stop);
            solution = next;
            session = next_session.unwrap();
            let Some(pending) = session.test_tsab_pending_elite_objective() else { continue };
            let snapshot = session.test_tsab_trajectory_snapshot();
            if pending >= solution.objectives[0] || snapshot.5 <= pending || snapshot.2.is_empty() {
                continue;
            }
            let policy = session.test_tsab_canonical_path();
            // Interrupt precisely between the two independent oracles. The
            // pending elite must remain private and recoverable on resumption.
            session.test_require_tsab_validation_on_resume(true);
            let (interrupted, partial, next_session) = adaptive_slice(&schedule, seed, 0, Some(&solution), Some(session), &stop, &stop);
            assert!(stop.load(Ordering::Acquire));
            assert_eq!(partial.work_steps, 0);
            assert_eq!(partial.schedule_tsab.fast_full_validations, 1);
            assert_eq!(partial.schedule_tsab.fast_pending_promotions, 0);
            assert_eq!(interrupted.starts, solution.starts);
            let mut session = next_session.unwrap();
            assert_eq!(session.test_tsab_pending_elite_objective(), Some(pending));
            assert!(session.test_tsab_fast_validation_due_on_resume());
            assert_eq!(session.test_tsab_trajectory_snapshot(), snapshot);
            assert_eq!(session.test_tsab_canonical_path(), policy);
            stop.store(false, Ordering::Release);
            let token = ScheduleEliteSolveToken::new();
            session.test_enable_search_elite_shadow(&token);
            let (promoted, metrics, next_session) = solve_schedule_capped_persistent_hybrid_with_strategy(
                &schedule,
                seed,
                seed,
                0,
                1,
                &stop,
                &stop,
                0,
                Some(&interrupted),
                Some(session),
                false,
                Some(&token),
                false,
                false,
                ScheduleJsspSearchStrategy::TsabAdaptive,
                None,
                &mut |_| {},
            );
            let mut session = next_session.unwrap();
            assert_eq!(metrics.work_steps, 0);
            assert_eq!(metrics.schedule_tsab.fast_full_validations, 2);
            assert_eq!(metrics.schedule_tsab.fast_pending_promotions, 1);
            assert_eq!(metrics.schedule_tsab.fast_oracle_mismatches, 0);
            assert_eq!(promoted.objectives, [pending]);
            replay(&schedule, &promoted);
            assert_eq!(session.test_tsab_trajectory_snapshot(), snapshot);
            assert_eq!(session.test_tsab_canonical_path(), policy);
            assert!(session.test_tsab_state_matches_full_oracle());
            let candidate = session.take_search_elite_candidate().expect("promotion captures the verified pending schedule");
            let mut archive = ScheduleEliteArchive::new();
            session.merge_search_elite_batch(&mut archive, vec![candidate], &stop).unwrap();
            assert_eq!(archive.best().unwrap().objective(), pending);
            assert!(pending < snapshot.5, "shadow candidate must be the elite, not the current private trajectory");
            return;
        }
    }
    panic!("deterministic fixtures must contain a pending improvement followed by a worse private state with live tabu memory");
}

#[test]
fn adaptive_tsab_replays_every_seed_and_preserves_the_incumbent_on_a_prearmed_stop() {
    let schedule = job_shop(5, 4);
    for seed in 0..8 {
        let stop = AtomicBool::new(false);
        let (initial, _, session) = adaptive_slice(&schedule, seed, 0, None, None, &stop, &stop);
        let (solution, _, session) = adaptive_slice(&schedule, seed, 96, Some(&initial), session, &stop, &stop);
        replay(&schedule, &solution);
        assert!(solution.objectives[0] <= initial.objectives[0]);
        let session = session.expect("retained session");
        let (threshold, _) = session.test_tsab_adaptive_policy();
        assert_eq!(threshold, 128);
        stop.store(true, Ordering::Release);
        let (stopped, metrics, _) = adaptive_slice(&schedule, seed, 16, Some(&solution), Some(session), &stop, &stop);
        replay(&schedule, &stopped);
        assert_eq!(metrics.work_steps, 0);
        assert_eq!(stopped.starts, solution.starts);
        assert_eq!(stopped.objectives, solution.objectives);
    }
}

fn local_jsplib_schedule(name: &str) -> Schedule {
    let text = std::fs::read_to_string(format!("data/jssp/jsplib/{name}")).expect("local JSPLIB instance");
    let values: Vec<usize> = text
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .flat_map(|line| line.split_whitespace())
        .map(|word| word.parse().expect("JSPLIB integer"))
        .collect();
    let (jobs, machines) = (values[0], values[1]);
    assert_eq!(values.len(), 2 + jobs * machines * 2);
    let mut groups = vec![Vec::new(); machines];
    let mut intervals = Vec::new();
    let horizon = values[2..].chunks_exact(2).map(|pair| pair[1] as i64).sum();
    for (operation, pair) in values[2..].chunks_exact(2).enumerate() {
        groups[pair[0]].push(operation);
        intervals.push(IntervalVar { duration: pair[1] as i64, horizon, modes: Vec::new(), optional: false });
    }
    Schedule {
        intervals,
        precedences: (0..jobs)
            .flat_map(|job| (0..machines - 1).map(move |operation| (job * machines + operation, job * machines + operation + 1)))
            .collect(),
        resources: groups.into_iter().map(Resource::NoOverlap).collect(),
        minimize_makespan: true,
    }
}

#[test]
#[ignore = "short paired exact-N5 probe diagnostic; requires the local JSPLIB corpus"]
fn adaptive_tsab_forward_probe_fixed_work_diagnostic() {
    const WORK: u64 = 1_000;
    for name in ["orb10", "la38", "ta80"] {
        let schedule = local_jsplib_schedule(name);
        let stop = AtomicBool::new(false);
        let (mut solution, _, session) = adaptive_slice(&schedule, 0, 0, None, None, &stop, &stop);
        let (mut reference, _, reference_session) = adaptive_slice(&schedule, 0, 0, None, None, &stop, &stop);
        let mut reference_session = reference_session.unwrap();
        reference_session.test_use_generic_adaptive_probes();
        let mut reference_session = Some(reference_session);
        let mut session = session;
        let mut work = 0;
        let mut transitions = 0;
        let mut generic_elapsed = std::time::Duration::ZERO;
        let mut forward_elapsed = std::time::Duration::ZERO;
        let mut generic_cone_work = 0;
        let mut forward_cone_work = 0;
        let mut forward_attempts = 0;
        let mut forward_fallbacks = 0;
        while work < WORK {
            let quota = (WORK - work).min(128);
            let started = std::time::Instant::now();
            let (next_reference, reference_metrics, next_reference_session) =
                adaptive_slice(&schedule, 0, quota, Some(&reference), reference_session, &stop, &stop);
            generic_elapsed += started.elapsed();
            let started = std::time::Instant::now();
            let (next, metrics, next_session) = adaptive_slice(&schedule, 0, quota, Some(&solution), session, &stop, &stop);
            forward_elapsed += started.elapsed();
            assert!(metrics.work_steps > 0);
            assert_eq!(metrics.work_steps, reference_metrics.work_steps);
            assert_eq!(metrics.tabu_steps, reference_metrics.tabu_steps);
            assert_eq!(next.starts, next_reference.starts);
            assert_eq!(next.objectives, next_reference.objectives);
            assert!(next_session.as_ref().unwrap().test_same_tabu_decisions(next_reference_session.as_ref().unwrap()));
            work += metrics.work_steps;
            transitions += metrics.tabu_steps;
            generic_cone_work += reference_metrics.dirty_cone_operations;
            forward_cone_work += metrics.dirty_cone_operations;
            forward_attempts += metrics.schedule_tsab.forward_probe_attempts;
            forward_fallbacks += metrics.schedule_tsab.forward_probe_fallbacks;
            solution = next;
            reference = next_reference;
            session = next_session;
            reference_session = next_reference_session;
        }
        replay(&schedule, &solution);
        eprintln!(
            "forward-probe {name} work={work} transitions={transitions} objective={} generic_wall_ms={:.3} forward_wall_ms={:.3} generic_cone_work={generic_cone_work} forward_cone_work={forward_cone_work} attempts={forward_attempts} fallbacks={forward_fallbacks}",
            solution.objectives[0], generic_elapsed.as_secs_f64() * 1_000.0, forward_elapsed.as_secs_f64() * 1_000.0,
        );
    }
}

#[test]
#[ignore = "bounded fixed-work JSSP quality diagnostic; requires the local JSPLIB corpus"]
fn adaptive_tsab_jsplib_fixed_work_diagnostic() {
    use qayd::engines::ls::lists::solve_schedule_capped_persistent;
    const WORK: u64 = 20_000;
    for name in ["ft06", "orb10"] {
        let schedule = local_jsplib_schedule(name);
        for seed in 0..5 {
            for adaptive in [false, true] {
                let stop = AtomicBool::new(false);
                let mut solution = None::<CollectionSolution>;
                let mut session = None::<ScheduleSearchSession>;
                let mut work = 0;
                let mut transitions = 0;
                let mut escapes = 0;
                let mut aspirations = 0;
                let started = std::time::Instant::now();
                while work < WORK {
                    let slice = (WORK - work).min(128);
                    let (next, metrics, next_session) = if adaptive {
                        adaptive_slice(&schedule, seed, slice, solution.as_ref(), session, &stop, &stop)
                    } else {
                        solve_schedule_capped_persistent(
                            &schedule,
                            seed,
                            seed,
                            0,
                            1,
                            &stop,
                            slice,
                            solution.as_ref(),
                            session,
                            false,
                            &mut |_| {},
                        )
                    };
                    assert!(metrics.work_steps != 0, "fixed-work diagnostic must make bounded progress");
                    work += metrics.work_steps;
                    transitions += metrics.tabu_steps;
                    escapes += metrics.schedule_tsab.restart_oracle_commits;
                    aspirations += metrics.tabu_aspirations;
                    assert_eq!(metrics.schedule_tsab.fast_oracle_mismatches, 0);
                    solution = Some(next);
                    session = next_session;
                }
                if adaptive {
                    let finalization = AtomicBool::new(false);
                    stop.store(true, Ordering::Release);
                    let (next, _, _) = adaptive_slice(&schedule, seed, 0, solution.as_ref(), session, &stop, &finalization);
                    solution = Some(next);
                }
                let solution = solution.expect("complete diagnostic incumbent");
                replay(&schedule, &solution);
                eprintln!("fixed-work {name} seed={seed} adaptive={adaptive} objective={} work={work} transitions={transitions} escapes={escapes} aspirations={aspirations} wall_ms={}", solution.objectives[0], started.elapsed().as_millis());
            }
        }
    }
}
