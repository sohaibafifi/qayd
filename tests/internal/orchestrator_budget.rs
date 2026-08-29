use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use qayd::orchestrator::{verify_final_supervised_with_budget, worker_iteration_quota, SolveBudget, SolveError, TerminationReason};

#[test]
fn worker_iteration_quotas_preserve_finite_totals() {
    for workers in [1usize, 2, 4, 9] {
        for total in [0u64, 1, 2, 7, 31, u64::MAX - 1] {
            let quotas = (0..workers).map(|worker| worker_iteration_quota(total, worker, workers)).collect::<Vec<_>>();
            assert_eq!(quotas.iter().map(|&quota| u128::from(quota)).sum::<u128>(), u128::from(total));
            let smallest = *quotas.iter().min().expect("at least one worker");
            let largest = *quotas.iter().max().expect("at least one worker");
            assert!(largest - smallest <= 1, "uneven quota for total {total} across {workers} workers: {quotas:?}");
        }
    }
}

#[test]
fn unbounded_iteration_quota_remains_unbounded_for_every_worker() {
    for workers in [1usize, 2, 4, 9] {
        for worker in 0..workers {
            assert_eq!(worker_iteration_quota(u64::MAX, worker, workers), u64::MAX);
        }
    }
}

#[test]
fn finalization_reserve_records_and_enforces_the_deadline() {
    let budget = SolveBudget::new(Some(Duration::from_millis(8)));
    let search_stop = budget.search_stop();

    assert!(search_stop.flag().load(Ordering::Acquire));
    assert_eq!(budget.termination_reason(), TerminationReason::Deadline);
    assert!(!budget.expired(), "the early search cutoff must not fire the request's external stop token");

    let guard = Instant::now();
    while !budget.expired() && guard.elapsed() < Duration::from_millis(250) {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(budget.expired(), "the request deadline did not stop finalization");
    assert_eq!(budget.termination_reason(), TerminationReason::Deadline);
}

#[test]
fn larger_finalization_floor_is_opt_in_and_preserves_a_short_search_slice() {
    let ordinary_budget = SolveBudget::new(Some(Duration::from_secs(2)));
    let ordinary = ordinary_budget.search_stop();
    let schedule_budget = SolveBudget::new(Some(Duration::from_secs(2)));
    let schedule = schedule_budget.search_stop_with_finalization_reserve(Duration::from_millis(2_500));

    let guard = Instant::now();
    while !schedule.flag().load(Ordering::Acquire) && guard.elapsed() < Duration::from_millis(500) {
        std::thread::sleep(Duration::from_millis(1));
    }

    assert!(schedule.flag().load(Ordering::Acquire), "the schedule-specific reserve did not stop search early");
    assert!(!ordinary.flag().load(Ordering::Acquire), "the ordinary engine reserve changed");
    assert!(!schedule_budget.expired(), "the early cutoff must leave the parent finalization budget alive");
    assert_eq!(schedule_budget.termination_reason(), TerminationReason::Deadline);
}

#[test]
fn larger_finalization_floor_still_observes_hard_cancellation_immediately() {
    let budget = SolveBudget::new(Some(Duration::from_secs(5)));
    let search = budget.search_stop_with_finalization_reserve(Duration::from_millis(2_500));
    assert!(!search.flag().load(Ordering::Acquire));

    budget.cancel_with(TerminationReason::MemoryLimit);
    let guard = Instant::now();
    while !search.flag().load(Ordering::Acquire) && guard.elapsed() < Duration::from_millis(100) {
        std::thread::sleep(Duration::from_millis(1));
    }

    assert!(search.flag().load(Ordering::Acquire));
    assert!(budget.hard_cancelled());
    assert_eq!(budget.termination_reason(), TerminationReason::MemoryLimit);
}

#[test]
fn non_cooperative_final_replay_cannot_escape_the_bounded_grace() {
    let budget = SolveBudget::new(None);
    budget.cancel_with(TerminationReason::Deadline);
    let started = Instant::now();

    let result = verify_final_supervised_with_budget(&budget, |_| {
        std::thread::sleep(Duration::from_millis(600));
        Ok(())
    });

    assert!(matches!(result, Err(SolveError::Interrupted(_))));
    assert!(started.elapsed() < Duration::from_millis(500), "a non-cooperative replay outlived its 250 ms grace");
}

#[test]
fn hard_cancellation_overrides_a_soft_finalization_deadline() {
    for reason in [TerminationReason::ExternalCancellation, TerminationReason::MemoryLimit, TerminationReason::EventSink] {
        let budget = SolveBudget::new(Some(Duration::from_millis(8)));
        let _search_stop = budget.search_stop();
        assert_eq!(budget.termination_reason(), TerminationReason::Deadline);

        budget.cancel_with(reason);

        assert!(budget.hard_cancelled());
        assert!(budget.expired());
        assert_eq!(budget.termination_reason(), reason);
    }
}

#[test]
fn optional_phase_expiration_does_not_cancel_the_request() {
    let budget = SolveBudget::new(None);
    let phase = budget.optional_phase_stop(Duration::from_millis(5));
    let guard = Instant::now();
    while !phase.flag().load(Ordering::Acquire) && guard.elapsed() < Duration::from_millis(100) {
        std::thread::sleep(Duration::from_millis(1));
    }

    assert!(phase.flag().load(Ordering::Acquire));
    assert!(!budget.expired());
    assert_eq!(budget.termination_reason(), TerminationReason::Running);

    let phase = budget.optional_phase_stop(Duration::from_secs(1));
    budget.cancel();
    let guard = Instant::now();
    while !phase.flag().load(Ordering::Acquire) && guard.elapsed() < Duration::from_millis(100) {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(phase.flag().load(Ordering::Acquire));
}

#[test]
fn final_replay_grace_is_bounded_and_observes_hard_cancellation() {
    let budget = SolveBudget::new(Some(Duration::from_millis(8)));
    let _search_stop = budget.search_stop();
    let grace = budget.finalization_grace_stop();
    assert!(!grace.flag().load(Ordering::Acquire));

    budget.cancel_with(TerminationReason::ExternalCancellation);
    let guard = Instant::now();
    while !grace.flag().load(Ordering::Acquire) && guard.elapsed() < Duration::from_millis(100) {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(grace.flag().load(Ordering::Acquire));
    assert!(guard.elapsed() < Duration::from_millis(100));

    let budget = SolveBudget::new(Some(Duration::from_millis(8)));
    let _search_stop = budget.search_stop();
    let grace = budget.finalization_grace_stop();
    let guard = Instant::now();
    while !grace.flag().load(Ordering::Acquire) && guard.elapsed() < Duration::from_secs(1) {
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(grace.flag().load(Ordering::Acquire));
    assert!(guard.elapsed() < Duration::from_secs(1));
}
