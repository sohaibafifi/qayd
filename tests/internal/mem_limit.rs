//! Memory-limit accounting and degrade-policy tests.
//!
//! The internal engine suite installs the tracking allocator here so memory
//! accounting is exercised without making solver internals public.
//!
//! Hard-stop semantics (a run halted by the shared stop flag reports
//! SATISFIABLE/UNKNOWN, never OPTIMUM/UNSATISFIABLE) are already covered by
//! `parallel_split_interruption_is_not_a_proof` and
//! `parallel_probe_interruption_is_not_a_proof` in `tests/xcsp.rs`. The memory
//! watcher sets the same flag, so those tests cover the hard limit's report path.

use std::hint::black_box;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::orchestrator::{execute_specialized, SolveLimits, SolveRequest, TerminationReason};
use qayd::mem::{self, Pressure};

#[global_allocator]
static ALLOC: mem::Tracking = mem::Tracking;

#[test]
fn allocator_counts_live_bytes() {
    let allocated_before = mem::allocated_total();
    let deallocated_before = mem::deallocated_total();
    let buf: Vec<u8> = black_box(vec![7u8; 8 * 1024 * 1024]);
    let allocated_during = mem::allocated_total();
    assert!(allocated_during >= allocated_before + buf.len(), "allocation was not counted: {allocated_before} -> {allocated_during}");
    assert!(mem::peak() >= mem::live(), "peak below live");
    let len = buf.len();
    drop(black_box(buf));
    let deallocated_after = mem::deallocated_total();
    assert!(deallocated_after >= deallocated_before + len, "deallocation was not counted: {deallocated_before} -> {deallocated_after}");
}

#[test]
fn classify_thresholds() {
    // limit 1000: soft at 85% (850), hard at 95% (950).
    assert_eq!(mem::classify(849, 1000), Pressure::None);
    assert_eq!(mem::classify(850, 1000), Pressure::Soft);
    assert_eq!(mem::classify(949, 1000), Pressure::Soft);
    assert_eq!(mem::classify(950, 1000), Pressure::Hard);
    // limit 0 means unlimited: never degrades.
    assert_eq!(mem::classify(usize::MAX, 0), Pressure::None);
}

#[test]
fn degrade_budget_halves_with_floor() {
    assert_eq!(mem::degrade_budget(8000), 4000);
    assert_eq!(mem::degrade_budget(5000), 2500);
    // Never below the floor, even from a small or already-floored budget.
    assert_eq!(mem::degrade_budget(3000), mem::LEARNED_FLOOR);
    assert_eq!(mem::degrade_budget(100), mem::LEARNED_FLOOR);
    assert_eq!(mem::degrade_budget(mem::LEARNED_FLOOR), mem::LEARNED_FLOOR);
}

#[test]
fn orchestrator_budget_owns_memory_cancellation() {
    let limit = u64::try_from(mem::live()).unwrap_or(u64::MAX).saturating_add(4 * 1024 * 1024);
    let request = SolveRequest { limits: SolveLimits { memory_bytes: Some(limit), ..SolveLimits::default() }, ..SolveRequest::default() };
    execute_specialized(&request, Arc::new(AtomicBool::new(false)), |budget| {
        let allocation = black_box(vec![9u8; 8 * 1024 * 1024]);
        let started = Instant::now();
        while !budget.expired() && started.elapsed() < Duration::from_secs(2) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(budget.expired(), "memory monitor did not cancel the shared budget");
        assert_eq!(budget.termination_reason(), TerminationReason::MemoryLimit);
        drop(black_box(allocation));
        Ok(())
    })
    .unwrap();
}
