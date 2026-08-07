use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use qayd::orchestrator::{SolveBudget, TerminationReason};

#[test]
fn finalization_reserve_records_and_enforces_the_deadline() {
    let budget = SolveBudget::new(Some(Duration::from_millis(8)));
    let search_stop = budget.search_stop();

    assert!(search_stop.flag().load(Ordering::Acquire));
    assert_eq!(budget.termination_reason(), TerminationReason::Deadline);

    let guard = Instant::now();
    while !budget.expired() && guard.elapsed() < Duration::from_millis(250) {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(budget.expired(), "the request deadline did not stop finalization");
    assert_eq!(budget.termination_reason(), TerminationReason::Deadline);
}
