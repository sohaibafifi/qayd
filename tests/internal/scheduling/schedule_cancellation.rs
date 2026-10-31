//! Coordinate a real cancelling thread with progress, rather than elapsed time.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::sync_channel;

use qayd::engines::ls::lists::audit_with_schedule_checkpoints;

pub(super) fn cancel_after_checkpoints<R>(stop: &AtomicBool, checkpoints: usize, work: impl FnOnce() -> R) -> R {
    assert!(checkpoints > 1, "cancellation must follow completed checkpoints");
    assert!(!stop.load(Ordering::Acquire));
    let (reached, wait_for_checkpoint) = sync_channel::<()>(0);
    let (cancelled, wait_for_cancellation) = sync_channel::<()>(0);

    std::thread::scope(|scope| {
        let canceller = scope.spawn(move || {
            if wait_for_checkpoint.recv().is_err() {
                return false;
            }
            stop.store(true, Ordering::Release);
            cancelled.send(()).expect("the checkpoint observer must await cancellation");
            true
        });
        let mut remaining = checkpoints;
        let result = audit_with_schedule_checkpoints(
            move || {
                if remaining == 0 {
                    return;
                }
                remaining -= 1;
                if remaining == 0 {
                    reached.send(()).expect("cancelling thread must be listening");
                    wait_for_cancellation.recv().expect("cancelling thread must set the stop flag");
                }
            },
            work,
        );
        // Dropping the observer also releases a canceller whose checkpoint was
        // never reached, so a missing polling point fails without hanging.
        assert!(canceller.join().unwrap(), "the operation finished before its cancellation checkpoint");
        assert!(stop.load(Ordering::Acquire));
        result
    })
}
