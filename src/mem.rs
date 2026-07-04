//! Process-wide live-allocation accounting and a memory-limit degrade policy.
//!
//! [`Tracking`] wraps the system allocator and keeps a running count of live
//! bytes (plus a peak high-water mark). It is installed as the `#[global_allocator]`
//! only in the `qayd` binary, so tests, the Python cdylib and other frontends keep
//! the untouched system allocator: there [`live`] and [`peak`] read zero and every
//! limit check no-ops (the limit defaults to 0 = unlimited).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
/// Hard memory budget in bytes; 0 means unlimited (the default everywhere the
/// tracking allocator is not installed).
static LIMIT: AtomicUsize = AtomicUsize::new(0);
/// Whether to emit `c mem ...` degrade notices (mirrors the CLI `-v` flag).
static VERBOSE: AtomicBool = AtomicBool::new(false);
static SOFT_NOTICED: AtomicBool = AtomicBool::new(false);

/// Allocator that forwards to [`System`] and accounts live bytes with Relaxed
/// ordering (the counter is advisory, never a correctness dependency).
pub struct Tracking;

unsafe impl GlobalAlloc for Tracking {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            record_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        LIVE.fetch_sub(layout.size(), Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let out = System.realloc(ptr, layout, new_size);
        if !out.is_null() {
            if new_size >= layout.size() {
                record_alloc(new_size - layout.size());
            } else {
                LIVE.fetch_sub(layout.size() - new_size, Relaxed);
            }
        }
        out
    }
}

fn record_alloc(bytes: usize) {
    let live = LIVE.fetch_add(bytes, Relaxed) + bytes;
    PEAK.fetch_max(live, Relaxed);
}

/// Current live (allocated minus freed) bytes.
pub fn live() -> usize {
    LIVE.load(Relaxed)
}

/// Peak live bytes since process start.
pub fn peak() -> usize {
    PEAK.load(Relaxed)
}

/// Configured hard budget in bytes (0 = unlimited).
pub fn limit() -> usize {
    LIMIT.load(Relaxed)
}

/// Install a hard budget of `mb` megabytes (0 = unlimited).
pub fn set_limit_mb(mb: usize) {
    LIMIT.store(mb.saturating_mul(1024 * 1024), Relaxed);
}

/// Enable degrade-notice printing.
pub fn set_verbose(verbose: bool) {
    VERBOSE.store(verbose, Relaxed);
}

/// Minimum learned-clause budget the soft limit will degrade to, so search stays
/// functional under memory pressure.
pub const LEARNED_FLOOR: usize = 2000;

/// Memory-pressure verdict for a live/limit pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pressure {
    /// Below the soft threshold, or no limit configured.
    None,
    /// At or above ~85%: degrade (reduce the learned-clause DB, halve the budget).
    Soft,
    /// At or above ~95%: stop search cleanly, keeping the incumbent.
    Hard,
}

/// Pure degrade decision: classify `live` bytes against a hard `limit`.
pub fn classify(live: usize, limit: usize) -> Pressure {
    if limit == 0 {
        return Pressure::None;
    }
    if live >= limit / 20 * 19 {
        Pressure::Hard
    } else if live >= limit / 20 * 17 {
        Pressure::Soft
    } else {
        Pressure::None
    }
}

/// Live-vs-configured-limit pressure right now.
pub fn pressure() -> Pressure {
    classify(live(), limit())
}

/// `true` once live memory has reached the hard (~95%) threshold.
pub fn over_hard() -> bool {
    pressure() == Pressure::Hard
}

/// Halve a learned-clause budget, never below [`LEARNED_FLOOR`]. Pure.
pub fn degrade_budget(current: usize) -> usize {
    (current / 2).max(LEARNED_FLOOR)
}

/// Emit the one-time soft-limit notice (verbose only). Goes to stderr because the
/// solver holds the stdout lock across the whole run, so a worker thread cannot
/// write the `w` stream without deadlocking.
pub fn note_soft() {
    if VERBOSE.load(Relaxed) && !SOFT_NOTICED.swap(true, Relaxed) {
        eprintln!("c mem soft limit reached at {} MB; halving learned-clause budget", live() / (1024 * 1024));
    }
}
