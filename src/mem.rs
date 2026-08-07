//! Process-wide live-allocation accounting and a memory-limit degrade policy.
//!
//! [`Tracking`] wraps the system allocator and keeps a running count of live
//! bytes (plus a peak high-water mark). It is installed as the `#[global_allocator]`
//! only in the `qayd` binary. Other frontends fall back to process resident
//! memory so the orchestrator can still enforce a hard limit.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static DEALLOCATED: AtomicUsize = AtomicUsize::new(0);
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
        #[cfg(test)]
        DEALLOCATED.fetch_add(layout.size(), Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let out = System.realloc(ptr, layout, new_size);
        if !out.is_null() {
            if new_size >= layout.size() {
                record_alloc(new_size - layout.size());
            } else {
                let released = layout.size() - new_size;
                LIVE.fetch_sub(released, Relaxed);
                #[cfg(test)]
                DEALLOCATED.fetch_add(released, Relaxed);
            }
        }
        out
    }
}

fn record_alloc(bytes: usize) {
    let live = LIVE.fetch_add(bytes, Relaxed) + bytes;
    PEAK.fetch_max(live, Relaxed);
    #[cfg(test)]
    ALLOCATED.fetch_add(bytes, Relaxed);
}

/// Current live (allocated minus freed) bytes.
pub fn live() -> usize {
    LIVE.load(Relaxed)
}

/// Peak live bytes since process start.
pub fn peak() -> usize {
    PEAK.load(Relaxed)
}

#[cfg(test)]
pub(crate) fn allocated_total() -> usize {
    ALLOCATED.load(Relaxed)
}

#[cfg(test)]
pub(crate) fn deallocated_total() -> usize {
    DEALLOCATED.load(Relaxed)
}

/// Memory usage used by the shared budget monitor. A process with the tracking
/// allocator uses its precise live heap counter. Other embedding surfaces use
/// resident set size instead.
pub fn monitored() -> Option<usize> {
    let tracked = live();
    if tracked > 0 {
        Some(tracked)
    } else {
        resident_set_size()
    }
}

/// Configured hard budget in bytes (0 = unlimited).
pub fn limit() -> usize {
    LIMIT.load(Relaxed)
}

/// Install a hard budget of `mb` megabytes (0 = unlimited).
pub fn set_limit_mb(mb: usize) {
    set_limit_bytes(mb.saturating_mul(1024 * 1024));
}

/// Install a hard budget in bytes (0 = unlimited).
pub fn set_limit_bytes(bytes: usize) {
    LIMIT.store(bytes, Relaxed);
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
    classify(monitored().unwrap_or(0), limit())
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
        eprintln!("c mem soft limit reached at {} MB; halving learned-clause budget", monitored().unwrap_or(0) / (1024 * 1024));
    }
}

#[cfg(target_os = "linux")]
fn resident_set_size() -> Option<usize> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages = statm.split_whitespace().nth(1)?.parse::<usize>().ok()?;
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    (page_size > 0).then(|| pages.saturating_mul(page_size as usize))
}

#[cfg(target_os = "macos")]
#[allow(deprecated)]
fn resident_set_size() -> Option<usize> {
    let mut info = unsafe { std::mem::zeroed::<libc::mach_task_basic_info_data_t>() };
    let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
    let status = unsafe {
        libc::task_info(
            libc::mach_task_self_,
            libc::MACH_TASK_BASIC_INFO as libc::task_flavor_t,
            (&mut info as *mut libc::mach_task_basic_info_data_t).cast::<libc::integer_t>(),
            &mut count,
        )
    };
    if status != libc::KERN_SUCCESS {
        return None;
    }
    let resident = unsafe { std::ptr::addr_of!(info.resident_size).read_unaligned() };
    usize::try_from(resident).ok()
}

#[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
fn resident_set_size() -> Option<usize> {
    let mut usage = unsafe { std::mem::zeroed::<libc::rusage>() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
        return None;
    }
    usize::try_from(usage.ru_maxrss).ok()?.checked_mul(1024)
}

#[cfg(not(unix))]
fn resident_set_size() -> Option<usize> {
    None
}
