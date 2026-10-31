use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, Weak};
use std::time::{Duration, Instant};

const MIN_FINALIZATION_RESERVE: Duration = Duration::from_millis(10);
const MAX_FINALIZATION_RESERVE: Duration = Duration::from_millis(250);
const FINALIZATION_GRACE: Duration = Duration::from_millis(250);
const SEARCH_STOP_POLL: Duration = Duration::from_millis(2);
const MAX_WARM_START_TIME: Duration = Duration::from_secs(5);
const WARM_START_BUDGET_DIVISOR: u32 = 10;

/// One wall-clock and cancellation budget shared by compilation, verification,
/// fallbacks, and every engine in a solve plan.
#[derive(Clone)]
pub struct SolveBudget {
    inner: Arc<BudgetInner>,
}

struct BudgetInner {
    id: u64,
    started: Instant,
    limit: Option<Duration>,
    stop: Arc<AtomicBool>,
    hard_stop: Arc<AtomicBool>,
    non_memory_cancelled: AtomicBool,
    reason: Arc<AtomicU8>,
    timer: Arc<TimerSignal>,
    /// Resident memory already owned by an embedding process when live memory
    /// enforcement begins. The tracking allocator reports solver-live bytes
    /// directly and therefore keeps a zero baseline.
    memory_baseline: AtomicUsize,
    memory_limit: AtomicUsize,
    memory_monitor_started: AtomicBool,
}

/// Search-local cancellation flag that stops before the request deadline and
/// leaves a bounded slice of the same wall-clock budget for canonical replay.
/// External, memory, and event cancellation still propagate immediately.
pub(crate) struct SearchStop {
    stop: Arc<AtomicBool>,
    done: Option<Arc<TimerSignal>>,
    worker: Option<std::thread::JoinHandle<()>>,
}

/// Two cooperative boundaries inside one optional warm-start allowance. Search
/// stops first; canonical transfer verification may use only the reserved tail.
pub(crate) struct WarmStartStops {
    search: SearchStop,
    transfer: SearchStop,
}

impl WarmStartStops {
    pub(crate) fn search(&self) -> &AtomicBool {
        self.search.flag()
    }

    pub(crate) fn transfer(&self) -> &AtomicBool {
        self.transfer.flag()
    }
}

impl SearchStop {
    pub(crate) fn flag(&self) -> &AtomicBool {
        &self.stop
    }

    #[cfg(test)]
    pub(crate) fn flag_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.stop)
    }
}

impl Drop for SearchStop {
    fn drop(&mut self) {
        if let Some(done) = &self.done {
            let mut stopped = done.stopped.lock().unwrap();
            *stopped = true;
            done.wake.notify_all();
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

static NEXT_BUDGET_ID: AtomicU64 = AtomicU64::new(1);
static ACTIVE_MEMORY_LIMITS: LazyLock<Mutex<BTreeMap<u64, usize>>> = LazyLock::new(|| Mutex::new(BTreeMap::new()));

#[derive(Default)]
struct TimerSignal {
    stopped: Mutex<bool>,
    wake: Condvar,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TerminationReason {
    Running = 0,
    Deadline = 1,
    ExternalCancellation = 2,
    MemoryLimit = 3,
    ConflictLimit = 4,
    IterationLimit = 5,
    EventSink = 6,
    Engine = 7,
}

impl TerminationReason {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Deadline,
            2 => Self::ExternalCancellation,
            3 => Self::MemoryLimit,
            4 => Self::ConflictLimit,
            5 => Self::IterationLimit,
            6 => Self::EventSink,
            7 => Self::Engine,
            _ => Self::Running,
        }
    }
}

impl SolveBudget {
    pub fn new(limit: Option<Duration>) -> Self {
        Self::with_stop(limit, Arc::new(AtomicBool::new(false)))
    }

    pub fn with_stop(limit: Option<Duration>, stop: Arc<AtomicBool>) -> Self {
        let started = Instant::now();
        let immediately_expired = limit == Some(Duration::ZERO);
        let initially_stopped = stop.load(Ordering::Acquire);
        let hard_stop = Arc::new(AtomicBool::new(initially_stopped));
        let reason = Arc::new(AtomicU8::new(if immediately_expired {
            TerminationReason::Deadline as u8
        } else if initially_stopped {
            TerminationReason::ExternalCancellation as u8
        } else {
            TerminationReason::Running as u8
        }));
        if immediately_expired {
            stop.store(true, Ordering::Release);
        }
        let timer = Arc::new(TimerSignal::default());
        if let Some(duration) = limit.filter(|duration| !duration.is_zero()) {
            spawn_deadline(started, duration, Arc::downgrade(&stop), Arc::downgrade(&reason), Arc::clone(&timer));
        }
        Self {
            inner: Arc::new(BudgetInner {
                id: NEXT_BUDGET_ID.fetch_add(1, Ordering::Relaxed),
                started,
                limit,
                stop,
                hard_stop,
                non_memory_cancelled: AtomicBool::new(initially_stopped),
                reason,
                timer,
                memory_baseline: AtomicUsize::new(0),
                memory_limit: AtomicUsize::new(0),
                memory_monitor_started: AtomicBool::new(false),
            }),
        }
    }

    pub fn stop(&self) -> &AtomicBool {
        &self.inner.stop
    }

    pub fn stop_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.inner.stop)
    }

    pub fn cancel(&self) {
        self.cancel_with(TerminationReason::ExternalCancellation);
    }

    pub fn cancel_with(&self, reason: TerminationReason) {
        if reason == TerminationReason::Running {
            return;
        }
        if reason != TerminationReason::Deadline {
            if reason != TerminationReason::MemoryLimit {
                self.inner.non_memory_cancelled.store(true, Ordering::Release);
            }
            self.inner.hard_stop.store(true, Ordering::Release);
        }
        record_reason(&self.inner.reason, reason);
        self.inner.stop.store(true, Ordering::Release);
        if reason != TerminationReason::Deadline {
            self.wake_timer();
        }
        self.trace_phase("cancelled");
    }

    pub fn expired(&self) -> bool {
        self.inner.stop.load(Ordering::Acquire)
    }

    pub fn elapsed(&self) -> Duration {
        self.inner.started.elapsed()
    }

    pub fn remaining(&self) -> Option<Duration> {
        self.inner.limit.map(|limit| limit.saturating_sub(self.elapsed()))
    }

    pub fn termination_reason(&self) -> TerminationReason {
        let reason = TerminationReason::from_u8(self.inner.reason.load(Ordering::Acquire));
        if reason == TerminationReason::Running && self.expired() {
            TerminationReason::ExternalCancellation
        } else {
            reason
        }
    }

    pub(crate) fn hard_cancelled(&self) -> bool {
        self.inner.hard_stop.load(Ordering::Acquire)
    }

    /// A memory stop may retain evidence verified before the stop, but a later
    /// user, event-sink, or engine cancellation still forbids that promotion.
    pub(crate) fn memory_only_cancelled(&self) -> bool {
        self.hard_cancelled()
            && self.termination_reason() == TerminationReason::MemoryLimit
            && !self.inner.non_memory_cancelled.load(Ordering::Acquire)
    }

    /// Sparse diagnostic transitions, emitted immediately to the caller's
    /// stderr capture. Disabled by default and never polled from a search loop.
    pub(crate) fn trace_phase(&self, phase: &str) {
        static ENABLED: LazyLock<bool> = LazyLock::new(|| std::env::var_os("QAYD_SOLVE_TRACE").is_some_and(|value| value == "1"));
        if *ENABLED {
            eprintln!(
                "qayd_phase pid={} budget={} elapsed_us={} phase={} reason={:?}",
                std::process::id(),
                self.inner.id,
                self.elapsed().as_micros(),
                phase,
                self.termination_reason()
            );
        }
    }

    #[cfg(test)]
    pub(crate) fn audit_deadline_worker_start(already_elapsed: bool) -> bool {
        audit_deadline_worker_start(already_elapsed)
    }

    #[cfg(test)]
    pub(crate) fn audit_memory_registration_after_monitor_drop() -> bool {
        let budget = Self::new(None);
        let id = budget.inner.id;
        budget.inner.memory_limit.store(usize::MAX, Ordering::Release);
        update_process_memory_limit(id, usize::MAX);
        // The memory monitor temporarily owns this same strong reference
        // while sampling and waiting. Its final drop must own cleanup even
        // when the last public budget handle disappeared first.
        let monitor_reference = Arc::clone(&budget.inner);
        drop(budget);
        let retained_while_owned = ACTIVE_MEMORY_LIMITS.lock().unwrap().contains_key(&id);
        drop(monitor_reference);
        let released = !ACTIVE_MEMORY_LIMITS.lock().unwrap().contains_key(&id);
        // Keep the reproduction isolated even with the pre-fix destructor.
        update_process_memory_limit(id, 0);
        retained_while_owned && released
    }

    /// Derive an engine-search flag from this request budget. With a deadline,
    /// search stops slightly early so final semantic verification can finish
    /// without escaping the request budget. Without a deadline the original
    /// cancellation flag is shared directly.
    pub(crate) fn search_stop(&self) -> SearchStop {
        self.search_stop_inner(None)
    }

    /// Derive an engine-search flag while requesting a larger minimum tail for
    /// finalization. Requests longer than the minimum finalization reserve keep
    /// one cooperative search slice; hard cancellation and the parent deadline
    /// remain authoritative.
    pub(crate) fn search_stop_with_finalization_reserve(&self, reserve_floor: Duration) -> SearchStop {
        self.search_stop_inner(Some(reserve_floor))
    }

    fn search_stop_inner(&self, reserve_floor: Option<Duration>) -> SearchStop {
        let Some(remaining) = self.remaining() else {
            return SearchStop { stop: self.stop_handle(), done: None, worker: None };
        };
        let default_reserve = finalization_reserve(remaining);
        let reserve = reserve_floor.map_or(default_reserve, |floor| {
            let largest_reserve =
                if remaining <= MIN_FINALIZATION_RESERVE { remaining } else { remaining.saturating_sub(MIN_FINALIZATION_RESERVE) };
            default_reserve.max(floor.min(largest_reserve))
        });
        let search_duration = remaining.saturating_sub(reserve);
        let stop = Arc::new(AtomicBool::new(search_duration.is_zero() || self.expired()));
        if stop.load(Ordering::Acquire) {
            if !self.expired() {
                let _ = self.inner.reason.compare_exchange(
                    TerminationReason::Running as u8,
                    TerminationReason::Deadline as u8,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
            }
            return SearchStop { stop, done: None, worker: None };
        }

        let parent = self.stop_handle();
        let reason = Arc::clone(&self.inner.reason);
        let child = Arc::clone(&stop);
        let done = Arc::new(TimerSignal::default());
        let worker_done = Arc::clone(&done);
        let started = self.inner.started;
        let search_limit = self.inner.limit.expect("search deadline requires a request limit").saturating_sub(reserve);
        let worker = std::thread::spawn(move || {
            let mut finished = worker_done.stopped.lock().unwrap();
            loop {
                if *finished {
                    return;
                }
                if parent.load(Ordering::Acquire) {
                    child.store(true, Ordering::Release);
                    return;
                }
                if started.elapsed() >= search_limit {
                    let _ = reason.compare_exchange(
                        TerminationReason::Running as u8,
                        TerminationReason::Deadline as u8,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    );
                    child.store(true, Ordering::Release);
                    return;
                }
                let wait = SEARCH_STOP_POLL.min(search_limit.saturating_sub(started.elapsed()));
                let (next, _) = worker_done.wake.wait_timeout(finished, wait).unwrap();
                finished = next;
            }
        });
        SearchStop { stop, done: Some(done), worker: Some(worker) }
    }

    /// Derive a bounded, optional phase from this request budget. Expiring the
    /// child stops only that phase, while cancellation of the parent request
    /// still propagates immediately. This is intended for heuristic work that
    /// must never consume the whole exact-search budget.
    pub(crate) fn optional_phase_stop(&self, duration: Duration) -> SearchStop {
        let started = Instant::now();
        let duration = self.remaining().map_or(duration, |remaining| duration.min(remaining));
        let stop = Arc::new(AtomicBool::new(duration.is_zero() || self.expired()));
        if stop.load(Ordering::Acquire) {
            return SearchStop { stop, done: None, worker: None };
        }

        let parent = self.stop_handle();
        let child = Arc::clone(&stop);
        let done = Arc::new(TimerSignal::default());
        let worker_done = Arc::clone(&done);
        let worker = std::thread::spawn(move || {
            let mut finished = worker_done.stopped.lock().unwrap();
            loop {
                if *finished {
                    return;
                }
                if parent.load(Ordering::Acquire) || started.elapsed() >= duration {
                    child.store(true, Ordering::Release);
                    return;
                }
                let wait = SEARCH_STOP_POLL.min(duration.saturating_sub(started.elapsed()));
                let (next, _) = worker_done.wake.wait_timeout(finished, wait).unwrap();
                finished = next;
            }
        });
        SearchStop { stop, done: Some(done), worker: Some(worker) }
    }

    /// Reserve the bulk of the remaining request for the exact stage. Without
    /// a request deadline, a warm start still has a fixed upper bound.
    pub(crate) fn warm_start_stop(&self) -> Option<SearchStop> {
        let duration = self.warm_start_duration();
        (!duration.is_zero()).then(|| self.optional_phase_stop(duration))
    }

    /// Bound optional search and reserve a short tail of the same allowance for
    /// canonical repair and transfer verification.
    pub(crate) fn warm_start_stops(&self) -> Option<WarmStartStops> {
        self.optional_search_stops(self.warm_start_duration())
    }

    /// Improve an admitted additive-penalty incumbent, leaving at least two
    /// thirds of the remaining request for exact search and final replay.
    pub(crate) fn penalty_refinement_stops(&self) -> Option<WarmStartStops> {
        let duration = self.remaining().map_or(MAX_WARM_START_TIME, |remaining| (remaining / 3).min(Duration::from_secs(60)));
        self.optional_search_stops(duration)
    }

    fn optional_search_stops(&self, duration: Duration) -> Option<WarmStartStops> {
        if duration.is_zero() {
            return None;
        }
        let transfer_reserve = if duration <= MIN_FINALIZATION_RESERVE {
            duration
        } else {
            (duration / 4).clamp(MIN_FINALIZATION_RESERVE, MAX_FINALIZATION_RESERVE)
        };
        let search_duration = duration.saturating_sub(transfer_reserve);
        Some(WarmStartStops { search: self.optional_phase_stop(search_duration), transfer: self.optional_phase_stop(duration) })
    }

    /// A short, cooperative grace token for one final canonical replay after a
    /// soft wall-clock deadline. Hard cancellation remains live and takes
    /// priority over `Deadline`; the grace itself is capped independently of the
    /// requested search time.
    pub(crate) fn finalization_grace_stop(&self) -> SearchStop {
        let started = Instant::now();
        let stop = Arc::new(AtomicBool::new(self.hard_cancelled()));
        if stop.load(Ordering::Acquire) {
            return SearchStop { stop, done: None, worker: None };
        }
        let hard_stop = Arc::clone(&self.inner.hard_stop);
        let child = Arc::clone(&stop);
        let done = Arc::new(TimerSignal::default());
        let worker_done = Arc::clone(&done);
        let worker = std::thread::spawn(move || {
            let mut finished = worker_done.stopped.lock().unwrap();
            loop {
                if *finished {
                    return;
                }
                if hard_stop.load(Ordering::Acquire) || started.elapsed() >= FINALIZATION_GRACE {
                    child.store(true, Ordering::Release);
                    return;
                }
                let wait = SEARCH_STOP_POLL.min(FINALIZATION_GRACE.saturating_sub(started.elapsed()));
                let (next, _) = worker_done.wake.wait_timeout(finished, wait).unwrap();
                finished = next;
            }
        });
        SearchStop { stop, done: Some(done), worker: Some(worker) }
    }

    fn configure_memory_limit(&self, limit: Option<u64>) {
        let bytes = limit.map(|value| usize::try_from(value).unwrap_or(usize::MAX)).unwrap_or(0);
        let previous = self.inner.memory_limit.swap(bytes, Ordering::AcqRel);
        if previous == bytes {
            return;
        }
        let baseline = if bytes > 0 && crate::mem::live() == 0 { crate::mem::monitored().unwrap_or(0) } else { 0 };
        self.inner.memory_baseline.store(baseline, Ordering::Release);
        update_process_memory_limit(self.inner.id, process_memory_limit(baseline, bytes));
        if bytes > 0 {
            if monitored_solve_bytes(&self.inner).is_some_and(|usage| crate::mem::classify(usage, bytes) == crate::mem::Pressure::Hard) {
                self.cancel_with(TerminationReason::MemoryLimit);
                return;
            }
            if !self.inner.memory_monitor_started.swap(true, Ordering::AcqRel) && crate::mem::monitored().is_some() {
                spawn_memory_monitor(Arc::downgrade(&self.inner));
            }
        }
    }

    fn warm_start_duration(&self) -> Duration {
        match self.remaining() {
            Some(remaining) => (remaining / WARM_START_BUDGET_DIVISOR).min(MAX_WARM_START_TIME),
            None => MAX_WARM_START_TIME,
        }
    }

    fn wake_timer(&self) {
        let mut stopped = self.inner.timer.stopped.lock().unwrap();
        *stopped = true;
        self.inner.timer.wake.notify_all();
    }
}

fn finalization_reserve(remaining: Duration) -> Duration {
    if remaining <= MIN_FINALIZATION_RESERVE {
        return remaining;
    }
    (remaining / 20).clamp(MIN_FINALIZATION_RESERVE, MAX_FINALIZATION_RESERVE)
}

impl Drop for BudgetInner {
    fn drop(&mut self) {
        if self.memory_limit.load(Ordering::Acquire) > 0 {
            update_process_memory_limit(self.id, 0);
        }
        let mut stopped = self.timer.stopped.lock().unwrap();
        *stopped = true;
        self.timer.wake.notify_all();
    }
}

fn spawn_deadline(started: Instant, duration: Duration, stop: Weak<AtomicBool>, reason: Weak<AtomicU8>, signal: Arc<TimerSignal>) {
    std::thread::spawn(move || {
        if !wait_for_deadline(&signal, started, duration) {
            return;
        }
        let (Some(stop), Some(reason)) = (stop.upgrade(), reason.upgrade()) else {
            return;
        };
        let recorded = reason
            .compare_exchange(TerminationReason::Running as u8, TerminationReason::Deadline as u8, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
            || reason.load(Ordering::Acquire) == TerminationReason::Deadline as u8;
        if recorded {
            stop.store(true, Ordering::Release);
        }
    });
}

fn wait_for_deadline(signal: &TimerSignal, started: Instant, duration: Duration) -> bool {
    let mut stopped = signal.stopped.lock().unwrap();
    while !*stopped {
        let remaining = duration.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            return true;
        }
        let (next, _) = signal.wake.wait_timeout(stopped, remaining).unwrap();
        stopped = next;
    }
    false
}

#[cfg(test)]
fn audit_deadline_worker_start(already_elapsed: bool) -> bool {
    let signal = Arc::new(TimerSignal::default());
    let duration = Duration::from_secs(60);
    let armed = if already_elapsed {
        Instant::now().checked_sub(duration).expect("test deadline fits the monotonic clock")
    } else {
        Instant::now()
    };
    // A deferred worker is modelled by an already elapsed arm timestamp. The
    // watchdog cancels a wrongly restarted relative wait instead of hanging
    // the regression test for sixty seconds.
    let watchdog_signal = Arc::clone(&signal);
    let watchdog = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(50));
        *watchdog_signal.stopped.lock().unwrap() = true;
        watchdog_signal.wake.notify_all();
    });
    let expired = wait_for_deadline(&signal, armed, duration);
    watchdog.join().unwrap();
    expired
}

fn record_reason(reason: &AtomicU8, next: TerminationReason) {
    loop {
        let current = TerminationReason::from_u8(reason.load(Ordering::Acquire));
        let replace =
            current == TerminationReason::Running || (current == TerminationReason::Deadline && next != TerminationReason::Deadline);
        if !replace {
            return;
        }
        if reason.compare_exchange(current as u8, next as u8, Ordering::AcqRel, Ordering::Acquire).is_ok() {
            return;
        }
    }
}

pub(crate) fn apply_memory_limit(limit: Option<u64>, budget: &SolveBudget) {
    budget.configure_memory_limit(limit);
}

fn update_process_memory_limit(id: u64, bytes: usize) {
    let mut limits = ACTIVE_MEMORY_LIMITS.lock().unwrap();
    if bytes == 0 {
        limits.remove(&id);
    } else {
        limits.insert(id, bytes);
    }
    let effective = limits.values().copied().min().unwrap_or(0);
    crate::mem::set_limit_bytes(effective);
}

fn spawn_memory_monitor(inner: Weak<BudgetInner>) {
    std::thread::spawn(move || loop {
        let Some(inner) = inner.upgrade() else { return };
        if inner.hard_stop.load(Ordering::Acquire) {
            return;
        }
        let limit = inner.memory_limit.load(Ordering::Acquire);
        if limit == 0 {
            return;
        }
        if crate::mem::classify(monitored_solve_bytes(&inner).unwrap_or(0), limit) == crate::mem::Pressure::Hard {
            inner.hard_stop.store(true, Ordering::Release);
            record_reason(&inner.reason, TerminationReason::MemoryLimit);
            inner.stop.store(true, Ordering::Release);
            inner.timer.wake.notify_all();
            return;
        }
        let stopped = inner.timer.stopped.lock().unwrap();
        let (stopped, _) = inner.timer.wake.wait_timeout(stopped, Duration::from_millis(20)).unwrap();
        if *stopped {
            return;
        }
    });
}

fn monitored_solve_bytes(inner: &BudgetInner) -> Option<usize> {
    let baseline = inner.memory_baseline.load(Ordering::Acquire);
    crate::mem::monitored().map(|usage| usage.saturating_sub(baseline))
}

/// Translate an incremental embedded-process allowance into the absolute
/// threshold consumed by the process-wide allocator pressure hooks. The budget
/// monitor itself still classifies the exact baseline-relative usage.
fn process_memory_limit(baseline: usize, allowance: usize) -> usize {
    if allowance == 0 || baseline == 0 {
        return allowance;
    }
    let hard = baseline.saturating_add(allowance / 20 * 19);
    usize::try_from((hard as u128).saturating_mul(20).div_ceil(19)).unwrap_or(usize::MAX)
}
