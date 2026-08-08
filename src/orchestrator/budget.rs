use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, Weak};
use std::time::{Duration, Instant};

const MIN_FINALIZATION_RESERVE: Duration = Duration::from_millis(10);
const MAX_FINALIZATION_RESERVE: Duration = Duration::from_millis(250);
const FINALIZATION_GRACE: Duration = Duration::from_millis(250);
const SEARCH_STOP_POLL: Duration = Duration::from_millis(2);

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
            spawn_deadline(duration, Arc::downgrade(&stop), Arc::downgrade(&reason), Arc::clone(&timer));
        }
        Self {
            inner: Arc::new(BudgetInner {
                id: NEXT_BUDGET_ID.fetch_add(1, Ordering::Relaxed),
                started: Instant::now(),
                limit,
                stop,
                hard_stop,
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
            self.inner.hard_stop.store(true, Ordering::Release);
        }
        record_reason(&self.inner.reason, reason);
        self.inner.stop.store(true, Ordering::Release);
        if reason != TerminationReason::Deadline {
            self.wake_timer();
        }
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

    /// Derive an engine-search flag from this request budget. With a deadline,
    /// search stops slightly early so final semantic verification can finish
    /// without escaping the request budget. Without a deadline the original
    /// cancellation flag is shared directly.
    pub(crate) fn search_stop(&self) -> SearchStop {
        let Some(remaining) = self.remaining() else {
            return SearchStop { stop: self.stop_handle(), done: None, worker: None };
        };
        let reserve = finalization_reserve(remaining);
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
        let worker = std::thread::spawn(move || {
            let started = Instant::now();
            let mut finished = worker_done.stopped.lock().unwrap();
            loop {
                if *finished {
                    return;
                }
                if parent.load(Ordering::Acquire) {
                    child.store(true, Ordering::Release);
                    return;
                }
                if started.elapsed() >= search_duration {
                    let _ = reason.compare_exchange(
                        TerminationReason::Running as u8,
                        TerminationReason::Deadline as u8,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    );
                    child.store(true, Ordering::Release);
                    return;
                }
                let wait = SEARCH_STOP_POLL.min(search_duration.saturating_sub(started.elapsed()));
                let (next, _) = worker_done.wake.wait_timeout(finished, wait).unwrap();
                finished = next;
            }
        });
        SearchStop { stop, done: Some(done), worker: Some(worker) }
    }

    /// A short, cooperative grace token for one final canonical replay after a
    /// soft wall-clock deadline. Hard cancellation remains live and takes
    /// priority over `Deadline`; the grace itself is capped independently of the
    /// requested search time.
    pub(crate) fn finalization_grace_stop(&self) -> SearchStop {
        let stop = Arc::new(AtomicBool::new(self.hard_cancelled()));
        if stop.load(Ordering::Acquire) {
            return SearchStop { stop, done: None, worker: None };
        }
        let hard_stop = Arc::clone(&self.inner.hard_stop);
        let child = Arc::clone(&stop);
        let done = Arc::new(TimerSignal::default());
        let worker_done = Arc::clone(&done);
        let worker = std::thread::spawn(move || {
            let started = Instant::now();
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

impl Drop for SolveBudget {
    fn drop(&mut self) {
        if Arc::strong_count(&self.inner) == 1 {
            if self.inner.memory_limit.load(Ordering::Acquire) > 0 {
                update_process_memory_limit(self.inner.id, 0);
            }
            self.wake_timer();
        }
    }
}

fn spawn_deadline(duration: Duration, stop: Weak<AtomicBool>, reason: Weak<AtomicU8>, signal: Arc<TimerSignal>) {
    std::thread::spawn(move || {
        let stopped = signal.stopped.lock().unwrap();
        let (stopped, timeout) = signal.wake.wait_timeout_while(stopped, duration, |stopped| !*stopped).unwrap();
        if *stopped || !timeout.timed_out() {
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
