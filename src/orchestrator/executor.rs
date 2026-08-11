//! Reusable scoped worker execution.
//!
//! Engines provide worker inputs and a worker function. This module owns thread
//! lifecycle, external cancellation propagation, deterministic report ordering,
//! and a bounded progress path. Progress is deliberately `latest-wins`: at most
//! one unconsumed event is retained, while final worker reports are never
//! dropped.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use super::types::EventControl;
use crate::search::SolveStats;

/// Worker capacity assigned by a prepared orchestration node.
///
/// Engines may choose CP-specific or LS-specific cooperation, but they do not
/// infer their concurrency from the solve request. This value is the single
/// runtime source of truth for the number of trajectories to execute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WorkerAllocation(std::num::NonZeroUsize);

impl WorkerAllocation {
    pub(crate) const fn single() -> Self {
        Self(std::num::NonZeroUsize::MIN)
    }

    pub(crate) fn portfolio(workers: usize) -> Self {
        Self(std::num::NonZeroUsize::new(workers).expect("validated solve requests have at least one worker"))
    }

    pub(crate) const fn workers(self) -> usize {
        self.0.get()
    }
}

/// One coalesced progress event from a worker.
#[derive(Debug)]
pub struct WorkerEvent<E> {
    pub worker: usize,
    pub sequence: u64,
    pub payload: E,
}

/// Final result of one worker. Reports are returned in worker-index order.
#[derive(Debug)]
pub struct WorkerReport<R> {
    pub worker: usize,
    pub seed: u64,
    pub result: R,
}

/// Generic execution summary, independent of any engine result semantics.
#[derive(Debug)]
pub struct ExecutionSummary<R> {
    pub reports: Vec<WorkerReport<R>>,
    pub externally_cancelled: bool,
    pub stopped_by_handler: bool,
}

struct LatestState<E> {
    event: Option<WorkerEvent<E>>,
    notified: bool,
}

impl<E> Default for LatestState<E> {
    fn default() -> Self {
        Self { event: None, notified: false }
    }
}

enum ExecutionMessage<R> {
    Latest,
    Finished(WorkerReport<R>),
}

/// Context supplied to one engine worker.
pub struct WorkerContext<E, R> {
    worker: usize,
    seed: u64,
    cancel: Arc<AtomicBool>,
    sequence: Arc<AtomicU64>,
    latest: Arc<Mutex<LatestState<E>>>,
    tx: mpsc::Sender<ExecutionMessage<R>>,
}

/// Minimal context for workers that never publish progress. Keeping this path
/// free of the coalescing queue matters for tight CP portfolios, while worker
/// lifecycle and cancellation remain owned by the same executor module.
pub struct SilentWorkerContext {
    worker: usize,
    seed: u64,
    cancel: Arc<AtomicBool>,
}

impl SilentWorkerContext {
    pub fn worker(&self) -> usize {
        self.worker
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn stop(&self) -> &AtomicBool {
        &self.cancel
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }
}

impl<E, R> WorkerContext<E, R> {
    pub fn worker(&self) -> usize {
        self.worker
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn stop(&self) -> &AtomicBool {
        &self.cancel
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }

    /// Publish low-frequency progress without creating an unbounded queue.
    ///
    /// If producers outrun the consumer, this replaces the pending payload. The
    /// final report still travels through the lossless completion path.
    pub fn publish_latest(&self, payload: E) -> bool {
        let sequence = self.sequence.fetch_add(1, Ordering::Relaxed);
        let notify = {
            let mut latest = self.latest.lock().unwrap();
            latest.event = Some(WorkerEvent { worker: self.worker, sequence, payload });
            if latest.notified {
                false
            } else {
                latest.notified = true;
                true
            }
        };
        !notify || self.tx.send(ExecutionMessage::Latest).is_ok()
    }
}

/// Execute scoped workers with lossless final reports and coalesced progress.
///
/// `cancel` is shared with the engine so an exact proof, an engine failure, the
/// event consumer, or the external stop flag can terminate the same worker set.
pub fn execute_workers<W, E, R, Error, Run, Handle>(
    inputs: Vec<W>,
    external_stop: &AtomicBool,
    cancel: Arc<AtomicBool>,
    base_seed: u64,
    run: Run,
    mut handle: Handle,
) -> Result<ExecutionSummary<R>, Error>
where
    W: Send,
    E: Send,
    R: Send,
    Run: Fn(WorkerContext<E, R>, W) -> R + Sync,
    Handle: FnMut(WorkerEvent<E>) -> Result<EventControl, Error>,
{
    let worker_count = inputs.len();
    if worker_count == 0 {
        return Ok(ExecutionSummary {
            reports: Vec::new(),
            externally_cancelled: external_stop.load(Ordering::Acquire),
            stopped_by_handler: false,
        });
    }

    if external_stop.load(Ordering::Acquire) {
        cancel.store(true, Ordering::Release);
    }

    let latest = Arc::new(Mutex::new(LatestState::default()));
    let sequence = Arc::new(AtomicU64::new(0));
    let (tx, rx) = mpsc::channel();
    let mut reports = (0..worker_count).map(|_| None).collect::<Vec<Option<WorkerReport<R>>>>();
    let mut completed = 0usize;
    let mut stopped_by_handler = false;
    let mut handler_error = None;

    std::thread::scope(|scope| {
        let monitor_cancel = Arc::clone(&cancel);
        scope.spawn(move || {
            while !monitor_cancel.load(Ordering::Acquire) {
                if external_stop.load(Ordering::Acquire) {
                    monitor_cancel.store(true, Ordering::Release);
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        });

        let run = &run;
        for (worker, input) in inputs.into_iter().enumerate() {
            let worker_tx = tx.clone();
            let context = WorkerContext {
                worker,
                seed: base_seed.wrapping_add(worker as u64),
                cancel: Arc::clone(&cancel),
                sequence: Arc::clone(&sequence),
                latest: Arc::clone(&latest),
                tx: worker_tx.clone(),
            };
            scope.spawn(move || {
                let seed = context.seed();
                let result = run(context, input);
                let _ = worker_tx.send(ExecutionMessage::Finished(WorkerReport { worker, seed, result }));
            });
        }
        drop(tx);

        while completed < worker_count {
            let Ok(message) = rx.recv() else {
                break;
            };
            match message {
                ExecutionMessage::Latest => {
                    let event = {
                        let mut latest = latest.lock().unwrap();
                        let event = latest.event.take();
                        latest.notified = false;
                        event
                    };
                    if handler_error.is_none() && !stopped_by_handler {
                        if let Some(event) = event {
                            match handle(event) {
                                Ok(EventControl::Continue) => {}
                                Ok(EventControl::Stop) => {
                                    stopped_by_handler = true;
                                    cancel.store(true, Ordering::Release);
                                }
                                Err(error) => {
                                    handler_error = Some(error);
                                    cancel.store(true, Ordering::Release);
                                }
                            }
                        }
                    }
                }
                ExecutionMessage::Finished(report) => {
                    let worker = report.worker;
                    if reports[worker].replace(report).is_none() {
                        completed += 1;
                    }
                }
            }
        }
        cancel.store(true, Ordering::Release);
    });

    if let Some(error) = handler_error {
        return Err(error);
    }
    Ok(ExecutionSummary {
        reports: reports.into_iter().flatten().collect(),
        externally_cancelled: external_stop.load(Ordering::Acquire),
        stopped_by_handler,
    })
}

/// Execute workers that do not publish progress events.
pub fn execute_workers_silent<W, R, Run>(
    inputs: Vec<W>,
    external_stop: &AtomicBool,
    cancel: Arc<AtomicBool>,
    base_seed: u64,
    run: Run,
) -> ExecutionSummary<R>
where
    W: Send,
    R: Send,
    Run: Fn(SilentWorkerContext, W) -> R + Sync,
{
    let worker_count = inputs.len();
    if worker_count == 0 {
        return ExecutionSummary {
            reports: Vec::new(),
            externally_cancelled: external_stop.load(Ordering::Acquire),
            stopped_by_handler: false,
        };
    }
    if external_stop.load(Ordering::Acquire) {
        cancel.store(true, Ordering::Release);
    }

    let (tx, rx) = mpsc::channel();
    let mut reports = (0..worker_count).map(|_| None).collect::<Vec<Option<WorkerReport<R>>>>();
    std::thread::scope(|scope| {
        let monitor_cancel = Arc::clone(&cancel);
        scope.spawn(move || {
            while !monitor_cancel.load(Ordering::Acquire) {
                if external_stop.load(Ordering::Acquire) {
                    monitor_cancel.store(true, Ordering::Release);
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        });

        let run = &run;
        for (worker, input) in inputs.into_iter().enumerate() {
            let worker_tx = tx.clone();
            let context = SilentWorkerContext { worker, seed: base_seed.wrapping_add(worker as u64), cancel: Arc::clone(&cancel) };
            scope.spawn(move || {
                let seed = context.seed();
                let result = run(context, input);
                let _ = worker_tx.send(WorkerReport { worker, seed, result });
            });
        }
        drop(tx);
        for report in rx {
            let worker = report.worker;
            reports[worker] = Some(report);
        }
        cancel.store(true, Ordering::Release);
    });

    ExecutionSummary {
        reports: reports.into_iter().flatten().collect(),
        externally_cancelled: external_stop.load(Ordering::Acquire),
        stopped_by_handler: false,
    }
}

/// Merge all search counters from one worker into an aggregate report.
pub fn merge_search_stats(total: &mut SolveStats, part: SolveStats) {
    total.solutions = total.solutions.saturating_add(part.solutions);
    total.nodes = total.nodes.saturating_add(part.nodes);
    total.failures = total.failures.saturating_add(part.failures);
    total.learned_lits = total.learned_lits.saturating_add(part.learned_lits);
    total.vivified_clauses = total.vivified_clauses.saturating_add(part.vivified_clauses);
    total.vivified_lits = total.vivified_lits.saturating_add(part.vivified_lits);
    total.binary_clause_visits = total.binary_clause_visits.saturating_add(part.binary_clause_visits);
    total.watched_clause_visits = total.watched_clause_visits.saturating_add(part.watched_clause_visits);
    total.watched_literal_scans = total.watched_literal_scans.saturating_add(part.watched_literal_scans);
    total.binary_implications = total.binary_implications.saturating_add(part.binary_implications);
}
