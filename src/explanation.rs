//! Immutable deferred propagation reasons and opt-in explanation diagnostics.
//!
//! Raw reasons contain domain predicates and historical provider-owned data.
//! Atom-numbered translations live only in the consuming LCG engine. Retained
//! byte counters are conservative payload estimates, not allocator measurements
//! or process RSS, and exclude generic eager reasons and engine side arrays.

use std::fmt;
use std::sync::{Arc, Mutex};

use crate::store::Premise;

/// Maximum historical entries in the initial deferred Boolean-linear family.
pub const MAX_DEFERRED_PREMISES: usize = 256;

/// Immutable historical explanation provider. It must never read a live Store.
///
/// Implementations reconstruct the complete conjunction from owned history.
/// `max_premises` bounds both the output length and historical template visits;
/// providers must not return a partial conjunction on cancellation.
pub trait DeferredReasonSource: Send + Sync {
    /// Reconstruct the complete historical antecedent conjunction.
    fn premises(&self) -> Vec<Premise>;
    /// Upper bound on premises and historical entries visited during expansion.
    fn max_premises(&self) -> usize;
    /// Retained provider graph payload upper bound in bytes, including any
    /// referenced templates and historical snapshots. Shared data may therefore
    /// be counted once per raw handle instead of once per unique allocation.
    fn estimated_bytes(&self) -> usize;
}

/// Opt-in explanation counters aggregated across worker clones.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExplanationStats {
    pub created_handles: u64,
    pub lcg_requests: u64,
    pub lcg_first_expansions: u64,
    pub lcg_cache_hits: u64,
    pub checker_expansions: u64,
    pub materialized_premises: u64,
    pub eager_premise_buffers: u64,
    pub eager_premises: u64,
    /// Eager primary or synchronisation antecedent buffers allocated by LCG.
    pub eager_lcg_buffers: u64,
    pub eager_lcg_literals: u64,
    /// Current per-handle raw graph upper bound; shared history can be counted
    /// repeatedly. This is not unique allocated storage.
    pub retained_raw_bytes: u64,
    /// Current shared fallback literal payload bytes, each fallback counted once.
    pub retained_fallback_bytes: u64,
    /// Current cached literal payload bytes, each cache counted once.
    pub retained_cache_bytes: u64,
    /// Peak of the tracked deferred payload upper bound. Excludes generic eager
    /// reasons, their premise vectors, allocator overhead and atom-reason arrays.
    pub peak_retained_bytes: u64,
}

/// Shared metrics retained after solver moves and aggregated across its clones.
#[derive(Clone, Default)]
pub struct ExplanationMetricsHandle {
    stats: Arc<Mutex<ExplanationStats>>,
}

impl ExplanationMetricsHandle {
    pub fn stats(&self) -> ExplanationStats {
        *self.stats.lock().expect("explanation metrics lock poisoned")
    }

    pub(crate) fn eager_premises(&self, count: usize) {
        let mut stats = self.stats.lock().expect("explanation metrics lock poisoned");
        stats.eager_premise_buffers = stats.eager_premise_buffers.saturating_add(1);
        stats.eager_premises = stats.eager_premises.saturating_add(count as u64);
    }

    pub(crate) fn eager_lcg(&self, count: usize) {
        let mut stats = self.stats.lock().expect("explanation metrics lock poisoned");
        stats.eager_lcg_buffers = stats.eager_lcg_buffers.saturating_add(1);
        stats.eager_lcg_literals = stats.eager_lcg_literals.saturating_add(count as u64);
    }

    pub(crate) fn request(&self, cached: bool) {
        let mut stats = self.stats.lock().expect("explanation metrics lock poisoned");
        stats.lcg_requests = stats.lcg_requests.saturating_add(1);
        if cached {
            stats.lcg_cache_hits = stats.lcg_cache_hits.saturating_add(1);
        }
    }

    pub(crate) fn retain(&self, kind: RetainedKind, bytes: usize) -> RetainedExplanation {
        let bytes = bytes as u64;
        let mut stats = self.stats.lock().expect("explanation metrics lock poisoned");
        let counter = match kind {
            RetainedKind::Raw => &mut stats.retained_raw_bytes,
            RetainedKind::Fallback => &mut stats.retained_fallback_bytes,
            RetainedKind::Cache => &mut stats.retained_cache_bytes,
        };
        *counter = counter.saturating_add(bytes);
        let retained = stats.retained_raw_bytes.saturating_add(stats.retained_fallback_bytes).saturating_add(stats.retained_cache_bytes);
        stats.peak_retained_bytes = stats.peak_retained_bytes.max(retained);
        RetainedExplanation { handle: self.clone(), kind, bytes }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum RetainedKind {
    Raw,
    Fallback,
    Cache,
}

/// Lifetime marker owned once by a raw handle, shared fallback, or local cache.
pub(crate) struct RetainedExplanation {
    handle: ExplanationMetricsHandle,
    kind: RetainedKind,
    bytes: u64,
}

impl Drop for RetainedExplanation {
    fn drop(&mut self) {
        let mut stats = self.handle.stats.lock().expect("explanation metrics lock poisoned");
        let counter = match self.kind {
            RetainedKind::Raw => &mut stats.retained_raw_bytes,
            RetainedKind::Fallback => &mut stats.retained_fallback_bytes,
            RetainedKind::Cache => &mut stats.retained_cache_bytes,
        };
        *counter = counter.saturating_sub(self.bytes);
    }
}

struct DeferredReasonInner {
    source: Arc<dyn DeferredReasonSource>,
    metrics: Option<ExplanationMetricsHandle>,
    _retained: Option<RetainedExplanation>,
}

/// Cloneable raw historical reason. No translated literals or expansion cache
/// are stored here, so solver clones cannot share atom-numbered reasons.
#[derive(Clone)]
pub struct DeferredReason(Arc<DeferredReasonInner>);

impl fmt::Debug for DeferredReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("DeferredReason").field("max_premises", &self.max_premises()).finish_non_exhaustive()
    }
}

impl DeferredReason {
    pub(crate) fn new(source: Arc<dyn DeferredReasonSource>, metrics: Option<ExplanationMetricsHandle>) -> Self {
        let retained = metrics.as_ref().map(|handle| {
            let mut stats = handle.stats.lock().expect("explanation metrics lock poisoned");
            stats.created_handles = stats.created_handles.saturating_add(1);
            drop(stats);
            handle.retain(RetainedKind::Raw, source.estimated_bytes().saturating_add(std::mem::size_of::<DeferredReasonInner>()))
        });
        Self(Arc::new(DeferredReasonInner { source, metrics, _retained: retained }))
    }

    pub fn max_premises(&self) -> usize {
        self.0.source.max_premises()
    }

    pub(crate) fn metrics(&self) -> Option<&ExplanationMetricsHandle> {
        self.0.metrics.as_ref()
    }

    pub(crate) fn expand_checker(&self) -> Vec<Premise> {
        self.expand(true)
    }

    pub(crate) fn expand_lcg(&self) -> Vec<Premise> {
        self.expand(false)
    }

    fn expand(&self, checker: bool) -> Vec<Premise> {
        let premises = self.0.source.premises();
        if let Some(handle) = &self.0.metrics {
            let mut stats = handle.stats.lock().expect("explanation metrics lock poisoned");
            if checker {
                stats.checker_expansions = stats.checker_expansions.saturating_add(1);
            } else {
                stats.lcg_first_expansions = stats.lcg_first_expansions.saturating_add(1);
            }
            stats.materialized_premises = stats.materialized_premises.saturating_add(premises.len() as u64);
        }
        premises
    }
}
