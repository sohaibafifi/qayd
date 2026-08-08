//! Finalization of exact decision-engine outcomes.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::search::SolveStats;

use super::{
    CandidateSolution, EngineKind, EngineReport, EventControl, EventSink, ProofClaim, ProvenConclusion, SolveBudget, SolveError,
    SolveEvent, SolveResult, SolveStatus, TerminationReason, VerificationLevel,
};

/// Synchronize a borrowed caller cancellation source with event publication.
/// The asynchronous budget monitor remains responsible for search and replay;
/// this adapter closes the sub-poll race around synchronous callbacks.
pub(crate) struct ExternalStopEventSink<'a> {
    external_stop: &'a AtomicBool,
    budget: &'a SolveBudget,
    target: &'a mut dyn EventSink,
    final_evidence_published: bool,
}

impl<'a> ExternalStopEventSink<'a> {
    pub(crate) fn new(external_stop: &'a AtomicBool, budget: &'a SolveBudget, target: &'a mut dyn EventSink) -> Self {
        Self { external_stop, budget, target, final_evidence_published: false }
    }

    pub(crate) fn final_evidence_published(&self) -> bool {
        self.final_evidence_published
    }
}

impl EventSink for ExternalStopEventSink<'_> {
    fn emit(&mut self, event: SolveEvent) -> Result<EventControl, SolveError> {
        if self.external_stop.load(Ordering::Acquire) {
            self.budget.cancel_with(TerminationReason::ExternalCancellation);
            return Ok(EventControl::Stop);
        }
        let is_final_evidence = matches!(&event, SolveEvent::Candidate(candidate) if candidate.verification() == VerificationLevel::Final)
            || matches!(&event, SolveEvent::Proof(_));
        let control = self.target.emit(event);
        if control.is_ok() && is_final_evidence {
            self.final_evidence_published = true;
        }
        if self.external_stop.load(Ordering::Acquire) {
            self.budget.cancel_with(TerminationReason::ExternalCancellation);
            return control.map(|_| EventControl::Stop);
        }
        control
    }
}

/// Raw outcome of an exact SAT/UNSAT decision engine.
///
/// Any candidate must already have been replayed by the engine adapter and be
/// safe for final publication. `complete` means that absence of a candidate is
/// a complete-search UNSAT result rather than interruption.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DecisionOutcome {
    engine: EngineKind,
    candidate: Option<CandidateSolution>,
    complete: bool,
    search: SolveStats,
    elapsed: Duration,
    metadata: Vec<(String, String)>,
}

impl DecisionOutcome {
    /// Build an outcome for a complete decision procedure. Construction is
    /// rejected for heuristic engines, so they cannot mint UNSAT proofs by
    /// merely setting a boolean completion flag.
    pub(crate) fn exact(
        engine: EngineKind,
        candidate: Option<CandidateSolution>,
        complete: bool,
        search: SolveStats,
        elapsed: Duration,
        metadata: Vec<(String, String)>,
    ) -> Result<Self, SolveError> {
        if !engine.can_prove_complete() {
            return Err(SolveError::InvalidResult(format!("engine {} is not authorized as an exact decision procedure", engine.name())));
        }
        Ok(Self { engine, candidate, complete, search, elapsed, metadata })
    }
}

/// Derive SAT/UNSAT/UNKNOWN semantics and publish the final decision events.
pub(crate) fn finalize_decision_result(outcome: DecisionOutcome, budget: &SolveBudget) -> Result<SolveResult, SolveError> {
    if !outcome.engine.can_prove_complete() {
        return Err(SolveError::InvalidResult(format!(
            "engine {} is not authorized as an exact decision procedure",
            outcome.engine.name()
        )));
    }
    if let Some(candidate) = &outcome.candidate {
        if candidate.verification() != VerificationLevel::Final {
            return Err(SolveError::InvalidResult("decision-engine candidate was not verified for final publication".to_string()));
        }
        if candidate.source() != outcome.engine {
            return Err(SolveError::InvalidResult("decision-engine candidate source does not match its report".to_string()));
        }
        if !candidate.objectives().is_empty() {
            return Err(SolveError::InvalidResult("decision-engine candidate unexpectedly carries an objective".to_string()));
        }
    }

    let status = if outcome.candidate.is_some() {
        SolveStatus::Satisfiable
    } else if outcome.complete {
        SolveStatus::Unsatisfiable
    } else {
        SolveStatus::Unknown
    };
    let proof =
        (status == SolveStatus::Unsatisfiable).then(|| ProofClaim::complete_search(outcome.engine, ProvenConclusion::Unsatisfiable, 0));
    let report = EngineReport {
        engine: Some(outcome.engine),
        search: outcome.search,
        elapsed: outcome.elapsed,
        improvements: u64::from(outcome.candidate.is_some()),
        metadata: outcome.metadata,
    };
    let message = (status == SolveStatus::Unknown).then(|| {
        if budget.expired() {
            format!("decision search stopped: {:?}", budget.termination_reason())
        } else {
            "decision search did not complete".to_string()
        }
    });
    let result = SolveResult { status, primal: outcome.candidate, bounds: Vec::new(), proof, reports: vec![report], message };
    result.validate_contract()?;
    Ok(result)
}

/// Publish final result events in protocol order. A consumer stop is terminal
/// for this sequence on every backend.
pub(crate) fn publish_result_events(result: &SolveResult, budget: &SolveBudget, sink: &mut dyn EventSink) -> Result<(), SolveError> {
    if budget.hard_cancelled() {
        return Ok(());
    }
    if let Some(candidate) = result.primal.clone() {
        if !emit(sink, budget, SolveEvent::Candidate(candidate))? {
            return Ok(());
        }
    }
    for bound in result.bounds.iter().cloned() {
        if !emit(sink, budget, SolveEvent::Bound(bound))? {
            return Ok(());
        }
    }
    if let Some(proof) = result.proof.clone() {
        if !emit(sink, budget, SolveEvent::Proof(proof))? {
            return Ok(());
        }
    }
    for report in result.reports.iter().cloned() {
        if !emit(sink, budget, SolveEvent::Finished(report))? {
            return Ok(());
        }
    }
    Ok(())
}

fn emit(sink: &mut dyn EventSink, budget: &SolveBudget, event: SolveEvent) -> Result<bool, SolveError> {
    if budget.hard_cancelled() {
        return Ok(false);
    }
    match sink.emit(event)? {
        EventControl::Continue => Ok(true),
        EventControl::Stop => {
            budget.cancel_with(TerminationReason::EventSink);
            Ok(false)
        }
    }
}
