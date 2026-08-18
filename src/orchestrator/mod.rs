//! Frontend-neutral solve control plane.
//!
//! Engines own algorithms. This module owns requests, budgets, executable plans,
//! cross-engine events, verification boundaries, and final result semantics.

mod budget;
mod collection;
mod cp;
mod diagnostics;
mod executor;
mod finalize;
mod integer_search;
mod plan;
mod sat;
mod session;
mod solve;
mod types;
mod verify;

pub(crate) use budget::WarmStartStops;
pub use budget::{SolveBudget, TerminationReason};
pub(crate) use collection::*;
pub(crate) use cp::*;
pub use diagnostics::{
    count_model_solutions_with_external_stop, enumerate_model_mus_with_external_stop, explain_model_mus_with_external_stop,
    extract_model_mus_with_external_stop, ModelMusAtom, ModelMusEnumeration, ModelMusExplanation, ModelMusResult, MusAtomRelation,
};
pub use executor::{
    execute_workers, execute_workers_silent, merge_search_stats, ExecutionSummary, SilentWorkerContext, WorkerContext, WorkerEvent,
    WorkerReport,
};
pub(crate) use executor::{worker_iteration_quota, WorkerAllocation};
pub(crate) use finalize::*;
pub use plan::{DecompositionMerge, EnginePlan, ExecutablePlan};
pub(crate) use sat::*;
pub use session::{SemanticNogood, SemanticNogoodLiteral, SemanticNogoodRelation, SemanticRawNogood, SemanticSolveSession};
#[cfg(test)]
pub(crate) use solve::{audit_interrupt_next_transfer_replay, execute_specialized, solve_model_with_budget};
pub use solve::{
    compile_model_plan, solve_model, solve_model_silent, solve_model_with_external_stop, solve_model_with_stop, PreparedSolvePlan,
};
pub use types::{
    Assignment, Bound, CandidateSolution, CpControls, EngineKind, EngineReport, EventCallback, EventControl, EventSink, IgnoreEvents,
    IntervalValue, LinearBackendMode, LinearControls, ModelFamily, ObjectiveSense, OptimalityGap, ProofClaim, ProofKind, ProofRequest,
    ProvenConclusion, RoutingControls, SatBackendMode, SatControls, SatPreprocess, SearchStats, SemanticAssumption, SemanticAssumptionOp,
    SolveError, SolveEvent, SolveLimits, SolveMode, SolveRequest, SolveResult, SolveStatus, VerificationLevel,
};
#[cfg(test)]
pub(crate) use verify::verify_final_supervised_with_budget;
pub use verify::{audit_semantic_verification_calls, verify_semantic_assignment};
pub(crate) use verify::{
    evaluate_int_expr_interruptible, verify_final_with_budget, verify_semantic_assignment_interruptible,
    verify_semantic_assignment_validated_interruptible,
};
