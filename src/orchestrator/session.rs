//! Persistent exact sessions for semantic integer models.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::lcg::clause::SharedClausePool;
use crate::lcg::lit::{AtomKind, AtomTable, Lit};
use crate::model::{CompiledCp, Constraint, IntDomain, IntExpr, Model, ModelPackage, Objective};
use crate::search::{Assumption, AssumptionOp};

use super::{
    solve_physical_exact_with_budget, CandidateSolution, EventControl, EventSink, PhysicalObjectiveTier, PhysicalSolveInput, ProofRequest,
    SemanticAssumptionOp, SolveBudget, SolveError, SolveEvent, SolveMode, SolveRequest, SolveResult, SolveStatus, TerminationReason,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticNogoodRelation {
    Eq,
    Ne,
    Ge,
    Lt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemanticNogoodLiteral {
    pub variable: usize,
    pub relation: SemanticNogoodRelation,
    pub value: i32,
}

pub type SemanticRawNogood = (u32, Vec<u32>);
pub type SemanticNogood = (u32, Vec<SemanticNogoodLiteral>);

pub struct SemanticSolveSession {
    package: ModelPackage,
    compiled: Option<CompiledCp>,
    clauses: Arc<SharedClausePool>,
    next_worker: usize,
}

impl SemanticSolveSession {
    pub fn new(package: ModelPackage) -> Result<Self, SolveError> {
        Ok(Self { package, compiled: None, clauses: Arc::new(SharedClausePool::default()), next_worker: 0 })
    }

    pub fn learned_nogoods(&self) -> usize {
        self.clauses.len()
    }

    pub fn clear_nogoods(&mut self) {
        self.clauses = Arc::new(SharedClausePool::default());
        self.next_worker = 0;
    }

    pub fn raw_nogoods(&self, limit: Option<usize>) -> Vec<SemanticRawNogood> {
        self.clauses
            .snapshot(limit.unwrap_or(0))
            .into_iter()
            .map(|(lbd, literals)| (lbd, literals.iter().map(|literal| literal.code()).collect()))
            .collect()
    }

    pub fn nogoods(&self, limit: Option<usize>) -> Result<Vec<SemanticNogood>, SolveError> {
        let clauses = self.clauses.snapshot(limit.unwrap_or(0));
        if clauses.is_empty() {
            return Ok(Vec::new());
        }
        let compiled = self
            .compiled
            .as_ref()
            .ok_or_else(|| SolveError::InvalidResult("an uncompiled session contains learned nogoods".to_string()))?;
        let atoms = Self::atom_table(compiled, &self.clauses);
        clauses
            .into_iter()
            .map(|(lbd, literals)| {
                let literals =
                    literals.iter().map(|literal| Self::decode_literal(compiled, &atoms, *literal)).collect::<Result<Vec<_>, _>>()?;
                Ok((lbd, literals))
            })
            .collect()
    }

    pub fn solve_with_external_stop(
        &mut self,
        request: &SolveRequest,
        external_stop: &AtomicBool,
        sink: &mut dyn EventSink,
    ) -> Result<SolveResult, SolveError> {
        request.validate()?;
        let budget = SolveBudget::new(request.limits.time);
        if external_stop.load(Ordering::Acquire) {
            budget.cancel_with(TerminationReason::ExternalCancellation);
        }
        let done = AtomicBool::new(false);
        let result = std::thread::scope(|scope| {
            scope.spawn(|| {
                while !done.load(Ordering::Acquire) {
                    if external_stop.load(Ordering::Acquire) {
                        budget.cancel_with(TerminationReason::ExternalCancellation);
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
            });
            let (mut result, final_evidence_published) = {
                let mut monitored_sink = super::ExternalStopEventSink::new(external_stop, &budget, sink);
                let result = self.solve_with_budget(request, &budget, &mut monitored_sink);
                (result, monitored_sink.final_evidence_published())
            };
            if external_stop.load(Ordering::Acquire) && !final_evidence_published {
                budget.cancel_with(TerminationReason::ExternalCancellation);
                result = Err(SolveError::Interrupted("session external cancellation won finalization".to_string()));
            }
            done.store(true, Ordering::Release);
            result
        });
        let result = match result {
            Err(SolveError::Interrupted(_)) => Ok(session_stopped_result(&budget)),
            result => result,
        }?;
        if request.proof == ProofRequest::Require && result.proof.is_none() {
            return Err(SolveError::InvalidResult("a required session proof was not produced".to_string()));
        }
        Ok(result)
    }

    fn solve_with_budget(
        &mut self,
        request: &SolveRequest,
        budget: &SolveBudget,
        sink: &mut dyn EventSink,
    ) -> Result<SolveResult, SolveError> {
        if request.mode == SolveMode::LocalSearch {
            return Err(SolveError::InvalidRequest("semantic solve sessions support exact mode only".to_string()));
        }
        if request.threads != 1 {
            return Err(SolveError::InvalidRequest("semantic solve sessions currently use one exact worker".to_string()));
        }
        if request.list_hint.is_some()
            || request.schedule_cdcl
            || request.routing != super::RoutingControls::default()
            || request.sat != super::SatControls::default()
            || request.cp != super::CpControls::default()
            || request.limits.iterations.is_some()
        {
            return Err(SolveError::InvalidRequest(
                "semantic solve sessions do not accept list, scheduling, routing, SAT, portfolio, or iteration controls".to_string(),
            ));
        }
        if budget.expired() {
            return Ok(session_stopped_result(budget));
        }
        self.prepare(request, budget)?;
        let compiled = self.compiled.as_ref().expect("successful session preparation installs a compiled plan");
        let Some((_, branch_order)) = compiled
            .search_guidance_interruptible(&request.hints, &request.branch_order, budget.stop())
            .map_err(|error| SolveError::InvalidRequest(error.reason))?
        else {
            return Err(SolveError::Interrupted("session search guidance construction was interrupted".to_string()));
        };
        let mut assumptions = Vec::with_capacity(request.assumptions.len());
        for assumption in &request.assumptions {
            if budget.expired() {
                return Err(SolveError::Interrupted("session assumption mapping was interrupted".to_string()));
            }
            let variable = compiled.int_variables().get(assumption.variable).copied().ok_or_else(|| {
                SolveError::InvalidRequest(format!("assumption references unknown integer variable {}", assumption.variable))
            })?;
            assumptions.push(Assumption {
                var: variable,
                op: match assumption.operation {
                    SemanticAssumptionOp::Eq => AssumptionOp::Eq,
                    SemanticAssumptionOp::Ne => AssumptionOp::Ne,
                    SemanticAssumptionOp::Le => AssumptionOp::Le,
                    SemanticAssumptionOp::Lt => AssumptionOp::Lt,
                    SemanticAssumptionOp::Ge => AssumptionOp::Ge,
                    SemanticAssumptionOp::Gt => AssumptionOp::Gt,
                },
                value: assumption.value,
            });
        }
        let mut hints = Vec::with_capacity(request.hints.len());
        for &(index, value) in &request.hints {
            if budget.expired() {
                return Err(SolveError::Interrupted("session hint mapping was interrupted".to_string()));
            }
            let variable = compiled
                .int_variables()
                .get(index)
                .copied()
                .ok_or_else(|| SolveError::InvalidRequest(format!("hint references unknown integer variable {index}")))?;
            hints.push((variable, value));
        }
        let mut objectives = Vec::with_capacity(compiled.objectives().len());
        for objective in compiled.objectives() {
            if budget.expired() {
                return Err(SolveError::Interrupted("session objective mapping was interrupted".to_string()));
            }
            objectives.push(PhysicalObjectiveTier { objective: objective.clone() });
        }
        let input = PhysicalSolveInput {
            problem: compiled.problem().clone(),
            visible_variables: self.package.model.int_vars().len(),
            objectives,
            assumptions,
            hints,
            branch_order,
            shared_clauses: Some(Arc::clone(&self.clauses)),
            first_worker: self.next_worker,
        };
        let mut replay = ReplayEvents { model: &self.package.model, budget, target: sink };
        let output = solve_physical_exact_with_budget(input, request, budget, &mut replay)?;
        self.next_worker = output.next_worker;
        let mut result = output.result;
        result.primal = result.primal.map(|candidate| replay_candidate(&self.package.model, budget, &candidate)).transpose()?;
        if request.proof == ProofRequest::Require && result.proof.is_none() {
            return Err(SolveError::InvalidResult("a required session proof was not produced".to_string()));
        }
        result.validate_model_contract(&self.package.model)?;
        super::publish_result_events(&result, budget, replay.target)?;
        Ok(result)
    }

    fn prepare(&mut self, request: &SolveRequest, budget: &SolveBudget) -> Result<(), SolveError> {
        if let Some(compiled) = &self.compiled {
            Self::preflight_memory(compiled.estimated_bytes(), request)?;
            super::budget::apply_memory_limit(request.limits.memory_bytes, budget);
            if budget.expired() {
                return Err(SolveError::Interrupted("session solve budget expired during request setup".to_string()));
            }
            return Ok(());
        }
        if !self.package.validate_interruptible(budget.stop()).map_err(|errors| SolveError::Compile(errors.join("; ")))? {
            return Err(SolveError::Interrupted("session semantic validation was interrupted".to_string()));
        }
        if !self.package.model.sets().is_empty() || !self.package.model.lists().is_empty() || !self.package.model.intervals().is_empty() {
            return Err(SolveError::Unsupported("semantic solve sessions currently accept integer models only".to_string()));
        }
        if budget.expired() {
            return Err(SolveError::Interrupted("session model preparation was interrupted".to_string()));
        }
        let semantic_estimate = CompiledCp::estimate_semantic_bytes_interruptible(&self.package.model, budget.stop())
            .ok_or_else(|| SolveError::Interrupted("session preliminary CP memory estimation was interrupted".to_string()))?;
        Self::preflight_memory(semantic_estimate, request)?;

        let mut prepared = self.package.model.clone();
        if budget.expired() {
            return Err(SolveError::Interrupted("session model preparation was interrupted".to_string()));
        }
        if !materialize_objectives_interruptible(&mut prepared, budget.stop())? {
            return Err(SolveError::Interrupted("session objective materialization was interrupted".to_string()));
        }
        let estimated_bytes = CompiledCp::estimate_semantic_bytes_interruptible(&prepared, budget.stop())
            .ok_or_else(|| SolveError::Interrupted("session CP memory estimation was interrupted".to_string()))?;
        Self::preflight_memory(estimated_bytes, request)?;
        super::budget::apply_memory_limit(request.limits.memory_bytes, budget);
        if budget.expired() {
            return Err(SolveError::Interrupted("session solve budget expired after CP memory preflight".to_string()));
        }
        let compiled = CompiledCp::compile_with_estimate_interruptible(&prepared, estimated_bytes, budget.stop())
            .map_err(|error| SolveError::Compile(error.reason))?
            .ok_or_else(|| SolveError::Interrupted("session CP compilation was interrupted".to_string()))?;

        self.package.model = prepared;
        self.compiled = Some(compiled);
        Ok(())
    }

    fn preflight_memory(estimated_bytes: u64, request: &SolveRequest) -> Result<(), SolveError> {
        if request.limits.memory_bytes.is_some_and(|limit| estimated_bytes > limit) {
            return Err(SolveError::Compile(format!(
                "estimated session CP backend requires {estimated_bytes} bytes, above the memory limit"
            )));
        }
        Ok(())
    }

    fn atom_table(compiled: &CompiledCp, clauses: &SharedClausePool) -> AtomTable {
        let solver = &compiled.problem().solver;
        let variables = compiled.problem().search.as_slice();
        let count = solver.store.num_vars();
        let mut active = (0..count).map(|index| solver.store.is_relevant(crate::ids::VarId(index as u32))).collect::<Vec<_>>();
        for variable in variables {
            active[variable.index()] = true;
        }
        AtomTable::build_active_sparse_with_registry(
            count,
            |variable| active[variable.index()],
            |variable| solver.store.size(variable) == 2 && solver.store.contains(variable, -1) && solver.store.contains(variable, 1),
            |variable| solver.store.sparse_values(variable),
            |variable| (solver.store.min(variable), solver.store.max(variable)),
            clauses.lazy_atoms(),
        )
    }

    fn decode_literal(compiled: &CompiledCp, atoms: &AtomTable, literal: Lit) -> Result<SemanticNogoodLiteral, SolveError> {
        let (variable, relation, value) = match atoms.decode(literal.atom()) {
            AtomKind::Ge { var, k } if literal.is_positive() => (var, SemanticNogoodRelation::Ge, k),
            AtomKind::Ge { var, k } => (var, SemanticNogoodRelation::Lt, k),
            AtomKind::Eq { var, v } if literal.is_positive() => (var, SemanticNogoodRelation::Eq, v),
            AtomKind::Eq { var, v } => (var, SemanticNogoodRelation::Ne, v),
        };
        let variable = compiled
            .int_variables()
            .iter()
            .position(|candidate| *candidate == variable)
            .ok_or_else(|| SolveError::InvalidResult("session nogood references a non-semantic variable".to_string()))?;
        Ok(SemanticNogoodLiteral { variable, relation, value })
    }
}

fn materialize_objectives_interruptible(model: &mut Model, stop: &AtomicBool) -> Result<bool, SolveError> {
    let objectives = std::mem::take(&mut model.objectives);
    for objective in objectives {
        if stop.load(Ordering::Acquire) {
            return Ok(false);
        }
        let Objective::IntExpr { minimize, expr } = objective else {
            return Err(SolveError::Unsupported("semantic integer session received a non-integer objective".to_string()));
        };
        let variable = match &expr {
            IntExpr::Variable(variable) => *variable,
            _ => {
                let Some((lo, hi)) = expression_bounds_interruptible(model, &expr, stop) else {
                    return Ok(false);
                };
                let lo = i32::try_from(lo).map_err(|_| SolveError::Compile("session objective lower bound is outside i32".to_string()))?;
                let hi = i32::try_from(hi).map_err(|_| SolveError::Compile("session objective upper bound is outside i32".to_string()))?;
                if stop.load(Ordering::Acquire) {
                    return Ok(false);
                }
                let variable = model.int_range(lo, hi);
                model.add_constraint(Constraint::Intension(IntExpr::Eq(Box::new(IntExpr::Variable(variable)), Box::new(expr))));
                variable
            }
        };
        model.add_objective(Objective::IntExpr { minimize, expr: IntExpr::Variable(variable) });
    }
    Ok(!stop.load(Ordering::Acquire))
}

fn domain_bounds(domain: &IntDomain) -> (i64, i64) {
    match domain {
        IntDomain::Bool => (0, 1),
        IntDomain::Range { lo, hi } => (i64::from(*lo), i64::from(*hi)),
        IntDomain::Set(values) => (
            i64::from(*values.iter().min().expect("validated integer domain is non-empty")),
            i64::from(*values.iter().max().expect("validated integer domain is non-empty")),
        ),
    }
}

fn expression_bounds_interruptible(model: &Model, expression: &IntExpr, stop: &AtomicBool) -> Option<(i64, i64)> {
    if stop.load(Ordering::Acquire) {
        return None;
    }
    Some(match expression {
        IntExpr::Constant(value) => (*value, *value),
        IntExpr::Variable(variable) => domain_bounds(&model.int_vars()[variable.0]),
        IntExpr::Neg(value) => {
            let (lo, hi) = expression_bounds_interruptible(model, value, stop)?;
            (hi.saturating_neg(), lo.saturating_neg())
        }
        IntExpr::Abs(value) => {
            let (lo, hi) = expression_bounds_interruptible(model, value, stop)?;
            (0, lo.saturating_abs().max(hi.saturating_abs()))
        }
        IntExpr::Add(values) => {
            let mut bounds = (0i64, 0i64);
            for value in values {
                let (value_lo, value_hi) = expression_bounds_interruptible(model, value, stop)?;
                bounds = (bounds.0.saturating_add(value_lo), bounds.1.saturating_add(value_hi));
            }
            bounds
        }
        IntExpr::Sub(left, right) => {
            let (left_lo, left_hi) = expression_bounds_interruptible(model, left, stop)?;
            let (right_lo, right_hi) = expression_bounds_interruptible(model, right, stop)?;
            (left_lo.saturating_sub(right_hi), left_hi.saturating_sub(right_lo))
        }
        IntExpr::Mul(values) => {
            let mut bounds = (1i64, 1i64);
            for value in values {
                let (right_lo, right_hi) = expression_bounds_interruptible(model, value, stop)?;
                let products = [
                    bounds.0.saturating_mul(right_lo),
                    bounds.0.saturating_mul(right_hi),
                    bounds.1.saturating_mul(right_lo),
                    bounds.1.saturating_mul(right_hi),
                ];
                bounds = (*products.iter().min().unwrap(), *products.iter().max().unwrap());
            }
            bounds
        }
        IntExpr::Min(values) => {
            let mut bounds = (i64::MIN, i64::MAX);
            let mut first = true;
            for value in values {
                let value = expression_bounds_interruptible(model, value, stop)?;
                bounds = if first { value } else { (bounds.0.min(value.0), bounds.1.min(value.1)) };
                first = false;
            }
            bounds
        }
        IntExpr::Max(values) => {
            let mut bounds = (i64::MIN, i64::MAX);
            let mut first = true;
            for value in values {
                let value = expression_bounds_interruptible(model, value, stop)?;
                bounds = if first { value } else { (bounds.0.max(value.0), bounds.1.max(value.1)) };
                first = false;
            }
            bounds
        }
        IntExpr::IfThenElse(_, then_value, else_value) => {
            let then_bounds = expression_bounds_interruptible(model, then_value, stop)?;
            let else_bounds = expression_bounds_interruptible(model, else_value, stop)?;
            (then_bounds.0.min(else_bounds.0), then_bounds.1.max(else_bounds.1))
        }
        IntExpr::Div(_, _) | IntExpr::Mod(_, _) => (i64::from(i32::MIN), i64::from(i32::MAX)),
        IntExpr::Eq(_, _)
        | IntExpr::Ne(_, _)
        | IntExpr::Lt(_, _)
        | IntExpr::Le(_, _)
        | IntExpr::Gt(_, _)
        | IntExpr::Ge(_, _)
        | IntExpr::Not(_)
        | IntExpr::And(_)
        | IntExpr::Or(_)
        | IntExpr::Imp(_, _)
        | IntExpr::Iff(_, _) => (0, 1),
    })
}

struct ReplayEvents<'a> {
    model: &'a crate::model::Model,
    budget: &'a SolveBudget,
    target: &'a mut dyn EventSink,
}

impl EventSink for ReplayEvents<'_> {
    fn emit(&mut self, event: SolveEvent) -> Result<EventControl, SolveError> {
        let event = match event {
            SolveEvent::Candidate(candidate) => SolveEvent::Candidate(replay_candidate(self.model, self.budget, &candidate)?),
            other => other,
        };
        self.target.emit(event)
    }
}

fn replay_candidate(
    model: &crate::model::Model,
    budget: &SolveBudget,
    candidate: &CandidateSolution,
) -> Result<CandidateSolution, SolveError> {
    if candidate.verification() == super::VerificationLevel::Final {
        super::verify_final_with_budget(budget, |stop| replay_candidate_once(model, candidate, stop))
    } else {
        replay_candidate_once(model, candidate, budget.stop())
    }
}

fn replay_candidate_once(
    model: &crate::model::Model,
    candidate: &CandidateSolution,
    stop: &AtomicBool,
) -> Result<CandidateSolution, SolveError> {
    let objectives =
        super::verify_semantic_assignment_validated_interruptible(model, candidate.assignment(), candidate.objectives(), stop)?;
    let assignment = super::clone_assignment_interruptible(candidate.assignment(), stop)?;
    Ok(CandidateSolution::verified(assignment, objectives, candidate.source(), candidate.verification()))
}

fn session_stopped_result(budget: &SolveBudget) -> SolveResult {
    SolveResult {
        status: SolveStatus::Unknown,
        primal: None,
        bounds: Vec::new(),
        proof: None,
        reports: Vec::new(),
        message: Some(format!("session stopped during preparation or execution: {:?}", budget.termination_reason())),
    }
}
