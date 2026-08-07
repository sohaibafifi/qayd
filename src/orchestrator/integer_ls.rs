//! Semantic integer local-search compilation and execution.
//!
//! The frontend-neutral model is lowered once to the same physical CP root as
//! exact search. The local-search scorer is a specialized view of that root;
//! every assignment crossing the worker boundary is repaired on the CP root
//! and replayed against the semantic model before publication.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::constraints::linear::Relation as PhysicalRelation;
use crate::constraints::table::{Dfa, Mdd, MddArc};
use crate::engines::ls::cop::{solve_ls_capped, LocalRhs, LocalSearchOutcome, LocalSearchSpec, LsConfig};
use crate::expr::Expr;
use crate::ids::VarId;
use crate::model::{BoolLiteral, CompiledCp, Constraint, IntGlobalConstraint, Model, Relation, SetVarRef};

use super::{
    execute_workers, CandidateSolution, EngineKind, EngineReport, EventControl, EventSink, SolveBudget, SolveError, SolveEvent,
    SolveRequest, SolveResult, SolveStatus, TerminationReason, VerificationLevel,
};

#[derive(Clone)]
pub(crate) struct IntegerLocalSearchPlan {
    spec: LocalSearchSpec,
}

#[derive(Clone)]
struct Improvement {
    objective: i64,
    assignment: Vec<i32>,
}

const INTERNAL_REPLAY_INTERVAL: Duration = Duration::from_millis(25);

struct RepairContext<'a> {
    model: &'a Model,
    compiled: &'a CompiledCp,
    spec: &'a LocalSearchSpec,
    request: &'a SolveRequest,
    budget: &'a SolveBudget,
}

impl RepairContext<'_> {
    fn candidate(&self, values: &[i32], seed: u64, verification: VerificationLevel) -> Result<Option<CandidateSolution>, SolveError> {
        repair_candidate(self, values, seed, verification)
    }
}

pub(crate) fn compile(model: &Model, compiled: &CompiledCp) -> Result<IntegerLocalSearchPlan, SolveError> {
    if compiled.objectives().len() > 1 {
        return Err(SolveError::InvalidRequest("integer local search currently supports at most one objective tier".to_string()));
    }

    let mut spec = LocalSearchSpec::default();
    for &variable in compiled.int_variables() {
        spec.add_var(variable);
    }
    for set in compiled.sets() {
        for &membership in &set.membership {
            spec.add_var(membership);
        }
    }
    for constraint in model.constraints() {
        compile_constraint(&mut spec, model, compiled, constraint)?;
    }
    if spec.unsupported() > 0 {
        return Err(SolveError::Unsupported(format!(
            "integer local-search compilation rejected {} unsupported model construct(s)",
            spec.unsupported()
        )));
    }
    Ok(IntegerLocalSearchPlan { spec })
}

pub(crate) fn solve(
    model: &Model,
    compiled: &CompiledCp,
    plan: &IntegerLocalSearchPlan,
    request: &SolveRequest,
    budget: &SolveBudget,
    engine_stop: &AtomicBool,
    sink: &mut dyn EventSink,
) -> Result<SolveResult, SolveError> {
    let started = Instant::now();
    let config = LsConfig { gls: true, min_conflicts: true, kick_bandit: false };
    let mut problem = compiled.problem().clone();
    if problem.objective.is_none() {
        problem.objective = Some(crate::problem::Objective::Expr(true, Expr::Const(0)));
    }
    let max_iterations = request.limits.iterations.unwrap_or(u64::MAX);
    let inputs = (0..request.threads)
        .map(|worker| (problem.clone(), plan.spec.clone(), worker_iteration_quota(max_iterations, worker, request.threads)))
        .collect();
    let publish_assignments = request.publish_incumbent_assignments;
    let repair = RepairContext { model, compiled, spec: &plan.spec, request, budget };
    let minimizing = model.objectives().first().is_none_or(crate::model::Objective::is_minimize);
    let mut checkpoint_best: Option<CandidateSolution> = None;
    let mut checkpoint_replays = 0u64;
    let mut last_checkpoint = None;
    let execution = execute_workers(
        inputs,
        engine_stop,
        Arc::new(AtomicBool::new(false)),
        request.seed,
        |context, (problem, spec, worker_iterations)| {
            solve_ls_capped(problem, spec, context.stop(), context.seed(), config, worker_iterations, |objective, assignment, _source| {
                context.publish_latest(Improvement { objective, assignment: assignment.to_vec() });
            })
        },
        |event| {
            let now = Instant::now();
            // A coalesced engine incumbent that improves the last canonically
            // replayed objective must be checked immediately. Otherwise an
            // external stop can leave an older verified assignment behind even
            // though the final progress event already announced a better one.
            let improves_verified_objective =
                checkpoint_best.as_ref().and_then(|candidate| candidate.objectives().first()).is_some_and(|incumbent| {
                    if minimizing {
                        event.payload.objective < *incumbent
                    } else {
                        event.payload.objective > *incumbent
                    }
                });
            let replay = checkpoint_best.is_none()
                || publish_assignments
                || improves_verified_objective
                || last_checkpoint.is_none_or(|previous: Instant| now.duration_since(previous) >= INTERNAL_REPLAY_INTERVAL);
            if replay {
                last_checkpoint = Some(now);
                if let Some(candidate) = repair.candidate(
                    &event.payload.assignment,
                    request.seed.wrapping_add(event.worker as u64),
                    VerificationLevel::Transfer,
                )? {
                    checkpoint_replays = checkpoint_replays.saturating_add(1);
                    let improved = checkpoint_best.as_ref().is_none_or(|incumbent| candidate_better(&candidate, incumbent, minimizing));
                    if improved {
                        checkpoint_best = Some(candidate.clone());
                    }
                    if publish_assignments && improved && sink.emit(SolveEvent::Candidate(candidate))? == EventControl::Stop {
                        budget.cancel_with(TerminationReason::EventSink);
                        return Ok(EventControl::Stop);
                    }
                }
            }
            let control = sink.emit(SolveEvent::Progress {
                engine: EngineKind::IntegerLocalSearch,
                objectives: vec![event.payload.objective],
                elapsed: budget.elapsed(),
            })?;
            if control == EventControl::Stop {
                budget.cancel_with(TerminationReason::EventSink);
            }
            Ok(control)
        },
    )?;

    let mut best = checkpoint_best.map(promote_checkpoint_candidate);
    let mut iterations = 0u64;
    let mut moves = 0u64;
    let mut restarts = 0u64;
    let mut constraints = 0usize;
    let mut functionals = 0usize;
    let mut unsupported = 0usize;
    let mut rejected = 0usize;
    let mut last_rejection = None;

    for report in execution.reports {
        let LocalSearchOutcome {
            best: local_best,
            iterations: local_iterations,
            moves: local_moves,
            restarts: local_restarts,
            constraints: local_constraints,
            functionals: local_functionals,
            unsupported: local_unsupported,
        } = report.result;
        iterations = iterations.saturating_add(local_iterations);
        moves = moves.saturating_add(local_moves);
        restarts = restarts.saturating_add(local_restarts);
        constraints = constraints.max(local_constraints);
        functionals = functionals.max(local_functionals);
        unsupported = unsupported.max(local_unsupported);
        let Some((values, _)) = local_best else {
            continue;
        };
        match repair.candidate(&values, report.seed, VerificationLevel::Final) {
            Ok(Some(candidate)) if best.as_ref().is_none_or(|incumbent| candidate_better(&candidate, incumbent, minimizing)) => {
                best = Some(candidate);
            }
            Ok(_) => {}
            Err(error) => {
                rejected = rejected.saturating_add(1);
                last_rejection = Some(error.to_string());
            }
        }
    }

    let iteration_limit_reached = request.limits.iterations.is_some_and(|limit| iterations >= limit) && !budget.expired();

    let status = if best.is_some() { SolveStatus::Satisfiable } else { SolveStatus::Unknown };
    let mut metadata = vec![
        ("ls_moves".to_string(), moves.to_string()),
        ("ls_constraints".to_string(), constraints.to_string()),
        ("ls_functionals".to_string(), functionals.to_string()),
        ("ls_unsupported".to_string(), unsupported.to_string()),
        ("ls_rejected_incumbents".to_string(), rejected.to_string()),
        ("ls_checkpoint_replays".to_string(), checkpoint_replays.to_string()),
        ("workers".to_string(), request.threads.to_string()),
    ];
    if let Some(error) = &last_rejection {
        metadata.push(("ls_last_rejection".to_string(), error.clone()));
    }
    let message = if iteration_limit_reached {
        Some("integer local search reached the shared IterationLimit".to_string())
    } else if unsupported > 0 {
        Some(format!("integer local search declined {unsupported} model constructs"))
    } else if rejected > 0 && best.is_none() {
        Some(format!(
            "canonical replay rejected {rejected} local-search incumbents{}",
            last_rejection.as_ref().map_or(String::new(), |error| format!(": {error}"))
        ))
    } else if budget.expired() && best.is_none() {
        Some(format!("integer local search stopped: {:?}", budget.termination_reason()))
    } else {
        None
    };
    Ok(SolveResult {
        status,
        primal: best,
        bounds: Vec::new(),
        proof: None,
        reports: vec![EngineReport {
            engine: Some(EngineKind::IntegerLocalSearch),
            search: crate::search::SolveStats {
                solutions: u64::from(status == SolveStatus::Satisfiable),
                nodes: iterations,
                failures: restarts,
                ..crate::search::SolveStats::default()
            },
            elapsed: started.elapsed(),
            improvements: u64::from(status == SolveStatus::Satisfiable),
            metadata,
        }],
        message,
    })
}

fn candidate_better(candidate: &CandidateSolution, incumbent: &CandidateSolution, minimizing: bool) -> bool {
    match (candidate.objectives().first(), incumbent.objectives().first()) {
        (None, None) => false,
        (Some(candidate), Some(incumbent)) if minimizing => candidate < incumbent,
        (Some(candidate), Some(incumbent)) => candidate > incumbent,
        _ => false,
    }
}

fn promote_checkpoint_candidate(candidate: CandidateSolution) -> CandidateSolution {
    CandidateSolution::verified(
        candidate.assignment().clone(),
        candidate.objectives().to_vec(),
        candidate.source(),
        VerificationLevel::Final,
    )
}

fn worker_iteration_quota(total: u64, worker: usize, workers: usize) -> u64 {
    if total == u64::MAX {
        return u64::MAX;
    }
    let workers = u64::try_from(workers).unwrap_or(u64::MAX).max(1);
    let worker = u64::try_from(worker).unwrap_or(u64::MAX);
    total / workers + u64::from(worker < total % workers)
}

fn repair_candidate(
    context: &RepairContext<'_>,
    values: &[i32],
    seed: u64,
    verification: VerificationLevel,
) -> Result<Option<CandidateSolution>, SolveError> {
    if context.budget.expired() && verification == VerificationLevel::Transfer {
        return Ok(None);
    }
    let stop = context.budget.stop();
    let problem = context.compiled.problem();
    if values.len() != problem.search.len() {
        return Err(SolveError::InvalidResult(format!(
            "integer local-search assignment has {} values, expected {}",
            values.len(),
            problem.search.len()
        )));
    }
    let mut solver = problem.solver.clone();
    for (&variable, &value) in problem.search.iter().zip(values) {
        if context.spec.is_decision(variable) && !context.spec.is_derived(variable) {
            solver
                .store
                .fix(variable, value)
                .map_err(|_| SolveError::InvalidResult("local-search decision violates its CP domain".to_string()))?;
        }
    }
    solver.enqueue_all();
    solver.propagate().map_err(|_| SolveError::InvalidResult("local-search decisions violate the canonical CP root".to_string()))?;
    let completed = if problem.search.iter().all(|variable| solver.store.is_fixed(*variable)) {
        problem.search.iter().map(|variable| solver.store.value(*variable)).collect()
    } else {
        let conflict_limit = Some(context.request.limits.conflicts.unwrap_or(10_000).min(10_000));
        let (solution, _, _) = crate::search::decide_sat_assuming_seeded(
            &mut solver,
            &problem.search,
            &[],
            stop,
            seed,
            None,
            conflict_limit,
            Vec::new(),
            Vec::new(),
        );
        let Some(solution) = solution else {
            return Err(SolveError::InvalidResult("local-search decisions have no CP completion within the shared budget".to_string()));
        };
        solution
    };
    let candidate =
        super::cp::candidate_if_running(context.model, context.compiled, &completed, context.budget, verification)?.map(|candidate| {
            CandidateSolution::verified(
                candidate.assignment().clone(),
                candidate.objectives().to_vec(),
                EngineKind::IntegerLocalSearch,
                verification,
            )
        });
    if let Some(candidate) = &candidate {
        super::cp::verify_assumptions(candidate, &context.request.assumptions)?;
    }
    Ok(candidate)
}

fn compile_constraint(spec: &mut LocalSearchSpec, model: &Model, compiled: &CompiledCp, constraint: &Constraint) -> Result<(), SolveError> {
    let map = compiled.int_variables();
    match constraint {
        Constraint::Intension(expression) => spec.add_expr(expression_of(compiled, expression)?),
        Constraint::Selected { selector, constraint } => {
            let start = spec.begin_guarded_constraints();
            let result = compile_constraint(spec, model, compiled, constraint);
            spec.finish_guarded_constraints(start, map[selector.0]);
            result?;
        }
        Constraint::Linear { terms, relation, rhs } => spec.add_linear(
            terms.iter().map(|(coefficient, _)| *coefficient).collect(),
            terms.iter().map(|(_, variable)| map[variable.0]).collect(),
            physical_relation(*relation),
            *rhs,
        ),
        Constraint::Clause(literals) => spec.add_expr(clause_expression(map, literals)),
        Constraint::IntegerGlobal(global) => compile_global(spec, compiled, global)?,
        Constraint::SetSubset { subset, superset } => {
            for value in set_values(model, [*subset, *superset]) {
                match (membership(compiled, *subset, value), membership(compiled, *superset, value)) {
                    (Some(left), Some(right)) => spec.add_expr(Expr::Imp(
                        Box::new(Expr::Eq(Box::new(Expr::Var(left)), Box::new(Expr::Const(1)))),
                        Box::new(Expr::Eq(Box::new(Expr::Var(right)), Box::new(Expr::Const(1)))),
                    )),
                    (Some(left), None) => spec.add_linear(vec![1], vec![left], PhysicalRelation::Eq, 0),
                    (None, _) => {}
                }
            }
        }
        Constraint::SetDisjoint { left, right } => {
            for value in set_values(model, [*left, *right]) {
                if let (Some(left), Some(right)) = (membership(compiled, *left, value), membership(compiled, *right, value)) {
                    spec.add_linear(vec![1, 1], vec![left, right], PhysicalRelation::Le, 1);
                }
            }
        }
        Constraint::SetCardinality { set, min, max } => {
            let variables = compiled.sets()[set.0].membership.clone();
            spec.add_linear(vec![1; variables.len()], variables.clone(), PhysicalRelation::Ge, *min as i64);
            spec.add_linear(vec![1; variables.len()], variables, PhysicalRelation::Le, *max as i64);
        }
        Constraint::ListPartition { .. }
        | Constraint::ListPartitionWithCoverage { .. }
        | Constraint::SameList { .. }
        | Constraint::ItemPrecedence { .. }
        | Constraint::CollectionGlobal(_)
        | Constraint::ListLength { .. }
        | Constraint::ListItemSum { .. }
        | Constraint::ListReduction(_)
        | Constraint::IntervalPrecedence { .. }
        | Constraint::IntervalAlternative { .. }
        | Constraint::IntervalEndpointRelation { .. }
        | Constraint::IntervalResource(_) => {
            return Err(SolveError::Compile("integer local-search compiler received a collection or interval constraint".to_string()));
        }
    }
    Ok(())
}

fn compile_global(spec: &mut LocalSearchSpec, compiled: &CompiledCp, global: &IntGlobalConstraint) -> Result<(), SolveError> {
    let map = compiled.int_variables();
    let vars = |ids: &[crate::model::IntVarRef]| ids.iter().map(|variable| map[variable.0]).collect::<Vec<_>>();
    match global {
        IntGlobalConstraint::AllDifferent { variables, except } if except.is_empty() => spec.add_all_different(vars(variables)),
        IntGlobalConstraint::AllDifferent { variables, except } => spec.add_all_different_except(vars(variables), except.clone()),
        IntGlobalConstraint::AllEqual(variables) => spec.add_all_equal(vars(variables)),
        IntGlobalConstraint::Ordered { variables, relation } => {
            for pair in vars(variables).windows(2) {
                spec.add_linear(vec![1, -1], pair.to_vec(), physical_relation(*relation), 0);
            }
        }
        IntGlobalConstraint::Instantiation { variables, values } => {
            for (variable, value) in vars(variables).into_iter().zip(values) {
                spec.add_linear(vec![1], vec![variable], PhysicalRelation::Eq, i64::from(*value));
            }
        }
        IntGlobalConstraint::Minimum { target, variables } | IntGlobalConstraint::Maximum { target, variables } => {
            let values = vars(variables).into_iter().map(Expr::Var).collect();
            let extremum = if matches!(global, IntGlobalConstraint::Minimum { .. }) { Expr::Min(values) } else { Expr::Max(values) };
            spec.add_expr(Expr::Eq(Box::new(Expr::Var(map[target.0])), Box::new(extremum)));
        }
        IntGlobalConstraint::Element { array, index, value } => {
            spec.add_element(vars(array), map[index.0], map[value.0], 0);
        }
        IntGlobalConstraint::ElementConst { array, index, value } => {
            let tuples = array.iter().enumerate().map(|(index, value)| vec![index as i32, *value]).collect();
            spec.add_extension(vec![map[index.0], map[value.0]], tuples, true);
        }
        IntGlobalConstraint::Count { variables, value, relation, count } => {
            spec.add_count(vars(variables), vec![*value], physical_relation(*relation), LocalRhs::Const(*count));
        }
        IntGlobalConstraint::Cardinality { variables, values, lower, upper, closed } => {
            spec.add_cardinality(vars(variables), values.clone(), lower.clone(), upper.clone(), *closed);
        }
        IntGlobalConstraint::NValues { variables, relation, count } => {
            spec.add_n_values(vars(variables), physical_relation(*relation), LocalRhs::Const(*count));
        }
        IntGlobalConstraint::Table { variables, tuples, positive } => {
            spec.add_extension(vars(variables), tuples.clone(), *positive);
        }
        IntGlobalConstraint::Regular { variables, automaton } => spec.add_regular(
            vars(variables),
            Dfa {
                n_states: automaton.states,
                start: automaton.start,
                accept: automaton.accepting.clone(),
                transitions: automaton.transitions.clone(),
            },
        ),
        IntGlobalConstraint::Mdd { variables, mdd } => spec.add_mdd(
            vars(variables),
            Mdd {
                layers: mdd
                    .layers
                    .iter()
                    .map(|layer| layer.iter().map(|arc| MddArc { from: arc.from, value: arc.value, to: arc.to }).collect())
                    .collect(),
                nodes_per_layer: mdd.nodes_per_layer.clone(),
            },
        ),
        IntGlobalConstraint::Lex { left, right, strict } => {
            spec.add_lex_chain(vec![vars(left), vars(right)], *strict);
        }
        IntGlobalConstraint::LexChain { rows, strict } => {
            spec.add_lex_chain(rows.iter().map(|row| vars(row)).collect(), *strict);
        }
        IntGlobalConstraint::Channel { left, right } => spec.add_channel_inverse(vars(left), 0, vars(right), 0),
        IntGlobalConstraint::Circuit { successors, .. } => spec.add_circuit(vars(successors)),
        IntGlobalConstraint::NoOverlap { starts, durations } => spec.add_no_overlap(
            vars(starts).into_iter().map(|start| vec![start]).collect(),
            durations.iter().copied().map(|duration| vec![Expr::Const(duration)]).collect(),
            false,
        ),
        IntGlobalConstraint::OptionalNoOverlap { starts, durations, presences } => spec.add_no_overlap(
            vars(starts).into_iter().map(|start| vec![start]).collect(),
            durations
                .iter()
                .zip(presences)
                .map(|(&duration, presence)| {
                    vec![presence
                        .map_or(Expr::Const(duration), |presence| Expr::Mul(vec![Expr::Const(duration), Expr::Var(map[presence.0])]))]
                })
                .collect(),
            true,
        ),
        IntGlobalConstraint::AlternativeChannel { shared_start, starts, presences, .. } => {
            let presences = vars(presences);
            spec.add_linear(vec![1; presences.len()], presences.clone(), PhysicalRelation::Eq, 1);
            for (&start, &presence) in starts.iter().zip(&presences) {
                spec.add_expr(Expr::Imp(
                    Box::new(Expr::Eq(Box::new(Expr::Var(presence)), Box::new(Expr::Const(1)))),
                    Box::new(Expr::Eq(Box::new(Expr::Var(map[shared_start.0])), Box::new(Expr::Var(map[start.0])))),
                ));
            }
        }
        IntGlobalConstraint::Cumulative { starts, durations, demands, capacity } => spec.add_cumulative_rhs(
            vars(starts),
            durations.iter().copied().map(LocalRhs::Const).collect(),
            demands.iter().copied().map(LocalRhs::Const).collect(),
            LocalRhs::Const(*capacity),
        ),
        IntGlobalConstraint::CumulativeVar { starts, durations, demands, capacity } => {
            spec.add_cumulative(vars(starts), vars(durations), vars(demands), LocalRhs::Var(map[capacity.0]))
        }
        IntGlobalConstraint::BinPacking { items, sizes, capacities } => {
            spec.add_bin_packing(vars(items), sizes.clone(), capacities.iter().copied().map(LocalRhs::Const).collect(), false)
        }
        IntGlobalConstraint::BinLoads { items, sizes, loads } => {
            spec.add_bin_packing(vars(items), sizes.clone(), vars(loads).into_iter().map(LocalRhs::Var).collect(), true)
        }
        IntGlobalConstraint::Knapsack { variables, weights, profits, weight_relation, weight_limit, profit_relation, profit_limit } => {
            let variables = vars(variables);
            spec.add_linear(weights.clone(), variables.clone(), physical_relation(*weight_relation), *weight_limit);
            spec.add_linear(profits.clone(), variables, physical_relation(*profit_relation), *profit_limit);
        }
        IntGlobalConstraint::ValuePrecedence { variables, values, covered } => {
            spec.add_precedence(vars(variables), values.clone(), *covered);
        }
    }
    Ok(())
}

fn expression_of(compiled: &CompiledCp, expression: &crate::model::IntExpr) -> Result<Expr, SolveError> {
    compiled.compile_expression(expression).map_err(|error| SolveError::Compile(error.reason))
}

fn physical_relation(relation: Relation) -> PhysicalRelation {
    match relation {
        Relation::Eq => PhysicalRelation::Eq,
        Relation::Ne => PhysicalRelation::Ne,
        Relation::Le => PhysicalRelation::Le,
        Relation::Lt => PhysicalRelation::Lt,
        Relation::Ge => PhysicalRelation::Ge,
        Relation::Gt => PhysicalRelation::Gt,
    }
}

fn clause_expression(map: &[VarId], literals: &[BoolLiteral]) -> Expr {
    Expr::Or(
        literals
            .iter()
            .map(|literal| {
                let variable = Expr::Var(map[literal.variable.0]);
                if literal.positive {
                    variable
                } else {
                    Expr::Not(Box::new(variable))
                }
            })
            .collect(),
    )
}

fn membership(compiled: &CompiledCp, set: SetVarRef, value: i32) -> Option<VarId> {
    let set = &compiled.sets()[set.0];
    set.values.binary_search(&value).ok().map(|index| set.membership[index])
}

fn set_values<const N: usize>(model: &Model, sets: [SetVarRef; N]) -> std::collections::BTreeSet<i32> {
    sets.iter().flat_map(|set| model.sets()[set.0].possible.iter().copied()).collect()
}
