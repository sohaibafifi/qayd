//! Structured interval scheduling backend.
//!
//! This module owns the reusable scheduling-engine orchestration: lowering the
//! schedule IR to intervals, choosing the exact CDCL or chronological
//! backend, maintaining makespan bounds, and replaying optional-mode incumbents.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Arc;

use crate::model::list::{Resource, Schedule};
use crate::constraints::intension;
use crate::constraints::interval as interval_constraints;
use crate::expr;
use crate::ids::{IntervalId, VarId};
use crate::search::{self, Objective as SearchObjective, SearchControl, SolveStats};
use crate::Solver;

/// Runtime options for the domain scheduling engine.
#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    /// Seed for CDCL-backed searches.
    pub seed: u64,
    /// Use the CDCL backend for optional-mode schedules. It is correct but not
    /// yet the strongest default on this model shape.
    pub optional_modes_cdcl: bool,
}

/// Solver status returned by the scheduling engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Optimal,
    Satisfiable,
    Unsatisfiable,
    Unknown,
}

impl Status {
    /// Stable status text for frontends.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Optimal => "OPTIMAL",
            Self::Satisfiable => "SATISFIABLE",
            Self::Unsatisfiable => "UNSATISFIABLE",
            Self::Unknown => "UNKNOWN",
        }
    }
}

/// Structured interval schedule result.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub status: Status,
    pub objective: Option<i64>,
    pub starts: Vec<i64>,
    /// Chosen machine per operation, or `-1` for fixed intervals.
    pub machines: Vec<i64>,
    pub stats: SolveStats,
}

/// Mode-interval layout of an optional-mode schedule. Built deterministically,
/// so VarId / IntervalId / disjunctive-pair indices match across fresh solver
/// instances created from the same schedule. That property is required to replay
/// the incumbent assignment on a clean solver after optimization.
struct ModedBuild {
    /// op_modes[op] = [(machine, mode interval id, duration)].
    op_modes: Vec<Vec<(usize, IntervalId, i32)>>,
    all_modes: Vec<IntervalId>,
    all_durations: Vec<i32>,
}

/// Solve a supported interval schedule. Returns `Ok(None)` when the
/// schedule shape is not supported by this exact backend.
pub fn solve<F>(schedule: &Schedule, stop: &AtomicBool, options: Options, on_improve: F) -> Result<Option<Outcome>, String>
where
    F: FnMut(i64),
{
    if schedule.intervals.iter().any(|iv| !iv.modes.is_empty()) {
        return solve_optional_modes(schedule, stop, options, on_improve);
    }
    solve_fixed_intervals(schedule, stop, options, on_improve)
}

fn solve_fixed_intervals<F>(schedule: &Schedule, stop: &AtomicBool, options: Options, mut on_improve: F) -> Result<Option<Outcome>, String>
where
    F: FnMut(i64),
{
    if !schedule.intervals.iter().all(|iv| iv.modes.is_empty()) {
        return Ok(None);
    }
    let resources_supported =
        schedule.resources.iter().all(|resource| matches!(resource, Resource::NoOverlap(_) | Resource::Cumulative { .. }));
    if !resources_supported {
        return Ok(None);
    }

    let mut solver = Solver::new();
    let mut intervals: Vec<IntervalId> = Vec::with_capacity(schedule.intervals.len());
    let mut durations_i32 = Vec::with_capacity(schedule.intervals.len());
    for interval in &schedule.intervals {
        let duration = checked_i32(interval.duration, "interval duration")?;
        let start_max = checked_start_max(interval.horizon, interval.duration, "interval")?;
        intervals.push(solver.store.new_interval(0, start_max, duration));
        durations_i32.push(duration);
    }
    for &(before, after) in &schedule.precedences {
        let before = *intervals.get(before).ok_or("schedule precedence references a missing interval")?;
        let after = *intervals.get(after).ok_or("schedule precedence references a missing interval")?;
        interval_constraints::interval_precedence(&mut solver, before, after);
    }
    for resource in &schedule.resources {
        match resource {
            Resource::NoOverlap(group) => {
                let group_ids = group
                    .iter()
                    .map(|&index| intervals.get(index).copied().ok_or("no-overlap references a missing interval"))
                    .collect::<Result<Vec<_>, _>>()?;
                interval_constraints::no_overlap(&mut solver, &group_ids);
            }
            Resource::Cumulative { demands, capacity } => {
                let group_ids = demands
                    .iter()
                    .map(|&(index, _)| intervals.get(index).copied().ok_or("cumulative references a missing interval"))
                    .collect::<Result<Vec<_>, _>>()?;
                let group_demands =
                    demands.iter().map(|&(_, demand)| checked_i32(demand, "cumulative demand")).collect::<Result<Vec<_>, _>>()?;
                let cap = checked_i32(*capacity, "cumulative capacity")?;
                interval_constraints::cumulative(&mut solver, &group_ids, &group_demands, cap);
            }
            Resource::MachineNoOverlap => {}
        }
    }

    if schedule.minimize_makespan {
        let max_horizon = checked_i32(schedule.intervals.iter().map(|iv| iv.horizon).max().unwrap_or(0), "schedule horizon")?;
        let makespan = solver.store.new_var_range(0, max_horizon);
        for (&iv, &dur) in intervals.iter().zip(&durations_i32) {
            let start = solver.store.interval_start_var(iv);
            intension::intension(&mut solver, expr::ge(expr::var(makespan), expr::add(vec![expr::var(start), expr::int(i64::from(dur))])));
        }
        let mut search_vars: Vec<VarId> = intervals.iter().map(|&iv| solver.store.interval_start_var(iv)).collect();
        let n_starts = search_vars.len();
        search_vars.extend((0..solver.store.disjunctive_pair_count()).map(|i| solver.store.disjunctive_order_var(i)));
        search_vars.push(makespan);

        let (best, stats, complete) = search::optimize_seeded(
            &mut solver,
            &search_vars,
            SearchObjective::Var(makespan),
            true,
            stop,
            options.seed,
            None,
            None,
            &[],
            None,
            |value, _| on_improve(value),
        );
        let outcome = match (best, complete) {
            (Some((assignment, value)), complete) => Outcome {
                status: if complete { Status::Optimal } else { Status::Satisfiable },
                objective: Some(value),
                starts: assignment[..n_starts].iter().map(|&s| i64::from(s)).collect(),
                machines: vec![-1; schedule.intervals.len()],
                stats,
            },
            (None, true) => Outcome { status: Status::Unsatisfiable, objective: None, starts: Vec::new(), machines: Vec::new(), stats },
            (None, false) => Outcome { status: Status::Unknown, objective: None, starts: Vec::new(), machines: Vec::new(), stats },
        };
        return Ok(Some(outcome));
    }

    let mut starts: Option<Vec<i64>> = None;
    let (stats, complete) = search::solve_domains_interruptible(
        &mut solver,
        |_, domain| {
            starts = Some(domain.interval_starts.iter().map(|start| i64::from(start.unwrap_or(0))).collect());
            SearchControl::Stop
        },
        stop,
    );
    let outcome = match (starts, complete) {
        (Some(starts), _) => {
            Outcome { status: Status::Satisfiable, objective: None, starts, machines: vec![-1; schedule.intervals.len()], stats }
        }
        (None, true) => Outcome { status: Status::Unsatisfiable, objective: None, starts: Vec::new(), machines: Vec::new(), stats },
        (None, false) => Outcome { status: Status::Unknown, objective: None, starts: Vec::new(), machines: Vec::new(), stats },
    };
    Ok(Some(outcome))
}

fn solve_optional_modes<F>(schedule: &Schedule, stop: &AtomicBool, options: Options, mut on_improve: F) -> Result<Option<Outcome>, String>
where
    F: FnMut(i64),
{
    let all_moded = schedule.intervals.iter().all(|iv| !iv.modes.is_empty());
    let resources_ok = schedule.resources.iter().all(|resource| matches!(resource, Resource::MachineNoOverlap));
    if !all_moded || !resources_ok {
        return Ok(None);
    }

    let mut solver = Solver::new();
    let ModedBuild { op_modes, all_modes, all_durations } = build_moded_schedule(schedule, &mut solver)?;
    let op_count = op_modes.len();

    if schedule.minimize_makespan && options.optional_modes_cdcl {
        return solve_optional_modes_cdcl(schedule, solver, op_modes, all_modes, all_durations, stop, options.seed, on_improve).map(Some);
    }

    let minimize = schedule.minimize_makespan;
    let makespan_ub = Arc::new(AtomicI32::new(i32::MAX));
    if minimize {
        interval_constraints::makespan_bound(&mut solver, &all_modes, &all_durations, Arc::clone(&makespan_ub));
    }
    let bound_on_improve = Arc::clone(&makespan_ub);
    let op_index: Vec<Vec<(usize, usize, i32)>> =
        op_modes.iter().map(|modes| modes.iter().map(|&(machine, id, duration)| (machine, id.index(), duration)).collect()).collect();
    let mut best: Option<(i64, Vec<i64>, Vec<i64>)> = None;
    let (stats, complete) = search::solve_domains_interruptible(
        &mut solver,
        |_, domain| {
            let mut starts = vec![0i64; op_count];
            let mut chosen = vec![-1i64; op_count];
            let mut makespan = 0i64;
            for (op, modes) in op_index.iter().enumerate() {
                for &(machine, idx, duration) in modes {
                    if let Some(start) = domain.interval_starts[idx] {
                        starts[op] = i64::from(start);
                        chosen[op] = machine as i64;
                        makespan = makespan.max(i64::from(start) + i64::from(duration));
                    }
                }
            }
            if best.as_ref().is_none_or(|(value, _, _)| makespan < *value) {
                if minimize {
                    on_improve(makespan);
                    bound_on_improve.store(i32::try_from(makespan.saturating_sub(1)).unwrap_or(i32::MAX), Ordering::Relaxed);
                }
                best = Some((makespan, starts, chosen));
            }
            if minimize {
                SearchControl::Continue
            } else {
                SearchControl::Stop
            }
        },
        stop,
    );
    let outcome = match (best, complete) {
        (Some((objective, starts, machines)), _) if minimize => Outcome {
            status: if complete { Status::Optimal } else { Status::Satisfiable },
            objective: Some(objective),
            starts,
            machines,
            stats,
        },
        (Some((_, starts, machines)), _) => Outcome { status: Status::Satisfiable, objective: None, starts, machines, stats },
        (None, true) => Outcome { status: Status::Unsatisfiable, objective: None, starts: Vec::new(), machines: Vec::new(), stats },
        (None, false) => Outcome { status: Status::Unknown, objective: None, starts: Vec::new(), machines: Vec::new(), stats },
    };
    Ok(Some(outcome))
}

#[allow(clippy::too_many_arguments)]
fn solve_optional_modes_cdcl<F>(
    schedule: &Schedule,
    mut solver: Solver,
    op_modes: Vec<Vec<(usize, IntervalId, i32)>>,
    all_modes: Vec<IntervalId>,
    all_durations: Vec<i32>,
    stop: &AtomicBool,
    seed: u64,
    mut on_improve: F,
) -> Result<Outcome, String>
where
    F: FnMut(i64),
{
    let op_count = op_modes.len();
    let max_horizon = checked_i32(schedule.intervals.iter().map(|iv| iv.horizon).max().unwrap_or(0), "schedule horizon")?;
    let makespan = solver.store.new_var_range(0, max_horizon);
    let start_vars: Vec<VarId> = all_modes.iter().map(|&id| solver.store.interval_start_var(id)).collect();
    let presence_vars: Vec<VarId> =
        all_modes.iter().map(|&id| solver.store.interval_presence_var(id).expect("mode interval is optional")).collect();
    for ((&start, &present), &dur) in start_vars.iter().zip(&presence_vars).zip(&all_durations) {
        intension::intension(
            &mut solver,
            expr::imp(
                expr::eq(expr::var(present), expr::int(1)),
                expr::ge(expr::var(makespan), expr::add(vec![expr::var(start), expr::int(i64::from(dur))])),
            ),
        );
    }

    let mut op_modes_flat: Vec<Vec<(usize, usize)>> = Vec::with_capacity(op_count);
    let mut flat = 0usize;
    for modes in &op_modes {
        let mut row = Vec::with_capacity(modes.len());
        for &(machine, _, _) in modes {
            row.push((machine, flat));
            flat += 1;
        }
        op_modes_flat.push(row);
    }
    let n_modes = all_modes.len();
    let n_orders = solver.store.disjunctive_pair_count();
    let mut search_vars: Vec<VarId> = presence_vars;
    search_vars.extend((0..n_orders).map(|i| solver.store.disjunctive_order_var(i)));
    search_vars.push(makespan);

    let (best, stats, complete) = search::optimize_seeded(
        &mut solver,
        &search_vars,
        SearchObjective::Var(makespan),
        true,
        stop,
        seed,
        None,
        None,
        &[],
        None,
        |value, _| on_improve(value),
    );

    let Some((assignment, value)) = best else {
        return Ok(if complete {
            Outcome { status: Status::Unsatisfiable, objective: None, starts: Vec::new(), machines: Vec::new(), stats }
        } else {
            Outcome { status: Status::Unknown, objective: None, starts: Vec::new(), machines: Vec::new(), stats }
        });
    };

    let mut replay = Solver::new();
    let replay_build = build_moded_schedule(schedule, &mut replay)?;
    let replay_present: Vec<VarId> =
        replay_build.all_modes.iter().map(|&id| replay.store.interval_presence_var(id).expect("mode interval is optional")).collect();
    for (i, &present) in replay_present.iter().enumerate() {
        replay.store.fix(present, assignment[i]).map_err(|_| "internal error: replaying the optimal mode schedule was inconsistent")?;
    }
    for k in 0..n_orders {
        let order_var = replay.store.disjunctive_order_var(k);
        replay
            .store
            .fix(order_var, assignment[n_modes + k])
            .map_err(|_| "internal error: replaying the optimal mode schedule was inconsistent")?;
    }
    replay.propagate().map_err(|_| "internal error: replaying the optimal mode schedule was inconsistent")?;

    let mut starts = vec![0i64; op_count];
    let mut chosen = vec![-1i64; op_count];
    let mut makespan_value = 0i64;
    for (op, modes) in op_modes_flat.iter().enumerate() {
        for &(machine, flat_idx) in modes {
            if assignment[flat_idx] == 1 {
                let start = i64::from(replay.store.interval_start_min(replay_build.all_modes[flat_idx]));
                starts[op] = start;
                chosen[op] = machine as i64;
                makespan_value = makespan_value.max(start + i64::from(all_durations[flat_idx]));
            }
        }
    }
    if makespan_value != value {
        return Err("internal error: replayed mode schedule makespan does not match the reported value".to_string());
    }
    Ok(Outcome {
        status: if complete { Status::Optimal } else { Status::Satisfiable },
        objective: Some(makespan_value),
        starts,
        machines: chosen,
        stats,
    })
}

fn build_moded_schedule(schedule: &Schedule, solver: &mut Solver) -> Result<ModedBuild, String> {
    let mut op_modes: Vec<Vec<(usize, IntervalId, i32)>> = Vec::with_capacity(schedule.intervals.len());
    let mut all_modes: Vec<IntervalId> = Vec::new();
    let mut all_durations: Vec<i32> = Vec::new();
    for interval in &schedule.intervals {
        let mut modes = Vec::with_capacity(interval.modes.len());
        for mode in &interval.modes {
            let duration = checked_i32(mode.duration, "mode duration")?;
            let start_max = checked_start_max(interval.horizon, mode.duration, "mode interval")?;
            let id = solver.store.new_optional_interval(0, start_max, duration);
            modes.push((mode.machine, id, duration));
            all_modes.push(id);
            all_durations.push(duration);
        }
        let ids: Vec<IntervalId> = modes.iter().map(|&(_, id, _)| id).collect();
        if ids.is_empty() {
            return Err("mode schedule operation has no eligible mode".to_string());
        }
        interval_constraints::exactly_one_mode(solver, &ids);
        op_modes.push(modes);
    }
    let mut machines: BTreeSet<usize> = BTreeSet::new();
    for modes in &op_modes {
        for &(machine, _, _) in modes {
            machines.insert(machine);
        }
    }
    for machine in machines {
        let group: Vec<IntervalId> = op_modes.iter().flatten().filter(|&&(m, _, _)| m == machine).map(|&(_, id, _)| id).collect();
        interval_constraints::no_overlap(solver, &group);
    }
    for &(before, after) in &schedule.precedences {
        let before_modes = op_modes.get(before).ok_or("schedule precedence references a missing interval")?;
        let after_modes = op_modes.get(after).ok_or("schedule precedence references a missing interval")?;
        for &(_, ib, _) in before_modes {
            for &(_, ia, _) in after_modes {
                interval_constraints::interval_precedence(solver, ib, ia);
            }
        }
    }
    Ok(ModedBuild { op_modes, all_modes, all_durations })
}

fn checked_i32(value: i64, name: &str) -> Result<i32, String> {
    i32::try_from(value).map_err(|_| format!("{name} is outside the i32 domain range"))
}

fn checked_start_max(horizon: i64, duration: i64, name: &str) -> Result<i32, String> {
    if duration < 0 {
        return Err(format!("{name} duration must be non-negative"));
    }
    let start_max = horizon.checked_sub(duration).ok_or_else(|| format!("{name} horizon minus duration overflows"))?;
    if start_max < 0 {
        return Err(format!("{name} duration exceeds its horizon"));
    }
    checked_i32(start_max, "interval start upper bound")
}
