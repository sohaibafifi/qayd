use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::engines::ls::schedule_ir::PrecedenceDag;
use crate::mix64;
use crate::model::list::{CollectionSolution, IntervalVar, Resource, Schedule};

use super::resource_schedule::{
    GenerationScheme, Justification, PriorityRule, PrioritySgs, ResourceAlnsBudget, ResourceMoveAcceptance, ResourceMoveOutcome,
    ResourceScheduleMetrics, ResourceScheduleProblem, ResourceScheduleState,
};
use super::schedule_state::{
    CriticalNeighborhood, DispatchRule, JobShopProblem, JobShopState, MoveAcceptance, MoveOutcome, ScheduleStateMetrics,
};

/// Scheduling-search measurements shared with the collection profiler.
pub(crate) struct ScheduleConstructionMetrics {
    pub(crate) elapsed: Duration,
    pub(crate) first_feasible: Option<Duration>,
    pub(crate) candidates: u64,
    pub(crate) work_steps: u64,
    pub(crate) constructor: &'static str,
    pub(crate) moves_considered: u64,
    pub(crate) moves_accepted: u64,
    pub(crate) incumbent_improvements: u64,
    pub(crate) incumbent_injections: u64,
    pub(crate) cycle_rejections: u64,
    pub(crate) window_rejections: u64,
    pub(crate) objective_rejections: u64,
    pub(crate) reconstructions: u64,
    pub(crate) critical_path_updates: u64,
    pub(crate) delta_evaluations: u64,
    pub(crate) full_evaluations: u64,
    pub(crate) full_fallbacks: u64,
    pub(crate) topological_rebuilds: u64,
    pub(crate) oracle_validations: u64,
    pub(crate) oracle_mismatches: u64,
    pub(crate) dirty_cone_operations: u64,
    pub(crate) max_dirty_cone: u64,
    pub(crate) workspace_growths: u64,
    pub(crate) workspace_rollbacks: u64,
    pub(crate) alns_generation_attempts: u64,
    pub(crate) alns_moves_generated: u64,
    pub(crate) resource_profile_checks: u64,
    pub(crate) resource_candidate_scheduling_attempts: u64,
    pub(crate) resource_event_visits: u64,
    pub(crate) resource_peak_profile_events: usize,
    pub(crate) precedence_rejections: u64,
    pub(crate) infeasible_rejections: u64,
    pub(crate) justification_attempts: u64,
}

struct ConstructedSchedule {
    starts: Vec<i64>,
    chosen: Vec<usize>,
    present: Vec<bool>,
}

struct ConstructedIncumbent {
    schedule: ConstructedSchedule,
    durations: Vec<i64>,
    machines: Vec<i64>,
    score: (i64, i64),
    solution: CollectionSolution,
}

#[derive(Clone, Copy)]
enum SgsRule {
    Stable,
    LongestRemainingPath,
    ShortestProcessingTime,
    LongestProcessingTime,
    MostSuccessors,
    Seeded,
}

const SGS_RULES: [SgsRule; 5] = [
    SgsRule::Stable,
    SgsRule::LongestRemainingPath,
    SgsRule::ShortestProcessingTime,
    SgsRule::LongestProcessingTime,
    SgsRule::MostSuccessors,
];

#[derive(Clone, Copy)]
struct Interrupted;

type Interruptible<T> = Result<T, Interrupted>;

fn checkpoint(stop: &AtomicBool) -> Interruptible<()> {
    if stop.load(Ordering::Acquire) {
        Err(Interrupted)
    } else {
        Ok(())
    }
}

fn collect_interruptible<T>(values: impl IntoIterator<Item = T>, stop: &AtomicBool) -> Interruptible<Vec<T>> {
    checkpoint(stop)?;
    let mut values = values.into_iter();
    let (lower, upper) = values.size_hint();
    let mut collected = Vec::with_capacity(upper.unwrap_or(lower));
    loop {
        checkpoint(stop)?;
        let Some(value) = values.next() else {
            break;
        };
        collected.push(value);
    }
    checkpoint(stop)?;
    Ok(collected)
}

fn sift_down(values: &mut [i64], mut root: usize, end: usize, stop: &AtomicBool) -> Interruptible<()> {
    loop {
        checkpoint(stop)?;
        let left = root.saturating_mul(2).saturating_add(1);
        if left >= end {
            return Ok(());
        }
        let right = left + 1;
        let larger = if right < end && values[right] > values[left] { right } else { left };
        if values[root] >= values[larger] {
            return Ok(());
        }
        values.swap(root, larger);
        root = larger;
    }
}

fn sort_dedup_interruptible(values: &mut Vec<i64>, stop: &AtomicBool) -> Interruptible<()> {
    checkpoint(stop)?;
    if values.len() < 2 {
        return Ok(());
    }
    let len = values.len();
    for root in (0..len / 2).rev() {
        checkpoint(stop)?;
        sift_down(values, root, len, stop)?;
    }
    for end in (1..len).rev() {
        checkpoint(stop)?;
        values.swap(0, end);
        sift_down(values, 0, end, stop)?;
    }

    let mut write = 1usize;
    for read in 1..values.len() {
        checkpoint(stop)?;
        if values[read] != values[write - 1] {
            values[write] = values[read];
            write += 1;
        }
    }
    values.truncate(write);
    checkpoint(stop)
}

/// Score of an interval schedule: total constraint violation (bound, precedence,
/// resource), then the makespan. Smaller is better, violation first.
/// Duration of an interval under the chosen mode (the fixed duration when it has
/// no modes).
fn iv_duration(iv: &IntervalVar, mode: usize) -> i64 {
    if iv.modes.is_empty() {
        iv.duration
    } else {
        iv.modes[mode].duration
    }
}

/// Chosen machine of an interval, or `-1` when it has no modes (no machine).
fn iv_machine(iv: &IntervalVar, mode: usize) -> i64 {
    if iv.modes.is_empty() {
        -1
    } else {
        i64::try_from(iv.modes[mode].machine).expect("validated schedule mode machine fits i64")
    }
}

fn machines_are_representable(sched: &Schedule, stop: &AtomicBool) -> Interruptible<bool> {
    for interval in &sched.intervals {
        checkpoint(stop)?;
        for mode in &interval.modes {
            checkpoint(stop)?;
            if i64::try_from(mode.machine).is_err() {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Inclusive start bounds implied by the selected mode.
fn iv_start_window(iv: &IntervalVar, mode: usize) -> (i64, i64) {
    if iv.modes.is_empty() {
        (0, iv.horizon.saturating_sub(iv.duration))
    } else {
        iv.modes[mode].start_window
    }
}

/// Stable semantic identity implied by the selected mode.
fn iv_mode_reference(iv: &IntervalVar, mode: usize) -> Option<usize> {
    (!iv.modes.is_empty()).then(|| iv.modes[mode].reference).flatten()
}

fn random_start((start_min, start_max): (i64, i64), random: u64) -> i64 {
    debug_assert!(start_min <= start_max);
    let width = i128::from(start_max) - i128::from(start_min) + 1;
    let offset = i128::from(random) % width;
    i64::try_from(i128::from(start_min) + offset).expect("sampled start stays inside its i64 window")
}

/// Durations and machines implied by the chosen mode of each interval.
fn mode_view(sched: &Schedule, chosen: &[usize], stop: &AtomicBool) -> Interruptible<(Vec<i64>, Vec<i64>)> {
    checkpoint(stop)?;
    let mut durations = Vec::with_capacity(chosen.len());
    checkpoint(stop)?;
    let mut machines = Vec::with_capacity(chosen.len());
    for (interval, &mode) in sched.intervals.iter().zip(chosen) {
        checkpoint(stop)?;
        durations.push(iv_duration(interval, mode));
        machines.push(iv_machine(interval, mode));
    }
    checkpoint(stop)?;
    Ok((durations, machines))
}

fn mode_references(sched: &Schedule, chosen: &[usize], present: &[bool], stop: &AtomicBool) -> Interruptible<Vec<Option<usize>>> {
    collect_interruptible(
        sched
            .intervals
            .iter()
            .zip(chosen)
            .zip(present)
            .map(|((interval, &mode), &present)| present.then(|| iv_mode_reference(interval, mode)).flatten()),
        stop,
    )
}

fn overlaps(start: i64, duration: i64, other_start: i64, other_duration: i64) -> bool {
    start < other_start.saturating_add(other_duration) && other_start < start.saturating_add(duration)
}

/// Earliest start not conflicting with already scheduled operations. Every
/// rejection advances to an existing end event, so the loop is finite and its
/// candidate count is a useful construction-work metric.
#[allow(clippy::too_many_arguments)]
fn earliest_resource_start(
    sched: &Schedule,
    index: usize,
    duration: i64,
    machine: i64,
    earliest: i64,
    start_window: (i64, i64),
    durations: &[i64],
    machines: &[i64],
    starts: &[i64],
    present: &[bool],
    scheduled: &[bool],
    candidates: &mut u64,
    stop: &AtomicBool,
) -> Interruptible<Option<i64>> {
    checkpoint(stop)?;
    let (start_min, latest) = start_window;
    let mut start = earliest.max(start_min);
    while start <= latest {
        checkpoint(stop)?;
        *candidates = candidates.saturating_add(1);
        let end = start.saturating_add(duration);
        let mut advance: Option<i64> = None;
        for resource in &sched.resources {
            checkpoint(stop)?;
            match resource {
                Resource::NoOverlap(group) => {
                    let mut applies = false;
                    for &member in group {
                        checkpoint(stop)?;
                        if member == index {
                            applies = true;
                            break;
                        }
                    }
                    if !applies {
                        continue;
                    }
                    for &other in group {
                        checkpoint(stop)?;
                        if other != index
                            && present.get(other).copied().unwrap_or(false)
                            && scheduled.get(other).copied().unwrap_or(false)
                            && overlaps(start, duration, starts[other], durations[other])
                        {
                            advance =
                                Some(advance.map_or(starts[other] + durations[other], |old| old.max(starts[other] + durations[other])));
                        }
                    }
                }
                Resource::MachineNoOverlap if machine >= 0 => {
                    for other in 0..sched.intervals.len() {
                        checkpoint(stop)?;
                        if other != index
                            && present[other]
                            && scheduled[other]
                            && machines[other] == machine
                            && overlaps(start, duration, starts[other], durations[other])
                        {
                            advance =
                                Some(advance.map_or(starts[other] + durations[other], |old| old.max(starts[other] + durations[other])));
                        }
                    }
                }
                Resource::Cumulative { demands, capacity } => {
                    let mut demand = 0;
                    for &(task, task_demand) in demands {
                        checkpoint(stop)?;
                        if task == index {
                            demand = task_demand;
                            break;
                        }
                    }
                    if demand > *capacity {
                        return Ok(None);
                    }
                    if demand == 0 {
                        continue;
                    }
                    checkpoint(stop)?;
                    let mut events = vec![start];
                    for &(other, _) in demands {
                        checkpoint(stop)?;
                        if other != index && present[other] && scheduled[other] {
                            let other_start = starts[other];
                            let other_end = other_start.saturating_add(durations[other]);
                            if start < other_start && other_start < end {
                                events.push(other_start);
                            }
                            if start < other_end && other_end < end {
                                events.push(other_end);
                            }
                        }
                    }
                    sort_dedup_interruptible(&mut events, stop)?;
                    for event in events {
                        checkpoint(stop)?;
                        let mut usage = i128::from(demand);
                        let mut next_end: Option<i64> = None;
                        for &(other, other_demand) in demands {
                            checkpoint(stop)?;
                            if other != index
                                && present[other]
                                && scheduled[other]
                                && starts[other] <= event
                                && event < starts[other].saturating_add(durations[other])
                            {
                                usage += i128::from(other_demand);
                                let other_end = starts[other].saturating_add(durations[other]);
                                next_end = Some(next_end.map_or(other_end, |old| old.min(other_end)));
                            }
                        }
                        if usage > i128::from(*capacity) {
                            advance = Some(advance.map_or(next_end.unwrap_or(end), |old| old.max(next_end.unwrap_or(end))));
                            break;
                        }
                    }
                }
                Resource::MachineNoOverlap => {}
            }
        }
        let Some(next) = advance else {
            checkpoint(stop)?;
            return Ok(Some(start));
        };
        start = next.max(start.saturating_add(1));
    }
    checkpoint(stop)?;
    Ok(None)
}

fn ready_priority(
    rule: SgsRule,
    operation: usize,
    durations: &[i64],
    bottom: &[i64],
    successors: &[Vec<usize>],
    seed: u64,
) -> (i128, i128, u64, usize) {
    let duration = durations[operation];
    let successor_count = successors[operation].len();
    match rule {
        SgsRule::Stable => (0, 0, 0, operation),
        SgsRule::LongestRemainingPath => (-i128::from(bottom[operation]), i128::from(duration), 0, operation),
        SgsRule::ShortestProcessingTime => (i128::from(duration), -i128::from(bottom[operation]), 0, operation),
        SgsRule::LongestProcessingTime => (-i128::from(duration), -i128::from(bottom[operation]), 0, operation),
        SgsRule::MostSuccessors => (-i128::try_from(successor_count).unwrap_or(i128::MAX), -i128::from(bottom[operation]), 0, operation),
        SgsRule::Seeded => (0, 0, mix64(seed ^ u64::try_from(operation).unwrap_or(u64::MAX)), operation),
    }
}

fn select_ready(
    ready: &[usize],
    rule: SgsRule,
    durations: &[i64],
    bottom: &[i64],
    successors: &[Vec<usize>],
    seed: u64,
    stop: &AtomicBool,
) -> Interruptible<usize> {
    let mut selected = 0usize;
    for position in 1..ready.len() {
        checkpoint(stop)?;
        if ready_priority(rule, ready[position], durations, bottom, successors, seed)
            < ready_priority(rule, ready[selected], durations, bottom, successors, seed)
        {
            selected = position;
        }
    }
    Ok(selected)
}

/// Serial schedule generation scheme. Eligible operations are selected by one
/// generic priority rule and placed at their earliest resource-feasible event.
/// For flexible operations, the mode with the earliest completion is selected.
fn serial_schedule(sched: &Schedule, rule: SgsRule, seed: u64, stop: &AtomicBool) -> (Interruptible<Option<ConstructedSchedule>>, u64) {
    if let Err(interrupted) = checkpoint(stop) {
        return (Err(interrupted), 0);
    }
    let n = sched.intervals.len();
    let present = match collect_interruptible(sched.intervals.iter().map(|interval| !interval.optional), stop) {
        Ok(present) => present,
        Err(interrupted) => return (Err(interrupted), 0),
    };
    let mut indegree = match collect_interruptible(std::iter::repeat_n(0usize, n), stop) {
        Ok(indegree) => indegree,
        Err(interrupted) => return (Err(interrupted), 0),
    };
    let mut successors = match collect_interruptible((0..n).map(|_| Vec::new()), stop) {
        Ok(successors) => successors,
        Err(interrupted) => return (Err(interrupted), 0),
    };
    for &(before, after) in &sched.precedences {
        if let Err(interrupted) = checkpoint(stop) {
            return (Err(interrupted), 0);
        }
        if present[before] && present[after] {
            successors[before].push(after);
        }
    }
    for list in &mut successors {
        if let Err(interrupted) = checkpoint(stop) {
            return (Err(interrupted), 0);
        }
        list.sort_unstable();
        list.dedup();
        for &successor in list.iter() {
            indegree[successor] = indegree[successor].saturating_add(1);
        }
    }
    let base_durations = match collect_interruptible(
        sched.intervals.iter().map(|interval| interval.modes.iter().map(|mode| mode.duration).min().unwrap_or(interval.duration)),
        stop,
    ) {
        Ok(durations) => durations,
        Err(interrupted) => return (Err(interrupted), 0),
    };
    let bottom = match PrecedenceDag::compile(successors.clone(), stop).and_then(|dag| dag.remaining_paths(&base_durations, stop)) {
        Some(bottom) => bottom,
        None if stop.load(Ordering::Acquire) => return (Err(Interrupted), 0),
        None => return (Ok(None), 0),
    };
    let mut ready = Vec::new();
    for index in 0..n {
        if let Err(interrupted) = checkpoint(stop) {
            return (Err(interrupted), 0);
        }
        if present[index] && indegree[index] == 0 {
            ready.push(index);
        }
    }
    let mut chosen = match collect_interruptible(std::iter::repeat_n(0usize, n), stop) {
        Ok(chosen) => chosen,
        Err(interrupted) => return (Err(interrupted), 0),
    };
    let (mut durations, mut machines) = match mode_view(sched, &chosen, stop) {
        Ok(view) => view,
        Err(interrupted) => return (Err(interrupted), 0),
    };
    let mut starts = match collect_interruptible(std::iter::repeat_n(0i64, n), stop) {
        Ok(starts) => starts,
        Err(interrupted) => return (Err(interrupted), 0),
    };
    let mut scheduled = match collect_interruptible(present.iter().map(|value| !*value), stop) {
        Ok(scheduled) => scheduled,
        Err(interrupted) => return (Err(interrupted), 0),
    };
    let mut scheduled_count = 0usize;
    let mut candidates = 0u64;
    while !ready.is_empty() {
        if let Err(interrupted) = checkpoint(stop) {
            return (Err(interrupted), candidates);
        }
        let ready_position = match select_ready(&ready, rule, &base_durations, &bottom, &successors, seed, stop) {
            Ok(position) => position,
            Err(interrupted) => return (Err(interrupted), candidates),
        };
        let index = ready.swap_remove(ready_position);
        let mut earliest = 0;
        for &(before, after) in &sched.precedences {
            if let Err(interrupted) = checkpoint(stop) {
                return (Err(interrupted), candidates);
            }
            if after == index && present[before] {
                earliest = earliest.max(starts[before].saturating_add(durations[before]));
            }
        }
        let mode_count = sched.intervals[index].modes.len().max(1);
        let mut best: Option<(i64, i64, usize, i64, i64)> = None;
        for mode in 0..mode_count {
            if let Err(interrupted) = checkpoint(stop) {
                return (Err(interrupted), candidates);
            }
            let duration = iv_duration(&sched.intervals[index], mode);
            let machine = iv_machine(&sched.intervals[index], mode);
            let start_window = iv_start_window(&sched.intervals[index], mode);
            let start = match earliest_resource_start(
                sched,
                index,
                duration,
                machine,
                earliest,
                start_window,
                &durations,
                &machines,
                &starts,
                &present,
                &scheduled,
                &mut candidates,
                stop,
            ) {
                Ok(start) => start,
                Err(interrupted) => return (Err(interrupted), candidates),
            };
            let Some(start) = start else {
                continue;
            };
            let candidate = (start.saturating_add(duration), start, mode, duration, machine);
            if best.is_none_or(|current| candidate < current) {
                best = Some(candidate);
            }
        }
        let Some((_, start, mode, duration, machine)) = best else {
            return (Ok(None), candidates);
        };
        if let Err(interrupted) = checkpoint(stop) {
            return (Err(interrupted), candidates);
        }
        chosen[index] = mode;
        durations[index] = duration;
        machines[index] = machine;
        starts[index] = start;
        scheduled[index] = true;
        scheduled_count += 1;
        for &successor in &successors[index] {
            if let Err(interrupted) = checkpoint(stop) {
                return (Err(interrupted), candidates);
            }
            indegree[successor] -= 1;
            if indegree[successor] == 0 {
                ready.push(successor);
            }
        }
    }
    let mut required_count = 0usize;
    for &is_present in &present {
        if let Err(interrupted) = checkpoint(stop) {
            return (Err(interrupted), candidates);
        }
        required_count += usize::from(is_present);
    }
    if scheduled_count != required_count {
        return (Ok(None), candidates);
    }
    if let Err(interrupted) = checkpoint(stop) {
        return (Err(interrupted), candidates);
    }
    (Ok(Some(ConstructedSchedule { starts, chosen, present })), candidates)
}

fn construct_schedule(
    sched: &Schedule,
    seed: u64,
    stop: &AtomicBool,
    report: &mut dyn FnMut(i64),
) -> (Interruptible<Option<ConstructedIncumbent>>, u64, Option<Duration>) {
    let started = Instant::now();
    let mut candidates = 0u64;
    let mut first_feasible = None;
    let mut best: Option<ConstructedIncumbent> = None;
    let attempts = SGS_RULES
        .iter()
        .copied()
        .map(|rule| (rule, 0))
        .chain((0..3u64).map(|lane| (SgsRule::Seeded, mix64(seed ^ lane.wrapping_mul(0x9e37_79b9_7f4a_7c15)))));

    let interrupted_result = |best: Option<ConstructedIncumbent>, interrupted, candidates, first_feasible| {
        if let Some(incumbent) = best {
            (Ok(Some(incumbent)), candidates, first_feasible)
        } else {
            (Err(interrupted), candidates, first_feasible)
        }
    };

    for (rule, rule_seed) in attempts {
        if let Err(interrupted) = checkpoint(stop) {
            return interrupted_result(best, interrupted, candidates, first_feasible);
        }
        let (constructed, evaluated) = serial_schedule(sched, rule, rule_seed, stop);
        candidates = candidates.saturating_add(evaluated);
        let schedule = match constructed {
            Ok(Some(schedule)) => schedule,
            Ok(None) => continue,
            Err(interrupted) => return interrupted_result(best, interrupted, candidates, first_feasible),
        };
        let (durations, machines) = match mode_view(sched, &schedule.chosen, stop) {
            Ok(view) => view,
            Err(interrupted) => return interrupted_result(best, interrupted, candidates, first_feasible),
        };
        let score = match schedule_score(sched, &schedule.chosen, &durations, &machines, &schedule.starts, &schedule.present, stop) {
            Ok(score) if score.0 == 0 => score,
            Ok(_) => continue,
            Err(interrupted) => return interrupted_result(best, interrupted, candidates, first_feasible),
        };
        let solution = match snapshot_solution(sched, &schedule.chosen, &schedule.starts, &schedule.present, score, stop) {
            Ok(solution) => solution,
            Err(interrupted) => return interrupted_result(best, interrupted, candidates, first_feasible),
        };
        first_feasible.get_or_insert_with(|| started.elapsed());
        if best.as_ref().is_none_or(|incumbent| score < incumbent.score) {
            best = Some(ConstructedIncumbent { schedule, durations, machines, score, solution });
            report(score.1);
        }
    }

    (Ok(best), candidates, first_feasible)
}

/// Overlap (>= 0) of two intervals' time spans.
fn pair_overlap(starts: &[i64], dur: &[i64], present: &[bool], i: usize, j: usize) -> i64 {
    if !present[i] || !present[j] {
        return 0;
    }
    ((starts[i] + dur[i]).min(starts[j] + dur[j]) - starts[i].max(starts[j])).max(0)
}

fn schedule_score(
    sched: &Schedule,
    chosen: &[usize],
    dur: &[i64],
    mach: &[i64],
    starts: &[i64],
    present: &[bool],
    stop: &AtomicBool,
) -> Interruptible<(i64, i64)> {
    checkpoint(stop)?;
    let mut viol = 0i64;
    let mut makespan = 0i64;
    for (i, iv) in sched.intervals.iter().enumerate() {
        checkpoint(stop)?;
        if !present[i] {
            continue;
        }
        let end = starts[i].saturating_add(dur[i]);
        let (start_min, start_max) = iv_start_window(iv, chosen[i]);
        if starts[i] < start_min {
            viol = viol.saturating_add(start_min.saturating_sub(starts[i]));
        }
        if starts[i] > start_max {
            viol = viol.saturating_add(starts[i].saturating_sub(start_max));
        }
        makespan = makespan.max(end);
    }
    for &(a, b) in &sched.precedences {
        checkpoint(stop)?;
        if present[a] && present[b] {
            viol = viol.saturating_add((starts[a].saturating_add(dur[a]) - starts[b]).max(0));
        }
    }
    for res in &sched.resources {
        checkpoint(stop)?;
        match res {
            Resource::NoOverlap(ivs) => {
                for x in 0..ivs.len() {
                    checkpoint(stop)?;
                    for y in (x + 1)..ivs.len() {
                        checkpoint(stop)?;
                        viol = viol.saturating_add(pair_overlap(starts, dur, present, ivs[x], ivs[y]));
                    }
                }
            }
            Resource::MachineNoOverlap => {
                // Group moded intervals by their chosen machine; intervals on the
                // same machine may not overlap.
                let n = sched.intervals.len();
                for i in 0..n {
                    checkpoint(stop)?;
                    if mach[i] < 0 {
                        continue;
                    }
                    for j in (i + 1)..n {
                        checkpoint(stop)?;
                        if mach[i] == mach[j] {
                            viol = viol.saturating_add(pair_overlap(starts, dur, present, i, j));
                        }
                    }
                }
            }
            Resource::Cumulative { demands, capacity } => {
                viol = viol.saturating_add(cumulative_overload(demands, *capacity, dur, starts, present, stop)?);
            }
        }
    }
    checkpoint(stop)?;
    Ok((viol, makespan))
}

/// Total resource-overload area of a cumulative resource (usage above capacity,
/// integrated over time). Usage changes only at interval boundaries.
fn cumulative_overload(
    demands: &[(usize, i64)],
    capacity: i64,
    dur: &[i64],
    starts: &[i64],
    present: &[bool],
    stop: &AtomicBool,
) -> Interruptible<i64> {
    checkpoint(stop)?;
    let mut times: Vec<i64> = Vec::with_capacity(demands.len() * 2);
    for &(i, _) in demands {
        checkpoint(stop)?;
        if !present[i] {
            continue;
        }
        times.push(starts[i]);
        times.push(starts[i].saturating_add(dur[i]));
    }
    sort_dedup_interruptible(&mut times, stop)?;
    let mut total = 0i128;
    for w in times.windows(2) {
        checkpoint(stop)?;
        let (t0, t1) = (w[0], w[1]);
        let mut usage = 0i128;
        for &(i, demand) in demands {
            checkpoint(stop)?;
            if present[i] && starts[i] <= t0 && t0 < starts[i].saturating_add(dur[i]) {
                usage += i128::from(demand);
            }
        }
        let over = (usage - i128::from(capacity)).max(0);
        let width = i128::from(t1) - i128::from(t0);
        total = total.saturating_add(over.saturating_mul(width)).min(i128::from(i64::MAX));
    }
    checkpoint(stop)?;
    Ok(i64::try_from(total).expect("non-negative cumulative overload is capped at i64::MAX"))
}

/// Earliest start of each interval respecting precedence only (a longest-path
/// forward pass); resources are left to the search to fix.
fn earliest_starts(sched: &Schedule, chosen: &[usize], dur: &[i64], present: &[bool], stop: &AtomicBool) -> Interruptible<Vec<i64>> {
    let mut starts =
        collect_interruptible(sched.intervals.iter().zip(chosen).map(|(interval, &mode)| iv_start_window(interval, mode).0), stop)?;
    for _ in 0..sched.intervals.len() {
        checkpoint(stop)?;
        let mut changed = false;
        for &(a, b) in &sched.precedences {
            checkpoint(stop)?;
            if !present[a] || !present[b] {
                continue;
            }
            let need = starts[a].saturating_add(dur[a]);
            if need > starts[b] {
                starts[b] = need;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    checkpoint(stop)?;
    Ok(starts)
}

/// Candidate start times for shifting interval `i`: its precedence-earliest, the
/// ends of other intervals (for resource packing), and 0; clamped to its window.
fn schedule_candidates(
    sched: &Schedule,
    chosen: &[usize],
    dur: &[i64],
    starts: &[i64],
    present: &[bool],
    i: usize,
    stop: &AtomicBool,
) -> Interruptible<Vec<i64>> {
    checkpoint(stop)?;
    let (lo, hi) = iv_start_window(&sched.intervals[i], chosen[i]);
    let mut est = 0i64;
    for &(before, after) in &sched.precedences {
        checkpoint(stop)?;
        if after == i && present[before] && present[after] {
            est = est.max(starts[before] + dur[before]);
        }
    }
    let mut cands = vec![lo, est];
    for (j, &sj) in starts.iter().enumerate() {
        checkpoint(stop)?;
        if j != i && present[j] {
            cands.push(sj + dur[j]);
            cands.push(sj.saturating_sub(dur[i]));
        }
    }
    for c in &mut cands {
        checkpoint(stop)?;
        *c = (*c).clamp(lo, hi);
    }
    sort_dedup_interruptible(&mut cands, stop)?;
    checkpoint(stop)?;
    Ok(cands)
}

fn unknown_solution() -> CollectionSolution {
    CollectionSolution {
        lists: Vec::new(),
        objectives: Vec::new(),
        feasible: false,
        starts: Vec::new(),
        presences: Vec::new(),
        machines: Vec::new(),
        modes: Vec::new(),
        bound: None,
    }
}

fn snapshot_solution(
    sched: &Schedule,
    chosen: &[usize],
    starts: &[i64],
    present: &[bool],
    score: (i64, i64),
    stop: &AtomicBool,
) -> Interruptible<CollectionSolution> {
    checkpoint(stop)?;
    if score.0 != 0 {
        return Ok(unknown_solution());
    }

    let starts = collect_interruptible(starts.iter().copied(), stop)?;
    let presences = collect_interruptible(present.iter().copied(), stop)?;
    let machines = collect_interruptible(
        sched
            .intervals
            .iter()
            .zip(chosen)
            .zip(present)
            .map(|((interval, &mode), &is_present)| if is_present { iv_machine(interval, mode) } else { -1 }),
        stop,
    )?;
    let modes = mode_references(sched, chosen, present, stop)?;
    checkpoint(stop)?;
    let objectives = if sched.minimize_makespan { vec![score.1] } else { Vec::new() };
    checkpoint(stop)?;
    Ok(CollectionSolution { lists: Vec::new(), objectives, feasible: true, starts, presences, machines, modes, bound: None })
}

fn add_state_metrics(total: &mut ScheduleStateMetrics, metrics: ScheduleStateMetrics) {
    total.construction_candidates = total.construction_candidates.saturating_add(metrics.construction_candidates);
    total.reconstructions = total.reconstructions.saturating_add(metrics.reconstructions);
    total.moves_considered = total.moves_considered.saturating_add(metrics.moves_considered);
    total.moves_accepted = total.moves_accepted.saturating_add(metrics.moves_accepted);
    total.cycle_rejections = total.cycle_rejections.saturating_add(metrics.cycle_rejections);
    total.window_rejections = total.window_rejections.saturating_add(metrics.window_rejections);
    total.objective_rejections = total.objective_rejections.saturating_add(metrics.objective_rejections);
    total.critical_path_updates = total.critical_path_updates.saturating_add(metrics.critical_path_updates);
    total.delta_evaluations = total.delta_evaluations.saturating_add(metrics.delta_evaluations);
    total.full_evaluations = total.full_evaluations.saturating_add(metrics.full_evaluations);
    total.full_fallbacks = total.full_fallbacks.saturating_add(metrics.full_fallbacks);
    total.topological_rebuilds = total.topological_rebuilds.saturating_add(metrics.topological_rebuilds);
    total.oracle_validations = total.oracle_validations.saturating_add(metrics.oracle_validations);
    total.oracle_mismatches = total.oracle_mismatches.saturating_add(metrics.oracle_mismatches);
    total.dirty_cone_operations = total.dirty_cone_operations.saturating_add(metrics.dirty_cone_operations);
    total.max_dirty_cone = total.max_dirty_cone.max(metrics.max_dirty_cone);
    total.workspace_growths = total.workspace_growths.saturating_add(metrics.workspace_growths);
}

fn job_shop_result(
    best: Option<CollectionSolution>,
    construction_elapsed: Duration,
    first_feasible: Option<Duration>,
    metrics: ScheduleStateMetrics,
    incumbent_improvements: u64,
    incumbent_injections: u64,
) -> (CollectionSolution, ScheduleConstructionMetrics) {
    (
        best.unwrap_or_else(unknown_solution),
        ScheduleConstructionMetrics {
            elapsed: construction_elapsed,
            first_feasible,
            candidates: metrics.construction_candidates,
            work_steps: metrics.moves_considered,
            constructor: "giffler-thompson-critical-path",
            moves_considered: metrics.moves_considered,
            moves_accepted: metrics.moves_accepted,
            incumbent_improvements,
            incumbent_injections,
            cycle_rejections: metrics.cycle_rejections,
            window_rejections: metrics.window_rejections,
            objective_rejections: metrics.objective_rejections,
            reconstructions: metrics.reconstructions,
            critical_path_updates: metrics.critical_path_updates,
            delta_evaluations: metrics.delta_evaluations,
            full_evaluations: metrics.full_evaluations,
            full_fallbacks: metrics.full_fallbacks,
            topological_rebuilds: metrics.topological_rebuilds,
            oracle_validations: metrics.oracle_validations,
            oracle_mismatches: metrics.oracle_mismatches,
            dirty_cone_operations: metrics.dirty_cone_operations,
            max_dirty_cone: metrics.max_dirty_cone,
            workspace_growths: metrics.workspace_growths,
            workspace_rollbacks: 0,
            alns_generation_attempts: 0,
            alns_moves_generated: 0,
            resource_profile_checks: 0,
            resource_candidate_scheduling_attempts: 0,
            resource_event_visits: 0,
            resource_peak_profile_events: 0,
            precedence_rejections: 0,
            infeasible_rejections: 0,
            justification_attempts: 0,
        },
    )
}

fn generic_schedule_metrics(
    elapsed: Duration,
    first_feasible: Option<Duration>,
    construction_candidates: u64,
    moves_considered: u64,
    moves_accepted: u64,
    incumbent_improvements: u64,
) -> ScheduleConstructionMetrics {
    ScheduleConstructionMetrics {
        elapsed,
        first_feasible,
        candidates: construction_candidates,
        work_steps: moves_considered,
        constructor: "priority-sgs",
        moves_considered,
        moves_accepted,
        incumbent_improvements,
        incumbent_injections: 0,
        cycle_rejections: 0,
        window_rejections: 0,
        objective_rejections: 0,
        reconstructions: 0,
        critical_path_updates: 0,
        delta_evaluations: 0,
        full_evaluations: 0,
        full_fallbacks: 0,
        topological_rebuilds: 0,
        oracle_validations: 0,
        oracle_mismatches: 0,
        dirty_cone_operations: 0,
        max_dirty_cone: 0,
        workspace_growths: 0,
        workspace_rollbacks: 0,
        alns_generation_attempts: 0,
        alns_moves_generated: 0,
        resource_profile_checks: 0,
        resource_candidate_scheduling_attempts: 0,
        resource_event_visits: 0,
        resource_peak_profile_events: 0,
        precedence_rejections: 0,
        infeasible_rejections: 0,
        justification_attempts: 0,
    }
}

fn retain_job_shop_incumbent(
    state: &JobShopState,
    best_objective: &mut Option<i64>,
    best: &mut Option<CollectionSolution>,
    first_feasible: &mut Option<Duration>,
    started: Instant,
    report: &mut dyn FnMut(i64),
) -> bool {
    let objective = state.makespan();
    if best_objective.is_none_or(|current| objective < current) {
        *best_objective = Some(objective);
        *best = Some(state.to_solution());
        first_feasible.get_or_insert_with(|| started.elapsed());
        report(objective);
        true
    } else {
        false
    }
}

fn complete_schedule_incumbent(sched: &Schedule, incumbent: Option<&CollectionSolution>) -> Option<CollectionSolution> {
    incumbent
        .filter(|solution| {
            solution.feasible
                && solution.starts.len() == sched.intervals.len()
                && solution.presences.len() == sched.intervals.len()
                && solution.machines.len() == sched.intervals.len()
                && solution.modes.len() == sched.intervals.len()
                && (!sched.minimize_makespan || solution.objectives.len() == 1)
        })
        .cloned()
}

/// Structured search for mandatory fixed-assignment job shops. Recognition is
/// deliberately conservative; every unsupported schedule remains on the
/// generic SGS and interval-move fallback below.
fn solve_job_shop(
    sched: &Schedule,
    seed: u64,
    stop: &AtomicBool,
    max_iterations: u64,
    repeat_until_stopped: bool,
    initial_incumbent: Option<&CollectionSolution>,
    report: &mut dyn FnMut(i64),
) -> Option<(CollectionSolution, ScheduleConstructionMetrics)> {
    let started = Instant::now();
    let problem = match JobShopProblem::recognize(sched, stop) {
        Ok(Some(problem)) => problem,
        Ok(None) => return None,
        Err(_) => {
            return Some(job_shop_result(None, started.elapsed(), None, ScheduleStateMetrics::default(), 0, 0));
        }
    };
    let operation_count = problem.operation_count();
    let max_attempts = (operation_count / 8).clamp(8, 32);
    let default_moves = u64::try_from(operation_count).unwrap_or(u64::MAX).saturating_mul(256).clamp(2_048, 100_000);
    let max_moves = if max_iterations != u64::MAX {
        max_iterations
    } else if repeat_until_stopped {
        u64::MAX
    } else {
        default_moves
    };
    let mut construction_elapsed = Duration::ZERO;
    let mut total = ScheduleStateMetrics::default();
    let mut best = complete_schedule_incumbent(sched, initial_incumbent);
    let mut best_objective = best.as_ref().and_then(|solution| solution.objectives.first().copied());
    let mut first_feasible = None;
    let mut incumbent_improvements = 0u64;
    let mut incumbent_injections = 0u64;
    let mut moves_used = 0u64;
    let mut attempt = 0usize;
    let mut injected_machine_sequences = best.as_ref().map(|incumbent| {
        let mut sequences = vec![Vec::new(); problem.machine_count()];
        for operation in 0..problem.operation_count() {
            sequences[problem.machine(operation)].push(operation);
        }
        for sequence in &mut sequences {
            sequence.sort_by_key(|&operation| (incumbent.starts[operation], operation));
        }
        sequences
    });

    'attempts: loop {
        if checkpoint(stop).is_err() || moves_used >= max_moves || (attempt >= max_attempts && !repeat_until_stopped) {
            break;
        }
        let current_attempt = attempt;
        attempt = attempt.saturating_add(1);
        let rule = DispatchRule::ALL.get(current_attempt).copied().unwrap_or(DispatchRule::Randomized);
        let attempt_seed = mix64(seed ^ u64::try_from(current_attempt).unwrap_or(u64::MAX).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let construction_started = Instant::now();
        let injection_attempt = injected_machine_sequences.is_some();
        let (constructed, partial_metrics) = if let Some(machine_sequences) = injected_machine_sequences.take() {
            (JobShopState::from_machine_sequences(&problem, machine_sequences, stop), ScheduleStateMetrics::default())
        } else {
            JobShopState::giffler_thompson_profiled(&problem, attempt_seed, rule, stop)
        };
        let mut state = match constructed {
            Ok(Some(state)) => {
                incumbent_injections = incumbent_injections.saturating_add(u64::from(injection_attempt));
                state
            }
            Ok(None) => {
                construction_elapsed = construction_elapsed.saturating_add(construction_started.elapsed());
                add_state_metrics(&mut total, partial_metrics);
                continue;
            }
            Err(_) => {
                construction_elapsed = construction_elapsed.saturating_add(construction_started.elapsed());
                add_state_metrics(&mut total, partial_metrics);
                break;
            }
        };
        construction_elapsed = construction_elapsed.saturating_add(construction_started.elapsed());
        incumbent_improvements = incumbent_improvements.saturating_add(u64::from(retain_job_shop_incumbent(
            &state,
            &mut best_objective,
            &mut best,
            &mut first_feasible,
            started,
            report,
        )));
        if checkpoint(stop).is_err() {
            add_state_metrics(&mut total, state.metrics());
            break;
        }

        let neighborhoods = match current_attempt % 3 {
            0 => [CriticalNeighborhood::N5, CriticalNeighborhood::N1, CriticalNeighborhood::N6],
            1 => [CriticalNeighborhood::N6, CriticalNeighborhood::N5, CriticalNeighborhood::N1],
            _ => [CriticalNeighborhood::N1, CriticalNeighborhood::N6, CriticalNeighborhood::N5],
        };
        let mut kicks = 0u64;
        let move_capacity = operation_count.saturating_mul(3);
        let mut movements = Vec::with_capacity(move_capacity);
        loop {
            if checkpoint(stop).is_err() || moves_used >= max_moves {
                add_state_metrics(&mut total, state.metrics());
                break 'attempts;
            }
            let mut accepted = false;
            if state.fill_critical_move_union(&neighborhoods, &mut movements, stop).is_err() {
                add_state_metrics(&mut total, state.metrics());
                break 'attempts;
            }
            if !movements.is_empty() {
                let offset = usize::try_from(mix64(attempt_seed ^ moves_used)).unwrap_or(usize::MAX) % movements.len();
                movements.rotate_left(offset);
            }
            for &movement in &movements {
                if checkpoint(stop).is_err() || moves_used >= max_moves {
                    add_state_metrics(&mut total, state.metrics());
                    break 'attempts;
                }
                moves_used = moves_used.saturating_add(1);
                match state.consider_move(movement, MoveAcceptance::Improving, stop) {
                    Ok(MoveOutcome::Accepted { .. }) => {
                        accepted = true;
                        incumbent_improvements = incumbent_improvements.saturating_add(u64::from(retain_job_shop_incumbent(
                            &state,
                            &mut best_objective,
                            &mut best,
                            &mut first_feasible,
                            started,
                            report,
                        )));
                        break;
                    }
                    Ok(MoveOutcome::Rejected(_)) => {}
                    Err(_) => {
                        add_state_metrics(&mut total, state.metrics());
                        break 'attempts;
                    }
                }
            }
            if accepted {
                continue;
            }
            if kicks >= 3 {
                break;
            }

            if state.fill_critical_moves(CriticalNeighborhood::N6, &mut movements, stop).is_err() {
                add_state_metrics(&mut total, state.metrics());
                break 'attempts;
            }
            if movements.is_empty() {
                break;
            }
            let offset =
                usize::try_from(mix64(attempt_seed ^ kicks.wrapping_mul(0xd1b5_4a32_d192_ed03))).unwrap_or(usize::MAX) % movements.len();
            movements.rotate_left(offset);
            let mut kicked = false;
            for &movement in &movements {
                if checkpoint(stop).is_err() || moves_used >= max_moves {
                    add_state_metrics(&mut total, state.metrics());
                    break 'attempts;
                }
                moves_used = moves_used.saturating_add(1);
                match state.consider_move(movement, MoveAcceptance::Always, stop) {
                    Ok(MoveOutcome::Accepted { .. }) => {
                        kicked = true;
                        kicks = kicks.saturating_add(1);
                        incumbent_improvements = incumbent_improvements.saturating_add(u64::from(retain_job_shop_incumbent(
                            &state,
                            &mut best_objective,
                            &mut best,
                            &mut first_feasible,
                            started,
                            report,
                        )));
                        break;
                    }
                    Ok(MoveOutcome::Rejected(_)) => {}
                    Err(_) => {
                        add_state_metrics(&mut total, state.metrics());
                        break 'attempts;
                    }
                }
            }
            if !kicked {
                break;
            }
        }
        add_state_metrics(&mut total, state.metrics());
    }

    Some(job_shop_result(best, construction_elapsed, first_feasible, total, incumbent_improvements, incumbent_injections))
}

fn add_resource_metrics(total: &mut ResourceScheduleMetrics, metrics: ResourceScheduleMetrics) {
    total.serial_constructions = total.serial_constructions.saturating_add(metrics.serial_constructions);
    total.parallel_constructions = total.parallel_constructions.saturating_add(metrics.parallel_constructions);
    total.reconstructions = total.reconstructions.saturating_add(metrics.reconstructions);
    total.construction_candidates = total.construction_candidates.saturating_add(metrics.construction_candidates);
    total.candidate_scheduling_attempts = total.candidate_scheduling_attempts.saturating_add(metrics.candidate_scheduling_attempts);
    total.profile_checks = total.profile_checks.saturating_add(metrics.profile_checks);
    total.event_visits = total.event_visits.saturating_add(metrics.event_visits);
    total.peak_profile_events = total.peak_profile_events.max(metrics.peak_profile_events);
    total.moves_considered = total.moves_considered.saturating_add(metrics.moves_considered);
    total.moves_accepted = total.moves_accepted.saturating_add(metrics.moves_accepted);
    total.precedence_rejections = total.precedence_rejections.saturating_add(metrics.precedence_rejections);
    total.infeasible_rejections = total.infeasible_rejections.saturating_add(metrics.infeasible_rejections);
    total.objective_rejections = total.objective_rejections.saturating_add(metrics.objective_rejections);
    total.left_justifications = total.left_justifications.saturating_add(metrics.left_justifications);
    total.right_justifications = total.right_justifications.saturating_add(metrics.right_justifications);
    total.double_justifications = total.double_justifications.saturating_add(metrics.double_justifications);
    total.delta_evaluations = total.delta_evaluations.saturating_add(metrics.delta_evaluations);
    total.full_workspace_evaluations = total.full_workspace_evaluations.saturating_add(metrics.full_workspace_evaluations);
    total.delta_activities_rescheduled = total.delta_activities_rescheduled.saturating_add(metrics.delta_activities_rescheduled);
    total.workspace_rollbacks = total.workspace_rollbacks.saturating_add(metrics.workspace_rollbacks);
    total.oracle_validations = total.oracle_validations.saturating_add(metrics.oracle_validations);
    total.oracle_mismatches = total.oracle_mismatches.saturating_add(metrics.oracle_mismatches);
    total.alns_generation_attempts = total.alns_generation_attempts.saturating_add(metrics.alns_generation_attempts);
    total.alns_moves_generated = total.alns_moves_generated.saturating_add(metrics.alns_moves_generated);
    total.workspace_growths = total.workspace_growths.saturating_add(metrics.workspace_growths);
}

fn retain_resource_incumbent(
    state: &ResourceScheduleState,
    best_objective: &mut Option<i64>,
    best: &mut Option<CollectionSolution>,
    first_feasible: &mut Option<Duration>,
    started: Instant,
    report: &mut dyn FnMut(i64),
) -> bool {
    let objective = state.makespan();
    if best_objective.is_none_or(|current| objective < current) {
        *best_objective = Some(objective);
        *best = Some(state.to_solution());
        first_feasible.get_or_insert_with(|| started.elapsed());
        report(objective);
        true
    } else {
        false
    }
}

#[derive(Default)]
struct ResourceSearchSummary {
    metrics: ResourceScheduleMetrics,
    steps: u64,
    incumbent_improvements: u64,
    incumbent_injections: u64,
    justification_attempts: u64,
}

fn resource_schedule_result(
    best: Option<CollectionSolution>,
    construction_elapsed: Duration,
    first_feasible: Option<Duration>,
    summary: ResourceSearchSummary,
) -> (CollectionSolution, ScheduleConstructionMetrics) {
    let ResourceSearchSummary { metrics, steps, incumbent_improvements, incumbent_injections, justification_attempts } = summary;
    (
        best.unwrap_or_else(unknown_solution),
        ScheduleConstructionMetrics {
            elapsed: construction_elapsed,
            first_feasible,
            candidates: metrics.construction_candidates,
            work_steps: steps,
            constructor: "resource-priority-sgs",
            moves_considered: metrics.moves_considered,
            moves_accepted: metrics.moves_accepted,
            incumbent_improvements,
            incumbent_injections,
            cycle_rejections: 0,
            window_rejections: 0,
            objective_rejections: metrics.objective_rejections,
            reconstructions: metrics.reconstructions,
            critical_path_updates: 0,
            delta_evaluations: metrics.delta_evaluations,
            full_evaluations: metrics.full_workspace_evaluations.saturating_add(metrics.oracle_validations),
            full_fallbacks: metrics.full_workspace_evaluations,
            topological_rebuilds: 0,
            oracle_validations: metrics.oracle_validations,
            oracle_mismatches: metrics.oracle_mismatches,
            dirty_cone_operations: metrics.delta_activities_rescheduled,
            max_dirty_cone: 0,
            workspace_growths: metrics.workspace_growths,
            workspace_rollbacks: metrics.workspace_rollbacks,
            alns_generation_attempts: metrics.alns_generation_attempts,
            alns_moves_generated: metrics.alns_moves_generated,
            resource_profile_checks: metrics.profile_checks,
            resource_candidate_scheduling_attempts: metrics.candidate_scheduling_attempts,
            resource_event_visits: metrics.event_visits,
            resource_peak_profile_events: metrics.peak_profile_events,
            precedence_rejections: metrics.precedence_rejections,
            infeasible_rejections: metrics.infeasible_rejections,
            justification_attempts,
        },
    )
}

/// Structured search for mandatory fixed-duration RCPSP schedules. Every
/// priority move is decoded into a complete schedule before it can replace the
/// incumbent. Unsupported interval shapes continue through the generic path.
fn solve_resource_schedule(
    sched: &Schedule,
    seed: u64,
    stop: &AtomicBool,
    max_iterations: u64,
    repeat_until_stopped: bool,
    initial_incumbent: Option<&CollectionSolution>,
    report: &mut dyn FnMut(i64),
) -> Option<(CollectionSolution, ScheduleConstructionMetrics)> {
    let started = Instant::now();
    let problem = match ResourceScheduleProblem::recognize(sched, stop) {
        Ok(Some(problem)) => problem,
        Ok(None) => return None,
        Err(_) => {
            return Some(resource_schedule_result(None, started.elapsed(), None, ResourceSearchSummary::default()));
        }
    };
    let activity_count = problem.activity_count();
    let default_steps = u64::try_from(activity_count).unwrap_or(u64::MAX).saturating_mul(256).clamp(2_048, 100_000);
    let internal_limit = if max_iterations != u64::MAX {
        max_iterations
    } else if repeat_until_stopped {
        u64::MAX
    } else {
        default_steps
    };
    let max_attempts = PriorityRule::ALL.len().saturating_mul(2).saturating_add(6);
    let movement_batch = activity_count.saturating_mul(8).clamp(32, 1_024);
    let mut construction_elapsed = Duration::ZERO;
    let mut total = ResourceScheduleMetrics::default();
    let mut best = complete_schedule_incumbent(sched, initial_incumbent);
    let mut best_objective = best.as_ref().and_then(|solution| solution.objectives.first().copied());
    let mut first_feasible = None;
    let mut search_steps = 0u64;
    let mut incumbent_improvements = 0u64;
    let mut incumbent_injections = 0u64;
    let mut justification_attempts = 0u64;
    let mut attempt = 0usize;
    let mut injected_priority = best.as_ref().map(|incumbent| {
        let mut order = (0..problem.activity_count()).collect::<Vec<_>>();
        order.sort_by_key(|&activity| (incumbent.starts[activity], activity));
        order
    });

    'attempts: loop {
        if checkpoint(stop).is_err() || search_steps >= internal_limit || (attempt >= max_attempts && !repeat_until_stopped) {
            break;
        }
        let current_attempt = attempt;
        attempt = attempt.saturating_add(1);
        let rule = PriorityRule::ALL.get(current_attempt % PriorityRule::ALL.len()).copied().unwrap_or(PriorityRule::Randomized);
        let attempt_seed = mix64(seed ^ u64::try_from(current_attempt).unwrap_or(u64::MAX).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let construction_started = Instant::now();
        let injection_attempt = injected_priority.is_some();
        let priority_result = if let Some(order) = injected_priority.take() {
            PrioritySgs::compile(&problem, order, stop)
        } else {
            PrioritySgs::dispatch(&problem, attempt_seed, rule, stop)
        };
        let priority = match priority_result {
            Ok(Some(priority)) => priority,
            Ok(None) => {
                construction_elapsed = construction_elapsed.saturating_add(construction_started.elapsed());
                continue;
            }
            Err(_) => break,
        };
        let scheme = if current_attempt.is_multiple_of(2) { GenerationScheme::Serial } else { GenerationScheme::Parallel };
        let mut state = match ResourceScheduleState::construct(&problem, priority, scheme, stop) {
            Ok(Some(state)) => {
                incumbent_injections = incumbent_injections.saturating_add(u64::from(injection_attempt));
                state
            }
            Ok(None) => {
                construction_elapsed = construction_elapsed.saturating_add(construction_started.elapsed());
                continue;
            }
            Err(_) => break,
        };
        construction_elapsed = construction_elapsed.saturating_add(construction_started.elapsed());
        incumbent_improvements = incumbent_improvements.saturating_add(u64::from(retain_resource_incumbent(
            &state,
            &mut best_objective,
            &mut best,
            &mut first_feasible,
            started,
            report,
        )));

        let justifications = match current_attempt % 3 {
            0 => [Justification::Double, Justification::Left, Justification::Right],
            1 => [Justification::Right, Justification::Double, Justification::Left],
            _ => [Justification::Left, Justification::Right, Justification::Double],
        };
        let mut next_justification = 0usize;
        let mut kicks = 0u64;
        let alns_budget = ResourceAlnsBudget::bounded(activity_count);
        let mut movements = Vec::with_capacity(movement_batch);
        let mut alns_movements = Vec::with_capacity(alns_budget.max_moves);
        let mut stop_all = false;
        loop {
            if checkpoint(stop).is_err() || search_steps >= internal_limit {
                stop_all = true;
                break;
            }
            let neighborhood_offset = usize::try_from(mix64(attempt_seed ^ search_steps)).unwrap_or(usize::MAX);
            if state.fill_bounded_moves_from(neighborhood_offset, movement_batch, &mut movements, stop).is_err() {
                stop_all = true;
                break;
            }
            let mut accepted = false;
            for &movement in &movements {
                if checkpoint(stop).is_err() || search_steps >= internal_limit {
                    stop_all = true;
                    break;
                }
                search_steps = search_steps.saturating_add(1);
                match state.consider_move(movement, ResourceMoveAcceptance::Improving, stop) {
                    Ok(ResourceMoveOutcome::Accepted { .. }) => {
                        accepted = true;
                        next_justification = 0;
                        incumbent_improvements = incumbent_improvements.saturating_add(u64::from(retain_resource_incumbent(
                            &state,
                            &mut best_objective,
                            &mut best,
                            &mut first_feasible,
                            started,
                            report,
                        )));
                        break;
                    }
                    Ok(ResourceMoveOutcome::Rejected(_)) => {}
                    Err(_) => {
                        stop_all = true;
                        break;
                    }
                }
            }
            if stop_all {
                break;
            }
            if accepted {
                continue;
            }

            if next_justification < justifications.len() && search_steps < internal_limit {
                let kind = justifications[next_justification];
                next_justification += 1;
                search_steps = search_steps.saturating_add(1);
                justification_attempts = justification_attempts.saturating_add(1);
                match state.justify(kind, stop) {
                    Ok(Some(outcome)) => {
                        if outcome.changed {
                            incumbent_improvements = incumbent_improvements.saturating_add(u64::from(retain_resource_incumbent(
                                &state,
                                &mut best_objective,
                                &mut best,
                                &mut first_feasible,
                                started,
                                report,
                            )));
                            continue;
                        }
                    }
                    Ok(None) => {}
                    Err(_) => {
                        stop_all = true;
                        break;
                    }
                }
            }

            if kicks < 2 && search_steps < internal_limit {
                let alns_seed = mix64(attempt_seed ^ search_steps ^ kicks.wrapping_mul(0xa076_1d64_78bd_642f));
                if state.fill_alns_segment_moves(alns_seed, alns_budget, &mut alns_movements, stop).is_err() {
                    stop_all = true;
                    break;
                }
                for &movement in &alns_movements {
                    if checkpoint(stop).is_err() || search_steps >= internal_limit {
                        stop_all = true;
                        break;
                    }
                    search_steps = search_steps.saturating_add(1);
                    match state.consider_move(movement, ResourceMoveAcceptance::Improving, stop) {
                        Ok(ResourceMoveOutcome::Accepted { .. }) => {
                            accepted = true;
                            next_justification = 0;
                            incumbent_improvements = incumbent_improvements.saturating_add(u64::from(retain_resource_incumbent(
                                &state,
                                &mut best_objective,
                                &mut best,
                                &mut first_feasible,
                                started,
                                report,
                            )));
                            break;
                        }
                        Ok(ResourceMoveOutcome::Rejected(_)) => {}
                        Err(_) => {
                            stop_all = true;
                            break;
                        }
                    }
                }
                if stop_all {
                    break;
                }
                if accepted {
                    continue;
                }
                if let Some(&movement) = alns_movements.first() {
                    if search_steps >= internal_limit {
                        break;
                    }
                    search_steps = search_steps.saturating_add(1);
                    match state.consider_move(movement, ResourceMoveAcceptance::Always, stop) {
                        Ok(ResourceMoveOutcome::Accepted { .. }) => {
                            kicks = kicks.saturating_add(1);
                            next_justification = 0;
                            incumbent_improvements = incumbent_improvements.saturating_add(u64::from(retain_resource_incumbent(
                                &state,
                                &mut best_objective,
                                &mut best,
                                &mut first_feasible,
                                started,
                                report,
                            )));
                            continue;
                        }
                        Ok(ResourceMoveOutcome::Rejected(_)) => {}
                        Err(_) => {
                            stop_all = true;
                            break;
                        }
                    }
                }
            }

            if kicks >= 2 || search_steps >= internal_limit {
                break;
            }
            let kick_offset = usize::try_from(mix64(attempt_seed ^ kicks.wrapping_mul(0xd1b5_4a32_d192_ed03))).unwrap_or(usize::MAX);
            if state.fill_bounded_moves_from(kick_offset, movement_batch, &mut movements, stop).is_err() {
                stop_all = true;
                break;
            }
            if movements.is_empty() {
                break;
            }
            search_steps = search_steps.saturating_add(1);
            match state.consider_move(movements[0], ResourceMoveAcceptance::Always, stop) {
                Ok(ResourceMoveOutcome::Accepted { .. }) => {
                    kicks = kicks.saturating_add(1);
                    next_justification = 0;
                    incumbent_improvements = incumbent_improvements.saturating_add(u64::from(retain_resource_incumbent(
                        &state,
                        &mut best_objective,
                        &mut best,
                        &mut first_feasible,
                        started,
                        report,
                    )));
                }
                Ok(ResourceMoveOutcome::Rejected(_)) => break,
                Err(_) => {
                    stop_all = true;
                    break;
                }
            }
        }
        add_resource_metrics(&mut total, state.metrics());
        if stop_all {
            break 'attempts;
        }
    }

    Some(resource_schedule_result(
        best,
        construction_elapsed,
        first_feasible,
        ResourceSearchSummary { metrics: total, steps: search_steps, incumbent_improvements, incumbent_injections, justification_attempts },
    ))
}

/// Solve an interval-scheduling subproblem by local search. The decisions are
/// each interval's start time and, for flexible (moded) intervals, its mode
/// (which sets the duration and machine). Moves shift a start or switch a mode.
pub(crate) fn solve_schedule(
    sched: &Schedule,
    seed: u64,
    stop: &AtomicBool,
    report: &mut dyn FnMut(i64),
) -> (CollectionSolution, ScheduleConstructionMetrics) {
    solve_schedule_capped(sched, seed, stop, u64::MAX, false, None, report)
}

pub(crate) fn solve_schedule_capped(
    sched: &Schedule,
    seed: u64,
    stop: &AtomicBool,
    max_iterations: u64,
    repeat_until_stopped: bool,
    initial_incumbent: Option<&CollectionSolution>,
    report: &mut dyn FnMut(i64),
) -> (CollectionSolution, ScheduleConstructionMetrics) {
    if !matches!(machines_are_representable(sched, stop), Ok(true)) {
        return (unknown_solution(), generic_schedule_metrics(Duration::ZERO, None, 0, 0, 0, 0));
    }
    if let Some(result) = solve_job_shop(sched, seed, stop, max_iterations, repeat_until_stopped, initial_incumbent, report) {
        return result;
    }
    if let Some(result) = solve_resource_schedule(sched, seed, stop, max_iterations, repeat_until_stopped, initial_incumbent, report) {
        return result;
    }

    let construction_started = Instant::now();
    let n = sched.intervals.len();
    let injected = complete_schedule_incumbent(sched, initial_incumbent);
    let (constructed, candidates, constructed_first_feasible) = construct_schedule(sched, seed, stop, report);
    let construction_elapsed = construction_started.elapsed();
    let Ok(Some(ConstructedIncumbent {
        schedule: ConstructedSchedule { mut starts, mut chosen, mut present },
        durations: mut dur,
        machines: mut mach,
        score: mut cur,
        solution: mut best_solution,
    })) = constructed
    else {
        return (injected.unwrap_or_else(unknown_solution), generic_schedule_metrics(construction_elapsed, None, candidates, 0, 0, 0));
    };
    let first_feasible = best_solution.feasible.then_some(constructed_first_feasible.unwrap_or(construction_elapsed));
    let mut moves_considered = 0u64;
    let mut moves_accepted = 0u64;
    let mut incumbent_improvements = u64::from(first_feasible.is_some());
    let mut incumbent_score = cur;
    if let Some(injected) = injected {
        let injected_score = (0, injected.objectives.first().copied().unwrap_or_default());
        if injected_score < incumbent_score {
            incumbent_score = injected_score;
            best_solution = injected;
        }
    }

    // Large schedules stay on the compact constructor. The pairwise shift
    // engine is retained only for small instances until the critical-path
    // neighbourhoods replace it.
    if n > 48 || checkpoint(stop).is_err() {
        return (
            best_solution,
            generic_schedule_metrics(
                construction_elapsed,
                first_feasible,
                candidates,
                moves_considered,
                moves_accepted,
                incumbent_improvements,
            ),
        );
    }

    const RESTART_AFTER: u64 = 25;
    let mut search_best = cur;
    let mut since_improve = 0u64;
    let mut iter = 0u64;

    'search: loop {
        if checkpoint(stop).is_err() || moves_considered >= max_iterations {
            break;
        }
        iter += 1;
        let mut moved = false;
        'scan: for i in 0..n {
            if checkpoint(stop).is_err() {
                break 'search;
            }

            // (a) Shift the start under the current mode.
            let candidates = match schedule_candidates(sched, &chosen, &dur, &starts, &present, i, stop) {
                Ok(candidates) => candidates,
                Err(_) => break 'search,
            };
            for t in candidates {
                if checkpoint(stop).is_err() {
                    break 'search;
                }
                if t == starts[i] {
                    continue;
                }
                if moves_considered >= max_iterations {
                    break 'search;
                }
                moves_considered = moves_considered.saturating_add(1);
                let old = starts[i];
                starts[i] = t;
                let trial = match schedule_score(sched, &chosen, &dur, &mach, &starts, &present, stop) {
                    Ok(trial) => trial,
                    Err(_) => {
                        starts[i] = old;
                        break 'search;
                    }
                };
                if trial < cur {
                    if trial < incumbent_score {
                        let snapshot = match snapshot_solution(sched, &chosen, &starts, &present, trial, stop) {
                            Ok(snapshot) => snapshot,
                            Err(_) => {
                                starts[i] = old;
                                break 'search;
                            }
                        };
                        incumbent_score = trial;
                        best_solution = snapshot;
                        incumbent_improvements = incumbent_improvements.saturating_add(1);
                    }
                    cur = trial;
                    moves_accepted = moves_accepted.saturating_add(1);
                    moved = true;
                    break 'scan;
                }
                starts[i] = old;
            }

            // (b) Switch this interval's mode (machine / duration).
            let modes = sched.intervals[i].modes.len();
            for m in 0..modes {
                if checkpoint(stop).is_err() {
                    break 'search;
                }
                if m == chosen[i] {
                    continue;
                }
                if moves_considered >= max_iterations {
                    break 'search;
                }
                moves_considered = moves_considered.saturating_add(1);
                let (old_mode, old_duration, old_machine, old_start) = (chosen[i], dur[i], mach[i], starts[i]);
                chosen[i] = m;
                dur[i] = iv_duration(&sched.intervals[i], m);
                mach[i] = iv_machine(&sched.intervals[i], m);
                let (start_min, start_max) = iv_start_window(&sched.intervals[i], m);
                starts[i] = old_start.clamp(start_min, start_max);
                let trial = match schedule_score(sched, &chosen, &dur, &mach, &starts, &present, stop) {
                    Ok(trial) => trial,
                    Err(_) => {
                        chosen[i] = old_mode;
                        dur[i] = old_duration;
                        mach[i] = old_machine;
                        starts[i] = old_start;
                        break 'search;
                    }
                };
                if trial < cur {
                    if trial < incumbent_score {
                        let snapshot = match snapshot_solution(sched, &chosen, &starts, &present, trial, stop) {
                            Ok(snapshot) => snapshot,
                            Err(_) => {
                                chosen[i] = old_mode;
                                dur[i] = old_duration;
                                mach[i] = old_machine;
                                starts[i] = old_start;
                                break 'search;
                            }
                        };
                        incumbent_score = trial;
                        best_solution = snapshot;
                        incumbent_improvements = incumbent_improvements.saturating_add(1);
                    }
                    cur = trial;
                    moves_accepted = moves_accepted.saturating_add(1);
                    moved = true;
                    break 'scan;
                }
                chosen[i] = old_mode;
                dur[i] = old_duration;
                mach[i] = old_machine;
                starts[i] = old_start;
            }

            if sched.intervals[i].optional {
                if checkpoint(stop).is_err() || moves_considered >= max_iterations {
                    break 'search;
                }
                moves_considered = moves_considered.saturating_add(1);
                present[i] = !present[i];
                let trial = match schedule_score(sched, &chosen, &dur, &mach, &starts, &present, stop) {
                    Ok(trial) => trial,
                    Err(_) => {
                        present[i] = !present[i];
                        break 'search;
                    }
                };
                if trial < cur {
                    if trial < incumbent_score {
                        let snapshot = match snapshot_solution(sched, &chosen, &starts, &present, trial, stop) {
                            Ok(snapshot) => snapshot,
                            Err(_) => {
                                present[i] = !present[i];
                                break 'search;
                            }
                        };
                        incumbent_score = trial;
                        best_solution = snapshot;
                        incumbent_improvements = incumbent_improvements.saturating_add(1);
                    }
                    cur = trial;
                    moves_accepted = moves_accepted.saturating_add(1);
                    moved = true;
                    break 'scan;
                }
                present[i] = !present[i];
            }
        }
        if moved {
            continue;
        }

        // Local optimum: record, then kick or restart.
        if cur < search_best {
            search_best = cur;
            if cur.0 == 0 && checkpoint(stop).is_ok() {
                report(cur.1);
            }
            since_improve = 0;
        } else {
            since_improve += 1;
        }
        if moves_considered >= max_iterations {
            break;
        }
        moves_considered = moves_considered.saturating_add(1);
        if since_improve >= RESTART_AFTER {
            // Random modes, then precedence-earliest starts with jitter.
            for (i, selected) in chosen.iter_mut().enumerate() {
                if checkpoint(stop).is_err() {
                    break 'search;
                }
                let modes = sched.intervals[i].modes.len().max(1);
                *selected = (mix64(seed ^ mix64(iter ^ 0x9e37).wrapping_add(i as u64)) % modes as u64) as usize;
            }
            (dur, mach) = match mode_view(sched, &chosen, stop) {
                Ok(view) => view,
                Err(_) => break 'search,
            };
            for (i, interval) in sched.intervals.iter().enumerate() {
                if checkpoint(stop).is_err() {
                    break 'search;
                }
                present[i] = !interval.optional || mix64(seed ^ iter.wrapping_add(i as u64)) & 1 == 1;
            }
            starts = match earliest_starts(sched, &chosen, &dur, &present, stop) {
                Ok(starts) => starts,
                Err(_) => break 'search,
            };
            for (i, start) in starts.iter_mut().enumerate() {
                if checkpoint(stop).is_err() {
                    break 'search;
                }
                let window = iv_start_window(&sched.intervals[i], chosen[i]);
                *start = random_start(window, mix64(seed ^ mix64(iter).wrapping_add(i as u64)));
            }
            since_improve = 0;
        } else if n > 0 {
            if checkpoint(stop).is_err() {
                break;
            }
            let i = (mix64(seed ^ mix64(iter)) % n as u64) as usize;
            starts[i] = random_start(iv_start_window(&sched.intervals[i], chosen[i]), mix64(seed ^ mix64(iter ^ 0x5151)));
        }
        cur = match schedule_score(sched, &chosen, &dur, &mach, &starts, &present, stop) {
            Ok(score) => score,
            Err(_) => break,
        };
        moves_accepted = moves_accepted.saturating_add(1);
        if cur < incumbent_score {
            let snapshot = match snapshot_solution(sched, &chosen, &starts, &present, cur, stop) {
                Ok(snapshot) => snapshot,
                Err(_) => break,
            };
            incumbent_score = cur;
            best_solution = snapshot;
            incumbent_improvements = incumbent_improvements.saturating_add(1);
        }
    }

    (
        best_solution,
        generic_schedule_metrics(
            construction_elapsed,
            first_feasible,
            candidates,
            moves_considered,
            moves_accepted,
            incumbent_improvements,
        ),
    )
}
