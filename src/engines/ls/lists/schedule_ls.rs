use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::mix64;
use crate::model::list::{CollectionSolution, IntervalVar, Resource, Schedule};

/// Construction-only measurements shared with the collection profiler.
pub(super) struct ScheduleConstructionMetrics {
    pub(super) elapsed: Duration,
    pub(super) first_feasible: Option<Duration>,
    pub(super) candidates: u64,
}

struct ConstructedSchedule {
    starts: Vec<i64>,
    chosen: Vec<usize>,
    present: Vec<bool>,
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
        iv.modes[mode].machine as i64
    }
}

/// Durations and machines implied by the chosen mode of each interval.
fn mode_view(sched: &Schedule, chosen: &[usize]) -> (Vec<i64>, Vec<i64>) {
    let dur = sched.intervals.iter().zip(chosen).map(|(iv, &m)| iv_duration(iv, m)).collect();
    let mach = sched.intervals.iter().zip(chosen).map(|(iv, &m)| iv_machine(iv, m)).collect();
    (dur, mach)
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
    durations: &[i64],
    machines: &[i64],
    starts: &[i64],
    present: &[bool],
    scheduled: &[bool],
    candidates: &mut u64,
) -> Option<i64> {
    let latest = sched.intervals[index].horizon.saturating_sub(duration);
    let mut start = earliest.max(0);
    while start <= latest {
        *candidates = candidates.saturating_add(1);
        let end = start.saturating_add(duration);
        let mut advance: Option<i64> = None;
        for resource in &sched.resources {
            match resource {
                Resource::NoOverlap(group) if group.contains(&index) => {
                    for &other in group {
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
                    let demand = demands.iter().find_map(|&(task, demand)| (task == index).then_some(demand)).unwrap_or(0);
                    if demand > *capacity {
                        return None;
                    }
                    if demand == 0 {
                        continue;
                    }
                    let mut events = vec![start];
                    for &(other, _) in demands {
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
                    events.sort_unstable();
                    events.dedup();
                    for event in events {
                        let mut usage = demand;
                        let mut next_end: Option<i64> = None;
                        for &(other, other_demand) in demands {
                            if other != index
                                && present[other]
                                && scheduled[other]
                                && starts[other] <= event
                                && event < starts[other].saturating_add(durations[other])
                            {
                                usage = usage.saturating_add(other_demand);
                                let other_end = starts[other].saturating_add(durations[other]);
                                next_end = Some(next_end.map_or(other_end, |old| old.min(other_end)));
                            }
                        }
                        if usage > *capacity {
                            advance = Some(advance.map_or(next_end.unwrap_or(end), |old| old.max(next_end.unwrap_or(end))));
                            break;
                        }
                    }
                }
                Resource::NoOverlap(_) | Resource::MachineNoOverlap => {}
            }
        }
        let Some(next) = advance else {
            return Some(start);
        };
        start = next.max(start.saturating_add(1));
    }
    None
}

/// Serial schedule generation scheme. Eligible operations are taken in stable
/// topological order and placed at their earliest resource-feasible event. For
/// flexible operations, the mode with the earliest completion is selected.
fn serial_schedule(sched: &Schedule, stop: &AtomicBool) -> (Option<ConstructedSchedule>, u64) {
    let n = sched.intervals.len();
    let present: Vec<bool> = sched.intervals.iter().map(|interval| !interval.optional).collect();
    let mut indegree = vec![0usize; n];
    let mut successors = vec![Vec::new(); n];
    for &(before, after) in &sched.precedences {
        if present[before] && present[after] {
            indegree[after] = indegree[after].saturating_add(1);
            successors[before].push(after);
        }
    }
    let mut ready = BinaryHeap::new();
    for index in 0..n {
        if present[index] && indegree[index] == 0 {
            ready.push(Reverse(index));
        }
    }
    let mut chosen = vec![0usize; n];
    let (mut durations, mut machines) = mode_view(sched, &chosen);
    let mut starts = vec![0i64; n];
    let mut scheduled = present.iter().map(|value| !*value).collect::<Vec<_>>();
    let mut scheduled_count = 0usize;
    let mut candidates = 0u64;
    while let Some(Reverse(index)) = ready.pop() {
        if stop.load(Ordering::Relaxed) {
            return (None, candidates);
        }
        let earliest = sched
            .precedences
            .iter()
            .filter_map(|&(before, after)| (after == index && present[before]).then_some(starts[before].saturating_add(durations[before])))
            .max()
            .unwrap_or(0);
        let mode_count = sched.intervals[index].modes.len().max(1);
        let mut best: Option<(i64, i64, usize, i64, i64)> = None;
        for mode in 0..mode_count {
            let duration = iv_duration(&sched.intervals[index], mode);
            let machine = iv_machine(&sched.intervals[index], mode);
            let Some(start) = earliest_resource_start(
                sched,
                index,
                duration,
                machine,
                earliest,
                &durations,
                &machines,
                &starts,
                &present,
                &scheduled,
                &mut candidates,
            ) else {
                continue;
            };
            let candidate = (start.saturating_add(duration), start, mode, duration, machine);
            if best.is_none_or(|current| candidate < current) {
                best = Some(candidate);
            }
        }
        let Some((_, start, mode, duration, machine)) = best else {
            return (None, candidates);
        };
        chosen[index] = mode;
        durations[index] = duration;
        machines[index] = machine;
        starts[index] = start;
        scheduled[index] = true;
        scheduled_count += 1;
        for &successor in &successors[index] {
            indegree[successor] -= 1;
            if indegree[successor] == 0 {
                ready.push(Reverse(successor));
            }
        }
    }
    if scheduled_count != present.iter().filter(|&&value| value).count() {
        return (None, candidates);
    }
    (Some(ConstructedSchedule { starts, chosen, present }), candidates)
}

/// Overlap (>= 0) of two intervals' time spans.
fn pair_overlap(starts: &[i64], dur: &[i64], present: &[bool], i: usize, j: usize) -> i64 {
    if !present[i] || !present[j] {
        return 0;
    }
    ((starts[i] + dur[i]).min(starts[j] + dur[j]) - starts[i].max(starts[j])).max(0)
}

fn schedule_score(sched: &Schedule, dur: &[i64], mach: &[i64], starts: &[i64], present: &[bool]) -> (i64, i64) {
    let mut viol = 0i64;
    let mut makespan = 0i64;
    for (i, iv) in sched.intervals.iter().enumerate() {
        if !present[i] {
            continue;
        }
        let end = starts[i].saturating_add(dur[i]);
        if starts[i] < 0 {
            viol = viol.saturating_add(-starts[i]);
        }
        if end > iv.horizon {
            viol = viol.saturating_add(end - iv.horizon);
        }
        makespan = makespan.max(end);
    }
    for &(a, b) in &sched.precedences {
        if present[a] && present[b] {
            viol = viol.saturating_add((starts[a].saturating_add(dur[a]) - starts[b]).max(0));
        }
    }
    for res in &sched.resources {
        match res {
            Resource::NoOverlap(ivs) => {
                for x in 0..ivs.len() {
                    for y in (x + 1)..ivs.len() {
                        viol = viol.saturating_add(pair_overlap(starts, dur, present, ivs[x], ivs[y]));
                    }
                }
            }
            Resource::MachineNoOverlap => {
                // Group moded intervals by their chosen machine; intervals on the
                // same machine may not overlap.
                let n = sched.intervals.len();
                for i in 0..n {
                    if mach[i] < 0 {
                        continue;
                    }
                    for j in (i + 1)..n {
                        if mach[i] == mach[j] {
                            viol = viol.saturating_add(pair_overlap(starts, dur, present, i, j));
                        }
                    }
                }
            }
            Resource::Cumulative { demands, capacity } => {
                viol = viol.saturating_add(cumulative_overload(demands, *capacity, dur, starts, present));
            }
        }
    }
    (viol, makespan)
}

/// Total resource-overload area of a cumulative resource (usage above capacity,
/// integrated over time). Usage changes only at interval boundaries.
fn cumulative_overload(demands: &[(usize, i64)], capacity: i64, dur: &[i64], starts: &[i64], present: &[bool]) -> i64 {
    let mut times: Vec<i64> = Vec::with_capacity(demands.len() * 2);
    for &(i, _) in demands {
        if !present[i] {
            continue;
        }
        times.push(starts[i]);
        times.push(starts[i].saturating_add(dur[i]));
    }
    times.sort_unstable();
    times.dedup();
    let mut total = 0i64;
    for w in times.windows(2) {
        let (t0, t1) = (w[0], w[1]);
        let usage: i64 = demands.iter().filter(|&&(i, _)| present[i] && starts[i] <= t0 && t0 < starts[i] + dur[i]).map(|&(_, d)| d).sum();
        let over = (usage - capacity).max(0);
        total = total.saturating_add(over.saturating_mul(t1 - t0));
    }
    total
}

/// Earliest start of each interval respecting precedence only (a longest-path
/// forward pass); resources are left to the search to fix.
fn earliest_starts(sched: &Schedule, dur: &[i64], present: &[bool]) -> Vec<i64> {
    let mut s = vec![0i64; sched.intervals.len()];
    for _ in 0..sched.intervals.len() {
        let mut changed = false;
        for &(a, b) in &sched.precedences {
            if !present[a] || !present[b] {
                continue;
            }
            let need = s[a].saturating_add(dur[a]);
            if need > s[b] {
                s[b] = need;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    s
}

/// Candidate start times for shifting interval `i`: its precedence-earliest, the
/// ends of other intervals (for resource packing), and 0; clamped to its window.
fn schedule_candidates(sched: &Schedule, dur: &[i64], starts: &[i64], present: &[bool], i: usize) -> Vec<i64> {
    let hi = (sched.intervals[i].horizon - dur[i]).max(0);
    let est =
        sched.precedences.iter().filter(|&&(a, b)| b == i && present[a] && present[b]).map(|&(a, _)| starts[a] + dur[a]).max().unwrap_or(0);
    let mut cands = vec![0, est];
    for (j, &sj) in starts.iter().enumerate() {
        if j != i && present[j] {
            cands.push(sj + dur[j]);
            cands.push((sj - dur[i]).max(0));
        }
    }
    for c in &mut cands {
        *c = (*c).clamp(0, hi);
    }
    cands.sort_unstable();
    cands.dedup();
    cands
}

/// Solve an interval-scheduling subproblem by local search. The decisions are
/// each interval's start time and, for flexible (moded) intervals, its mode
/// (which sets the duration and machine). Moves shift a start or switch a mode.
pub(super) fn solve_schedule(
    sched: &Schedule,
    seed: u64,
    stop: &AtomicBool,
    report: &mut dyn FnMut(i64),
) -> (CollectionSolution, ScheduleConstructionMetrics) {
    let construction_started = Instant::now();
    let n = sched.intervals.len();
    let (constructed, candidates) = serial_schedule(sched, stop);
    let construction_elapsed = construction_started.elapsed();
    let Some(ConstructedSchedule { mut starts, mut chosen, mut present }) = constructed else {
        return (
            CollectionSolution {
                lists: Vec::new(),
                objectives: Vec::new(),
                feasible: false,
                starts: Vec::new(),
                presences: Vec::new(),
                machines: Vec::new(),
                bound: None,
            },
            ScheduleConstructionMetrics { elapsed: construction_elapsed, first_feasible: None, candidates },
        );
    };
    let (mut dur, mut mach) = mode_view(sched, &chosen);
    let mut cur = schedule_score(sched, &dur, &mach, &starts, &present);
    let mut best_starts = starts.clone();
    let mut best_chosen = chosen.clone();
    let mut best_present = present.clone();
    let mut best = cur;
    if best.0 == 0 {
        report(best.1);
    }
    let first_feasible = (best.0 == 0).then_some(construction_elapsed);
    // Phase 9 deliberately keeps large schedules on the compact constructor.
    // The pairwise shift engine is retained only for small instances until the
    // critical-path neighbourhoods of Phase 13 replace it.
    if n > 48 || stop.load(Ordering::Relaxed) {
        let machines: Vec<i64> = sched
            .intervals
            .iter()
            .zip(&best_chosen)
            .zip(&best_present)
            .map(|((iv, &mode), &is_present)| if is_present { iv_machine(iv, mode) } else { -1 })
            .collect();
        return (
            CollectionSolution {
                lists: Vec::new(),
                objectives: if best.0 == 0 && sched.minimize_makespan { vec![best.1] } else { Vec::new() },
                feasible: best.0 == 0,
                starts: if best.0 == 0 { best_starts } else { Vec::new() },
                presences: if best.0 == 0 { best_present } else { Vec::new() },
                machines: if best.0 == 0 { machines } else { Vec::new() },
                bound: None,
            },
            ScheduleConstructionMetrics { elapsed: construction_elapsed, first_feasible, candidates },
        );
    }
    const RESTART_AFTER: u64 = 25;
    let mut since_improve = 0u64;
    let mut iter = 0u64;

    while !stop.load(Ordering::Relaxed) {
        iter += 1;
        let mut moved = false;
        'scan: for i in 0..n {
            // (a) Shift the start under the current mode.
            for t in schedule_candidates(sched, &dur, &starts, &present, i) {
                if t == starts[i] {
                    continue;
                }
                let old = starts[i];
                starts[i] = t;
                let trial = schedule_score(sched, &dur, &mach, &starts, &present);
                if trial < cur {
                    cur = trial;
                    moved = true;
                    break 'scan;
                }
                starts[i] = old;
            }
            // (b) Switch this interval's mode (machine / duration).
            let modes = sched.intervals[i].modes.len();
            for m in 0..modes {
                if m == chosen[i] {
                    continue;
                }
                let (om, od, oma, os) = (chosen[i], dur[i], mach[i], starts[i]);
                chosen[i] = m;
                dur[i] = iv_duration(&sched.intervals[i], m);
                mach[i] = iv_machine(&sched.intervals[i], m);
                starts[i] = os.clamp(0, (sched.intervals[i].horizon - dur[i]).max(0));
                let trial = schedule_score(sched, &dur, &mach, &starts, &present);
                if trial < cur {
                    cur = trial;
                    moved = true;
                    break 'scan;
                }
                chosen[i] = om;
                dur[i] = od;
                mach[i] = oma;
                starts[i] = os;
            }
            if sched.intervals[i].optional {
                present[i] = !present[i];
                let trial = schedule_score(sched, &dur, &mach, &starts, &present);
                if trial < cur {
                    cur = trial;
                    moved = true;
                    break 'scan;
                }
                present[i] = !present[i];
            }
            if i % 64 == 0 && stop.load(Ordering::Relaxed) {
                break;
            }
        }
        if moved {
            continue;
        }
        // Local optimum: record, then kick or restart.
        if cur < best {
            best = cur;
            best_starts = starts.clone();
            best_chosen = chosen.clone();
            best_present = present.clone();
            if cur.0 == 0 {
                report(cur.1);
            }
            since_improve = 0;
        } else {
            since_improve += 1;
        }
        if since_improve >= RESTART_AFTER {
            // Random modes, then precedence-earliest starts with jitter.
            for (i, c) in chosen.iter_mut().enumerate() {
                let modes = sched.intervals[i].modes.len().max(1);
                *c = (mix64(seed ^ mix64(iter ^ 0x9e37).wrapping_add(i as u64)) % modes as u64) as usize;
            }
            (dur, mach) = mode_view(sched, &chosen);
            for (i, interval) in sched.intervals.iter().enumerate() {
                present[i] = !interval.optional || mix64(seed ^ iter.wrapping_add(i as u64)) & 1 == 1;
            }
            starts = earliest_starts(sched, &dur, &present);
            for (i, s) in starts.iter_mut().enumerate() {
                let hi = (sched.intervals[i].horizon - dur[i]).max(0);
                if hi > 0 {
                    *s = (mix64(seed ^ mix64(iter).wrapping_add(i as u64)) % (hi as u64 + 1)) as i64;
                }
            }
            since_improve = 0;
        } else if n > 0 {
            let i = (mix64(seed ^ mix64(iter)) % n as u64) as usize;
            let hi = (sched.intervals[i].horizon - dur[i]).max(0);
            if hi > 0 {
                starts[i] = (mix64(seed ^ mix64(iter ^ 0x5151)) % (hi as u64 + 1)) as i64;
            }
        }
        cur = schedule_score(sched, &dur, &mach, &starts, &present);
    }

    if cur < best {
        best = cur;
        best_starts = starts.clone();
        best_chosen = chosen.clone();
        best_present = present.clone();
    }
    let feasible = best.0 == 0;
    let machines: Vec<i64> = sched
        .intervals
        .iter()
        .zip(&best_chosen)
        .zip(&best_present)
        .map(|((iv, &m), &present)| if present { iv_machine(iv, m) } else { -1 })
        .collect();
    (
        CollectionSolution {
            lists: Vec::new(),
            objectives: if feasible && sched.minimize_makespan { vec![best.1] } else { Vec::new() },
            feasible,
            starts: if feasible { best_starts } else { Vec::new() },
            presences: if feasible { best_present } else { Vec::new() },
            machines: if feasible { machines } else { Vec::new() },
            bound: None,
        },
        ScheduleConstructionMetrics { elapsed: construction_elapsed, first_feasible, candidates },
    )
}
