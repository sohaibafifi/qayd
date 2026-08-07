use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::mix64;
use crate::model::list::{CollectionSolution, IntervalVar, Resource, Schedule};

/// Construction-only measurements shared with the collection profiler.
pub(crate) struct ScheduleConstructionMetrics {
    pub(crate) elapsed: Duration,
    pub(crate) first_feasible: Option<Duration>,
    pub(crate) candidates: u64,
}

struct ConstructedSchedule {
    starts: Vec<i64>,
    chosen: Vec<usize>,
    present: Vec<bool>,
}

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
        iv.modes[mode].machine as i64
    }
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
                        let mut usage = demand;
                        let mut next_end: Option<i64> = None;
                        for &(other, other_demand) in demands {
                            checkpoint(stop)?;
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

/// Serial schedule generation scheme. Eligible operations are taken in stable
/// topological order and placed at their earliest resource-feasible event. For
/// flexible operations, the mode with the earliest completion is selected.
fn serial_schedule(sched: &Schedule, stop: &AtomicBool) -> (Interruptible<Option<ConstructedSchedule>>, u64) {
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
            indegree[after] = indegree[after].saturating_add(1);
            successors[before].push(after);
        }
    }
    let mut ready = BinaryHeap::new();
    for index in 0..n {
        if let Err(interrupted) = checkpoint(stop) {
            return (Err(interrupted), 0);
        }
        if present[index] && indegree[index] == 0 {
            ready.push(Reverse(index));
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
    while let Some(Reverse(index)) = ready.pop() {
        if let Err(interrupted) = checkpoint(stop) {
            return (Err(interrupted), candidates);
        }
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
                ready.push(Reverse(successor));
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
    let mut total = 0i64;
    for w in times.windows(2) {
        checkpoint(stop)?;
        let (t0, t1) = (w[0], w[1]);
        let mut usage = 0i64;
        for &(i, demand) in demands {
            checkpoint(stop)?;
            if present[i] && starts[i] <= t0 && t0 < starts[i] + dur[i] {
                usage += demand;
            }
        }
        let over = (usage - capacity).max(0);
        total = total.saturating_add(over.saturating_mul(t1 - t0));
    }
    checkpoint(stop)?;
    Ok(total)
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

/// Solve an interval-scheduling subproblem by local search. The decisions are
/// each interval's start time and, for flexible (moded) intervals, its mode
/// (which sets the duration and machine). Moves shift a start or switch a mode.
pub(crate) fn solve_schedule(
    sched: &Schedule,
    seed: u64,
    stop: &AtomicBool,
    report: &mut dyn FnMut(i64),
) -> (CollectionSolution, ScheduleConstructionMetrics) {
    let construction_started = Instant::now();
    let n = sched.intervals.len();
    let (constructed, candidates) = serial_schedule(sched, stop);
    let construction_elapsed = construction_started.elapsed();
    let metrics = |first_feasible| ScheduleConstructionMetrics { elapsed: construction_elapsed, first_feasible, candidates };
    let Ok(Some(ConstructedSchedule { mut starts, mut chosen, mut present })) = constructed else {
        return (unknown_solution(), metrics(None));
    };
    let (mut dur, mut mach) = match mode_view(sched, &chosen, stop) {
        Ok(view) => view,
        Err(_) => return (unknown_solution(), metrics(None)),
    };
    let mut cur = match schedule_score(sched, &chosen, &dur, &mach, &starts, &present, stop) {
        Ok(score) => score,
        Err(_) => return (unknown_solution(), metrics(None)),
    };
    let mut best_solution = match snapshot_solution(sched, &chosen, &starts, &present, cur, stop) {
        Ok(solution) => solution,
        Err(_) => return (unknown_solution(), metrics(None)),
    };
    let first_feasible = best_solution.feasible.then_some(construction_elapsed);
    if best_solution.feasible && checkpoint(stop).is_ok() {
        report(cur.1);
    }
    let mut incumbent_score = cur;

    // Large schedules stay on the compact constructor. The pairwise shift
    // engine is retained only for small instances until the critical-path
    // neighbourhoods replace it.
    if n > 48 || checkpoint(stop).is_err() {
        return (best_solution, metrics(first_feasible));
    }

    const RESTART_AFTER: u64 = 25;
    let mut search_best = cur;
    let mut since_improve = 0u64;
    let mut iter = 0u64;

    'search: loop {
        if checkpoint(stop).is_err() {
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
                    }
                    cur = trial;
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
                    }
                    cur = trial;
                    moved = true;
                    break 'scan;
                }
                chosen[i] = old_mode;
                dur[i] = old_duration;
                mach[i] = old_machine;
                starts[i] = old_start;
            }

            if sched.intervals[i].optional {
                if checkpoint(stop).is_err() {
                    break 'search;
                }
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
                    }
                    cur = trial;
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
        if cur < incumbent_score {
            let snapshot = match snapshot_solution(sched, &chosen, &starts, &present, cur, stop) {
                Ok(snapshot) => snapshot,
                Err(_) => break,
            };
            incumbent_score = cur;
            best_solution = snapshot;
        }
    }

    (best_solution, metrics(first_feasible))
}
