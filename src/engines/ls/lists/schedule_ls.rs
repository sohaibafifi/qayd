use std::sync::atomic::{AtomicBool, Ordering};

use crate::mix64;
use crate::model::list::{CollectionSolution, IntervalVar, Resource, Schedule};

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

/// Overlap (>= 0) of two intervals' time spans.
fn pair_overlap(starts: &[i64], dur: &[i64], i: usize, j: usize) -> i64 {
    ((starts[i] + dur[i]).min(starts[j] + dur[j]) - starts[i].max(starts[j])).max(0)
}

fn schedule_score(sched: &Schedule, dur: &[i64], mach: &[i64], starts: &[i64]) -> (i64, i64) {
    let mut viol = 0i64;
    let mut makespan = 0i64;
    for (i, iv) in sched.intervals.iter().enumerate() {
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
        viol = viol.saturating_add((starts[a].saturating_add(dur[a]) - starts[b]).max(0));
    }
    for res in &sched.resources {
        match res {
            Resource::NoOverlap(ivs) => {
                for x in 0..ivs.len() {
                    for y in (x + 1)..ivs.len() {
                        viol = viol.saturating_add(pair_overlap(starts, dur, ivs[x], ivs[y]));
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
                            viol = viol.saturating_add(pair_overlap(starts, dur, i, j));
                        }
                    }
                }
            }
            Resource::Cumulative { demands, capacity } => {
                viol = viol.saturating_add(cumulative_overload(demands, *capacity, dur, starts));
            }
        }
    }
    (viol, makespan)
}

/// Total resource-overload area of a cumulative resource (usage above capacity,
/// integrated over time). Usage changes only at interval boundaries.
fn cumulative_overload(demands: &[(usize, i64)], capacity: i64, dur: &[i64], starts: &[i64]) -> i64 {
    let mut times: Vec<i64> = Vec::with_capacity(demands.len() * 2);
    for &(i, _) in demands {
        times.push(starts[i]);
        times.push(starts[i].saturating_add(dur[i]));
    }
    times.sort_unstable();
    times.dedup();
    let mut total = 0i64;
    for w in times.windows(2) {
        let (t0, t1) = (w[0], w[1]);
        let usage: i64 = demands.iter().filter(|&&(i, _)| starts[i] <= t0 && t0 < starts[i] + dur[i]).map(|&(_, d)| d).sum();
        let over = (usage - capacity).max(0);
        total = total.saturating_add(over.saturating_mul(t1 - t0));
    }
    total
}

/// Earliest start of each interval respecting precedence only (a longest-path
/// forward pass); resources are left to the search to fix.
fn earliest_starts(sched: &Schedule, dur: &[i64]) -> Vec<i64> {
    let mut s = vec![0i64; sched.intervals.len()];
    for _ in 0..sched.intervals.len() {
        let mut changed = false;
        for &(a, b) in &sched.precedences {
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
fn schedule_candidates(sched: &Schedule, dur: &[i64], starts: &[i64], i: usize) -> Vec<i64> {
    let hi = (sched.intervals[i].horizon - dur[i]).max(0);
    let est = sched.precedences.iter().filter(|&&(_, b)| b == i).map(|&(a, _)| starts[a] + dur[a]).max().unwrap_or(0);
    let mut cands = vec![0, est];
    for (j, &sj) in starts.iter().enumerate() {
        if j != i {
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
pub(super) fn solve_schedule(sched: &Schedule, seed: u64, stop: &AtomicBool, report: &mut dyn FnMut(i64)) -> CollectionSolution {
    let n = sched.intervals.len();
    let mut chosen = vec![0usize; n];
    let (mut dur, mut mach) = mode_view(sched, &chosen);
    let mut starts = earliest_starts(sched, &dur);
    let mut cur = schedule_score(sched, &dur, &mach, &starts);
    let mut best_starts = starts.clone();
    let mut best_chosen = chosen.clone();
    let mut best = cur;
    if best.0 == 0 {
        report(best.1);
    }
    const RESTART_AFTER: u64 = 25;
    let mut since_improve = 0u64;
    let mut iter = 0u64;

    while !stop.load(Ordering::Relaxed) {
        iter += 1;
        let mut moved = false;
        'scan: for i in 0..n {
            // (a) Shift the start under the current mode.
            for t in schedule_candidates(sched, &dur, &starts, i) {
                if t == starts[i] {
                    continue;
                }
                let old = starts[i];
                starts[i] = t;
                let trial = schedule_score(sched, &dur, &mach, &starts);
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
                let trial = schedule_score(sched, &dur, &mach, &starts);
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
            starts = earliest_starts(sched, &dur);
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
        cur = schedule_score(sched, &dur, &mach, &starts);
    }

    if cur < best {
        best = cur;
        best_starts = starts.clone();
        best_chosen = chosen.clone();
    }
    let feasible = best.0 == 0;
    let machines: Vec<i64> = sched.intervals.iter().zip(&best_chosen).map(|(iv, &m)| iv_machine(iv, m)).collect();
    CollectionSolution {
        lists: Vec::new(),
        objectives: if feasible && sched.minimize_makespan { vec![best.1] } else { Vec::new() },
        feasible,
        starts: if feasible { best_starts } else { Vec::new() },
        machines: if feasible { machines } else { Vec::new() },
    }
}
