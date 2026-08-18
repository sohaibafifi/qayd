//! Sparse mandatory-part profiles shared by scheduling propagators.
//!
//! A profile contains only event-delimited segments. Its memory is therefore
//! proportional to the number of tasks, never to the numeric time horizon.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProfileSegment {
    pub start: i128,
    pub end: i128,
    pub usage: i128,
}

/// Build a piecewise-constant resource profile from `(start, end, demand)`
/// mandatory parts. Zero-length and zero-demand parts have no effect.
pub(crate) fn build_profile(
    parts: impl IntoIterator<Item = (i128, i128, i128)>,
    events: &mut Vec<(i128, i128)>,
    segments: &mut Vec<ProfileSegment>,
) {
    events.clear();
    for (start, end, demand) in parts {
        if start >= end || demand == 0 {
            continue;
        }
        events.push((start, demand));
        events.push((end, -demand));
    }
    events.sort_unstable_by_key(|&(time, _)| time);
    segments.clear();
    if events.is_empty() {
        return;
    }

    let mut usage = 0i128;
    let mut previous = events[0].0;
    let mut index = 0usize;
    while index < events.len() {
        let time = events[index].0;
        if previous < time && usage != 0 {
            segments.push(ProfileSegment { start: previous, end: time, usage });
        }
        while index < events.len() && events[index].0 == time {
            usage += events[index].1;
            index += 1;
        }
        previous = time;
    }
}

pub(crate) fn peak_usage(segments: &[ProfileSegment]) -> i128 {
    segments.iter().map(|segment| segment.usage).max().unwrap_or(0)
}

/// Return the first event-delimited instant at which placing a task over
/// `[start, end)` would exceed `capacity`. `own_part` identifies the task's
/// contribution already present in the profile and prevents double counting.
pub(crate) fn first_overload(
    segments: &[ProfileSegment],
    start: i128,
    end: i128,
    demand: i128,
    own_part: Option<(i128, i128, i128)>,
    capacity: i128,
) -> Option<(i128, i128)> {
    if start >= end || demand == 0 {
        return None;
    }
    if demand > capacity {
        return Some((start, end));
    }
    for segment in segments {
        let overlap_start = start.max(segment.start);
        let overlap_end = end.min(segment.end);
        if overlap_start >= overlap_end {
            continue;
        }
        let own = own_part
            .filter(|(own_start, own_end, _)| segment.start >= *own_start && segment.end <= *own_end)
            .map_or(0, |(_, _, own_demand)| own_demand);
        if segment.usage - own + demand > capacity {
            return Some((overlap_start, segment.end));
        }
    }
    None
}

/// Earliest start in `[earliest, latest]` whose occupation does not overload
/// the profile. Conflicts jump directly to the end of an event segment, so the
/// running time depends on profile events rather than the numeric horizon.
pub(crate) fn earliest_feasible_start(
    segments: &[ProfileSegment],
    earliest: i128,
    latest: i128,
    duration: i128,
    demand: i128,
    own_part: Option<(i128, i128, i128)>,
    capacity: i128,
) -> Option<i128> {
    if earliest > latest {
        return None;
    }
    if duration <= 0 || demand == 0 {
        return Some(earliest);
    }
    if demand > capacity {
        return None;
    }
    let mut candidate = earliest;
    while candidate <= latest {
        match first_overload(segments, candidate, candidate + duration, demand, own_part, capacity) {
            Some((_, segment_end)) => candidate = segment_end,
            None => return Some(candidate),
        }
    }
    None
}

/// Latest start in `[earliest, latest]` whose occupation does not overload the
/// profile. Conflicts jump before the start of an event segment.
pub(crate) fn latest_feasible_start(
    segments: &[ProfileSegment],
    earliest: i128,
    latest: i128,
    duration: i128,
    demand: i128,
    own_part: Option<(i128, i128, i128)>,
    capacity: i128,
) -> Option<i128> {
    if earliest > latest {
        return None;
    }
    if duration <= 0 || demand == 0 {
        return Some(latest);
    }
    if demand > capacity {
        return None;
    }
    let mut candidate = latest;
    while candidate >= earliest {
        let end = candidate + duration;
        let conflict = segments.iter().rev().find(|segment| {
            let overlap_start = candidate.max(segment.start);
            let overlap_end = end.min(segment.end);
            if overlap_start >= overlap_end {
                return false;
            }
            let own = own_part
                .filter(|(own_start, own_end, _)| segment.start >= *own_start && segment.end <= *own_end)
                .map_or(0, |(_, _, own_demand)| own_demand);
            segment.usage - own + demand > capacity
        });
        match conflict {
            Some(segment) => candidate = segment.start - duration,
            None => return Some(candidate),
        }
    }
    None
}

/// Maximum height an interval may use throughout its mandatory part, after
/// subtracting its own minimum contribution from the shared profile.
pub(crate) fn mandatory_height_limit(segments: &[ProfileSegment], start: i128, end: i128, own_demand: i128, capacity: i128) -> i128 {
    let mut limit = capacity;
    for segment in segments {
        if start < segment.end && segment.start < end {
            limit = limit.min(capacity - (segment.usage - own_demand));
        }
    }
    limit
}

/// Store-independent view of a fixed cumulative task.
///
/// Bounds are converted to `i128` before duration is added. This matters at the
/// `i32` start boundary, where a perfectly valid task may end after
/// `i32::MAX` even though its start remains representable by the finite-domain
/// store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FixedCumulativeTask {
    earliest_start: i128,
    latest_start: i128,
    duration: i128,
    demand: i128,
}

impl FixedCumulativeTask {
    pub(crate) fn new(earliest_start: i32, latest_start: i32, duration: i64, demand: i64) -> Self {
        Self {
            earliest_start: i128::from(earliest_start),
            latest_start: i128::from(latest_start),
            duration: i128::from(duration),
            demand: i128::from(demand),
        }
    }

    pub(crate) fn earliest_start(self) -> i128 {
        self.earliest_start
    }

    pub(crate) fn compulsory_part(self) -> (i128, i128, i128) {
        (self.latest_start, self.earliest_start + self.duration, self.demand)
    }

    fn latest_completion(self) -> i128 {
        self.latest_start + self.duration
    }

    fn energy(self) -> i128 {
        self.duration * self.demand
    }
}

/// A subset of tasks whose energy cannot fit in `[window_start, window_end)`.
/// Member indices address the task slice passed to [`EnergeticWorkspace::analyse`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EnergyConflict {
    pub(crate) window_start: i128,
    pub(crate) window_end: i128,
    pub(crate) energy: i128,
    pub(crate) available_energy: i128,
}

/// Edge finding derived a lower bound beyond a task's latest start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EdgeBoundConflict {
    pub(crate) task: usize,
    pub(crate) lower_bound: i128,
    pub(crate) latest_start: i128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EnergeticConflict {
    Overload(EnergyConflict),
    EdgeBound(EdgeBoundConflict),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EnergeticAnalysis {
    Complete,
    Interrupted,
}

/// Reusable scratch for energetic overload checking and fixed-task edge
/// finding. The kernel knows only mathematical task bounds and resource
/// quantities. Store ids, interval presence, and explanation literals remain
/// responsibilities of the two CP adapters.
#[derive(Clone, Debug, Default)]
pub(crate) struct EnergeticWorkspace {
    energy: Vec<i128>,
    by_est: Vec<usize>,
    by_lct: Vec<usize>,
    lower_bounds: Vec<i128>,
    conflict_tasks: Vec<usize>,
}

impl EnergeticWorkspace {
    pub(crate) fn analyse(&mut self, tasks: &[FixedCumulativeTask], capacity: i128) -> Result<(), EnergeticConflict> {
        match self.analyse_core(tasks, capacity, &|| false)? {
            EnergeticAnalysis::Complete => Ok(()),
            EnergeticAnalysis::Interrupted => unreachable!("the non-interruptible energetic analysis cannot stop"),
        }
    }

    /// Analyse while polling `should_stop` at bounded intervals in the
    /// quadratic loops. An interrupted result carries no conflict and callers
    /// must not consume [`Self::lower_bounds`] until a later complete run.
    pub(crate) fn analyse_until(
        &mut self,
        tasks: &[FixedCumulativeTask],
        capacity: i128,
        should_stop: &dyn Fn() -> bool,
    ) -> Result<EnergeticAnalysis, EnergeticConflict> {
        self.analyse_core(tasks, capacity, should_stop)
    }

    fn analyse_core<F: Fn() -> bool + ?Sized>(
        &mut self,
        tasks: &[FixedCumulativeTask],
        capacity: i128,
        should_stop: &F,
    ) -> Result<EnergeticAnalysis, EnergeticConflict> {
        if should_stop() {
            self.conflict_tasks.clear();
            return Ok(EnergeticAnalysis::Interrupted);
        }

        self.resize(tasks.len());
        let mut work = 0usize;
        for (index, task) in tasks.iter().copied().enumerate() {
            self.energy[index] = task.energy();
            self.by_est[index] = index;
            self.by_lct[index] = index;
            self.lower_bounds[index] = task.earliest_start;
            if interruption_polled(should_stop, &mut work) {
                self.conflict_tasks.clear();
                return Ok(EnergeticAnalysis::Interrupted);
            }
        }

        self.by_est
            .sort_unstable_by(|&left, &right| tasks[right].earliest_start.cmp(&tasks[left].earliest_start).then_with(|| left.cmp(&right)));
        if should_stop() {
            self.conflict_tasks.clear();
            return Ok(EnergeticAnalysis::Interrupted);
        }
        for upper_task in 0..tasks.len() {
            let window_end = tasks[upper_task].latest_completion();
            let mut energy = 0i128;
            self.conflict_tasks.clear();
            for &task in &self.by_est {
                if interruption_polled(should_stop, &mut work) {
                    self.conflict_tasks.clear();
                    return Ok(EnergeticAnalysis::Interrupted);
                }
                if tasks[task].latest_completion() > window_end {
                    continue;
                }
                energy = energy.saturating_add(self.energy[task]);
                self.conflict_tasks.push(task);
                let window_start = tasks[task].earliest_start;
                let available_energy = capacity.saturating_mul(window_end - window_start);
                if energy > available_energy {
                    if should_stop() {
                        self.conflict_tasks.clear();
                        return Ok(EnergeticAnalysis::Interrupted);
                    }
                    return Err(EnergeticConflict::Overload(EnergyConflict { window_start, window_end, energy, available_energy }));
                }
            }
        }

        self.by_lct.sort_unstable_by(|&left, &right| {
            tasks[left].latest_completion().cmp(&tasks[right].latest_completion()).then_with(|| left.cmp(&right))
        });
        if should_stop() {
            self.conflict_tasks.clear();
            return Ok(EnergeticAnalysis::Interrupted);
        }
        for (task, current) in tasks.iter().copied().enumerate() {
            if interruption_polled(should_stop, &mut work) {
                self.conflict_tasks.clear();
                return Ok(EnergeticAnalysis::Interrupted);
            }
            if current.duration <= 0 || current.demand <= 0 {
                continue;
            }
            let mut omega_energy = 0i128;
            let mut omega_est = i128::MAX;
            let mut lower_bound = current.earliest_start;
            for &other in &self.by_lct {
                if interruption_polled(should_stop, &mut work) {
                    self.conflict_tasks.clear();
                    return Ok(EnergeticAnalysis::Interrupted);
                }
                if other == task {
                    continue;
                }
                omega_energy = omega_energy.saturating_add(self.energy[other]);
                omega_est = omega_est.min(tasks[other].earliest_start);
                let window_end = tasks[other].latest_completion();
                let combined_start = omega_est.min(current.earliest_start);
                let combined_energy = omega_energy.saturating_add(self.energy[task]);
                if combined_energy > capacity.saturating_mul(window_end - combined_start) {
                    let parallel_energy = (capacity - current.demand).saturating_mul(window_end - omega_est);
                    let residual_energy = omega_energy.saturating_sub(parallel_energy);
                    if residual_energy > 0 {
                        lower_bound = lower_bound.max(omega_est.saturating_add(ceil_div_positive(residual_energy, current.demand)));
                    }
                }
            }
            if should_stop() {
                self.conflict_tasks.clear();
                return Ok(EnergeticAnalysis::Interrupted);
            }
            if lower_bound > current.latest_start {
                return Err(EnergeticConflict::EdgeBound(EdgeBoundConflict { task, lower_bound, latest_start: current.latest_start }));
            }
            self.lower_bounds[task] = lower_bound;
        }
        if should_stop() {
            self.conflict_tasks.clear();
            return Ok(EnergeticAnalysis::Interrupted);
        }
        Ok(EnergeticAnalysis::Complete)
    }

    pub(crate) fn lower_bounds(&self) -> &[i128] {
        &self.lower_bounds
    }

    #[cfg(test)]
    pub(crate) fn conflict_tasks(&self) -> &[usize] {
        &self.conflict_tasks
    }

    fn resize(&mut self, len: usize) {
        self.energy.resize(len, 0);
        self.by_est.resize(len, 0);
        self.by_lct.resize(len, 0);
        self.lower_bounds.resize(len, 0);
        self.conflict_tasks.clear();
    }
}

const ENERGETIC_POLL_INTERVAL: usize = 64;

#[inline]
fn interruption_polled<F: Fn() -> bool + ?Sized>(should_stop: &F, work: &mut usize) -> bool {
    *work = work.wrapping_add(1);
    work.is_multiple_of(ENERGETIC_POLL_INTERVAL) && should_stop()
}

fn ceil_div_positive(numerator: i128, denominator: i128) -> i128 {
    numerator / denominator + i128::from(numerator % denominator != 0)
}
