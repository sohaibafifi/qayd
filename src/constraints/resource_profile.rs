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
