//! Opt-in unary edge finding for mandatory, positive fixed-duration intervals.
//!
//! A theta-lambda tree gives O(n log n) detection per direction. The eager
//! historical reasons currently scan theta, so total propagation also includes
//! O(n) work per explanation. Trees are rebuilt from immutable bound snapshots.

use std::sync::{Arc, Mutex};

use crate::ids::{IntervalId, VarId};
use crate::propagator::Inconsistency;
use crate::store::{Premise, Store};

/// Counters for the opt-in unary edge-finding experiment.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UnaryEdgeFindingStats {
    pub propagation_calls: u64,
    pub directional_passes: u64,
    pub tree_operations: u64,
    pub overloads: u64,
    pub bound_updates: u64,
    /// Actual integer start values removed by successful edge-finding updates.
    pub eliminated_starts: u64,
    pub reason_premises: u64,
    pub unsupported_calls: u64,
}

impl UnaryEdgeFindingStats {
    pub(crate) fn merge(&mut self, other: Self) {
        self.propagation_calls = self.propagation_calls.saturating_add(other.propagation_calls);
        self.directional_passes = self.directional_passes.saturating_add(other.directional_passes);
        self.tree_operations = self.tree_operations.saturating_add(other.tree_operations);
        self.overloads = self.overloads.saturating_add(other.overloads);
        self.bound_updates = self.bound_updates.saturating_add(other.bound_updates);
        self.eliminated_starts = self.eliminated_starts.saturating_add(other.eliminated_starts);
        self.reason_premises = self.reason_premises.saturating_add(other.reason_premises);
        self.unsupported_calls = self.unsupported_calls.saturating_add(other.unsupported_calls);
    }
}

/// Shared counters survive runtime solver clones and worker ownership changes.
#[derive(Clone, Debug, Default)]
pub struct UnaryEdgeFindingMetricsHandle(Arc<Mutex<UnaryEdgeFindingStats>>);

impl UnaryEdgeFindingMetricsHandle {
    pub fn stats(&self) -> UnaryEdgeFindingStats {
        *self.0.lock().expect("unary edge-finding metrics mutex poisoned")
    }

    pub(crate) fn record(&self, stats: UnaryEdgeFindingStats) {
        self.0.lock().expect("unary edge-finding metrics mutex poisoned").merge(stats);
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Snapshot {
    pub(crate) interval: IntervalId,
    pub(crate) start: VarId,
    pub(crate) lower: i32,
    pub(crate) upper: i32,
    pub(crate) duration: i128,
    pub(crate) est: i128,
    pub(crate) lct: i128,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum Membership {
    #[default]
    Empty,
    Theta,
    Lambda,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Responsible {
    pub(crate) value: i128,
    pub(crate) task: usize,
}

/// Prefer the smaller original index on equal values, independent of tree shape.
fn best(a: Option<Responsible>, b: Option<Responsible>) -> Option<Responsible> {
    match (a, b) {
        (None, value) | (value, None) => value,
        (Some(a), Some(b)) => Some(if a.value > b.value || (a.value == b.value && a.task < b.task) { a } else { b }),
    }
}

fn shifted(value: Option<Responsible>, amount: i128) -> Option<Responsible> {
    value.map(|value| Responsible { value: value.value + amount, task: value.task })
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Node {
    pub(crate) theta_duration: i128,
    pub(crate) theta_ect: Option<i128>,
    pub(crate) lambda_duration: Option<Responsible>,
    pub(crate) lambda_ect: Option<Responsible>,
}

impl Node {
    fn merge(left: Self, right: Self) -> Self {
        let theta_ect = left.theta_ect.map(|value| value + right.theta_duration).max(right.theta_ect);
        let lambda_duration =
            best(shifted(left.lambda_duration, right.theta_duration), shifted(right.lambda_duration, left.theta_duration));
        let cross = left.theta_ect.and_then(|ect| shifted(right.lambda_duration, ect));
        let right_theta = right.theta_ect.and_then(|ect| left.lambda_duration.map(|entry| Responsible { value: ect, task: entry.task }));
        let lambda_ect = best(best(best(right.lambda_ect, shifted(left.lambda_ect, right.theta_duration)), cross), right_theta);
        Self { theta_duration: left.theta_duration + right.theta_duration, theta_ect, lambda_duration, lambda_ect }
    }
}

/// Leaf order is nondecreasing earliest start. Each lambda summary contains
/// exactly one lambda task, together with the theta tasks in that subtree.
/// i128 covers every physically representable Vec of i32-duration tasks.
pub(crate) struct ThetaLambdaTree {
    nodes: Vec<Node>,
    leaves: Vec<usize>,
    pub(crate) membership: Vec<Membership>,
}

fn stopped(should_stop: Option<&dyn Fn() -> bool>) -> bool {
    should_stop.is_some_and(|stop| stop())
}

impl ThetaLambdaTree {
    pub(crate) fn new(tasks: &[Snapshot], stats: &mut UnaryEdgeFindingStats, should_stop: Option<&dyn Fn() -> bool>) -> Option<Self> {
        if stopped(should_stop) {
            return None;
        }
        let mut order = (0..tasks.len()).collect::<Vec<_>>();
        order.sort_unstable_by_key(|&index| (tasks[index].est, index));
        if stopped(should_stop) {
            return None;
        }
        let base = tasks.len().checked_next_power_of_two()?.max(1);
        let mut tree = Self {
            nodes: vec![Node::default(); base.checked_mul(2)?],
            leaves: vec![0; tasks.len()],
            membership: vec![Membership::Theta; tasks.len()],
        };
        for (position, index) in order.into_iter().enumerate() {
            if stopped(should_stop) {
                return None;
            }
            tree.leaves[index] = base + position;
            tree.nodes[base + position] = Node {
                theta_duration: tasks[index].duration,
                theta_ect: Some(tasks[index].est + tasks[index].duration),
                ..Node::default()
            };
            stats.tree_operations = stats.tree_operations.saturating_add(1);
        }
        for node in (1..base).rev() {
            if stopped(should_stop) {
                return None;
            }
            tree.nodes[node] = Node::merge(tree.nodes[node * 2], tree.nodes[node * 2 + 1]);
            stats.tree_operations = stats.tree_operations.saturating_add(1);
        }
        Some(tree)
    }

    pub(crate) fn root(&self) -> Node {
        self.nodes[1]
    }

    pub(crate) fn set(
        &mut self,
        tasks: &[Snapshot],
        task: usize,
        membership: Membership,
        stats: &mut UnaryEdgeFindingStats,
        should_stop: Option<&dyn Fn() -> bool>,
    ) -> bool {
        if stopped(should_stop) {
            return false;
        }
        self.membership[task] = membership;
        let mut node = self.leaves[task];
        self.nodes[node] = match membership {
            Membership::Empty => Node::default(),
            Membership::Theta => {
                Node { theta_duration: tasks[task].duration, theta_ect: Some(tasks[task].est + tasks[task].duration), ..Node::default() }
            }
            Membership::Lambda => Node {
                lambda_duration: Some(Responsible { value: tasks[task].duration, task }),
                lambda_ect: Some(Responsible { value: tasks[task].est + tasks[task].duration, task }),
                ..Node::default()
            },
        };
        stats.tree_operations = stats.tree_operations.saturating_add(1);
        while node > 1 {
            if stopped(should_stop) {
                return false;
            }
            node /= 2;
            self.nodes[node] = Node::merge(self.nodes[node * 2], self.nodes[node * 2 + 1]);
            stats.tree_operations = stats.tree_operations.saturating_add(1);
        }
        true
    }
}

fn premises(
    tasks: &[Snapshot],
    tree: &ThetaLambdaTree,
    target: Option<usize>,
    mirrored: bool,
    should_stop: Option<&dyn Fn() -> bool>,
) -> Option<Vec<Premise>> {
    let mut why = Vec::new();
    for (index, task) in tasks.iter().enumerate() {
        if stopped(should_stop) {
            return None;
        }
        if tree.membership[index] == Membership::Theta {
            why.push(Premise::Ge { var: task.start, bound: task.lower });
            why.push(Premise::Le { var: task.start, bound: task.upper });
        }
    }
    if let Some(index) = target {
        let task = tasks[index];
        why.push(if mirrored {
            Premise::Le { var: task.start, bound: task.upper }
        } else {
            Premise::Ge { var: task.start, bound: task.lower }
        });
    }
    (!stopped(should_stop)).then_some(why)
}

fn directional_pass(
    store: &mut Store,
    intervals: &[IntervalId],
    mirrored: bool,
    stats: &mut UnaryEdgeFindingStats,
    should_stop: Option<&dyn Fn() -> bool>,
) -> Result<Option<bool>, Inconsistency> {
    let mut tasks = Vec::with_capacity(intervals.len());
    for &interval in intervals {
        if stopped(should_stop) {
            return Ok(None);
        }
        let lower = store.interval_start_min(interval);
        let upper = store.interval_start_max(interval);
        let duration = i128::from(store.interval_duration(interval));
        let (est, lct) =
            if mirrored { (-i128::from(upper) - duration, -i128::from(lower)) } else { (i128::from(lower), i128::from(upper) + duration) };
        tasks.push(Snapshot { interval, start: store.interval_start_var(interval), lower, upper, duration, est, lct });
    }
    stats.directional_passes = stats.directional_passes.saturating_add(1);
    let Some(mut tree) = ThetaLambdaTree::new(&tasks, stats, should_stop) else {
        return Ok(None);
    };
    let mut deadlines = (0..tasks.len()).collect::<Vec<_>>();
    deadlines.sort_unstable_by_key(|&index| (std::cmp::Reverse(tasks[index].lct), index));
    let mut changed = false;
    for (position, &task) in deadlines.iter().enumerate() {
        if stopped(should_stop) {
            return Ok(None);
        }
        if tree.root().theta_ect.is_some_and(|ect| ect > tasks[task].lct) {
            let Some(why) = premises(&tasks, &tree, None, mirrored, should_stop) else {
                return Ok(None);
            };
            stats.overloads = stats.overloads.saturating_add(1);
            stats.reason_premises = stats.reason_premises.saturating_add(why.len() as u64);
            return Err(store.fail_because(why));
        }
        let Some(&next) = deadlines.get(position + 1) else {
            break;
        };
        if !tree.set(&tasks, task, Membership::Lambda, stats, should_stop) {
            return Ok(None);
        }
        while let Some(responsible) = tree.root().lambda_ect.filter(|ect| ect.value > tasks[next].lct) {
            if stopped(should_stop) {
                return Ok(None);
            }
            let target = tasks[responsible.task];
            let Some(bound) = tree.root().theta_ect else {
                break;
            };
            let raw_bound = if mirrored { -bound - target.duration } else { bound };
            let current = if mirrored { store.interval_start_max(target.interval) } else { store.interval_start_min(target.interval) };
            if if mirrored { raw_bound < i128::from(current) } else { raw_bound > i128::from(current) } {
                // Theta must complete by the next deadline. If Theta together
                // with this target cannot do so, the target follows every task
                // in Theta, hence its start is at least ECT(Theta). These frozen
                // bounds establish both facts without consulting later domains.
                let Some(why) = premises(&tasks, &tree, Some(responsible.task), mirrored, should_stop) else {
                    return Ok(None);
                };
                stats.reason_premises = stats.reason_premises.saturating_add(why.len() as u64);
                let Ok(bound) = i32::try_from(raw_bound) else {
                    return Err(store.fail_because(why));
                };
                let before = store.size(target.start);
                let updated = if mirrored {
                    store.set_interval_start_max_because(target.interval, bound, why)?
                } else {
                    store.set_interval_start_min_because(target.interval, bound, why)?
                };
                if updated {
                    stats.bound_updates = stats.bound_updates.saturating_add(1);
                    stats.eliminated_starts = stats.eliminated_starts.saturating_add((before - store.size(target.start)) as u64);
                    changed = true;
                }
            }
            if !tree.set(&tasks, responsible.task, Membership::Empty, stats, should_stop) {
                return Ok(None);
            }
        }
    }
    Ok(Some(changed))
}

pub(crate) fn propagate(
    store: &mut Store,
    intervals: &[IntervalId],
    should_stop: Option<&dyn Fn() -> bool>,
) -> Result<Option<bool>, Inconsistency> {
    let mut stats = UnaryEdgeFindingStats::default();
    let result = (|| {
        let Some(forward) = directional_pass(store, intervals, false, &mut stats, should_stop)? else {
            return Ok(None);
        };
        let Some(backward) = directional_pass(store, intervals, true, &mut stats, should_stop)? else {
            return Ok(None);
        };
        Ok(Some(forward || backward))
    })();
    store.record_unary_edge_finding(stats);
    result
}
