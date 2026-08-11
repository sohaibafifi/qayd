//! Shared physical graph primitives for scheduling heuristics.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};

/// Validated precedence DAG used by every integer scheduling warm start.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PrecedenceDag {
    predecessors: Vec<Vec<usize>>,
    successors: Vec<Vec<usize>>,
    topological: Vec<usize>,
}

impl PrecedenceDag {
    pub(crate) fn compile(mut successors: Vec<Vec<usize>>, stop: &AtomicBool) -> Option<Self> {
        let count = successors.len();
        let mut predecessors = vec![Vec::new(); count];
        for (operation, list) in successors.iter_mut().enumerate() {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            list.sort_unstable();
            if list.windows(2).any(|pair| pair[0] == pair[1]) || list.iter().any(|&successor| successor >= count || successor == operation)
            {
                return None;
            }
            for &successor in list.iter() {
                predecessors[successor].push(operation);
            }
        }
        for list in &mut predecessors {
            list.sort_unstable();
        }
        let topological = topological_order(&successors, false, stop)?;
        Some(Self { predecessors, successors, topological })
    }

    pub(crate) fn predecessors(&self, operation: usize) -> &[usize] {
        &self.predecessors[operation]
    }

    pub(crate) fn successors(&self, operation: usize) -> &[usize] {
        &self.successors[operation]
    }

    pub(crate) fn all_successors(&self) -> &[Vec<usize>] {
        &self.successors
    }

    pub(crate) fn topological(&self) -> &[usize] {
        &self.topological
    }

    pub(crate) fn terminals(&self) -> impl Iterator<Item = usize> + '_ {
        self.successors.iter().enumerate().filter_map(|(operation, successors)| successors.is_empty().then_some(operation))
    }

    pub(crate) fn remaining_paths(&self, durations: &[i64], stop: &AtomicBool) -> Option<Vec<i64>> {
        if durations.len() != self.successors.len() {
            return None;
        }
        let mut bottom = vec![0i64; durations.len()];
        for &operation in self.topological.iter().rev() {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            let tail = self.successors[operation].iter().map(|&successor| bottom[successor]).max().unwrap_or(0);
            bottom[operation] = durations[operation].checked_add(tail)?;
        }
        Some(bottom)
    }

    pub(crate) fn unique_order(successors: &[Vec<usize>], stop: &AtomicBool) -> Option<Vec<usize>> {
        topological_order(successors, true, stop)
    }
}

fn topological_order(successors: &[Vec<usize>], unique: bool, stop: &AtomicBool) -> Option<Vec<usize>> {
    let mut indegree = vec![0usize; successors.len()];
    for list in successors {
        if stop.load(Ordering::Acquire) {
            return None;
        }
        for &successor in list {
            let degree = indegree.get_mut(successor)?;
            *degree = degree.checked_add(1)?;
        }
    }
    let mut ready =
        indegree.iter().enumerate().filter_map(|(operation, &degree)| (degree == 0).then_some(operation)).collect::<BTreeSet<_>>();
    let mut order = Vec::with_capacity(successors.len());
    while !ready.is_empty() {
        if stop.load(Ordering::Acquire) || (unique && ready.len() != 1) {
            return None;
        }
        let operation = ready.pop_first()?;
        order.push(operation);
        for &successor in &successors[operation] {
            let degree = indegree.get_mut(successor)?;
            *degree = degree.checked_sub(1)?;
            if *degree == 0 {
                ready.insert(successor);
            }
        }
    }
    (order.len() == successors.len()).then_some(order)
}
