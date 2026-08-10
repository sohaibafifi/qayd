use std::sync::atomic::{AtomicBool, Ordering};

use super::alns::{routing_relink_candidate_work, routing_relink_structural_floor, state_clone_work, AlnsWorkBudget, CpuTimer};
use super::local_search::{full_score_raw, PerList, Score, State};
use crate::mix64;
use crate::model::list::CollectionModel;

const MIN_CAPACITY: usize = 6;
const MAX_CAPACITY: usize = 8;
const BOUNDARY: i64 = i64::MIN;
const ARCHIVE_WORK_FACTOR: u64 = 2_048;
const SELECTION_WORK_FACTOR: u64 = 1_024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct EliteWorkBudget {
    units: u64,
}

impl EliteWorkBudget {
    pub(super) const fn new(units: u64) -> Self {
        Self { units }
    }
}

pub(super) fn elite_archive_budget(model: &CollectionModel) -> EliteWorkBudget {
    elite_work_budget(model, ARCHIVE_WORK_FACTOR)
}

pub(super) fn elite_selection_budget(model: &CollectionModel) -> EliteWorkBudget {
    elite_work_budget(model, SELECTION_WORK_FACTOR)
}

fn elite_work_budget(model: &CollectionModel, factor: u64) -> EliteWorkBudget {
    let items = u64::try_from(model.items.len()).unwrap_or(u64::MAX);
    let lists = u64::try_from(model.lists.max(1)).unwrap_or(u64::MAX);
    let tiers = u64::try_from(model.objectives.len()).unwrap_or(u64::MAX);
    let footprint = items.saturating_add(lists.saturating_mul(2)).saturating_add(tiers).saturating_add(1);
    EliteWorkBudget::new(footprint.saturating_mul(factor).saturating_add(factor))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EliteOperationStatus {
    Complete,
    BudgetExhausted,
    Interrupted,
}

pub(super) struct EliteConsiderRun {
    pub(super) status: EliteOperationStatus,
    pub(super) inserted: bool,
    pub(super) work_units: u64,
    pub(super) cpu_nanos: u64,
}

pub(super) struct EliteSelectionRun<'a> {
    pub(super) status: EliteOperationStatus,
    pub(super) target: Option<&'a EliteEntry>,
    pub(super) work_units: u64,
    pub(super) cpu_nanos: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct EliteEdge {
    route: usize,
    from: i64,
    to: i64,
}

#[derive(Clone)]
pub(super) struct EliteEntry {
    lists: Vec<Vec<i32>>,
    score: Score,
    edges: Vec<EliteEdge>,
    hash: u64,
}

impl EliteEntry {
    pub(super) fn lists(&self) -> &[Vec<i32>] {
        &self.lists
    }

    pub(super) fn score(&self) -> &Score {
        &self.score
    }

    pub(super) const fn hash(&self) -> u64 {
        self.hash
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EliteTask {
    EdgeConstruction,
    EdgeCanonicalization,
    EdgeSorting,
    EdgeHashing,
    EdgeComparison,
    ListCloning,
    DistanceCalculation,
    ArchiveOrdering,
}

trait EliteHooks {
    fn before_task(&mut self, _task: EliteTask, _stop: &AtomicBool) {}
}

struct NoEliteHooks;

impl EliteHooks for NoEliteHooks {}

#[derive(Clone, Copy)]
struct EliteReservation(u64);

struct EliteWorkContext<'a, H> {
    budget: EliteWorkBudget,
    completed: u64,
    stop: &'a AtomicBool,
    hooks: &'a mut H,
}

impl<'a, H: EliteHooks> EliteWorkContext<'a, H> {
    fn new(budget: EliteWorkBudget, stop: &'a AtomicBool, hooks: &'a mut H) -> Self {
        Self { budget, completed: 0, stop, hooks }
    }

    fn begin(&mut self, task: EliteTask) -> Result<(), EliteOperationStatus> {
        self.hooks.before_task(task, self.stop);
        self.check()
    }

    fn check(&self) -> Result<(), EliteOperationStatus> {
        if self.stop.load(Ordering::Relaxed) {
            Err(EliteOperationStatus::Interrupted)
        } else if self.completed >= self.budget.units {
            Err(EliteOperationStatus::BudgetExhausted)
        } else {
            Ok(())
        }
    }

    fn reserve(&self, units: u64) -> Result<EliteReservation, EliteOperationStatus> {
        if self.stop.load(Ordering::Relaxed) {
            return Err(EliteOperationStatus::Interrupted);
        }
        if units > self.budget.units.saturating_sub(self.completed) {
            return Err(EliteOperationStatus::BudgetExhausted);
        }
        Ok(EliteReservation(units))
    }

    fn complete(&mut self, reservation: EliteReservation) {
        self.completed = self.completed.saturating_add(reservation.0);
    }

    fn unit(&mut self) -> Result<(), EliteOperationStatus> {
        let reservation = self.reserve(1)?;
        self.complete(reservation);
        Ok(())
    }
}

/// Small worker-local archive. Ordering is fully explicit, so hash-table
/// iteration cannot alter the search trajectory between runs.
pub(super) struct ElitePool {
    capacity: usize,
    preserve_route_identity: bool,
    canonicalize_reverse: bool,
    entries: Vec<EliteEntry>,
}

impl ElitePool {
    pub(super) fn new(capacity: usize, preserve_route_identity: bool, canonicalize_reverse: bool) -> Self {
        let capacity = capacity.clamp(MIN_CAPACITY, MAX_CAPACITY);
        Self { capacity, preserve_route_identity, canonicalize_reverse, entries: Vec::with_capacity(capacity) }
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(super) fn best(&self) -> Option<&EliteEntry> {
        self.entries.first()
    }

    pub(super) fn consider_bounded(
        &mut self,
        state: &State,
        score: &Score,
        budget: EliteWorkBudget,
        stop: &AtomicBool,
    ) -> EliteConsiderRun {
        self.consider_lists_with_hooks(&state.lists, score, budget, stop, &mut NoEliteHooks)
    }

    fn consider_lists_with_hooks<H: EliteHooks>(
        &mut self,
        lists: &[Vec<i32>],
        score: &Score,
        budget: EliteWorkBudget,
        stop: &AtomicBool,
        hooks: &mut H,
    ) -> EliteConsiderRun {
        let started = CpuTimer::start();
        let mut context = EliteWorkContext::new(budget, stop, hooks);
        let result = context.check().and_then(|()| self.consider_lists_inner(lists, score, &mut context));
        match result {
            Ok(inserted) => EliteConsiderRun {
                status: EliteOperationStatus::Complete,
                inserted,
                work_units: context.completed,
                cpu_nanos: started.elapsed_nanos(),
            },
            Err(status) => EliteConsiderRun { status, inserted: false, work_units: context.completed, cpu_nanos: started.elapsed_nanos() },
        }
    }

    fn consider_lists_inner<H: EliteHooks>(
        &mut self,
        lists: &[Vec<i32>],
        score: &Score,
        context: &mut EliteWorkContext<'_, H>,
    ) -> Result<bool, EliteOperationStatus> {
        context.unit()?;
        if score.violation != 0 {
            return Ok(false);
        }
        let edges = sorted_edges_bounded(lists, self.preserve_route_identity, self.canonicalize_reverse, context)?;
        let hash = stable_edge_hash_bounded(&edges, context)?;

        if let Some(index) = self.duplicate_index(hash, &edges, context)? {
            if score_cmp_bounded(score, &self.entries[index].score, context)?.is_ge() {
                return Ok(false);
            }
            let insertion = self.insertion_index(score, hash, Some(index), context)?;
            let entry = clone_entry_bounded(lists, score, edges, hash, context)?;
            self.commit_entry(Some(index), insertion, entry, context)?;
            return Ok(true);
        }

        if self.entries.is_empty() {
            let entry = clone_entry_bounded(lists, score, edges, hash, context)?;
            self.commit_entry(None, 0, entry, context)?;
            return Ok(true);
        }

        let (closest_distance, closest_index) = self.closest_entry(&edges, context)?.expect("the pool is non-empty");
        context.unit()?;
        let diversity_floor = (edges.len() / 16).max(2);
        if closest_distance < diversity_floor && score_cmp_bounded(score, &self.entries[closest_index].score, context)?.is_ge() {
            return Ok(false);
        }
        if closest_distance < diversity_floor {
            let insertion = self.insertion_index(score, hash, Some(closest_index), context)?;
            let entry = clone_entry_bounded(lists, score, edges, hash, context)?;
            self.commit_entry(Some(closest_index), insertion, entry, context)?;
            return Ok(true);
        }

        let insertion = self.insertion_index(score, hash, None, context)?;
        if self.entries.len() < self.capacity {
            let entry = clone_entry_bounded(lists, score, edges, hash, context)?;
            self.commit_entry(None, insertion, entry, context)?;
            return Ok(true);
        }

        let candidate = EliteCandidateView { score, edges: &edges, hash };
        let remove = self.least_diverse_with_candidate(candidate, insertion, context)?;
        if remove == insertion {
            return Ok(false);
        }
        let physical_remove = if remove < insertion { remove } else { remove.saturating_sub(1) };
        let final_insertion = insertion.saturating_sub(usize::from(physical_remove < insertion));
        let entry = clone_entry_bounded(lists, score, edges, hash, context)?;
        self.commit_entry(Some(physical_remove), final_insertion, entry, context)?;
        Ok(true)
    }

    pub(super) fn select_target_bounded(
        &self,
        source: &State,
        seed: u64,
        budget: EliteWorkBudget,
        stop: &AtomicBool,
    ) -> EliteSelectionRun<'_> {
        self.select_target_with_hooks(source, seed, budget, stop, &mut NoEliteHooks)
    }

    fn select_target_with_hooks<'a, H: EliteHooks>(
        &'a self,
        source: &State,
        seed: u64,
        budget: EliteWorkBudget,
        stop: &AtomicBool,
        hooks: &mut H,
    ) -> EliteSelectionRun<'a> {
        let started = CpuTimer::start();
        let mut context = EliteWorkContext::new(budget, stop, hooks);
        let result = context.check().and_then(|()| self.select_target_index(source, seed, &mut context));
        match result {
            Ok(index) => EliteSelectionRun {
                status: EliteOperationStatus::Complete,
                target: index.and_then(|index| self.entries.get(index)),
                work_units: context.completed,
                cpu_nanos: started.elapsed_nanos(),
            },
            Err(status) => EliteSelectionRun { status, target: None, work_units: context.completed, cpu_nanos: started.elapsed_nanos() },
        }
    }

    fn select_target_index<H: EliteHooks>(
        &self,
        source: &State,
        seed: u64,
        context: &mut EliteWorkContext<'_, H>,
    ) -> Result<Option<usize>, EliteOperationStatus> {
        let source_edges = sorted_edges_bounded(&source.lists, self.preserve_route_identity, self.canonicalize_reverse, context)?;
        let source_hash = stable_edge_hash_bounded(&source_edges, context)?;
        context.begin(EliteTask::ArchiveOrdering)?;
        let allocation = context.reserve(1)?;
        let mut ranked = Vec::with_capacity(self.entries.len());
        context.complete(allocation);
        for (index, entry) in self.entries.iter().enumerate() {
            context.unit()?;
            if entry.hash == source_hash && edges_equal_bounded(&entry.edges, &source_edges, context)? {
                continue;
            }
            let distance = edge_distance_sorted_bounded(&source_edges, &entry.edges, context)?;
            context.unit()?;
            ranked.push(EliteRank { distance, index });
        }
        sort_elite_ranks_bounded(&mut ranked, &self.entries, context)?;
        if ranked.is_empty() {
            return Ok(None);
        }
        context.unit()?;
        let candidates = ranked.len().div_ceil(2);
        let pick = (mix64(seed) % candidates as u64) as usize;
        Ok(Some(ranked[pick].index))
    }

    fn duplicate_index<H: EliteHooks>(
        &self,
        hash: u64,
        edges: &[EliteEdge],
        context: &mut EliteWorkContext<'_, H>,
    ) -> Result<Option<usize>, EliteOperationStatus> {
        for (index, entry) in self.entries.iter().enumerate() {
            context.unit()?;
            if entry.hash == hash && edges_equal_bounded(&entry.edges, edges, context)? {
                return Ok(Some(index));
            }
        }
        Ok(None)
    }

    fn closest_entry<H: EliteHooks>(
        &self,
        edges: &[EliteEdge],
        context: &mut EliteWorkContext<'_, H>,
    ) -> Result<Option<(usize, usize)>, EliteOperationStatus> {
        let mut closest = None;
        for (index, entry) in self.entries.iter().enumerate() {
            let distance = edge_distance_sorted_bounded(edges, &entry.edges, context)?;
            context.unit()?;
            if closest.is_none_or(|best: (usize, usize)| (distance, index) < best) {
                closest = Some((distance, index));
            }
        }
        Ok(closest)
    }

    fn insertion_index<H: EliteHooks>(
        &self,
        score: &Score,
        hash: u64,
        skip: Option<usize>,
        context: &mut EliteWorkContext<'_, H>,
    ) -> Result<usize, EliteOperationStatus> {
        context.begin(EliteTask::ArchiveOrdering)?;
        let mut insertion = 0usize;
        for (index, entry) in self.entries.iter().enumerate() {
            if Some(index) == skip {
                continue;
            }
            let order = score_hash_cmp_bounded(&entry.score, entry.hash, score, hash, context)?;
            if order.is_gt() {
                break;
            }
            insertion = insertion.saturating_add(1);
        }
        Ok(insertion)
    }

    fn least_diverse_with_candidate<H: EliteHooks>(
        &self,
        candidate: EliteCandidateView<'_>,
        insertion: usize,
        context: &mut EliteWorkContext<'_, H>,
    ) -> Result<usize, EliteOperationStatus> {
        let virtual_len = self.entries.len().saturating_add(1);
        let mut selected: Option<(usize, usize)> = None;
        for index in 1..virtual_len {
            let entry = self.virtual_entry(candidate, insertion, index);
            let mut nearest = usize::MAX;
            for other in 0..virtual_len {
                if other == index {
                    continue;
                }
                let other_entry = self.virtual_entry(candidate, insertion, other);
                nearest = nearest.min(edge_distance_sorted_bounded(entry.edges, other_entry.edges, context)?);
            }
            let replace = match selected {
                None => true,
                Some((best_nearest, best_index)) if nearest < best_nearest => true,
                Some((best_nearest, best_index)) if nearest == best_nearest => {
                    let best = self.virtual_entry(candidate, insertion, best_index);
                    match score_cmp_bounded(entry.score, best.score, context)? {
                        std::cmp::Ordering::Greater => true,
                        std::cmp::Ordering::Equal => {
                            context.unit()?;
                            entry.hash > best.hash
                        }
                        std::cmp::Ordering::Less => false,
                    }
                }
                Some(_) => false,
            };
            if replace {
                selected = Some((nearest, index));
            }
        }
        Ok(selected.expect("a non-best archive entry exists").1)
    }

    fn virtual_entry<'a>(&'a self, candidate: EliteCandidateView<'a>, insertion: usize, index: usize) -> EliteCandidateView<'a> {
        if index == insertion {
            candidate
        } else {
            let physical = index.saturating_sub(usize::from(index > insertion));
            let entry = &self.entries[physical];
            EliteCandidateView { score: &entry.score, edges: &entry.edges, hash: entry.hash }
        }
    }

    fn commit_entry<H: EliteHooks>(
        &mut self,
        remove: Option<usize>,
        insertion: usize,
        entry: EliteEntry,
        context: &mut EliteWorkContext<'_, H>,
    ) -> Result<(), EliteOperationStatus> {
        let after_removal = self.entries.len().saturating_sub(usize::from(remove.is_some()));
        let removal_shifts = remove.map_or(0, |index| self.entries.len().saturating_sub(index.saturating_add(1)));
        let insertion_shifts = after_removal.saturating_sub(insertion);
        let units = u64::try_from(removal_shifts.saturating_add(insertion_shifts).saturating_add(1)).unwrap_or(u64::MAX);
        let reservation = context.reserve(units)?;
        if let Some(index) = remove {
            self.entries.remove(index);
        }
        self.entries.insert(insertion, entry);
        context.complete(reservation);
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct EliteCandidateView<'a> {
    score: &'a Score,
    edges: &'a [EliteEdge],
    hash: u64,
}

#[derive(Clone, Copy)]
struct EliteRank {
    distance: usize,
    index: usize,
}

fn sorted_edges_bounded<H: EliteHooks>(
    lists: &[Vec<i32>],
    preserve_route_identity: bool,
    canonicalize_reverse: bool,
    context: &mut EliteWorkContext<'_, H>,
) -> Result<Vec<EliteEdge>, EliteOperationStatus> {
    context.begin(EliteTask::EdgeConstruction)?;
    let mut capacity = lists.len().saturating_mul(2);
    for route in lists {
        context.unit()?;
        capacity = capacity.saturating_add(route.len());
    }
    let allocation = context.reserve(1)?;
    let mut edges = Vec::with_capacity(capacity);
    context.complete(allocation);
    for (route_index, route) in lists.iter().enumerate() {
        context.unit()?;
        if route.is_empty() {
            continue;
        }
        let route_key = if preserve_route_identity { route_index } else { 0 };
        let reverse = if canonicalize_reverse { route_is_greater_than_reverse_bounded(route, context)? } else { false };
        if reverse {
            push_edge_bounded(
                &mut edges,
                EliteEdge { route: route_key, from: BOUNDARY, to: i64::from(*route.last().expect("route is non-empty")) },
                context,
            )?;
            for index in (1..route.len()).rev() {
                push_edge_bounded(
                    &mut edges,
                    EliteEdge { route: route_key, from: i64::from(route[index]), to: i64::from(route[index - 1]) },
                    context,
                )?;
            }
            push_edge_bounded(&mut edges, EliteEdge { route: route_key, from: i64::from(route[0]), to: BOUNDARY }, context)?;
        } else {
            push_edge_bounded(&mut edges, EliteEdge { route: route_key, from: BOUNDARY, to: i64::from(route[0]) }, context)?;
            for edge in route.windows(2) {
                push_edge_bounded(&mut edges, EliteEdge { route: route_key, from: i64::from(edge[0]), to: i64::from(edge[1]) }, context)?;
            }
            push_edge_bounded(
                &mut edges,
                EliteEdge { route: route_key, from: i64::from(*route.last().expect("route is non-empty")), to: BOUNDARY },
                context,
            )?;
        }
    }
    sort_edges_bounded(&mut edges, context)?;
    Ok(edges)
}

fn route_is_greater_than_reverse_bounded<H: EliteHooks>(
    route: &[i32],
    context: &mut EliteWorkContext<'_, H>,
) -> Result<bool, EliteOperationStatus> {
    context.begin(EliteTask::EdgeCanonicalization)?;
    for index in 0..route.len() {
        context.unit()?;
        match route[index].cmp(&route[route.len() - index - 1]) {
            std::cmp::Ordering::Less => return Ok(false),
            std::cmp::Ordering::Greater => return Ok(true),
            std::cmp::Ordering::Equal => {}
        }
    }
    Ok(false)
}

fn push_edge_bounded<H: EliteHooks>(
    edges: &mut Vec<EliteEdge>,
    edge: EliteEdge,
    context: &mut EliteWorkContext<'_, H>,
) -> Result<(), EliteOperationStatus> {
    context.unit()?;
    edges.push(edge);
    Ok(())
}

fn sort_edges_bounded<H: EliteHooks>(edges: &mut [EliteEdge], context: &mut EliteWorkContext<'_, H>) -> Result<(), EliteOperationStatus> {
    context.begin(EliteTask::EdgeSorting)?;
    if edges.len() < 2 {
        context.unit()?;
        return Ok(());
    }
    for root in (0..edges.len() / 2).rev() {
        sift_edges_down_bounded(edges, root, edges.len(), context)?;
    }
    for end in (1..edges.len()).rev() {
        context.unit()?;
        edges.swap(0, end);
        sift_edges_down_bounded(edges, 0, end, context)?;
    }
    Ok(())
}

fn sift_edges_down_bounded<H: EliteHooks>(
    edges: &mut [EliteEdge],
    mut root: usize,
    end: usize,
    context: &mut EliteWorkContext<'_, H>,
) -> Result<(), EliteOperationStatus> {
    loop {
        context.unit()?;
        let left = root.saturating_mul(2).saturating_add(1);
        if left >= end {
            return Ok(());
        }
        let mut largest = left;
        let right = left.saturating_add(1);
        if right < end {
            context.unit()?;
            if edges[right] > edges[left] {
                largest = right;
            }
        }
        context.unit()?;
        if edges[root] >= edges[largest] {
            return Ok(());
        }
        context.unit()?;
        edges.swap(root, largest);
        root = largest;
    }
}

fn stable_edge_hash_bounded<H: EliteHooks>(
    edges: &[EliteEdge],
    context: &mut EliteWorkContext<'_, H>,
) -> Result<u64, EliteOperationStatus> {
    context.begin(EliteTask::EdgeHashing)?;
    context.unit()?;
    let mut hash = mix64(edges.len() as u64);
    for (index, edge) in edges.iter().enumerate() {
        context.unit()?;
        let from = edge.from as u64;
        let to = edge.to as u64;
        hash = mix64(hash ^ mix64(edge.route as u64) ^ mix64(from) ^ mix64(to.rotate_left(29)) ^ mix64(index as u64));
    }
    Ok(hash)
}

fn edges_equal_bounded<H: EliteHooks>(
    left: &[EliteEdge],
    right: &[EliteEdge],
    context: &mut EliteWorkContext<'_, H>,
) -> Result<bool, EliteOperationStatus> {
    context.begin(EliteTask::EdgeComparison)?;
    context.unit()?;
    if left.len() != right.len() {
        return Ok(false);
    }
    for (left, right) in left.iter().zip(right) {
        context.unit()?;
        if left != right {
            return Ok(false);
        }
    }
    Ok(true)
}

fn edge_distance_sorted_bounded<H: EliteHooks>(
    left: &[EliteEdge],
    right: &[EliteEdge],
    context: &mut EliteWorkContext<'_, H>,
) -> Result<usize, EliteOperationStatus> {
    context.begin(EliteTask::DistanceCalculation)?;
    let (mut i, mut j, mut distance) = (0usize, 0usize, 0usize);
    while i < left.len() && j < right.len() {
        context.unit()?;
        match left[i].cmp(&right[j]) {
            std::cmp::Ordering::Less => {
                i += 1;
                distance += 1;
            }
            std::cmp::Ordering::Greater => {
                j += 1;
                distance += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    context.unit()?;
    Ok(distance + left.len() - i + right.len() - j)
}

fn score_cmp_bounded<H: EliteHooks>(
    left: &Score,
    right: &Score,
    context: &mut EliteWorkContext<'_, H>,
) -> Result<std::cmp::Ordering, EliteOperationStatus> {
    context.unit()?;
    let violation = left.violation.cmp(&right.violation);
    if !violation.is_eq() {
        return Ok(violation);
    }
    for (&left, &right) in left.tiers.iter().zip(&right.tiers) {
        context.unit()?;
        let order = left.cmp(&right);
        if !order.is_eq() {
            return Ok(order);
        }
    }
    context.unit()?;
    Ok(left.tiers.len().cmp(&right.tiers.len()))
}

fn score_hash_cmp_bounded<H: EliteHooks>(
    left_score: &Score,
    left_hash: u64,
    right_score: &Score,
    right_hash: u64,
    context: &mut EliteWorkContext<'_, H>,
) -> Result<std::cmp::Ordering, EliteOperationStatus> {
    let score = score_cmp_bounded(left_score, right_score, context)?;
    if !score.is_eq() {
        return Ok(score);
    }
    context.unit()?;
    Ok(left_hash.cmp(&right_hash))
}

fn clone_entry_bounded<H: EliteHooks>(
    lists: &[Vec<i32>],
    score: &Score,
    edges: Vec<EliteEdge>,
    hash: u64,
    context: &mut EliteWorkContext<'_, H>,
) -> Result<EliteEntry, EliteOperationStatus> {
    context.begin(EliteTask::ListCloning)?;
    let allocation = context.reserve(1)?;
    let mut cloned_lists = Vec::with_capacity(lists.len());
    context.complete(allocation);
    for route in lists {
        let allocation = context.reserve(1)?;
        let mut cloned_route = Vec::with_capacity(route.len());
        context.complete(allocation);
        for &item in route {
            context.unit()?;
            cloned_route.push(item);
        }
        context.unit()?;
        cloned_lists.push(cloned_route);
    }
    let mut cloned_score = Score { violation: score.violation, tiers: Default::default() };
    for &tier in &score.tiers {
        context.unit()?;
        cloned_score.tiers.push(tier);
    }
    Ok(EliteEntry { lists: cloned_lists, score: cloned_score, edges, hash })
}

fn sort_elite_ranks_bounded<H: EliteHooks>(
    ranks: &mut [EliteRank],
    entries: &[EliteEntry],
    context: &mut EliteWorkContext<'_, H>,
) -> Result<(), EliteOperationStatus> {
    context.begin(EliteTask::ArchiveOrdering)?;
    for index in 1..ranks.len() {
        let mut cursor = index;
        while cursor > 0 && elite_rank_cmp_bounded(ranks[cursor - 1], ranks[cursor], entries, context)?.is_gt() {
            context.unit()?;
            ranks.swap(cursor - 1, cursor);
            cursor -= 1;
        }
    }
    Ok(())
}

fn elite_rank_cmp_bounded<H: EliteHooks>(
    left: EliteRank,
    right: EliteRank,
    entries: &[EliteEntry],
    context: &mut EliteWorkContext<'_, H>,
) -> Result<std::cmp::Ordering, EliteOperationStatus> {
    context.unit()?;
    let distance = right.distance.cmp(&left.distance);
    if !distance.is_eq() {
        return Ok(distance);
    }
    let score = score_cmp_bounded(&entries[left.index].score, &entries[right.index].score, context)?;
    if !score.is_eq() {
        return Ok(score);
    }
    context.unit()?;
    let hash = entries[left.index].hash.cmp(&entries[right.index].hash);
    if !hash.is_eq() {
        return Ok(hash);
    }
    context.unit()?;
    Ok(left.index.cmp(&right.index))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PathRelinkStatus {
    Complete,
    BudgetExhausted,
    Interrupted,
    InfeasibleTarget,
}

pub(super) struct PathRelinkRun {
    pub(super) status: PathRelinkStatus,
    pub(super) candidate: Option<State>,
    pub(super) generated: u64,
    pub(super) evaluated: u64,
    pub(super) work_units: u64,
    pub(super) structural_work: u64,
    pub(super) canonical_rebuilds: u64,
    pub(super) cpu_nanos: u64,
    pub(super) steps: usize,
}

struct RelinkWork {
    budget: AlnsWorkBudget,
    generated: u64,
    evaluated: u64,
    structural: u64,
    canonical_rebuilds: u64,
    steps: usize,
}

#[derive(Clone, Copy)]
struct RelinkReservation {
    generated: u64,
    evaluated: u64,
    structural: u64,
    canonical_rebuilds: u64,
    steps: usize,
}

impl RelinkWork {
    const fn new(budget: AlnsWorkBudget) -> Self {
        Self { budget, generated: 0, evaluated: 0, structural: 0, canonical_rebuilds: 0, steps: 0 }
    }

    fn reserve_structural(&self, units: u64, stop: &AtomicBool) -> Result<RelinkReservation, PathRelinkStatus> {
        self.reserve(RelinkReservation { generated: units, evaluated: 0, structural: units, canonical_rebuilds: 0, steps: 0 }, stop)
    }

    fn reserve_candidate(&self, structural: u64, stop: &AtomicBool) -> Result<RelinkReservation, PathRelinkStatus> {
        self.reserve(
            RelinkReservation { generated: structural.saturating_add(1), evaluated: 1, structural, canonical_rebuilds: 1, steps: 1 },
            stop,
        )
    }

    fn reserve(&self, reservation: RelinkReservation, stop: &AtomicBool) -> Result<RelinkReservation, PathRelinkStatus> {
        if stop.load(Ordering::Relaxed) {
            return Err(PathRelinkStatus::Interrupted);
        }
        if reservation.generated > self.budget.generated.saturating_sub(self.generated)
            || reservation.evaluated > self.budget.evaluated.saturating_sub(self.evaluated)
        {
            return Err(PathRelinkStatus::BudgetExhausted);
        }
        Ok(reservation)
    }

    fn complete(&mut self, reservation: RelinkReservation) {
        self.generated = self.generated.saturating_add(reservation.generated);
        self.evaluated = self.evaluated.saturating_add(reservation.evaluated);
        self.structural = self.structural.saturating_add(reservation.structural);
        self.canonical_rebuilds = self.canonical_rebuilds.saturating_add(reservation.canonical_rebuilds);
        self.steps = self.steps.saturating_add(reservation.steps);
    }

    fn structural(&mut self, units: u64, stop: &AtomicBool) -> Result<(), PathRelinkStatus> {
        let reservation = self.reserve_structural(units, stop)?;
        self.complete(reservation);
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RelinkCloneKind {
    Source,
    Candidate,
}

trait RelinkHooks {
    fn before_source_score(&mut self, _stop: &AtomicBool) {}

    fn before_step_limit(&mut self, _stop: &AtomicBool) {}

    fn before_candidate_score(&mut self, _stop: &AtomicBool) {}

    fn interrupt_clone(&mut self, _kind: RelinkCloneKind, _copied_items: usize) -> bool {
        false
    }

    fn before_candidate_rebuild(&mut self, _stop: &AtomicBool) {}
}

struct NoRelinkHooks;

impl RelinkHooks for NoRelinkHooks {}

fn clone_lists_interruptible<H: RelinkHooks>(
    lists: &[Vec<i32>],
    additional_capacity: usize,
    stop: &AtomicBool,
    kind: RelinkCloneKind,
    hooks: &mut H,
) -> Option<Vec<Vec<i32>>> {
    let mut copy = Vec::with_capacity(lists.len());
    let mut copied_items = 0usize;
    for route in lists {
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        let mut copied_route = Vec::with_capacity(route.len().saturating_add(additional_capacity));
        for (index, &item) in route.iter().enumerate() {
            if index.is_multiple_of(1024) && stop.load(Ordering::Relaxed) {
                return None;
            }
            copied_route.push(item);
            copied_items = copied_items.saturating_add(1);
            if hooks.interrupt_clone(kind, copied_items) {
                return None;
            }
        }
        copy.push(copied_route);
    }
    Some(copy)
}

pub(super) fn path_relink_bounded(
    model: &CollectionModel,
    per: &PerList,
    source: &State,
    target: &EliteEntry,
    budget: AlnsWorkBudget,
    stop: &AtomicBool,
) -> PathRelinkRun {
    path_relink_bounded_with_hooks(model, per, source, target, budget, stop, &mut NoRelinkHooks)
}

fn path_relink_bounded_with_hooks<H: RelinkHooks>(
    model: &CollectionModel,
    per: &PerList,
    source: &State,
    target: &EliteEntry,
    budget: AlnsWorkBudget,
    stop: &AtomicBool,
    hooks: &mut H,
) -> PathRelinkRun {
    let started = CpuTimer::start();
    let mut work = RelinkWork::new(budget);
    let initial_status = if stop.load(Ordering::Relaxed) {
        Some(PathRelinkStatus::Interrupted)
    } else if budget.generated == 0 || budget.evaluated == 0 {
        Some(PathRelinkStatus::BudgetExhausted)
    } else {
        None
    };
    if let Some(status) = initial_status {
        return finish_path_relink_run(status, None, &work, &started);
    }

    hooks.before_source_score(stop);
    if stop.load(Ordering::Relaxed) {
        return finish_path_relink_run(PathRelinkStatus::Interrupted, None, &work, &started);
    }
    let mut best_score = full_score_raw(per, source);
    if stop.load(Ordering::Relaxed) {
        return finish_path_relink_run(PathRelinkStatus::Interrupted, None, &work, &started);
    }
    let mut best = None;
    hooks.before_step_limit(stop);
    if stop.load(Ordering::Relaxed) {
        return finish_path_relink_run(PathRelinkStatus::Interrupted, None, &work, &started);
    }
    let step_limit = relink_step_limit(model.items.len());
    let mut status = PathRelinkStatus::Complete;

    let mut lists = match work.reserve_structural(state_clone_work(model, source), stop) {
        Ok(reservation) => match clone_lists_interruptible(&source.lists, 1, stop, RelinkCloneKind::Source, hooks) {
            Some(lists) => {
                work.complete(reservation);
                lists
            }
            None => {
                status = PathRelinkStatus::Interrupted;
                Vec::new()
            }
        },
        Err(abort) => {
            status = abort;
            Vec::new()
        }
    };
    let mut target_reached = false;

    if status == PathRelinkStatus::Complete {
        for _ in 0..step_limit {
            let next = match next_target_move_bounded(&lists, &target.lists, &mut work, stop) {
                Ok(next) => next,
                Err(abort) => {
                    status = abort;
                    break;
                }
            };
            let Some((item, destination, position)) = next else {
                target_reached = true;
                break;
            };
            let source_location = match locate_item_bounded(&lists, item, &mut work, stop) {
                Ok(location) => location,
                Err(abort) => {
                    status = abort;
                    break;
                }
            };
            let Some((source_list, source_position)) = source_location else {
                status = PathRelinkStatus::InfeasibleTarget;
                break;
            };
            let Some(candidate_work) =
                routing_relink_candidate_work(model, per, &lists, source_list, source_position, destination, position, stop)
            else {
                status = PathRelinkStatus::Interrupted;
                break;
            };
            let reservation = match work.reserve_candidate(candidate_work, stop) {
                Ok(reservation) => reservation,
                Err(abort) => {
                    status = abort;
                    break;
                }
            };
            lists[source_list].remove(source_position);
            let position = position.min(lists[destination].len());
            lists[destination].insert(position, item);
            let Some(owned_lists) = clone_lists_interruptible(&lists, 0, stop, RelinkCloneKind::Candidate, hooks) else {
                status = PathRelinkStatus::Interrupted;
                break;
            };
            hooks.before_candidate_rebuild(stop);
            let Some(candidate) = State::from_lists_interruptible(model, per, owned_lists, stop) else {
                status = if stop.load(Ordering::Relaxed) { PathRelinkStatus::Interrupted } else { PathRelinkStatus::InfeasibleTarget };
                break;
            };
            hooks.before_candidate_score(stop);
            if stop.load(Ordering::Relaxed) {
                status = PathRelinkStatus::Interrupted;
                break;
            }
            let score = full_score_raw(per, &candidate);
            work.complete(reservation);
            per.metrics.record_candidate();
            if score.violation == 0 && score < best_score {
                best_score = score;
                best = Some(candidate);
            }
            if stop.load(Ordering::Relaxed) {
                status = PathRelinkStatus::Interrupted;
                break;
            }
        }
    }
    if status == PathRelinkStatus::Complete && !target_reached {
        status = match next_target_move_bounded(&lists, &target.lists, &mut work, stop) {
            Err(abort) => abort,
            Ok(Some(_)) => PathRelinkStatus::BudgetExhausted,
            Ok(None) => PathRelinkStatus::Complete,
        };
    }

    finish_path_relink_run(status, best, &work, &started)
}

fn finish_path_relink_run(status: PathRelinkStatus, candidate: Option<State>, work: &RelinkWork, started: &CpuTimer) -> PathRelinkRun {
    PathRelinkRun {
        status,
        candidate,
        generated: work.generated,
        evaluated: work.evaluated,
        work_units: work.generated.saturating_add(work.evaluated),
        structural_work: work.structural,
        canonical_rebuilds: work.canonical_rebuilds,
        cpu_nanos: started.elapsed_nanos(),
        steps: work.steps,
    }
}

fn relink_step_limit(items: usize) -> usize {
    (items / 4).clamp(1, 32)
}

fn locate_item_bounded(
    lists: &[Vec<i32>],
    item: i32,
    work: &mut RelinkWork,
    stop: &AtomicBool,
) -> Result<Option<(usize, usize)>, PathRelinkStatus> {
    for (list, route) in lists.iter().enumerate() {
        work.structural(1, stop)?;
        for (position, &value) in route.iter().enumerate() {
            work.structural(1, stop)?;
            if value == item {
                return Ok(Some((list, position)));
            }
        }
    }
    Ok(None)
}

fn next_target_move_bounded(
    current: &[Vec<i32>],
    target: &[Vec<i32>],
    work: &mut RelinkWork,
    stop: &AtomicBool,
) -> Result<Option<(i32, usize, usize)>, PathRelinkStatus> {
    work.structural(1, stop)?;
    if current.len() != target.len() {
        return Err(PathRelinkStatus::InfeasibleTarget);
    }
    let mut lengths_match = true;
    for (list, target_route) in target.iter().enumerate() {
        work.structural(1, stop)?;
        lengths_match &= current[list].len() == target_route.len();
        for (position, &item) in target_route.iter().enumerate() {
            work.structural(1, stop)?;
            if current[list].get(position) != Some(&item) {
                return match locate_item_bounded(current, item, work, stop)? {
                    Some(_) => Ok(Some((item, list, position))),
                    None => Err(PathRelinkStatus::InfeasibleTarget),
                };
            }
        }
    }
    if lengths_match {
        Ok(None)
    } else {
        Err(PathRelinkStatus::InfeasibleTarget)
    }
}

fn audit_has_exact_items(lists: &[Vec<i32>], expected: &[i32]) -> bool {
    lists.iter().map(Vec::len).sum::<usize>() == expected.len()
        && expected.iter().all(|expected_item| lists.iter().flatten().filter(|&&item| item == *expected_item).count() == 1)
}

#[cfg(test)]
struct AuditEliteHooks {
    interrupt_task: EliteTask,
    observed: bool,
}

#[cfg(test)]
impl EliteHooks for AuditEliteHooks {
    fn before_task(&mut self, task: EliteTask, stop: &AtomicBool) {
        if task == self.interrupt_task {
            self.observed = true;
            stop.store(true, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
fn audit_elite_task_interruption(task: EliteTask) -> bool {
    let base: Vec<i32> = (1..=8).collect();
    let mut candidate = base.clone();
    candidate.rotate_left(3);
    let mut pool = ElitePool::new(6, false, task == EliteTask::EdgeCanonicalization);
    if task == EliteTask::DistanceCalculation {
        assert!(audit_consider_lists(
            &mut pool,
            std::slice::from_ref(&base),
            &Score { violation: 0, tiers: std::iter::once(10).collect() },
        ));
    }
    let before_len = pool.len();
    let before_hash = pool.best().map(EliteEntry::hash);
    let stop = AtomicBool::new(false);
    let mut hooks = AuditEliteHooks { interrupt_task: task, observed: false };
    let run = pool.consider_lists_with_hooks(
        std::slice::from_ref(&candidate),
        &Score { violation: 0, tiers: std::iter::once(9).collect() },
        EliteWorkBudget::new(u64::MAX),
        &stop,
        &mut hooks,
    );
    hooks.observed
        && stop.load(Ordering::Relaxed)
        && run.status == EliteOperationStatus::Interrupted
        && !run.inserted
        && pool.len() == before_len
        && pool.best().map(EliteEntry::hash) == before_hash
}

#[cfg(test)]
fn audit_bounded_elite_operations() -> bool {
    let base: Vec<i32> = (1..=8).collect();
    let score = Score { violation: 0, tiers: std::iter::once(10).collect() };

    let mut preallocated = ElitePool::new(8, false, false);
    let allocated_capacity = preallocated.entries.capacity();
    let mut all_inserted = true;
    for shift in 0..8 {
        let mut route = base.clone();
        route.rotate_left(shift);
        all_inserted &= audit_consider_lists(
            &mut preallocated,
            std::slice::from_ref(&route),
            &Score { violation: 0, tiers: std::iter::once(20 + shift as i64).collect() },
        );
        all_inserted &= preallocated.entries.capacity() == allocated_capacity;
    }

    let mut budget_pool = ElitePool::new(6, false, false);
    let budget_stop = AtomicBool::new(false);
    let budgeted = budget_pool.consider_lists_with_hooks(
        std::slice::from_ref(&base),
        &score,
        EliteWorkBudget::new(1),
        &budget_stop,
        &mut NoEliteHooks,
    );
    let mut stopped_pool = ElitePool::new(6, false, false);
    let stopped = AtomicBool::new(true);
    let interrupted = stopped_pool.consider_lists_with_hooks(
        std::slice::from_ref(&base),
        &score,
        EliteWorkBudget::new(u64::MAX),
        &stopped,
        &mut NoEliteHooks,
    );

    let model = audit_flat_relink_model(base.len());
    let per = PerList::build(&model);
    let source = State::from_lists(&model, &per, vec![base.clone()]);
    let mut identical_pool = ElitePool::new(6, false, false);
    let archived = identical_pool.consider_bounded(&source, &score, elite_archive_budget(&model), &budget_stop);
    let no_target = identical_pool.select_target_bounded(&source, 17, elite_selection_budget(&model), &budget_stop);
    let selection_budget = identical_pool.select_target_bounded(&source, 17, EliteWorkBudget::new(0), &budget_stop);
    let selection_interrupted = identical_pool.select_target_bounded(&source, 17, elite_selection_budget(&model), &stopped);

    let mut target_route = base.clone();
    target_route.rotate_left(2);
    let target_state = State::from_lists(&model, &per, vec![target_route]);
    let mut target_pool = ElitePool::new(6, false, false);
    let target_archived = target_pool.consider_bounded(&target_state, &score, elite_archive_budget(&model), &budget_stop);
    let distance_stop = AtomicBool::new(false);
    let mut distance_hooks = AuditEliteHooks { interrupt_task: EliteTask::DistanceCalculation, observed: false };
    let selection_distance =
        target_pool.select_target_with_hooks(&source, 17, elite_selection_budget(&model), &distance_stop, &mut distance_hooks);

    allocated_capacity == 8
        && all_inserted
        && preallocated.len() == 8
        && budgeted.status == EliteOperationStatus::BudgetExhausted
        && budgeted.work_units == 1
        && budget_pool.is_empty()
        && interrupted.status == EliteOperationStatus::Interrupted
        && interrupted.work_units == 0
        && stopped_pool.is_empty()
        && archived.status == EliteOperationStatus::Complete
        && archived.inserted
        && archived.work_units <= elite_archive_budget(&model).units
        && no_target.status == EliteOperationStatus::Complete
        && no_target.target.is_none()
        && no_target.work_units <= elite_selection_budget(&model).units
        && selection_budget.status == EliteOperationStatus::BudgetExhausted
        && selection_budget.target.is_none()
        && selection_budget.work_units == 0
        && selection_interrupted.status == EliteOperationStatus::Interrupted
        && selection_interrupted.target.is_none()
        && selection_interrupted.work_units == 0
        && target_archived.status == EliteOperationStatus::Complete
        && target_archived.inserted
        && distance_hooks.observed
        && selection_distance.status == EliteOperationStatus::Interrupted
        && selection_distance.target.is_none()
        && [
            EliteTask::EdgeConstruction,
            EliteTask::EdgeCanonicalization,
            EliteTask::EdgeSorting,
            EliteTask::ListCloning,
            EliteTask::DistanceCalculation,
        ]
        .into_iter()
        .all(audit_elite_task_interruption)
}

#[doc(hidden)]
#[cfg(test)]
pub fn audit_elite_archive() -> bool {
    let score = |value| Score { violation: 0, tiers: std::iter::once(value).collect() };
    let mut pool = ElitePool::new(6, false, false);
    let base: Vec<i32> = (1..=8).collect();
    let first = audit_consider_lists(&mut pool, std::slice::from_ref(&base), &score(20));
    let duplicate_worse = audit_consider_lists(&mut pool, std::slice::from_ref(&base), &score(21));
    let duplicate_better = audit_consider_lists(&mut pool, std::slice::from_ref(&base), &score(19));
    for shift in 1..8 {
        let mut route = base.clone();
        route.rotate_left(shift);
        audit_consider_lists(&mut pool, &[route], &score(30 + shift as i64));
    }
    let assignment_a = vec![vec![1, 2], vec![3, 4]];
    let assignment_b = vec![vec![3, 4], vec![1, 2]];
    let mut homogeneous = ElitePool::new(6, false, false);
    let homogeneous_first = audit_consider_lists(&mut homogeneous, &assignment_a, &score(10));
    let homogeneous_duplicate = audit_consider_lists(&mut homogeneous, &assignment_b, &score(10));
    let mut heterogeneous = ElitePool::new(6, true, false);
    let heterogeneous_first = audit_consider_lists(&mut heterogeneous, &assignment_a, &score(10));
    let heterogeneous_assignment = audit_consider_lists(&mut heterogeneous, &assignment_b, &score(10));

    let route = vec![1, 2, 3, 4];
    let reverse = vec![4, 3, 2, 1];
    let mut symmetric = ElitePool::new(6, false, true);
    let symmetric_first = audit_consider_lists(&mut symmetric, std::slice::from_ref(&route), &score(10));
    let symmetric_duplicate = audit_consider_lists(&mut symmetric, std::slice::from_ref(&reverse), &score(10));
    let mut directed = ElitePool::new(6, false, false);
    let directed_first = audit_consider_lists(&mut directed, std::slice::from_ref(&route), &score(10));
    let directed_reverse = audit_consider_lists(&mut directed, std::slice::from_ref(&reverse), &score(10));

    first
        && !duplicate_worse
        && duplicate_better
        && pool.len() == 6
        && pool.best().is_some_and(|entry| entry.score.tiers[0] == 19)
        && pool.entries.iter().all(|entry| audit_has_exact_items(&entry.lists, &base))
        && homogeneous_first
        && !homogeneous_duplicate
        && heterogeneous_first
        && heterogeneous_assignment
        && symmetric_first
        && !symmetric_duplicate
        && directed_first
        && directed_reverse
        && audit_bounded_elite_operations()
}

#[cfg(test)]
fn audit_consider_lists(pool: &mut ElitePool, lists: &[Vec<i32>], score: &Score) -> bool {
    let stop = AtomicBool::new(false);
    let run = pool.consider_lists_with_hooks(lists, score, EliteWorkBudget::new(u64::MAX), &stop, &mut NoEliteHooks);
    assert_eq!(run.status, EliteOperationStatus::Complete);
    run.inserted
}

#[doc(hidden)]
pub fn audit_relink_bound(items: usize) -> usize {
    relink_step_limit(items)
}

#[doc(hidden)]
pub fn audit_path_relink() -> bool {
    use std::sync::Arc;

    use crate::model::list::{ExprArena, Iterable, ObjectiveTier, ReduceOp, Reduction};

    let mut matrix = vec![vec![10; 5]; 5];
    for (node, row) in matrix.iter_mut().enumerate() {
        row[node] = 0;
    }
    for (from, to) in [(0, 4), (4, 3), (3, 2), (2, 1), (1, 0)] {
        matrix[from][to] = 0;
    }
    let mut arena = ExprArena::default();
    let from = arena.arg(0);
    let to = arena.arg(1);
    let body = arena.matrix(Arc::new(matrix), from, to);
    let model = CollectionModel {
        items: vec![1, 2, 3, 4],
        lists: 1,
        objectives: vec![ObjectiveTier {
            minimize: true,
            terms: vec![Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list: 0, start: 0, end: 0 }, arena, body, coeff: 1 }],
            max_terms: None,
        }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let stop = AtomicBool::new(false);
    let per = PerList::build(&model);
    let source = State::from_lists(&model, &per, vec![vec![1, 2, 3, 4]]);
    let target_state = State::from_lists(&model, &per, vec![vec![4, 3, 2, 1]]);
    let mut pool = ElitePool::new(6, false, false);
    let target_score = full_score_raw(&per, &target_state);
    let archived = pool.consider_bounded(&target_state, &target_score, elite_archive_budget(&model), &stop);
    assert_eq!(archived.status, EliteOperationStatus::Complete);
    assert!(archived.inserted);
    let target = pool.best().expect("the target was inserted");
    let Some(floor) = routing_relink_structural_floor(&model, &per, &source, &stop) else {
        return false;
    };
    let run = path_relink_bounded(&model, &per, &source, target, AlnsWorkBudget::new(floor, 16), &stop);
    let generated_only = path_relink_bounded(&model, &per, &source, target, AlnsWorkBudget::new(1, 0), &stop);
    let interrupted_stop = AtomicBool::new(true);
    let interrupted = path_relink_bounded(&model, &per, &source, target, AlnsWorkBudget::new(16, 16), &interrupted_stop);
    run.candidate.is_some()
        && run.candidate.as_ref().is_some_and(|candidate| audit_has_exact_items(&candidate.lists, &model.items))
        && run.generated <= floor
        && run.evaluated == 1
        && run.steps == 1
        && run.work_units == run.generated + run.evaluated
        && run.structural_work < run.generated
        && run.canonical_rebuilds == 1
        && run.status == PathRelinkStatus::BudgetExhausted
        && generated_only.status == PathRelinkStatus::BudgetExhausted
        && generated_only.generated == 0
        && generated_only.evaluated == 0
        && generated_only.structural_work == 0
        && generated_only.canonical_rebuilds == 0
        && generated_only.steps == 0
        && generated_only.candidate.is_none()
        && interrupted.status == PathRelinkStatus::Interrupted
        && interrupted.generated == 0
        && interrupted.evaluated == 0
        && interrupted.structural_work == 0
        && interrupted.canonical_rebuilds == 0
}

#[cfg(test)]
#[derive(Default)]
struct AuditRelinkHooks {
    source_score_started: bool,
    step_limit_started: bool,
    interrupt_clone: Option<(RelinkCloneKind, usize)>,
    clone_started: bool,
    clone_interrupted: bool,
    stop_before_candidate_rebuild: bool,
    candidate_rebuild_started: bool,
    stop_before_candidate_score: bool,
    candidate_score_started: bool,
}

#[cfg(test)]
impl RelinkHooks for AuditRelinkHooks {
    fn before_source_score(&mut self, _stop: &AtomicBool) {
        self.source_score_started = true;
    }

    fn before_step_limit(&mut self, _stop: &AtomicBool) {
        self.step_limit_started = true;
    }

    fn before_candidate_score(&mut self, stop: &AtomicBool) {
        self.candidate_score_started = true;
        if self.stop_before_candidate_score {
            stop.store(true, Ordering::Relaxed);
        }
    }

    fn interrupt_clone(&mut self, kind: RelinkCloneKind, copied_items: usize) -> bool {
        self.clone_started = true;
        let interrupt = self.interrupt_clone.is_some_and(|(target, limit)| target == kind && copied_items >= limit);
        self.clone_interrupted |= interrupt;
        interrupt
    }

    fn before_candidate_rebuild(&mut self, stop: &AtomicBool) {
        self.candidate_rebuild_started = true;
        if self.stop_before_candidate_rebuild {
            stop.store(true, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
fn audit_flat_relink_model(item_count: usize) -> CollectionModel {
    CollectionModel {
        items: (1..=item_count).map(|item| i32::try_from(item).expect("audit item count fits i32")).collect(),
        lists: 1,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    }
}

#[cfg(test)]
fn audit_target_entry(lists: Vec<Vec<i32>>) -> EliteEntry {
    EliteEntry { lists, score: Score { violation: 0, tiers: std::iter::empty().collect() }, edges: Vec::new(), hash: 0 }
}

#[doc(hidden)]
#[cfg(test)]
pub fn audit_path_relink_interruption_accounting() -> bool {
    const ITEM_COUNT: usize = 2_048;

    let model = audit_flat_relink_model(ITEM_COUNT);
    let per = PerList::build(&model);
    let source = State::from_lists(&model, &per, vec![model.items.clone()]);
    let mut reversed = model.items.clone();
    reversed.reverse();
    let target = audit_target_entry(vec![reversed]);
    let budget = AlnsWorkBudget::new(u64::MAX, u64::MAX);

    let source_stop = AtomicBool::new(false);
    let mut source_hooks = AuditRelinkHooks { interrupt_clone: Some((RelinkCloneKind::Source, 1_025)), ..Default::default() };
    let source_run = path_relink_bounded_with_hooks(&model, &per, &source, &target, budget, &source_stop, &mut source_hooks);

    let candidate_stop = AtomicBool::new(false);
    let mut candidate_hooks = AuditRelinkHooks { interrupt_clone: Some((RelinkCloneKind::Candidate, 1_025)), ..Default::default() };
    let candidate_run = path_relink_bounded_with_hooks(&model, &per, &source, &target, budget, &candidate_stop, &mut candidate_hooks);

    let rebuild_stop = AtomicBool::new(false);
    let mut rebuild_hooks = AuditRelinkHooks { stop_before_candidate_rebuild: true, ..Default::default() };
    let rebuild_run = path_relink_bounded_with_hooks(&model, &per, &source, &target, budget, &rebuild_stop, &mut rebuild_hooks);

    let score_stop = AtomicBool::new(false);
    let mut score_hooks = AuditRelinkHooks { stop_before_candidate_score: true, ..Default::default() };
    let score_run = path_relink_bounded_with_hooks(&model, &per, &source, &target, budget, &score_stop, &mut score_hooks);

    let stopped_guard = AtomicBool::new(true);
    let mut stopped_guard_hooks = AuditRelinkHooks::default();
    let stopped_guard_run =
        path_relink_bounded_with_hooks(&model, &per, &source, &target, budget, &stopped_guard, &mut stopped_guard_hooks);
    let budget_guard = AtomicBool::new(false);
    let mut budget_guard_hooks = AuditRelinkHooks::default();
    let budget_guard_run = path_relink_bounded_with_hooks(
        &model,
        &per,
        &source,
        &target,
        AlnsWorkBudget::new(0, u64::MAX),
        &budget_guard,
        &mut budget_guard_hooks,
    );

    let headroom_stop = AtomicBool::new(false);
    let headroom_source = vec![vec![1, 2, 3], Vec::new()];
    let mut headroom_hooks = AuditRelinkHooks::default();
    let headroom = clone_lists_interruptible(&headroom_source, 1, &headroom_stop, RelinkCloneKind::Source, &mut headroom_hooks)
        .is_some_and(|lists| lists.iter().all(|route| route.capacity() >= route.len().saturating_add(1)));

    source_hooks.clone_interrupted
        && source_run.status == PathRelinkStatus::Interrupted
        && source_run.generated == 0
        && source_run.evaluated == 0
        && source_run.work_units == 0
        && source_run.structural_work == 0
        && source_run.canonical_rebuilds == 0
        && source_run.steps == 0
        && source_run.candidate.is_none()
        && candidate_hooks.clone_interrupted
        && candidate_run.status == PathRelinkStatus::Interrupted
        && candidate_run.generated > 0
        && candidate_run.generated == candidate_run.structural_work
        && candidate_run.evaluated == 0
        && candidate_run.work_units == candidate_run.generated
        && candidate_run.canonical_rebuilds == 0
        && candidate_run.steps == 0
        && candidate_run.candidate.is_none()
        && rebuild_hooks.candidate_rebuild_started
        && rebuild_stop.load(Ordering::Relaxed)
        && rebuild_run.status == PathRelinkStatus::Interrupted
        && rebuild_run.generated == candidate_run.generated
        && rebuild_run.generated == rebuild_run.structural_work
        && rebuild_run.evaluated == 0
        && rebuild_run.work_units == rebuild_run.generated
        && rebuild_run.canonical_rebuilds == 0
        && rebuild_run.steps == 0
        && rebuild_run.candidate.is_none()
        && score_hooks.candidate_score_started
        && score_stop.load(Ordering::Relaxed)
        && score_run.status == PathRelinkStatus::Interrupted
        && score_run.generated == candidate_run.generated
        && score_run.evaluated == 0
        && score_run.canonical_rebuilds == 0
        && score_run.steps == 0
        && score_run.candidate.is_none()
        && stopped_guard_run.status == PathRelinkStatus::Interrupted
        && stopped_guard_run.work_units == 0
        && !stopped_guard_hooks.source_score_started
        && !stopped_guard_hooks.step_limit_started
        && !stopped_guard_hooks.clone_started
        && budget_guard_run.status == PathRelinkStatus::BudgetExhausted
        && budget_guard_run.work_units == 0
        && !budget_guard_hooks.source_score_started
        && !budget_guard_hooks.step_limit_started
        && !budget_guard_hooks.clone_started
        && headroom
}

#[doc(hidden)]
#[cfg(test)]
pub fn audit_path_relink_large_partition() -> bool {
    const ITEM_COUNT: usize = 50_000;

    let model = audit_flat_relink_model(ITEM_COUNT);
    let per = PerList::build(&model);
    let source = State::from_lists(&model, &per, vec![model.items.clone()]);
    let target = audit_target_entry(vec![model.items.clone()]);
    let item_work = u64::try_from(ITEM_COUNT).expect("audit item count fits u64");
    let exact_generated = item_work.saturating_mul(2).saturating_add(3);
    let stop = AtomicBool::new(false);
    let run = path_relink_bounded(&model, &per, &source, &target, AlnsWorkBudget::new(exact_generated, 1), &stop);

    run.status == PathRelinkStatus::Complete
        && run.generated == exact_generated
        && run.structural_work == exact_generated
        && run.evaluated == 0
        && run.work_units == exact_generated
        && run.canonical_rebuilds == 0
        && run.steps == 0
        && run.candidate.is_none()
}
