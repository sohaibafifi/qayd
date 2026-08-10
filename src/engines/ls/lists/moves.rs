use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};

use super::alns::CpuTimer;
use super::eval::{eval_expr, violation_of, INFEASIBLE};
use super::incremental::{EvalScratch, EvaluationInterrupted, ListView};
use super::local_search::{
    active_list_indices, full_score, full_score_exact_lists, full_score_interruptible, full_score_raw, list_score_exact,
    score_with_replacements, score_with_replacements_interruptible, score_with_replacements_raw_interruptible, tier_values,
    ConstraintViolations, ListScore, ObjectiveReductionValues, PerList, ReductionDeltaKind, ReductionValues, Score, State, TrialList,
};
use crate::mix64;
use crate::model::list::{CollectionModel, Iterable, Reduction};

pub(super) fn snapshot(per: &PerList, state: &State) -> (Vec<Vec<i32>>, Score, bool) {
    let score = full_score_raw(per, state);
    let feasible = score.violation == 0;
    (state.lists.clone(), score, feasible)
}

/// Prefer feasible over infeasible, then better lexicographic score.
pub(super) fn better(feasible: bool, score: &Score, best_feasible: bool, best_score: &Score) -> bool {
    match (feasible, best_feasible) {
        (true, false) => true,
        (false, true) => false,
        _ => score < best_score,
    }
}

const MAX_OR_OPT: usize = 3;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum Move {
    Relocate { src: usize, src_pos: usize, dst: usize, dst_pos: usize },
    OrOpt { src: usize, start: usize, len: usize, dst: usize, dst_pos: usize },
    TwoOptStar { a: usize, cut_a: usize, b: usize, cut_b: usize },
    CrossExchange { a: usize, start_a: usize, len_a: usize, b: usize, start_b: usize, len_b: usize },
    Swap { a: usize, a_pos: usize, b: usize, b_pos: usize },
    Reverse { list: usize, i: usize, j: usize },
}

#[derive(Clone, Copy)]
enum ScoreMode {
    Guided,
    Raw,
}

/// Scratch storage shared across speculative move evaluations. None of the
/// buffers retains a view into the state, so one workspace can be reused for
/// every candidate in a scan or repair pass without rebuilding a route.
#[derive(Default)]
pub(super) struct MoveScoreWorkspace {
    scratch_a: EvalScratch,
    scratch_b: EvalScratch,
    score_scratch: EvalScratch,
    overrides: Vec<(i32, usize)>,
}

/// Score of the state if one list were replaced by candidate contents.
fn score_one_mode(
    per: &PerList,
    state: &State,
    list: usize,
    score: &ListScore,
    contents: &dyn ListView,
    scratch: &mut EvalScratch,
    mode: ScoreMode,
) -> Score {
    let stop = AtomicBool::new(false);
    score_one_mode_interruptible(per, state, list, score, contents, scratch, mode, &stop)
        .expect("an uninterrupted one-list score must complete")
}

#[allow(clippy::too_many_arguments)]
fn score_one_mode_interruptible(
    per: &PerList,
    state: &State,
    list: usize,
    score: &ListScore,
    contents: &dyn ListView,
    scratch: &mut EvalScratch,
    mode: ScoreMode,
    stop: &AtomicBool,
) -> Result<Score, EvaluationInterrupted> {
    let replacements = [TrialList { list, score, contents }];
    let result = match mode {
        ScoreMode::Guided => score_with_replacements_interruptible(per, state, &replacements, 0, scratch, stop),
        ScoreMode::Raw => score_with_replacements_raw_interruptible(per, state, &replacements, 0, scratch, stop),
    }?;
    per.metrics.record_candidate();
    Ok(result)
}

fn score_one(per: &PerList, state: &State, list: usize, score: &ListScore, contents: &dyn ListView, scratch: &mut EvalScratch) -> Score {
    score_one_mode(per, state, list, score, contents, scratch, ScoreMode::Guided)
}

/// Score of the state if two lists were replaced by candidate contents.
fn score_two_mode<'a>(
    per: &PerList,
    state: &'a State,
    left: TrialList<'a>,
    right: TrialList<'a>,
    global_delta: i64,
    scratch: &mut EvalScratch,
    mode: ScoreMode,
) -> Score {
    let stop = AtomicBool::new(false);
    score_two_mode_interruptible(per, state, left, right, global_delta, scratch, mode, &stop)
        .expect("an uninterrupted two-list score must complete")
}

#[allow(clippy::too_many_arguments)]
fn score_two_mode_interruptible<'a>(
    per: &PerList,
    state: &'a State,
    left: TrialList<'a>,
    right: TrialList<'a>,
    global_delta: i64,
    scratch: &mut EvalScratch,
    mode: ScoreMode,
    stop: &AtomicBool,
) -> Result<Score, EvaluationInterrupted> {
    let replacements = [left, right];
    let result = match mode {
        ScoreMode::Guided => score_with_replacements_interruptible(per, state, &replacements, global_delta, scratch, stop),
        ScoreMode::Raw => score_with_replacements_raw_interruptible(per, state, &replacements, global_delta, scratch, stop),
    }?;
    per.metrics.record_candidate();
    Ok(result)
}

fn score_two<'a>(
    per: &PerList,
    state: &'a State,
    left: TrialList<'a>,
    right: TrialList<'a>,
    global_delta: i64,
    scratch: &mut EvalScratch,
) -> Score {
    score_two_mode(per, state, left, right, global_delta, scratch, ScoreMode::Guided)
}

/// Copy `list` into `buf` (reusing its capacity) with one edit applied.
fn build_removed(buf: &mut Vec<i32>, list: &[i32], pos: usize) {
    buf.clear();
    buf.extend_from_slice(list);
    buf.remove(pos);
}
fn build_inserted(buf: &mut Vec<i32>, list: &[i32], pos: usize, item: i32) {
    buf.clear();
    buf.extend_from_slice(list);
    buf.insert(pos.min(list.len()), item);
}
fn build_replaced(buf: &mut Vec<i32>, list: &[i32], pos: usize, item: i32) {
    buf.clear();
    buf.extend_from_slice(list);
    buf[pos] = item;
}
fn build_reversed(buf: &mut Vec<i32>, list: &[i32], i: usize, j: usize) {
    buf.clear();
    buf.extend_from_slice(list);
    buf[i..=j].reverse();
}
fn build_segment_removed(buf: &mut Vec<i32>, list: &[i32], start: usize, len: usize) {
    buf.clear();
    buf.extend_from_slice(list);
    buf.drain(start..start + len);
}
fn build_segment_inserted(buf: &mut Vec<i32>, list: &[i32], pos: usize, items: &[i32]) {
    buf.clear();
    buf.extend_from_slice(list);
    let pos = pos.min(buf.len());
    for (offset, &item) in items.iter().enumerate() {
        buf.insert(pos + offset, item);
    }
}
fn build_segment_moved_within(buf: &mut Vec<i32>, list: &[i32], start: usize, len: usize, to: usize) {
    let items = segment_items(list, start, len);
    build_segment_removed(buf, list, start, len);
    let pos = to.min(buf.len());
    for (offset, &item) in items[..len].iter().enumerate() {
        buf.insert(pos + offset, item);
    }
}
fn build_two_opt_star(left: &mut Vec<i32>, right: &mut Vec<i32>, a: &[i32], cut_a: usize, b: &[i32], cut_b: usize) {
    left.clear();
    left.extend_from_slice(&a[..cut_a]);
    left.extend_from_slice(&b[cut_b..]);

    right.clear();
    right.extend_from_slice(&b[..cut_b]);
    right.extend_from_slice(&a[cut_a..]);
}
fn build_cross_exchange(left: &mut Vec<i32>, right: &mut Vec<i32>, a: (&[i32], usize, usize), b: (&[i32], usize, usize)) {
    let (a, start_a, len_a) = a;
    let (b, start_b, len_b) = b;
    left.clear();
    left.extend_from_slice(&a[..start_a]);
    left.extend_from_slice(&b[start_b..start_b + len_b]);
    left.extend_from_slice(&a[start_a + len_a..]);

    right.clear();
    right.extend_from_slice(&b[..start_b]);
    right.extend_from_slice(&a[start_a..start_a + len_a]);
    right.extend_from_slice(&b[start_b + len_b..]);
}
fn build_moved_within(buf: &mut Vec<i32>, list: &[i32], from: usize, to: usize, item: i32) {
    buf.clear();
    buf.extend_from_slice(list);
    buf.remove(from);
    buf.insert(to.min(buf.len()), item);
}

fn segment_items(list: &[i32], start: usize, len: usize) -> [i32; MAX_OR_OPT] {
    let mut items = [0; MAX_OR_OPT];
    items[..len].copy_from_slice(&list[start..start + len]);
    items
}

/// A single local edit to one list. Carries enough to (a) compute the affected
/// reductions' delta in O(edit) and (b) materialise the edited list for the
/// full-recompute fallback. Positions match the `build_*` helpers exactly.
#[derive(Clone, Copy, Debug)]
pub(super) enum Edit {
    Remove { pos: usize },
    Insert { pos: usize, item: i32 },
    MoveWithin { from: usize, to: usize },
    Replace { pos: usize, item: i32 },
    Reverse { i: usize, j: usize },
    SegmentRemove { start: usize, len: usize },
    SegmentInsert { pos: usize, items: [i32; MAX_OR_OPT], len: usize },
    SegmentMoveWithin { start: usize, len: usize, to: usize },
}

impl Edit {
    /// Length of the list after the edit.
    fn new_len(&self, len: usize) -> usize {
        match self {
            Edit::Remove { .. } => len - 1,
            Edit::Insert { .. } => len + 1,
            Edit::SegmentRemove { len: segment_len, .. } => len - segment_len,
            Edit::SegmentInsert { len: segment_len, .. } => len + segment_len,
            _ => len,
        }
    }

    /// Materialise the edited list into `buf` (the fallback path).
    fn apply(&self, buf: &mut Vec<i32>, list: &[i32]) {
        match *self {
            Edit::Remove { pos } => build_removed(buf, list, pos),
            Edit::Insert { pos, item } => build_inserted(buf, list, pos, item),
            Edit::MoveWithin { from, to } => build_moved_within(buf, list, from, to, list[from]),
            Edit::Replace { pos, item } => build_replaced(buf, list, pos, item),
            Edit::Reverse { i, j } => build_reversed(buf, list, i, j),
            Edit::SegmentRemove { start, len } => build_segment_removed(buf, list, start, len),
            Edit::SegmentInsert { pos, items, len } => build_segment_inserted(buf, list, pos, &items[..len]),
            Edit::SegmentMoveWithin { start, len, to } => build_segment_moved_within(buf, list, start, len, to),
        }
    }
}

struct EditView<'a> {
    base: &'a [i32],
    edit: Edit,
}

impl<'a> EditView<'a> {
    fn new(base: &'a [i32], edit: Edit) -> Self {
        Self { base, edit }
    }
}

impl ListView for EditView<'_> {
    fn len(&self) -> usize {
        self.edit.new_len(self.base.len())
    }

    fn at(&self, index: usize) -> i32 {
        match self.edit {
            Edit::Remove { pos } => self.base[if index < pos { index } else { index + 1 }],
            Edit::Insert { pos, item } => {
                let pos = pos.min(self.base.len());
                if index < pos {
                    self.base[index]
                } else if index == pos {
                    item
                } else {
                    self.base[index - 1]
                }
            }
            Edit::MoveWithin { from, to } => {
                let to = to.min(self.base.len() - 1);
                if from < to {
                    if index < from || index > to {
                        self.base[index]
                    } else if index == to {
                        self.base[from]
                    } else {
                        self.base[index + 1]
                    }
                } else if from > to {
                    if index < to || index > from {
                        self.base[index]
                    } else if index == to {
                        self.base[from]
                    } else {
                        self.base[index - 1]
                    }
                } else {
                    self.base[index]
                }
            }
            Edit::Replace { pos, item } => {
                if index == pos {
                    item
                } else {
                    self.base[index]
                }
            }
            Edit::Reverse { i, j } => {
                if (i..=j).contains(&index) {
                    self.base[j - (index - i)]
                } else {
                    self.base[index]
                }
            }
            Edit::SegmentRemove { start, len } => self.base[if index < start { index } else { index + len }],
            Edit::SegmentInsert { pos, items, len } => {
                let pos = pos.min(self.base.len());
                if index < pos {
                    self.base[index]
                } else if index < pos + len {
                    items[index - pos]
                } else {
                    self.base[index - len]
                }
            }
            Edit::SegmentMoveWithin { start, len, to } => {
                let post_len = self.base.len() - len;
                let to = to.min(post_len);
                if (to..to + len).contains(&index) {
                    self.base[start + index - to]
                } else {
                    let removed_index = if index < to { index } else { index - len };
                    self.base[if removed_index < start { removed_index } else { removed_index + len }]
                }
            }
        }
    }

    fn common_prefix_len(&self, _old: &[i32]) -> usize {
        match self.edit {
            Edit::Remove { pos } | Edit::Insert { pos, .. } | Edit::Replace { pos, .. } => pos,
            Edit::MoveWithin { from, to } => from.min(to),
            Edit::Reverse { i, .. } => i,
            Edit::SegmentRemove { start, .. } => start,
            Edit::SegmentInsert { pos, .. } => pos,
            Edit::SegmentMoveWithin { start, to, .. } => start.min(to),
        }
    }

    fn common_suffix_len(&self, _old: &[i32], _prefix: usize) -> usize {
        let old_len = self.base.len();
        match self.edit {
            Edit::Remove { pos } => old_len - pos - 1,
            Edit::Insert { pos, .. } => old_len - pos.min(old_len),
            Edit::MoveWithin { from, to } => old_len - from.max(to) - 1,
            Edit::Replace { pos, .. } => old_len - pos - 1,
            Edit::Reverse { j, .. } => old_len - j - 1,
            Edit::SegmentRemove { start, len } => old_len - start - len,
            Edit::SegmentInsert { pos, .. } => old_len - pos.min(old_len),
            Edit::SegmentMoveWithin { start, len, to } => old_len - (start + len).max(to + len),
        }
    }

    fn common_prefix_len_interruptible(&self, _old: &[i32], stop: &AtomicBool) -> Result<usize, EvaluationInterrupted> {
        if stop.load(Ordering::Relaxed) {
            Err(EvaluationInterrupted)
        } else {
            Ok(match self.edit {
                Edit::Remove { pos } | Edit::Insert { pos, .. } | Edit::Replace { pos, .. } => pos,
                Edit::MoveWithin { from, to } => from.min(to),
                Edit::Reverse { i, .. } => i,
                Edit::SegmentRemove { start, .. } => start,
                Edit::SegmentInsert { pos, .. } => pos,
                Edit::SegmentMoveWithin { start, to, .. } => start.min(to),
            })
        }
    }

    fn common_suffix_len_interruptible(&self, _old: &[i32], _prefix: usize, stop: &AtomicBool) -> Result<usize, EvaluationInterrupted> {
        if stop.load(Ordering::Relaxed) {
            return Err(EvaluationInterrupted);
        }
        let old_len = self.base.len();
        Ok(match self.edit {
            Edit::Remove { pos } => old_len - pos - 1,
            Edit::Insert { pos, .. } => old_len - pos.min(old_len),
            Edit::MoveWithin { from, to } => old_len - from.max(to) - 1,
            Edit::Replace { pos, .. } => old_len - pos - 1,
            Edit::Reverse { j, .. } => old_len - j - 1,
            Edit::SegmentRemove { start, len } => old_len - start - len,
            Edit::SegmentInsert { pos, .. } => old_len - pos.min(old_len),
            Edit::SegmentMoveWithin { start, len, to } => old_len - (start + len).max(to + len),
        })
    }
}

struct ChunkView<'a> {
    chunks: [&'a [i32]; 3],
    count: usize,
    len: usize,
    common_prefix: usize,
    common_suffix: usize,
}

impl<'a> ChunkView<'a> {
    fn two(first: &'a [i32], second: &'a [i32]) -> Self {
        Self { chunks: [first, second, &[]], count: 2, len: first.len() + second.len(), common_prefix: first.len(), common_suffix: 0 }
    }

    fn three(first: &'a [i32], second: &'a [i32], third: &'a [i32]) -> Self {
        Self {
            chunks: [first, second, third],
            count: 3,
            len: first.len() + second.len() + third.len(),
            common_prefix: first.len(),
            common_suffix: third.len(),
        }
    }
}

impl ListView for ChunkView<'_> {
    fn len(&self) -> usize {
        self.len
    }

    fn at(&self, mut index: usize) -> i32 {
        for chunk in &self.chunks[..self.count] {
            if index < chunk.len() {
                return chunk[index];
            }
            index -= chunk.len();
        }
        unreachable!("candidate chunk index out of range")
    }

    fn common_prefix_len(&self, _old: &[i32]) -> usize {
        self.common_prefix
    }

    fn common_suffix_len(&self, _old: &[i32], _prefix: usize) -> usize {
        self.common_suffix
    }

    fn common_prefix_len_interruptible(&self, _old: &[i32], stop: &AtomicBool) -> Result<usize, EvaluationInterrupted> {
        if stop.load(Ordering::Relaxed) {
            Err(EvaluationInterrupted)
        } else {
            Ok(self.common_prefix)
        }
    }

    fn common_suffix_len_interruptible(&self, _old: &[i32], _prefix: usize, stop: &AtomicBool) -> Result<usize, EvaluationInterrupted> {
        if stop.load(Ordering::Relaxed) {
            Err(EvaluationInterrupted)
        } else {
            Ok(self.common_suffix)
        }
    }
}

pub(super) struct SearchMemory {
    inactive: Vec<bool>,
}

impl SearchMemory {
    pub(super) fn new(routes: usize) -> Self {
        Self { inactive: vec![false; routes] }
    }

    pub(super) fn reset_all(&mut self) {
        self.inactive.fill(false);
    }

    pub(super) fn reset_touched(&mut self, mv: Move) {
        for list in mv.touched_lists() {
            if let Some(inactive) = self.inactive.get_mut(list) {
                *inactive = false;
            }
        }
    }

    fn skip(&self, list: usize) -> bool {
        self.inactive.get(list).copied().unwrap_or(false)
    }

    fn mark_inactive(&mut self, list: usize) {
        if let Some(inactive) = self.inactive.get_mut(list) {
            *inactive = true;
        }
    }
}

/// Routing neighbourhoods exposed to the time-sliced search scheduler.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum NeighborhoodKind {
    Relocate,
    Swap,
    OrOpt,
    TwoOptStar,
    CrossExchange,
    Reverse,
}

impl NeighborhoodKind {
    pub(crate) const ALL: [Self; 6] = [Self::Relocate, Self::Swap, Self::OrOpt, Self::TwoOptStar, Self::CrossExchange, Self::Reverse];

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Relocate => "relocate",
            Self::Swap => "swap",
            Self::OrOpt => "or-opt",
            Self::TwoOptStar => "two-opt-star",
            Self::CrossExchange => "cross-exchange",
            Self::Reverse => "reverse",
        }
    }

    pub(crate) const fn name(self) -> &'static str {
        self.as_str()
    }

    pub(crate) const fn index(self) -> usize {
        match self {
            Self::Relocate => 0,
            Self::Swap => 1,
            Self::OrOpt => 2,
            Self::TwoOptStar => 3,
            Self::CrossExchange => 4,
            Self::Reverse => 5,
        }
    }
}

/// Granular scans traverse only moves that create a k-nearest-neighbour edge.
/// Global scans traverse the complete neighbourhood, but retain a cursor so a
/// single call can never turn into an unbounded quadratic pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ScanMode {
    Granular,
    Global,
}

/// Deterministic work allowance for one scheduler slice. `generated` bounds
/// generation work, including cursor transitions that do not emit a move.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WorkBudget {
    pub(crate) generated: u64,
    pub(crate) evaluated: u64,
}

impl WorkBudget {
    pub(crate) const fn new(generated: u64, evaluated: u64) -> Self {
        Self { generated, evaluated }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ScanOutcome {
    Improved(Move),
    Complete,
    BudgetExhausted,
    Interrupted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct NeighborhoodRun {
    pub(super) outcome: ScanOutcome,
    /// Candidate moves emitted by the generator.
    pub(super) generated: u64,
    /// Cursor transitions charged against the generation budget.
    pub(super) generation_work: u64,
    pub(super) evaluated: u64,
    /// Thread CPU exported for profiling, never used for adaptive weighting,
    /// work budgets, or stopping conditions.
    pub(super) cpu_nanos: u64,
}

#[derive(Clone, Copy, Debug)]
struct RoutingLocation {
    route: usize,
    pos: usize,
    flat: usize,
}

struct RoutingIndex {
    locations: HashMap<i32, RoutingLocation>,
    ordered_items: Vec<i32>,
    empty_routes: Vec<usize>,
    max_route_len: usize,
}

impl RoutingIndex {
    fn location(&self, item: i32) -> Option<RoutingLocation> {
        self.locations.get(&item).copied()
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum RoutingIndexBuildPhase {
    Allocate,
    Locations,
    EmptyRoutes,
    Complete,
}

struct RoutingIndexBuilder {
    phase: RoutingIndexBuildPhase,
    locations: HashMap<i32, RoutingLocation>,
    ordered_items: Vec<i32>,
    empty_routes: Vec<usize>,
    route_at: usize,
    item_at: usize,
    empty_route_at: usize,
    flat: usize,
    max_route_len: usize,
}

impl RoutingIndexBuilder {
    fn new() -> Self {
        Self {
            phase: RoutingIndexBuildPhase::Allocate,
            locations: HashMap::new(),
            ordered_items: Vec::new(),
            empty_routes: Vec::new(),
            route_at: 0,
            item_at: 0,
            empty_route_at: 0,
            flat: 0,
            max_route_len: 0,
        }
    }

    /// Advance by one deterministic structural unit.  The caller admits and
    /// charges every call before allowing another one, so index construction is
    /// resumable under the same generation budget as candidate enumeration.
    fn advance(&mut self, state: &State, interchangeable_lists: bool, stop: &AtomicBool) -> IndexBuildStep {
        if stop.load(Ordering::Relaxed) {
            return IndexBuildStep::Interrupted;
        }
        match self.phase {
            RoutingIndexBuildPhase::Allocate => {
                let capacity = state.item_list.len();
                self.locations = HashMap::with_capacity(capacity);
                self.ordered_items = Vec::with_capacity(capacity);
                self.empty_routes = Vec::with_capacity(if interchangeable_lists { 1 } else { state.lists.len() });
                self.phase = RoutingIndexBuildPhase::Locations;
                IndexBuildStep::Progress
            }
            RoutingIndexBuildPhase::Locations => {
                let Some(route) = state.lists.get(self.route_at) else {
                    self.phase = RoutingIndexBuildPhase::EmptyRoutes;
                    return IndexBuildStep::Progress;
                };
                self.max_route_len = self.max_route_len.max(route.len());
                if let Some(&item) = route.get(self.item_at) {
                    self.locations.insert(item, RoutingLocation { route: self.route_at, pos: self.item_at, flat: self.flat });
                    self.ordered_items.push(item);
                    self.item_at += 1;
                    self.flat += 1;
                } else {
                    self.route_at += 1;
                    self.item_at = 0;
                }
                IndexBuildStep::Progress
            }
            RoutingIndexBuildPhase::EmptyRoutes => {
                let Some(route) = state.lists.get(self.empty_route_at) else {
                    self.phase = RoutingIndexBuildPhase::Complete;
                    return IndexBuildStep::Complete;
                };
                if route.is_empty() {
                    self.empty_routes.push(self.empty_route_at);
                }
                self.empty_route_at += 1;
                if (interchangeable_lists && !self.empty_routes.is_empty()) || self.empty_route_at >= state.lists.len() {
                    self.phase = RoutingIndexBuildPhase::Complete;
                    IndexBuildStep::Complete
                } else {
                    IndexBuildStep::Progress
                }
            }
            RoutingIndexBuildPhase::Complete => IndexBuildStep::Complete,
        }
    }

    fn finish(self) -> RoutingIndex {
        debug_assert!(self.phase == RoutingIndexBuildPhase::Complete);
        RoutingIndex {
            locations: self.locations,
            ordered_items: self.ordered_items,
            empty_routes: self.empty_routes,
            max_route_len: self.max_route_len,
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum IndexBuildStep {
    Progress,
    Complete,
    Interrupted,
}

/// Topology-dependent data shared by all six granular neighbourhoods.  It is
/// invalidated explicitly after every accepted topology change and is built at
/// most once for the resulting state.
pub(super) struct RoutingIndexCache {
    index: Option<RoutingIndex>,
    builder: Option<RoutingIndexBuilder>,
    interchangeable_lists: Option<bool>,
    generation: u64,
    #[cfg(test)]
    completed_builds: u64,
}

impl Default for RoutingIndexCache {
    fn default() -> Self {
        Self {
            index: None,
            builder: None,
            interchangeable_lists: None,
            generation: 1,
            #[cfg(test)]
            completed_builds: 0,
        }
    }
}

impl RoutingIndexCache {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn reset(&mut self) {
        self.index = None;
        self.builder = None;
        self.interchangeable_lists = None;
        self.generation = self.generation.wrapping_add(1).max(1);
    }

    fn is_ready(&self, interchangeable_lists: bool) -> bool {
        self.interchangeable_lists == Some(interchangeable_lists) && self.index.is_some()
    }

    fn advance(&mut self, state: &State, interchangeable_lists: bool, stop: &AtomicBool) -> IndexBuildStep {
        if self.interchangeable_lists != Some(interchangeable_lists) {
            self.reset();
            self.interchangeable_lists = Some(interchangeable_lists);
            self.builder = Some(RoutingIndexBuilder::new());
        }
        if self.index.is_some() {
            return IndexBuildStep::Complete;
        }
        let builder = self.builder.get_or_insert_with(RoutingIndexBuilder::new);
        let step = builder.advance(state, interchangeable_lists, stop);
        if step == IndexBuildStep::Complete {
            let completed = self.builder.take().expect("a completed routing index builder must exist");
            self.index = Some(completed.finish());
            #[cfg(test)]
            {
                self.completed_builds = self.completed_builds.saturating_add(1);
            }
        }
        step
    }

    fn index(&self) -> Option<&RoutingIndex> {
        self.index.as_ref()
    }
}

pub(super) struct RoutingScanWorkspace<'a> {
    index_cache: &'a mut RoutingIndexCache,
    memory: &'a mut RoutingScanMemory,
}

impl<'a> RoutingScanWorkspace<'a> {
    pub(super) fn new(index_cache: &'a mut RoutingIndexCache, memory: &'a mut RoutingScanMemory) -> Self {
        Self { index_cache, memory }
    }
}

/// Exact compact set for generated moves. Normal routing dimensions encode in
/// one `u64`; only dimensions whose mixed-radix key overflows fall back to the
/// full enum. This keeps the six persistent granular scans small without
/// accepting hash collisions or duplicate candidates.
struct CompactMoveSet {
    routes: u64,
    positions: u64,
    compact: HashSet<u64>,
    overflow: HashSet<Move>,
}

impl CompactMoveSet {
    fn new(routes: usize, max_route_len: usize) -> Self {
        let routes = u64::try_from(routes).unwrap_or(u64::MAX).max(1);
        let positions = max_route_len.checked_add(1).and_then(|len| u64::try_from(len).ok()).unwrap_or(u64::MAX).max(1);
        Self { routes, positions, compact: HashSet::new(), overflow: HashSet::new() }
    }

    fn append(value: usize, radix: u64, key: &mut u64) -> Option<()> {
        let value = u64::try_from(value).ok()?;
        if value >= radix {
            return None;
        }
        *key = key.checked_mul(radix)?.checked_add(value)?;
        Some(())
    }

    fn encode(&self, mv: Move) -> Option<u64> {
        let mut key = 0;
        let append_route = |value, key: &mut u64| Self::append(value, self.routes, key);
        let append_pos = |value, key: &mut u64| Self::append(value, self.positions, key);
        match mv {
            Move::Relocate { src, src_pos, dst, dst_pos } => {
                Self::append(0, NeighborhoodKind::ALL.len() as u64, &mut key)?;
                append_route(src, &mut key)?;
                append_pos(src_pos, &mut key)?;
                append_route(dst, &mut key)?;
                append_pos(dst_pos, &mut key)?;
            }
            Move::Swap { a, a_pos, b, b_pos } => {
                Self::append(1, NeighborhoodKind::ALL.len() as u64, &mut key)?;
                append_route(a, &mut key)?;
                append_pos(a_pos, &mut key)?;
                append_route(b, &mut key)?;
                append_pos(b_pos, &mut key)?;
            }
            Move::OrOpt { src, start, len, dst, dst_pos } => {
                Self::append(2, NeighborhoodKind::ALL.len() as u64, &mut key)?;
                append_route(src, &mut key)?;
                append_pos(start, &mut key)?;
                Self::append(len, (MAX_OR_OPT + 1) as u64, &mut key)?;
                append_route(dst, &mut key)?;
                append_pos(dst_pos, &mut key)?;
            }
            Move::TwoOptStar { a, cut_a, b, cut_b } => {
                Self::append(3, NeighborhoodKind::ALL.len() as u64, &mut key)?;
                append_route(a, &mut key)?;
                append_pos(cut_a, &mut key)?;
                append_route(b, &mut key)?;
                append_pos(cut_b, &mut key)?;
            }
            Move::CrossExchange { a, start_a, len_a, b, start_b, len_b } => {
                Self::append(4, NeighborhoodKind::ALL.len() as u64, &mut key)?;
                append_route(a, &mut key)?;
                append_pos(start_a, &mut key)?;
                Self::append(len_a, (MAX_OR_OPT + 1) as u64, &mut key)?;
                append_route(b, &mut key)?;
                append_pos(start_b, &mut key)?;
                Self::append(len_b, (MAX_OR_OPT + 1) as u64, &mut key)?;
            }
            Move::Reverse { list, i, j } => {
                Self::append(5, NeighborhoodKind::ALL.len() as u64, &mut key)?;
                append_route(list, &mut key)?;
                append_pos(i, &mut key)?;
                append_pos(j, &mut key)?;
            }
        }
        Some(key)
    }

    fn insert(&mut self, mv: Move) -> bool {
        self.encode(mv).map_or_else(|| self.overflow.insert(mv), |key| self.compact.insert(key))
    }
}

#[derive(Clone)]
enum GlobalCursor {
    Relocate { src: usize, src_pos: usize, dst: usize, dst_pos: usize },
    Swap { a: usize, a_pos: usize, b: usize, b_pos: usize },
    OrOpt { src: usize, start: usize, len: usize, dst: usize, dst_pos: usize },
    TwoOptStar { a: usize, cut_a: usize, b: usize, cut_b: usize },
    CrossExchange { a: usize, start_a: usize, len_a: usize, b: usize, start_b: usize, len_b: usize },
    Reverse { route: usize, i: usize, j: usize },
    Done,
}

impl GlobalCursor {
    fn new(kind: NeighborhoodKind) -> Self {
        match kind {
            NeighborhoodKind::Relocate => Self::Relocate { src: 0, src_pos: 0, dst: 0, dst_pos: 0 },
            NeighborhoodKind::Swap => Self::Swap { a: 0, a_pos: 0, b: 1, b_pos: 0 },
            NeighborhoodKind::OrOpt => Self::OrOpt { src: 0, start: 0, len: 2, dst: 0, dst_pos: 0 },
            NeighborhoodKind::TwoOptStar => Self::TwoOptStar { a: 0, cut_a: 0, b: 1, cut_b: 0 },
            NeighborhoodKind::CrossExchange => Self::CrossExchange { a: 0, start_a: 0, len_a: 1, b: 1, start_b: 0, len_b: 1 },
            NeighborhoodKind::Reverse => Self::Reverse { route: 0, i: 0, j: 1 },
        }
    }
}

enum GranularSpecialCursor {
    Relocate { empty_at: usize, item_at: usize },
    OrOpt { empty_at: usize, item_at: usize, len: usize },
    Done,
}

const MAX_GRANULAR_EMPTY_DESTINATIONS: usize = 8;

struct GranularCursor {
    special: GranularSpecialCursor,
    item_at: usize,
    neighbor_at: usize,
    variants: Vec<Move>,
    variant_at: usize,
    seen: CompactMoveSet,
}

impl GranularCursor {
    fn new(kind: NeighborhoodKind, state: &State, index: &RoutingIndex) -> Self {
        let special = if index.empty_routes.is_empty() {
            GranularSpecialCursor::Done
        } else {
            match kind {
                NeighborhoodKind::Relocate => GranularSpecialCursor::Relocate { empty_at: 0, item_at: 0 },
                NeighborhoodKind::OrOpt => GranularSpecialCursor::OrOpt { empty_at: 0, item_at: 0, len: 2 },
                _ => GranularSpecialCursor::Done,
            }
        };
        Self {
            special,
            item_at: 0,
            neighbor_at: 0,
            variants: Vec::new(),
            variant_at: 0,
            seen: CompactMoveSet::new(state.lists.len(), index.max_route_len),
        }
    }
}

/// Resumable state for one routing-neighbourhood scan.  The shared index
/// generation invalidates granular cursors after an accepted move, while
/// repeated budgeted calls over an unchanged solution resume exactly where the
/// prior call stopped.
pub(super) struct RoutingScanMemory {
    kind: Option<NeighborhoodKind>,
    mode: Option<ScanMode>,
    index_generation: u64,
    granular: Option<GranularCursor>,
    global: GlobalCursor,
    pending: Option<Move>,
}

impl Default for RoutingScanMemory {
    fn default() -> Self {
        Self { kind: None, mode: None, index_generation: 0, granular: None, global: GlobalCursor::Done, pending: None }
    }
}

impl RoutingScanMemory {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn reset(&mut self) {
        *self = Self::default();
    }
}

impl Move {
    fn touched_lists(self) -> [usize; 2] {
        match self {
            Move::Relocate { src, dst, .. } | Move::OrOpt { src, dst, .. } => [src, dst],
            Move::TwoOptStar { a, b, .. } | Move::CrossExchange { a, b, .. } | Move::Swap { a, b, .. } => [a, b],
            Move::Reverse { list, .. } => [list, list],
        }
    }
}

/// Delta of a per-item (order-independent) value reduction under `edit`, where
/// `value(item)` is the body contribution of one item.
fn items_value_delta(list: &[i32], edit: &Edit, value: impl Fn(i32) -> i64) -> i64 {
    match *edit {
        Edit::Remove { pos } => -value(list[pos]),
        Edit::Insert { item, .. } => value(item),
        Edit::Replace { pos, item } => value(item).saturating_sub(value(list[pos])),
        Edit::SegmentRemove { start, len } => list[start..start + len].iter().fold(0i64, |delta, &item| delta.saturating_sub(value(item))),
        Edit::SegmentInsert { items, len, .. } => items[..len].iter().fold(0i64, |delta, &item| delta.saturating_add(value(item))),
        Edit::MoveWithin { .. } | Edit::Reverse { .. } | Edit::SegmentMoveWithin { .. } => 0,
    }
}

/// Delta of one supported reduction's raw value under `edit`.
fn reduction_delta(r: &Reduction, kind: ReductionDeltaKind, list: &[i32], edit: &Edit) -> i64 {
    match kind {
        ReductionDeltaKind::ItemsCount => {
            items_value_delta(list, edit, |item| i64::from(eval_expr(&r.arena.exprs, r.body, &[i64::from(item)]) != 0))
        }
        ReductionDeltaKind::Used => {
            let old = list.len();
            i64::from(edit.new_len(old) > 0) - i64::from(old > 0)
        }
        ReductionDeltaKind::Unsupported => unreachable!("reduction_delta on an unsupported reduction"),
    }
}

#[allow(clippy::too_many_arguments)]
fn candidate_reduction_value_interruptible(
    per: &PerList,
    reduction: &Reduction,
    kind: ReductionDeltaKind,
    cache: &super::incremental::ReductionCache,
    old: &[i32],
    candidate: &dyn ListView,
    edit: Option<&Edit>,
    scratch: &mut EvalScratch,
    stop: &AtomicBool,
) -> Result<Option<i64>, EvaluationInterrupted> {
    if stop.load(Ordering::Relaxed) {
        return Err(EvaluationInterrupted);
    }
    // Count and Used are the only algebraic deltas that remain exact across
    // every i64 saturation boundary. Sum reductions use the ordered prefix and
    // suffix transforms in ReductionCache instead: adding a delta to an already
    // saturated total is not equivalent to replaying saturating_add.
    let exact_raw_delta = matches!(kind, ReductionDeltaKind::ItemsCount | ReductionDeltaKind::Used);
    let value = match (exact_raw_delta, edit) {
        (true, Some(edit)) => per.metrics.measure_delta(reduction, || {
            cache.raw_value().map(|raw| raw.saturating_add(reduction_delta(reduction, kind, old, edit)).saturating_mul(reduction.coeff))
        }),
        _ => per.metrics.measure_delta(reduction, || cache.candidate_value_interruptible(reduction, old, candidate, scratch, stop))?,
    };
    if stop.load(Ordering::Relaxed) {
        return Err(EvaluationInterrupted);
    }
    if matches!(reduction.iterable, Iterable::Scan { .. }) {
        per.metrics.record_incremental_scan(scratch.recomputed_scan_steps());
    }
    Ok(value)
}

/// Trial score from accepted-state reduction caches. Local edits retain the
/// existing O(edit) formulas for Count/Used; every other reduction recomputes
/// only its affected output span through `candidate`.
pub(super) fn trial_list_score_view(
    per: &PerList,
    state: &State,
    idx: usize,
    candidate: &dyn ListView,
    edit: Option<&Edit>,
    scratch: &mut EvalScratch,
) -> ListScore {
    let stop = AtomicBool::new(false);
    trial_list_score_view_interruptible(per, state, idx, candidate, edit, scratch, &stop)
        .expect("an uninterrupted trial-list score must complete")
}

#[allow(clippy::too_many_arguments)]
fn trial_list_score_view_interruptible(
    per: &PerList,
    state: &State,
    idx: usize,
    candidate: &dyn ListView,
    edit: Option<&Edit>,
    scratch: &mut EvalScratch,
    stop: &AtomicBool,
) -> Result<ListScore, EvaluationInterrupted> {
    if stop.load(Ordering::Relaxed) {
        return Err(EvaluationInterrupted);
    }
    let old = &state.lists[idx];
    let caches = &state.caches[idx];
    let mut violation = 0i64;
    let mut undefined_violation = 0i64;
    let mut objectives = tier_values(per.tiers, 0);
    let mut objective_reductions = ObjectiveReductionValues::with_capacity(per.tiers);
    for (tier, slot) in objectives.iter_mut().enumerate() {
        let mut values = ReductionValues::with_capacity(per.objective[idx][tier].len());
        for ((reduction, kind), cache) in per.objective[idx][tier].iter().zip(&per.objective_delta[idx][tier]).zip(&caches.objective[tier])
        {
            match candidate_reduction_value_interruptible(per, reduction, *kind, cache, old, candidate, edit, scratch, stop)? {
                Some(value) => {
                    *slot = slot.saturating_add(value);
                    values.push(Some(value));
                }
                None => {
                    violation = violation.saturating_add(INFEASIBLE);
                    undefined_violation = undefined_violation.saturating_add(INFEASIBLE);
                    values.push(None);
                }
            }
        }
        objective_reductions.push(values);
    }
    let mut constraint_violations = ConstraintViolations::with_capacity(per.constraints[idx].len());
    for ((constraint, kind), cache) in per.constraints[idx].iter().zip(&per.constraint_delta[idx]).zip(&caches.constraints) {
        match candidate_reduction_value_interruptible(per, &constraint.reduction, *kind, cache, old, candidate, edit, scratch, stop)? {
            Some(value) => {
                let constraint_violation = violation_of(value, constraint.op, constraint.rhs);
                violation = violation.saturating_add(constraint_violation);
                constraint_violations.push(constraint_violation);
            }
            None => {
                violation = violation.saturating_add(INFEASIBLE);
                undefined_violation = undefined_violation.saturating_add(INFEASIBLE);
                constraint_violations.push(0);
            }
        }
    }
    let edge_penalty = per.edge_penalty_interruptible(idx, candidate, stop)?;
    if stop.load(Ordering::Relaxed) {
        return Err(EvaluationInterrupted);
    }
    per.metrics.record_incremental_trial();
    Ok(ListScore { violation, objectives, constraint_violations, objective_reductions, undefined_violation, edge_penalty })
}

fn rotated_routes(mut routes: Vec<usize>, seed: u64) -> Vec<usize> {
    if seed != 0 && !routes.is_empty() {
        let offset = (mix64(seed) % routes.len() as u64) as usize;
        routes.rotate_left(offset);
    }
    routes
}

fn trial_list_score(per: &PerList, state: &State, idx: usize, edit: Edit, scratch: &mut EvalScratch) -> ListScore {
    let candidate = EditView::new(&state.lists[idx], edit);
    trial_list_score_view(per, state, idx, &candidate, Some(&edit), scratch)
}

fn trial_list_score_interruptible(
    per: &PerList,
    state: &State,
    idx: usize,
    edit: Edit,
    scratch: &mut EvalScratch,
    stop: &AtomicBool,
) -> Result<ListScore, EvaluationInterrupted> {
    let candidate = EditView::new(&state.lists[idx], edit);
    trial_list_score_view_interruptible(per, state, idx, &candidate, Some(&edit), scratch, stop)
}

/// Test oracle: for `model` with the given list contents, assert the incremental
/// trial score of every single-list edit equals a full recompute of the edited
/// list. Panics on the first mismatch; returns the number of edits checked.
/// Exposed for the integration oracle, which keeps the actual test out of `src/`.
#[doc(hidden)]
pub fn audit_incremental(model: &CollectionModel, lists: &[Vec<i32>]) -> usize {
    let per = PerList::build(model);
    let state = State::from_lists(model, &per, lists.to_vec());

    let mut scratch = EvalScratch::default();
    let mut right_scratch = EvalScratch::default();
    let mut score_scratch = EvalScratch::default();
    let mut full_buf = Vec::new();
    let mut checked = 0usize;
    let same = |a: &ListScore, b: &ListScore| a.violation == b.violation && a.objectives == b.objectives;
    for (idx, list) in lists.iter().enumerate() {
        let n = list.len();
        let mut edits: Vec<Edit> = Vec::new();
        for pos in 0..n {
            edits.push(Edit::Remove { pos });
        }
        for &item in &model.items {
            for pos in 0..=n {
                edits.push(Edit::Insert { pos, item });
            }
            for pos in 0..n {
                edits.push(Edit::Replace { pos, item });
            }
        }
        for from in 0..n {
            for to in 0..n {
                if from != to {
                    edits.push(Edit::MoveWithin { from, to });
                }
            }
        }
        for i in 0..n {
            for j in (i + 1)..n {
                edits.push(Edit::Reverse { i, j });
            }
        }
        for len in 2..=MAX_OR_OPT.min(n) {
            for start in 0..=n - len {
                edits.push(Edit::SegmentRemove { start, len });
                let items = segment_items(list, start, len);
                for pos in 0..=n {
                    edits.push(Edit::SegmentInsert { pos, items, len });
                }
                for to in 0..=n - len {
                    if to != start {
                        edits.push(Edit::SegmentMoveWithin { start, len, to });
                    }
                }
            }
        }
        for edit in edits {
            let view = EditView::new(list, edit);
            let inc = trial_list_score(&per, &state, idx, edit, &mut scratch);
            let incremental_total =
                score_with_replacements(&per, &state, &[TrialList { list: idx, score: &inc, contents: &view }], 0, &mut score_scratch);
            edit.apply(&mut full_buf, list);
            let full = list_score_exact(&per, idx, &full_buf);
            assert!(
                same(&inc, &full),
                "incremental != full for list {idx} edit {edit:?}: inc=({},{:?}) full=({},{:?})",
                inc.violation,
                inc.objectives,
                full.violation,
                full.objectives
            );
            let mut full_lists = lists.to_vec();
            full_lists[idx] = full_buf.clone();
            let full_total = full_score_exact_lists(&per, &full_lists, state.global_viol);
            assert!(
                incremental_total == full_total,
                "incremental total != full total for list {idx} edit {edit:?}: inc=({},{:?}) full=({},{:?})",
                incremental_total.violation,
                incremental_total.tiers,
                full_total.violation,
                full_total.tiers
            );
            checked += 1;
        }
    }

    for left in 0..lists.len() {
        for right in (left + 1)..lists.len() {
            for cut_left in 0..=lists[left].len() {
                for cut_right in 0..=lists[right].len() {
                    if cut_left == lists[left].len() && cut_right == lists[right].len() {
                        continue;
                    }
                    let left_view = ChunkView::two(&lists[left][..cut_left], &lists[right][cut_right..]);
                    let right_view = ChunkView::two(&lists[right][..cut_right], &lists[left][cut_left..]);
                    let left_score = trial_list_score_view(&per, &state, left, &left_view, None, &mut scratch);
                    let right_score = trial_list_score_view(&per, &state, right, &right_view, None, &mut right_scratch);
                    let incremental_total = score_with_replacements(
                        &per,
                        &state,
                        &[
                            TrialList { list: left, score: &left_score, contents: &left_view },
                            TrialList { list: right, score: &right_score, contents: &right_view },
                        ],
                        0,
                        &mut score_scratch,
                    );
                    let mut materialized_left = Vec::new();
                    let mut materialized_right = Vec::new();
                    build_two_opt_star(&mut materialized_left, &mut materialized_right, &lists[left], cut_left, &lists[right], cut_right);
                    let mut full_lists = lists.to_vec();
                    full_lists[left] = materialized_left;
                    full_lists[right] = materialized_right;
                    let full_total = full_score_exact_lists(&per, &full_lists, state.global_viol);
                    assert!(incremental_total == full_total, "2-opt* incremental total differs from full recomputation");
                    checked += 1;
                }
            }

            for len_left in 1..=MAX_OR_OPT.min(lists[left].len()) {
                for start_left in 0..=lists[left].len() - len_left {
                    for len_right in 1..=MAX_OR_OPT.min(lists[right].len()) {
                        for start_right in 0..=lists[right].len() - len_right {
                            let left_view = ChunkView::three(
                                &lists[left][..start_left],
                                &lists[right][start_right..start_right + len_right],
                                &lists[left][start_left + len_left..],
                            );
                            let right_view = ChunkView::three(
                                &lists[right][..start_right],
                                &lists[left][start_left..start_left + len_left],
                                &lists[right][start_right + len_right..],
                            );
                            let left_score = trial_list_score_view(&per, &state, left, &left_view, None, &mut scratch);
                            let right_score = trial_list_score_view(&per, &state, right, &right_view, None, &mut right_scratch);
                            let incremental_total = score_with_replacements(
                                &per,
                                &state,
                                &[
                                    TrialList { list: left, score: &left_score, contents: &left_view },
                                    TrialList { list: right, score: &right_score, contents: &right_view },
                                ],
                                0,
                                &mut score_scratch,
                            );
                            let mut materialized_left = Vec::new();
                            let mut materialized_right = Vec::new();
                            build_cross_exchange(
                                &mut materialized_left,
                                &mut materialized_right,
                                (&lists[left], start_left, len_left),
                                (&lists[right], start_right, len_right),
                            );
                            let mut full_lists = lists.to_vec();
                            full_lists[left] = materialized_left;
                            full_lists[right] = materialized_right;
                            let full_total = full_score_exact_lists(&per, &full_lists, state.global_viol);
                            assert!(incremental_total == full_total, "cross-exchange incremental total differs from full recomputation");
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    checked
}

fn route_prev_next(per: &PerList, state: &State, list: usize, pos: usize) -> Option<(i32, i32)> {
    let (start, end) = per.route_bounds.get(list).copied().flatten()?;
    let route = state.lists.get(list)?;
    let prev = if pos == 0 { start } else { *route.get(pos - 1)? };
    let next = if pos == route.len() { end } else { *route.get(pos)? };
    Some((prev, next))
}

fn route_before_after(per: &PerList, state: &State, list: usize, start_pos: usize, len: usize) -> Option<(i32, i32)> {
    let (start, end) = per.route_bounds.get(list).copied().flatten()?;
    let route = state.lists.get(list)?;
    let before = if start_pos == 0 { start } else { *route.get(start_pos - 1)? };
    let after_pos = start_pos + len;
    let after = if after_pos == route.len() { end } else { *route.get(after_pos)? };
    Some((before, after))
}

fn candidate_edge(per: &PerList, a: i32, b: i32) -> bool {
    per.candidates.as_ref().is_none_or(|candidates| candidates.contains(a, b))
}

fn candidate_insert(per: &PerList, prev: i32, next: i32, first: i32, last: i32) -> bool {
    candidate_edge(per, prev, first) || candidate_edge(per, last, next)
}

fn candidate_segment_insert(per: &PerList, state: &State, list: usize, pos: usize, first: i32, last: i32) -> bool {
    route_prev_next(per, state, list, pos).is_none_or(|(prev, next)| candidate_insert(per, prev, next, first, last))
}

fn candidate_two_opt_star(per: &PerList, state: &State, a: usize, cut_a: usize, b: usize, cut_b: usize) -> bool {
    match (route_prev_next(per, state, a, cut_a), route_prev_next(per, state, b, cut_b)) {
        (Some((a_prev, a_next)), Some((b_prev, b_next))) => candidate_edge(per, a_prev, b_next) || candidate_edge(per, b_prev, a_next),
        _ => true,
    }
}

fn candidate_cross_exchange(per: &PerList, state: &State, a: (usize, usize, usize), b: (usize, usize, usize)) -> bool {
    let (a, start_a, len_a) = a;
    let (b, start_b, len_b) = b;
    let route_a = &state.lists[a];
    let route_b = &state.lists[b];
    let first_a = route_a[start_a];
    let last_a = route_a[start_a + len_a - 1];
    let first_b = route_b[start_b];
    let last_b = route_b[start_b + len_b - 1];
    match (route_before_after(per, state, a, start_a, len_a), route_before_after(per, state, b, start_b, len_b)) {
        (Some((a_before, a_after)), Some((b_before, b_after))) => {
            candidate_edge(per, a_before, first_b)
                || candidate_edge(per, last_b, a_after)
                || candidate_edge(per, b_before, first_a)
                || candidate_edge(per, last_a, b_after)
        }
        _ => true,
    }
}

fn candidate_swap(per: &PerList, state: &State, a: usize, a_pos: usize, b: usize, b_pos: usize) -> bool {
    let a_item = state.lists[a][a_pos];
    let b_item = state.lists[b][b_pos];
    match (route_before_after(per, state, a, a_pos, 1), route_before_after(per, state, b, b_pos, 1)) {
        (Some((a_before, a_after)), Some((b_before, b_after))) => {
            candidate_edge(per, a_before, b_item)
                || candidate_edge(per, b_item, a_after)
                || candidate_edge(per, b_before, a_item)
                || candidate_edge(per, a_item, b_after)
        }
        _ => true,
    }
}

fn push_unique(moves: &mut Vec<Move>, seen: &mut CompactMoveSet, mv: Move) {
    if seen.insert(mv) {
        moves.push(mv);
    }
}

fn relocate_next_to(
    state: &State,
    moving: RoutingLocation,
    anchor: RoutingLocation,
    after: bool,
    moves: &mut Vec<Move>,
    seen: &mut CompactMoveSet,
) {
    if moving.flat == anchor.flat {
        return;
    }
    let dst_pos = if moving.route == anchor.route {
        let anchor_after_removal = anchor.pos - usize::from(moving.pos < anchor.pos);
        anchor_after_removal + usize::from(after)
    } else {
        anchor.pos + usize::from(after)
    };
    let post_len = state.lists[moving.route].len() - usize::from(moving.route == anchor.route);
    if moving.route == anchor.route && (dst_pos > post_len || dst_pos == moving.pos) {
        return;
    }
    push_unique(moves, seen, Move::Relocate { src: moving.route, src_pos: moving.pos, dst: anchor.route, dst_pos });
}

fn segment_starts_at_endpoint(route_len: usize, pos: usize, len: usize) -> impl Iterator<Item = usize> {
    let first = pos.checked_add(len).filter(|&end| end <= route_len).map(|_| pos);
    let last = pos.checked_add(1).and_then(|end| end.checked_sub(len));
    [first, last].into_iter().flatten()
}

#[allow(clippy::too_many_arguments)]
fn or_opt_next_to(
    state: &State,
    endpoint: RoutingLocation,
    anchor: RoutingLocation,
    len: usize,
    start: usize,
    after: bool,
    moves: &mut Vec<Move>,
    seen: &mut CompactMoveSet,
) {
    if endpoint.route == anchor.route && (start..start + len).contains(&anchor.pos) {
        return;
    }
    let dst_pos = if endpoint.route == anchor.route {
        let anchor_after_removal = anchor.pos - usize::from(anchor.pos >= start + len) * len;
        anchor_after_removal + usize::from(after)
    } else {
        anchor.pos + usize::from(after)
    };
    let post_len = state.lists[endpoint.route].len() - usize::from(endpoint.route == anchor.route) * len;
    if endpoint.route == anchor.route && (dst_pos > post_len || dst_pos == start) {
        return;
    }
    push_unique(moves, seen, Move::OrOpt { src: endpoint.route, start, len, dst: anchor.route, dst_pos });
}

fn push_two_opt_star(a: RoutingLocation, cut_a: usize, b: RoutingLocation, cut_b: usize, moves: &mut Vec<Move>, seen: &mut CompactMoveSet) {
    if a.route == b.route {
        return;
    }
    let mv = if a.route < b.route {
        Move::TwoOptStar { a: a.route, cut_a, b: b.route, cut_b }
    } else {
        Move::TwoOptStar { a: b.route, cut_a: cut_b, b: a.route, cut_b: cut_a }
    };
    push_unique(moves, seen, mv);
}

#[allow(clippy::too_many_arguments)]
fn push_cross(
    a: usize,
    start_a: usize,
    len_a: usize,
    b: usize,
    start_b: usize,
    len_b: usize,
    moves: &mut Vec<Move>,
    seen: &mut CompactMoveSet,
) {
    if a == b || (len_a == 1 && len_b == 1) {
        return;
    }
    let mv = if a < b {
        Move::CrossExchange { a, start_a, len_a, b, start_b, len_b }
    } else {
        Move::CrossExchange { a: b, start_a: start_b, len_a: len_b, b: a, start_b: start_a, len_b: len_a }
    };
    push_unique(moves, seen, mv);
}

fn cross_from_boundary(
    state: &State,
    boundary: RoutingLocation,
    endpoint: RoutingLocation,
    moves: &mut Vec<Move>,
    seen: &mut CompactMoveSet,
) {
    if boundary.route == endpoint.route {
        return;
    }
    let a_len = state.lists[boundary.route].len();
    let b_len = state.lists[endpoint.route].len();
    for len_a in 1..=MAX_OR_OPT.min(a_len) {
        for len_b in 1..=MAX_OR_OPT.min(b_len) {
            // boundary is immediately before the removed segment and endpoint
            // is the first item of the segment inserted in its place.
            if boundary.pos + 1 + len_a <= a_len && endpoint.pos + len_b <= b_len {
                push_cross(boundary.route, boundary.pos + 1, len_a, endpoint.route, endpoint.pos, len_b, moves, seen);
            }
            // boundary is immediately after the removed segment and endpoint
            // is the final item of the segment inserted in its place.
            if boundary.pos >= len_a && endpoint.pos + 1 >= len_b {
                push_cross(boundary.route, boundary.pos - len_a, len_a, endpoint.route, endpoint.pos + 1 - len_b, len_b, moves, seen);
            }
        }
    }
}

fn push_reverse(route: usize, i: usize, j: usize, moves: &mut Vec<Move>, seen: &mut CompactMoveSet) {
    if i < j {
        push_unique(moves, seen, Move::Reverse { list: route, i, j });
    }
}

const MAX_BOUNDARY_ROUTE_PROBES: usize = 8;

/// Moves that make `item` adjacent to a route boundary represented by a
/// candidate-graph node such as the depot.  Boundary nodes are shared by many
/// physical routes, so probe a deterministic bounded ring of destinations.
/// This restores inter-route beginning/end moves without multiplying every
/// candidate edge by the fleet size.
fn granular_boundary_moves(
    kind: NeighborhoodKind,
    per: &PerList,
    state: &State,
    index: &RoutingIndex,
    item: RoutingLocation,
    boundary: i32,
    seen: &mut CompactMoveSet,
) -> Vec<Move> {
    let mut moves = Vec::new();
    let routes = state.lists.len();
    if routes == 0 {
        return moves;
    }
    let source_len = state.lists[item.route].len();

    for offset in 0..routes.min(MAX_BOUNDARY_ROUTE_PROBES) {
        let route = (item.route + offset) % routes;
        if per.interchangeable_lists.get() && state.lists[route].is_empty() && index.empty_routes.first().copied() != Some(route) {
            continue;
        }
        let Some((start, end)) = per.route_bounds.get(route).copied().flatten() else {
            continue;
        };
        let at_start = start == boundary;
        let at_end = end == boundary;
        if !at_start && !at_end {
            continue;
        }
        let destination_len = state.lists[route].len();

        match kind {
            NeighborhoodKind::Relocate => {
                let end_pos = if route == item.route { source_len.saturating_sub(1) } else { destination_len };
                if at_start && (route != item.route || item.pos != 0) {
                    push_unique(&mut moves, seen, Move::Relocate { src: item.route, src_pos: item.pos, dst: route, dst_pos: 0 });
                }
                if at_end && (route != item.route || item.pos != end_pos) {
                    push_unique(&mut moves, seen, Move::Relocate { src: item.route, src_pos: item.pos, dst: route, dst_pos: end_pos });
                }
            }
            NeighborhoodKind::Swap => {
                if route != item.route {
                    if at_start && destination_len > 0 {
                        let (a, a_pos, b, b_pos) =
                            if item.route < route { (item.route, item.pos, route, 0) } else { (route, 0, item.route, item.pos) };
                        push_unique(&mut moves, seen, Move::Swap { a, a_pos, b, b_pos });
                    }
                    if at_end && destination_len > 0 {
                        let boundary_pos = destination_len - 1;
                        let (a, a_pos, b, b_pos) = if item.route < route {
                            (item.route, item.pos, route, boundary_pos)
                        } else {
                            (route, boundary_pos, item.route, item.pos)
                        };
                        push_unique(&mut moves, seen, Move::Swap { a, a_pos, b, b_pos });
                    }
                }
            }
            NeighborhoodKind::OrOpt => {
                for len in 2..=MAX_OR_OPT.min(source_len) {
                    for segment_start in segment_starts_at_endpoint(source_len, item.pos, len) {
                        let end_pos = if route == item.route { source_len - len } else { destination_len };
                        if at_start && segment_start == item.pos && (route != item.route || segment_start != 0) {
                            push_unique(
                                &mut moves,
                                seen,
                                Move::OrOpt { src: item.route, start: segment_start, len, dst: route, dst_pos: 0 },
                            );
                        }
                        if at_end && segment_start + len - 1 == item.pos && (route != item.route || segment_start != end_pos) {
                            push_unique(
                                &mut moves,
                                seen,
                                Move::OrOpt { src: item.route, start: segment_start, len, dst: route, dst_pos: end_pos },
                            );
                        }
                    }
                }
            }
            NeighborhoodKind::TwoOptStar => {
                if route != item.route {
                    let virtual_boundary = RoutingLocation { route, pos: 0, flat: usize::MAX };
                    if at_start {
                        push_two_opt_star(virtual_boundary, 0, item, item.pos, &mut moves, seen);
                    }
                    if at_end {
                        push_two_opt_star(virtual_boundary, destination_len, item, item.pos + 1, &mut moves, seen);
                    }
                }
            }
            NeighborhoodKind::CrossExchange => {
                if route != item.route && destination_len > 0 {
                    for source_segment_len in 1..=MAX_OR_OPT.min(source_len) {
                        for source_start in segment_starts_at_endpoint(source_len, item.pos, source_segment_len) {
                            for destination_segment_len in 1..=MAX_OR_OPT.min(destination_len) {
                                if at_start && source_start == item.pos {
                                    push_cross(
                                        item.route,
                                        source_start,
                                        source_segment_len,
                                        route,
                                        0,
                                        destination_segment_len,
                                        &mut moves,
                                        seen,
                                    );
                                }
                                if at_end && source_start + source_segment_len - 1 == item.pos {
                                    push_cross(
                                        item.route,
                                        source_start,
                                        source_segment_len,
                                        route,
                                        destination_len - destination_segment_len,
                                        destination_segment_len,
                                        &mut moves,
                                        seen,
                                    );
                                }
                            }
                        }
                    }
                }
            }
            NeighborhoodKind::Reverse => {
                if route == item.route {
                    if at_start && item.pos > 0 {
                        push_reverse(item.route, 0, item.pos, &mut moves, seen);
                    }
                    if at_end && item.pos + 1 < source_len {
                        push_reverse(item.route, item.pos, source_len - 1, &mut moves, seen);
                    }
                }
            }
        }
    }
    moves
}

fn granular_anchor_moves(
    kind: NeighborhoodKind,
    per: &PerList,
    state: &State,
    index: &RoutingIndex,
    edge: (i32, i32),
    seen: &mut CompactMoveSet,
) -> Vec<Move> {
    let mut moves = Vec::new();
    let left = index.location(edge.0);
    let right = index.location(edge.1);
    if let (Some(left), Some(right)) = (left, right) {
        match kind {
            NeighborhoodKind::Relocate => {
                for after in [false, true] {
                    relocate_next_to(state, left, right, after, &mut moves, seen);
                    relocate_next_to(state, right, left, after, &mut moves, seen);
                }
            }
            NeighborhoodKind::Swap => {
                for (moving, anchor) in [(left, right), (right, left)] {
                    for adjacent in [anchor.pos.checked_sub(1), anchor.pos.checked_add(1)] {
                        let Some(pos) = adjacent.filter(|&pos| pos < state.lists[anchor.route].len()) else {
                            continue;
                        };
                        if moving.route == anchor.route {
                            continue;
                        }
                        let mv = if moving.route < anchor.route {
                            Move::Swap { a: moving.route, a_pos: moving.pos, b: anchor.route, b_pos: pos }
                        } else {
                            Move::Swap { a: anchor.route, a_pos: pos, b: moving.route, b_pos: moving.pos }
                        };
                        push_unique(&mut moves, seen, mv);
                    }
                }
            }
            NeighborhoodKind::OrOpt => {
                for (endpoint, anchor) in [(left, right), (right, left)] {
                    let route_len = state.lists[endpoint.route].len();
                    for len in 2..=MAX_OR_OPT.min(route_len) {
                        for start in segment_starts_at_endpoint(route_len, endpoint.pos, len) {
                            // If the candidate endpoint is the segment's first
                            // item, inserting after the anchor creates anchor ->
                            // first. If it is the final item, inserting before the
                            // anchor creates final -> anchor.
                            if start == endpoint.pos {
                                or_opt_next_to(state, endpoint, anchor, len, start, true, &mut moves, seen);
                            }
                            if start + len - 1 == endpoint.pos {
                                or_opt_next_to(state, endpoint, anchor, len, start, false, &mut moves, seen);
                            }
                        }
                    }
                }
            }
            NeighborhoodKind::TwoOptStar => {
                push_two_opt_star(left, left.pos + 1, right, right.pos, &mut moves, seen);
                push_two_opt_star(left, left.pos, right, right.pos + 1, &mut moves, seen);
            }
            NeighborhoodKind::CrossExchange => {
                cross_from_boundary(state, left, right, &mut moves, seen);
                cross_from_boundary(state, right, left, &mut moves, seen);
            }
            NeighborhoodKind::Reverse => {
                if left.route == right.route {
                    push_reverse(left.route, left.pos + 1, right.pos, &mut moves, seen);
                    push_reverse(left.route, right.pos + 1, left.pos, &mut moves, seen);
                    if let Some(j) = right.pos.checked_sub(1) {
                        push_reverse(left.route, left.pos, j, &mut moves, seen);
                    }
                    if let Some(j) = left.pos.checked_sub(1) {
                        push_reverse(left.route, right.pos, j, &mut moves, seen);
                    }
                }
            }
        }
    }
    if let Some(item) = left {
        moves.extend(granular_boundary_moves(kind, per, state, index, item, edge.1, seen));
    }
    if let Some(item) = right {
        moves.extend(granular_boundary_moves(kind, per, state, index, item, edge.0, seen));
    }
    moves
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GenerationStep {
    Candidate(Move),
    Progress,
    Complete,
    Interrupted,
}

fn next_granular_special(
    state: &State,
    index: &RoutingIndex,
    cursor: &mut GranularSpecialCursor,
    seen: &mut CompactMoveSet,
) -> GenerationStep {
    let empty_route = |empty_at: usize| {
        let probes = index.empty_routes.len().min(MAX_GRANULAR_EMPTY_DESTINATIONS);
        if empty_at < probes {
            Some(index.empty_routes[empty_at.saturating_mul(index.empty_routes.len()) / probes])
        } else {
            None
        }
    };
    match cursor {
        GranularSpecialCursor::Relocate { empty_at, item_at } => {
            let Some(empty) = empty_route(*empty_at) else {
                *cursor = GranularSpecialCursor::Done;
                return GenerationStep::Progress;
            };
            let Some(&item) = index.ordered_items.get(*item_at) else {
                *empty_at += 1;
                *item_at = 0;
                return GenerationStep::Progress;
            };
            let location = index.location(item).expect("every ordered routing item has a location");
            let mv = Move::Relocate { src: location.route, src_pos: location.pos, dst: empty, dst_pos: 0 };
            *item_at += 1;
            if seen.insert(mv) {
                GenerationStep::Candidate(mv)
            } else {
                GenerationStep::Progress
            }
        }
        GranularSpecialCursor::OrOpt { empty_at, item_at, len } => {
            let Some(empty) = empty_route(*empty_at) else {
                *cursor = GranularSpecialCursor::Done;
                return GenerationStep::Progress;
            };
            let Some(&item) = index.ordered_items.get(*item_at) else {
                *empty_at += 1;
                *item_at = 0;
                *len = 2;
                return GenerationStep::Progress;
            };
            let location = index.location(item).expect("every ordered routing item has a location");
            let src_len = state.lists[location.route].len();
            if *len > MAX_OR_OPT.min(src_len) {
                *item_at += 1;
                *len = 2;
                return GenerationStep::Progress;
            }
            if location.pos + *len > src_len {
                *len += 1;
                return GenerationStep::Progress;
            }
            let mv = Move::OrOpt { src: location.route, start: location.pos, len: *len, dst: empty, dst_pos: 0 };
            *len += 1;
            if seen.insert(mv) {
                GenerationStep::Candidate(mv)
            } else {
                GenerationStep::Progress
            }
        }
        GranularSpecialCursor::Done => GenerationStep::Complete,
    }
}

fn next_granular(
    kind: NeighborhoodKind,
    per: &PerList,
    state: &State,
    index: &RoutingIndex,
    cursor: &mut GranularCursor,
    stop: &AtomicBool,
) -> GenerationStep {
    if stop.load(Ordering::Relaxed) {
        return GenerationStep::Interrupted;
    }
    if !matches!(cursor.special, GranularSpecialCursor::Done) {
        return next_granular_special(state, index, &mut cursor.special, &mut cursor.seen);
    }
    if let Some(&mv) = cursor.variants.get(cursor.variant_at) {
        cursor.variant_at += 1;
        return GenerationStep::Candidate(mv);
    }
    let Some(candidates) = &per.candidates else {
        return GenerationStep::Complete;
    };
    let Some(&item) = index.ordered_items.get(cursor.item_at) else {
        return GenerationStep::Complete;
    };
    let neighbors = candidates.routing_neighbors(item);
    let Some(&other) = neighbors.get(cursor.neighbor_at) else {
        cursor.item_at += 1;
        cursor.neighbor_at = 0;
        return GenerationStep::Progress;
    };
    cursor.neighbor_at += 1;
    // The routing graph is symmetric.  Emit an item-item edge only from its
    // canonical endpoint; a boundary has no ordered item entry and is emitted
    // from the physical endpoint unconditionally.
    if item == other || (index.location(other).is_some() && item > other) {
        return GenerationStep::Progress;
    }
    cursor.variants = granular_anchor_moves(kind, per, state, index, (item, other), &mut cursor.seen);
    cursor.variant_at = 0;
    GenerationStep::Progress
}

#[allow(clippy::too_many_lines)]
fn next_global(cursor: &mut GlobalCursor, state: &State, stop: &AtomicBool) -> GenerationStep {
    if stop.load(Ordering::Relaxed) {
        return GenerationStep::Interrupted;
    }
    let routes = state.lists.len();
    match cursor {
        GlobalCursor::Relocate { src, src_pos, dst, dst_pos } => {
            if *src >= routes {
                *cursor = GlobalCursor::Done;
                return GenerationStep::Progress;
            }
            let src_len = state.lists[*src].len();
            if *src_pos >= src_len {
                *src += 1;
                *src_pos = 0;
                *dst = 0;
                *dst_pos = 0;
                return GenerationStep::Progress;
            }
            if *dst >= routes {
                *src_pos += 1;
                *dst = 0;
                *dst_pos = 0;
                return GenerationStep::Progress;
            }
            let positions = if *dst == *src { src_len } else { state.lists[*dst].len() + 1 };
            if *dst_pos >= positions {
                *dst += 1;
                *dst_pos = 0;
                return GenerationStep::Progress;
            }
            let mv = Move::Relocate { src: *src, src_pos: *src_pos, dst: *dst, dst_pos: *dst_pos };
            *dst_pos += 1;
            if let Move::Relocate { src_pos, dst, dst_pos, .. } = mv {
                if dst == *src && dst_pos == src_pos {
                    return GenerationStep::Progress;
                }
            }
            GenerationStep::Candidate(mv)
        }
        GlobalCursor::Swap { a, a_pos, b, b_pos } => {
            if *a >= routes {
                *cursor = GlobalCursor::Done;
                return GenerationStep::Progress;
            }
            if *b >= routes {
                *a += 1;
                *a_pos = 0;
                *b = a.saturating_add(1);
                *b_pos = 0;
                return GenerationStep::Progress;
            }
            if *a_pos >= state.lists[*a].len() {
                *b += 1;
                *a_pos = 0;
                *b_pos = 0;
                return GenerationStep::Progress;
            }
            if *b_pos >= state.lists[*b].len() {
                *a_pos += 1;
                *b_pos = 0;
                return GenerationStep::Progress;
            }
            let mv = Move::Swap { a: *a, a_pos: *a_pos, b: *b, b_pos: *b_pos };
            *b_pos += 1;
            GenerationStep::Candidate(mv)
        }
        GlobalCursor::OrOpt { src, start, len, dst, dst_pos } => {
            if *src >= routes {
                *cursor = GlobalCursor::Done;
                return GenerationStep::Progress;
            }
            let src_len = state.lists[*src].len();
            let max_len = MAX_OR_OPT.min(src_len);
            if *len > max_len {
                *src += 1;
                *start = 0;
                *len = 2;
                *dst = 0;
                *dst_pos = 0;
                return GenerationStep::Progress;
            }
            if *start + *len > src_len {
                *len += 1;
                *start = 0;
                *dst = 0;
                *dst_pos = 0;
                return GenerationStep::Progress;
            }
            if *dst >= routes {
                *start += 1;
                *dst = 0;
                *dst_pos = 0;
                return GenerationStep::Progress;
            }
            let post_len = if *dst == *src { src_len - *len } else { state.lists[*dst].len() };
            if *dst_pos > post_len {
                *dst += 1;
                *dst_pos = 0;
                return GenerationStep::Progress;
            }
            let mv = Move::OrOpt { src: *src, start: *start, len: *len, dst: *dst, dst_pos: *dst_pos };
            *dst_pos += 1;
            if let Move::OrOpt { src, start, dst, dst_pos, .. } = mv {
                if src == dst && start == dst_pos {
                    return GenerationStep::Progress;
                }
            }
            GenerationStep::Candidate(mv)
        }
        GlobalCursor::TwoOptStar { a, cut_a, b, cut_b } => {
            if *a >= routes {
                *cursor = GlobalCursor::Done;
                return GenerationStep::Progress;
            }
            if *b >= routes {
                *a += 1;
                *cut_a = 0;
                *b = a.saturating_add(1);
                *cut_b = 0;
                return GenerationStep::Progress;
            }
            let len_a = state.lists[*a].len();
            let len_b = state.lists[*b].len();
            if *cut_a > len_a {
                *b += 1;
                *cut_a = 0;
                *cut_b = 0;
                return GenerationStep::Progress;
            }
            if *cut_b > len_b {
                *cut_a += 1;
                *cut_b = 0;
                return GenerationStep::Progress;
            }
            let mv = Move::TwoOptStar { a: *a, cut_a: *cut_a, b: *b, cut_b: *cut_b };
            *cut_b += 1;
            if let Move::TwoOptStar { cut_a, cut_b, .. } = mv {
                if cut_a == len_a && cut_b == len_b {
                    return GenerationStep::Progress;
                }
            }
            GenerationStep::Candidate(mv)
        }
        GlobalCursor::CrossExchange { a, start_a, len_a, b, start_b, len_b } => {
            if *a >= routes {
                *cursor = GlobalCursor::Done;
                return GenerationStep::Progress;
            }
            if *b >= routes {
                *a += 1;
                *start_a = 0;
                *len_a = 1;
                *b = a.saturating_add(1);
                *start_b = 0;
                *len_b = 1;
                return GenerationStep::Progress;
            }
            let route_a_len = state.lists[*a].len();
            let route_b_len = state.lists[*b].len();
            let max_a = MAX_OR_OPT.min(route_a_len);
            let max_b = MAX_OR_OPT.min(route_b_len);
            if *len_a > max_a {
                *b += 1;
                *start_a = 0;
                *len_a = 1;
                *start_b = 0;
                *len_b = 1;
                return GenerationStep::Progress;
            }
            if *start_a + *len_a > route_a_len {
                *len_a += 1;
                *start_a = 0;
                *start_b = 0;
                *len_b = 1;
                return GenerationStep::Progress;
            }
            if *len_b > max_b {
                *start_a += 1;
                *start_b = 0;
                *len_b = 1;
                return GenerationStep::Progress;
            }
            if *start_b + *len_b > route_b_len {
                *len_b += 1;
                *start_b = 0;
                return GenerationStep::Progress;
            }
            let mv = Move::CrossExchange { a: *a, start_a: *start_a, len_a: *len_a, b: *b, start_b: *start_b, len_b: *len_b };
            *start_b += 1;
            if let Move::CrossExchange { len_a, len_b, .. } = mv {
                if len_a == 1 && len_b == 1 {
                    return GenerationStep::Progress;
                }
            }
            GenerationStep::Candidate(mv)
        }
        GlobalCursor::Reverse { route, i, j } => {
            if *route >= routes {
                *cursor = GlobalCursor::Done;
                return GenerationStep::Progress;
            }
            let len = state.lists[*route].len();
            if *i + 1 >= len {
                *route += 1;
                *i = 0;
                *j = 1;
                return GenerationStep::Progress;
            }
            if *j >= len {
                *i += 1;
                *j = *i + 1;
                return GenerationStep::Progress;
            }
            let mv = Move::Reverse { list: *route, i: *i, j: *j };
            *j += 1;
            GenerationStep::Candidate(mv)
        }
        GlobalCursor::Done => GenerationStep::Complete,
    }
}

fn score_move_mode_interruptible(
    per: &PerList,
    state: &State,
    mv: Move,
    workspace: &mut MoveScoreWorkspace,
    mode: ScoreMode,
    stop: &AtomicBool,
) -> Result<Option<Score>, EvaluationInterrupted> {
    macro_rules! valid_or_none {
        ($value:expr) => {
            match $value {
                Some(value) => value,
                None => return Ok(None),
            }
        };
    }
    if stop.load(Ordering::Relaxed) {
        return Err(EvaluationInterrupted);
    }
    let MoveScoreWorkspace { scratch_a, scratch_b, score_scratch, overrides } = workspace;
    match mv {
        Move::Relocate { src, src_pos, dst, dst_pos } => {
            let source = valid_or_none!(state.lists.get(src));
            let &item = valid_or_none!(source.get(src_pos));
            if src == dst {
                if dst_pos >= state.lists[src].len() || dst_pos == src_pos {
                    return Ok(None);
                }
                let edit = Edit::MoveWithin { from: src_pos, to: dst_pos };
                let next = trial_list_score_interruptible(per, state, src, edit, scratch_a, stop)?;
                let view = EditView::new(&state.lists[src], edit);
                Ok(Some(score_one_mode_interruptible(per, state, src, &next, &view, score_scratch, mode, stop)?))
            } else {
                let destination = valid_or_none!(state.lists.get(dst));
                if dst_pos > destination.len() {
                    return Ok(None);
                }
                let source_edit = Edit::Remove { pos: src_pos };
                let destination_edit = Edit::Insert { pos: dst_pos, item };
                let left = trial_list_score_interruptible(per, state, src, source_edit, scratch_a, stop)?;
                let right = trial_list_score_interruptible(per, state, dst, destination_edit, scratch_b, stop)?;
                let left_view = EditView::new(&state.lists[src], source_edit);
                let right_view = EditView::new(&state.lists[dst], destination_edit);
                let global_delta = per.globals.delta_interruptible(&state.item_list, &[(item, dst)], stop)?;
                Ok(Some(score_two_mode_interruptible(
                    per,
                    state,
                    TrialList { list: src, score: &left, contents: &left_view },
                    TrialList { list: dst, score: &right, contents: &right_view },
                    global_delta,
                    score_scratch,
                    mode,
                    stop,
                )?))
            }
        }
        Move::Swap { a, a_pos, b, b_pos } => {
            if a == b {
                return Ok(None);
            }
            let left_route = valid_or_none!(state.lists.get(a));
            let right_route = valid_or_none!(state.lists.get(b));
            let (&left_item, &right_item) = (valid_or_none!(left_route.get(a_pos)), valid_or_none!(right_route.get(b_pos)));
            let left_edit = Edit::Replace { pos: a_pos, item: right_item };
            let right_edit = Edit::Replace { pos: b_pos, item: left_item };
            let left = trial_list_score_interruptible(per, state, a, left_edit, scratch_a, stop)?;
            let right = trial_list_score_interruptible(per, state, b, right_edit, scratch_b, stop)?;
            let left_view = EditView::new(&state.lists[a], left_edit);
            let right_view = EditView::new(&state.lists[b], right_edit);
            let global_delta = per.globals.delta_interruptible(&state.item_list, &[(left_item, b), (right_item, a)], stop)?;
            Ok(Some(score_two_mode_interruptible(
                per,
                state,
                TrialList { list: a, score: &left, contents: &left_view },
                TrialList { list: b, score: &right, contents: &right_view },
                global_delta,
                score_scratch,
                mode,
                stop,
            )?))
        }
        Move::OrOpt { src, start, len, dst, dst_pos } => {
            let source = valid_or_none!(state.lists.get(src));
            if !(2..=MAX_OR_OPT).contains(&len) || start + len > source.len() {
                return Ok(None);
            }
            let items = segment_items(source, start, len);
            if src == dst {
                let post_len = source.len() - len;
                if dst_pos > post_len || dst_pos == start {
                    return Ok(None);
                }
                let edit = Edit::SegmentMoveWithin { start, len, to: dst_pos };
                let next = trial_list_score_interruptible(per, state, src, edit, scratch_a, stop)?;
                let view = EditView::new(source, edit);
                Ok(Some(score_one_mode_interruptible(per, state, src, &next, &view, score_scratch, mode, stop)?))
            } else {
                let destination = valid_or_none!(state.lists.get(dst));
                if dst_pos > destination.len() {
                    return Ok(None);
                }
                let source_edit = Edit::SegmentRemove { start, len };
                let destination_edit = Edit::SegmentInsert { pos: dst_pos, items, len };
                let left = trial_list_score_interruptible(per, state, src, source_edit, scratch_a, stop)?;
                let right = trial_list_score_interruptible(per, state, dst, destination_edit, scratch_b, stop)?;
                let left_view = EditView::new(source, source_edit);
                let right_view = EditView::new(&state.lists[dst], destination_edit);
                overrides.clear();
                overrides.extend(items[..len].iter().map(|&item| (item, dst)));
                let global_delta = per.globals.delta_interruptible(&state.item_list, overrides, stop)?;
                Ok(Some(score_two_mode_interruptible(
                    per,
                    state,
                    TrialList { list: src, score: &left, contents: &left_view },
                    TrialList { list: dst, score: &right, contents: &right_view },
                    global_delta,
                    score_scratch,
                    mode,
                    stop,
                )?))
            }
        }
        Move::TwoOptStar { a, cut_a, b, cut_b } => {
            let left_route = valid_or_none!(state.lists.get(a));
            let right_route = valid_or_none!(state.lists.get(b));
            if a == b || cut_a > left_route.len() || cut_b > right_route.len() {
                return Ok(None);
            }
            if cut_a == state.lists[a].len() && cut_b == state.lists[b].len() {
                return Ok(None);
            }
            let left_view = ChunkView::two(&state.lists[a][..cut_a], &state.lists[b][cut_b..]);
            let right_view = ChunkView::two(&state.lists[b][..cut_b], &state.lists[a][cut_a..]);
            let left = trial_list_score_view_interruptible(per, state, a, &left_view, None, scratch_a, stop)?;
            let right = trial_list_score_view_interruptible(per, state, b, &right_view, None, scratch_b, stop)?;
            overrides.clear();
            for (index, &item) in state.lists[a][cut_a..].iter().enumerate() {
                if index.is_multiple_of(64) && stop.load(Ordering::Relaxed) {
                    return Err(EvaluationInterrupted);
                }
                overrides.push((item, b));
            }
            for (index, &item) in state.lists[b][cut_b..].iter().enumerate() {
                if index.is_multiple_of(64) && stop.load(Ordering::Relaxed) {
                    return Err(EvaluationInterrupted);
                }
                overrides.push((item, a));
            }
            let global_delta = per.globals.delta_interruptible(&state.item_list, overrides, stop)?;
            Ok(Some(score_two_mode_interruptible(
                per,
                state,
                TrialList { list: a, score: &left, contents: &left_view },
                TrialList { list: b, score: &right, contents: &right_view },
                global_delta,
                score_scratch,
                mode,
                stop,
            )?))
        }
        Move::CrossExchange { a, start_a, len_a, b, start_b, len_b } => {
            let left_route = valid_or_none!(state.lists.get(a));
            let right_route = valid_or_none!(state.lists.get(b));
            if a == b
                || len_a == 0
                || len_b == 0
                || len_a > MAX_OR_OPT
                || len_b > MAX_OR_OPT
                || start_a + len_a > left_route.len()
                || start_b + len_b > right_route.len()
            {
                return Ok(None);
            }
            let left_view =
                ChunkView::three(&state.lists[a][..start_a], &state.lists[b][start_b..start_b + len_b], &state.lists[a][start_a + len_a..]);
            let right_view =
                ChunkView::three(&state.lists[b][..start_b], &state.lists[a][start_a..start_a + len_a], &state.lists[b][start_b + len_b..]);
            let left = trial_list_score_view_interruptible(per, state, a, &left_view, None, scratch_a, stop)?;
            let right = trial_list_score_view_interruptible(per, state, b, &right_view, None, scratch_b, stop)?;
            overrides.clear();
            overrides.extend(state.lists[a][start_a..start_a + len_a].iter().map(|&item| (item, b)));
            overrides.extend(state.lists[b][start_b..start_b + len_b].iter().map(|&item| (item, a)));
            let global_delta = per.globals.delta_interruptible(&state.item_list, overrides, stop)?;
            Ok(Some(score_two_mode_interruptible(
                per,
                state,
                TrialList { list: a, score: &left, contents: &left_view },
                TrialList { list: b, score: &right, contents: &right_view },
                global_delta,
                score_scratch,
                mode,
                stop,
            )?))
        }
        Move::Reverse { list, i, j } => {
            let route = valid_or_none!(state.lists.get(list));
            if i >= j || j >= route.len() {
                return Ok(None);
            }
            let edit = Edit::Reverse { i, j };
            let next = trial_list_score_interruptible(per, state, list, edit, scratch_a, stop)?;
            let view = EditView::new(&state.lists[list], edit);
            Ok(Some(score_one_mode_interruptible(per, state, list, &next, &view, score_scratch, mode, stop)?))
        }
    }
}

fn score_move(
    per: &PerList,
    state: &State,
    mv: Move,
    workspace: &mut MoveScoreWorkspace,
    stop: &AtomicBool,
) -> Result<Option<Score>, EvaluationInterrupted> {
    score_move_mode_interruptible(per, state, mv, workspace, ScoreMode::Guided, stop)
}

/// Exact speculative score of `mv`, excluding adaptive GLS penalties. The
/// candidate is represented by edit and chunk views, so even Or-opt and
/// cross-exchange avoid cloning or rebuilding either touched route.
pub(super) fn score_move_raw(per: &PerList, state: &State, mv: Move, workspace: &mut MoveScoreWorkspace) -> Option<Score> {
    let stop = AtomicBool::new(false);
    score_move_mode_interruptible(per, state, mv, workspace, ScoreMode::Raw, &stop).expect("an uninterrupted raw move score must complete")
}

fn prepare_routing_scan(
    memory: &mut RoutingScanMemory,
    index_cache: &RoutingIndexCache,
    state: &State,
    kind: NeighborhoodKind,
    mode: ScanMode,
) -> bool {
    let index_generation = if mode == ScanMode::Granular { index_cache.generation } else { 0 };
    if memory.kind == Some(kind) && memory.mode == Some(mode) && memory.index_generation == index_generation {
        return true;
    }
    let granular = match mode {
        ScanMode::Granular => {
            let Some(index) = index_cache.index() else {
                return false;
            };
            Some(GranularCursor::new(kind, state, index))
        }
        ScanMode::Global => None,
    };
    memory.kind = Some(kind);
    memory.mode = Some(mode);
    memory.index_generation = index_generation;
    memory.pending = None;
    memory.global = GlobalCursor::new(kind);
    memory.granular = granular;
    true
}

fn next_routing_candidate(
    memory: &mut RoutingScanMemory,
    index_cache: &RoutingIndexCache,
    per: &PerList,
    state: &State,
    kind: NeighborhoodKind,
    mode: ScanMode,
    stop: &AtomicBool,
) -> GenerationStep {
    match mode {
        ScanMode::Granular => match (index_cache.index(), memory.granular.as_mut()) {
            (Some(index), Some(cursor)) => next_granular(kind, per, state, index, cursor, stop),
            _ => GenerationStep::Complete,
        },
        ScanMode::Global => next_global(&mut memory.global, state, stop),
    }
}

/// Search one bounded, resumable routing neighbourhood slice. Candidate
/// generation and scoring have independent deterministic budgets. A generated
/// but not yet evaluated move remains pending for the next call.
pub(super) fn search_routing_neighborhood(
    per: &PerList,
    state: &State,
    stop: &AtomicBool,
    workspace: RoutingScanWorkspace<'_>,
    kind: NeighborhoodKind,
    mode: ScanMode,
    budget: WorkBudget,
) -> NeighborhoodRun {
    let RoutingScanWorkspace { index_cache, memory } = workspace;
    let started = CpuTimer::start();
    if stop.load(Ordering::Relaxed) {
        return NeighborhoodRun {
            outcome: ScanOutcome::Interrupted,
            generated: 0,
            generation_work: 0,
            evaluated: 0,
            cpu_nanos: started.elapsed_nanos(),
        };
    }
    let mut generated = 0u64;
    let mut generation_work = 0u64;
    let mut evaluated = 0u64;

    if mode == ScanMode::Granular {
        let interchangeable_lists = per.interchangeable_lists.get();
        while !index_cache.is_ready(interchangeable_lists) {
            if generation_work >= budget.generated {
                return NeighborhoodRun {
                    outcome: ScanOutcome::BudgetExhausted,
                    generated,
                    generation_work,
                    evaluated,
                    cpu_nanos: started.elapsed_nanos(),
                };
            }
            match index_cache.advance(state, interchangeable_lists, stop) {
                IndexBuildStep::Progress | IndexBuildStep::Complete => generation_work += 1,
                IndexBuildStep::Interrupted => {
                    return NeighborhoodRun {
                        outcome: ScanOutcome::Interrupted,
                        generated,
                        generation_work,
                        evaluated,
                        cpu_nanos: started.elapsed_nanos(),
                    };
                }
            }
        }
    }
    if !prepare_routing_scan(memory, index_cache, state, kind, mode) {
        return NeighborhoodRun {
            outcome: ScanOutcome::Interrupted,
            generated,
            generation_work,
            evaluated,
            cpu_nanos: started.elapsed_nanos(),
        };
    }

    let current = match full_score_interruptible(per, state, stop) {
        Ok(score) => score,
        Err(EvaluationInterrupted) => {
            return NeighborhoodRun {
                outcome: ScanOutcome::Interrupted,
                generated,
                generation_work,
                evaluated,
                cpu_nanos: started.elapsed_nanos(),
            };
        }
    };
    let mut score_workspace = MoveScoreWorkspace::default();

    let outcome = loop {
        if stop.load(Ordering::Relaxed) {
            break ScanOutcome::Interrupted;
        }
        if memory.pending.is_none() {
            if generation_work >= budget.generated {
                break ScanOutcome::BudgetExhausted;
            }
            match next_routing_candidate(memory, index_cache, per, state, kind, mode, stop) {
                GenerationStep::Candidate(mv) => {
                    generation_work += 1;
                    generated += 1;
                    memory.pending = Some(mv);
                }
                GenerationStep::Progress => {
                    generation_work += 1;
                    continue;
                }
                GenerationStep::Complete => break ScanOutcome::Complete,
                GenerationStep::Interrupted => break ScanOutcome::Interrupted,
            }
        }
        if evaluated >= budget.evaluated {
            break ScanOutcome::BudgetExhausted;
        }
        let mv = memory.pending.expect("a pending routing candidate was just prepared");
        match score_move(per, state, mv, &mut score_workspace, stop) {
            Err(EvaluationInterrupted) => break ScanOutcome::Interrupted,
            Ok(score) => {
                memory.pending = None;
                evaluated += 1;
                if score.is_some_and(|score| score < current) {
                    break ScanOutcome::Improved(mv);
                }
            }
        }
    };

    NeighborhoodRun { outcome, generated, generation_work, evaluated, cpu_nanos: started.elapsed_nanos() }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RoutingAuditOutcome {
    Improved,
    Complete,
    BudgetExhausted,
    Interrupted,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RoutingAuditRun {
    pub(crate) outcome: RoutingAuditOutcome,
    pub(crate) generated: u64,
    pub(crate) generation_work: u64,
    pub(crate) evaluated: u64,
}

/// Test-only façade that exercises the real resumable scanner without making
/// its physical `PerList` and `State` representations part of the crate API.
#[cfg(test)]
pub(crate) struct RoutingNeighborhoodAudit {
    model: CollectionModel,
    per: PerList,
    state: State,
    index_cache: RoutingIndexCache,
    memory: RoutingScanMemory,
}

#[cfg(test)]
fn materialize_composite_move(lists: &[Vec<i32>], mv: Move) -> Vec<Vec<i32>> {
    let mut candidate = lists.to_vec();
    match mv {
        Move::OrOpt { src, start, len, dst, dst_pos } => {
            let segment: Vec<i32> = candidate[src].drain(start..start + len).collect();
            let insert_at = dst_pos.min(candidate[dst].len());
            candidate[dst].splice(insert_at..insert_at, segment);
        }
        Move::CrossExchange { a, start_a, len_a, b, start_b, len_b } => {
            let segment_a: Vec<i32> = candidate[a][start_a..start_a + len_a].to_vec();
            let segment_b: Vec<i32> = candidate[b][start_b..start_b + len_b].to_vec();
            candidate[a].splice(start_a..start_a + len_a, segment_b);
            candidate[b].splice(start_b..start_b + len_b, segment_a);
        }
        _ => unreachable!("composite raw-score audit only accepts Or-opt and cross-exchange"),
    }
    candidate
}

#[cfg(test)]
impl RoutingNeighborhoodAudit {
    pub(crate) fn new(model: &CollectionModel, lists: Vec<Vec<i32>>) -> Self {
        let per = PerList::build(model);
        let state = State::from_lists(model, &per, lists);
        Self { model: model.clone(), per, state, index_cache: RoutingIndexCache::new(), memory: RoutingScanMemory::new() }
    }

    pub(crate) fn run(&mut self, stop: &AtomicBool, kind: NeighborhoodKind, mode: ScanMode, budget: WorkBudget) -> RoutingAuditRun {
        let run = search_routing_neighborhood(
            &self.per,
            &self.state,
            stop,
            RoutingScanWorkspace::new(&mut self.index_cache, &mut self.memory),
            kind,
            mode,
            budget,
        );
        let outcome = match run.outcome {
            ScanOutcome::Improved(_) => RoutingAuditOutcome::Improved,
            ScanOutcome::Complete => RoutingAuditOutcome::Complete,
            ScanOutcome::BudgetExhausted => RoutingAuditOutcome::BudgetExhausted,
            ScanOutcome::Interrupted => RoutingAuditOutcome::Interrupted,
        };
        RoutingAuditRun { outcome, generated: run.generated, generation_work: run.generation_work, evaluated: run.evaluated }
    }

    pub(crate) fn reset(&mut self) {
        self.index_cache.reset();
        self.memory.reset();
    }

    pub(crate) fn set_interchangeable_lists(&mut self, interchangeable: bool) {
        self.per.interchangeable_lists.set(interchangeable);
        self.index_cache.reset();
        self.memory.reset();
    }

    pub(crate) fn completed_index_builds(&self) -> u64 {
        self.index_cache.completed_builds
    }

    pub(crate) fn cached_index_shape(&self) -> Option<(usize, usize)> {
        self.index_cache.index().map(|index| (index.locations.len(), index.ordered_items.len()))
    }

    pub(crate) fn granular_empty_destination_audit(&mut self, kind: NeighborhoodKind) -> Option<(u64, u64, Vec<usize>)> {
        let stop = AtomicBool::new(false);
        let interchangeable_lists = self.per.interchangeable_lists.get();
        while !self.index_cache.is_ready(interchangeable_lists) {
            if self.index_cache.advance(&self.state, interchangeable_lists, &stop) == IndexBuildStep::Interrupted {
                return None;
            }
        }
        let index = self.index_cache.index()?;
        let mut cursor = GranularCursor::new(kind, &self.state, index);
        let mut generated = 0u64;
        let mut work = 0u64;
        let mut destinations = HashSet::new();
        while !matches!(cursor.special, GranularSpecialCursor::Done) {
            work = work.saturating_add(1);
            if let GenerationStep::Candidate(mv) = next_granular_special(&self.state, index, &mut cursor.special, &mut cursor.seen) {
                generated = generated.saturating_add(1);
                match mv {
                    Move::Relocate { dst, .. } | Move::OrOpt { dst, .. } => {
                        destinations.insert(dst);
                    }
                    _ => unreachable!("the empty-destination cursor only emits relocate and Or-opt moves"),
                }
            }
        }
        let mut destinations: Vec<_> = destinations.into_iter().collect();
        destinations.sort_unstable();
        Some((generated, work, destinations))
    }

    pub(crate) fn boundary_move_counts(&mut self, kind: NeighborhoodKind, item: i32, boundary: i32) -> Option<(usize, usize)> {
        let stop = AtomicBool::new(false);
        let interchangeable_lists = self.per.interchangeable_lists.get();
        while !self.index_cache.is_ready(interchangeable_lists) {
            if self.index_cache.advance(&self.state, interchangeable_lists, &stop) == IndexBuildStep::Interrupted {
                return None;
            }
        }
        let index = self.index_cache.index()?;
        let location = index.location(item)?;
        let mut seen = CompactMoveSet::new(self.state.lists.len(), index.max_route_len);
        let moves = granular_boundary_moves(kind, &self.per, &self.state, index, location, boundary, &mut seen);
        let inter_route = moves.iter().filter(|mv| {
            let [left, right] = mv.touched_lists();
            left != right
        });
        Some((moves.len(), inter_route.count()))
    }

    pub(crate) fn anchor_boundary_coverage(
        &mut self,
        kind: NeighborhoodKind,
        item: i32,
        boundary: i32,
    ) -> Option<(usize, usize, usize, usize)> {
        let stop = AtomicBool::new(false);
        let interchangeable_lists = self.per.interchangeable_lists.get();
        while !self.index_cache.is_ready(interchangeable_lists) {
            if self.index_cache.advance(&self.state, interchangeable_lists, &stop) == IndexBuildStep::Interrupted {
                return None;
            }
        }
        let index = self.index_cache.index()?;
        let location = index.location(item)?;
        index.location(boundary)?;

        let mut boundary_seen = CompactMoveSet::new(self.state.lists.len(), index.max_route_len);
        let boundary_moves = granular_boundary_moves(kind, &self.per, &self.state, index, location, boundary, &mut boundary_seen);
        let mut all_seen = CompactMoveSet::new(self.state.lists.len(), index.max_route_len);
        let all_moves = granular_anchor_moves(kind, &self.per, &self.state, index, (item, boundary), &mut all_seen);
        let all: HashSet<_> = all_moves.iter().copied().collect();
        let present = boundary_moves.iter().filter(|mv| all.contains(mv)).count();
        Some((all_moves.len(), boundary_moves.len(), present, all.len()))
    }

    /// Compare the view-based unpenalized score with a fresh, independently
    /// materialized state for every valid Or-opt and cross-exchange candidate.
    /// Returns the number of candidates checked for each neighborhood.
    pub(crate) fn audit_composite_raw_scores(&self) -> (usize, usize, usize) {
        assert!(self.per.bump_gls(&self.state) > 0, "the raw-score oracle requires an active guided penalty");
        let mut raw_workspace = MoveScoreWorkspace::default();
        let mut guided_workspace = MoveScoreWorkspace::default();
        let mut or_opt = 0usize;
        let mut cross_exchange = 0usize;
        let mut guided_differences = 0usize;
        let stop = AtomicBool::new(false);
        let mut check = |mv: Move| {
            let incremental = score_move_raw(&self.per, &self.state, mv, &mut raw_workspace)
                .unwrap_or_else(|| panic!("valid composite move was rejected: {mv:?}"));
            let guided = score_move(&self.per, &self.state, mv, &mut guided_workspace, &stop)
                .expect("the raw-score audit is not interrupted")
                .unwrap_or_else(|| panic!("valid composite move was rejected by guided scoring: {mv:?}"));
            guided_differences += usize::from(guided != incremental);
            let materialized_lists = materialize_composite_move(&self.state.lists, mv);
            let materialized = State::from_lists(&self.model, &self.per, materialized_lists);
            let canonical = full_score_exact_lists(&self.per, &materialized.lists, materialized.global_viol);
            assert!(
                incremental == canonical,
                "raw incremental score differs for {mv:?}: incremental=({},{:?}) canonical=({},{:?})",
                incremental.violation,
                incremental.tiers,
                canonical.violation,
                canonical.tiers,
            );
        };

        for src in 0..self.state.lists.len() {
            let source_len = self.state.lists[src].len();
            for len in 2..=MAX_OR_OPT.min(source_len) {
                for start in 0..=source_len - len {
                    for dst in 0..self.state.lists.len() {
                        let destination_len = if src == dst { source_len - len } else { self.state.lists[dst].len() };
                        for dst_pos in 0..=destination_len {
                            if src == dst && dst_pos == start {
                                continue;
                            }
                            check(Move::OrOpt { src, start, len, dst, dst_pos });
                            or_opt += 1;
                        }
                    }
                }
            }
        }

        for a in 0..self.state.lists.len() {
            for b in a + 1..self.state.lists.len() {
                for len_a in 1..=MAX_OR_OPT.min(self.state.lists[a].len()) {
                    for start_a in 0..=self.state.lists[a].len() - len_a {
                        for len_b in 1..=MAX_OR_OPT.min(self.state.lists[b].len()) {
                            for start_b in 0..=self.state.lists[b].len() - len_b {
                                check(Move::CrossExchange { a, start_a, len_a, b, start_b, len_b });
                                cross_exchange += 1;
                            }
                        }
                    }
                }
            }
        }
        (or_opt, cross_exchange, guided_differences)
    }
}

/// First-improvement local move: return the first relocate / swap / or-opt /
/// 2-opt* / cross-exchange / reversal that strictly lowers the score, or `None`
/// at a local optimum (or when `stop` fires). The neighbourhood is scanned one
/// source route at a time: a route whose *entire* neighbourhood yields no
/// improvement is marked inactive (don't-look bit) and skipped until a later
/// applied move touches it again, so settled routes are not re-scanned every
/// pass. Candidate buffers are reused, so no allocation happens per candidate.
/// `stop` is polled per candidate so even very long lists honour the time limit.
pub(super) fn best_improving_move(
    per: &PerList,
    state: &State,
    stop: &AtomicBool,
    memory: &mut SearchMemory,
    scan_seed: u64,
) -> Option<Move> {
    let current = full_score(per, state);
    // Repair-capable moves (relocate / or-opt insert filters) only prune by the
    // geometric candidate lists once feasible: while a route still overflows, the
    // repair destination may not be a geometric neighbour, so they need the full
    // neighbourhood. Cost-refiner moves (2-opt* / cross / reverse) are the
    // O(k²·len^…) cost of a pass and never the *only* way to repair load (plain
    // relocate is a complete membership neighbourhood), so they prune by
    // candidates even while infeasible - that is what lets an infeasible-heavy
    // instance afford enough passes to reach feasibility.
    let use_candidates = per.has_edges && current.violation == 0 && per.candidates.is_some();
    let cand_cost = if per.infeas_cand { per.has_edges && per.candidates.is_some() } else { use_candidates };
    let interchangeable_lists = per.interchangeable_lists.get();
    let active_routes = active_list_indices(&state.lists, interchangeable_lists);
    let sources = rotated_routes(
        active_routes.iter().copied().filter(|&list| !interchangeable_lists || !state.lists[list].is_empty()).collect(),
        scan_seed,
    );
    let mut scratch_a = EvalScratch::default();
    let mut scratch_b = EvalScratch::default();
    let mut score_scratch = EvalScratch::default();
    let mut overrides = Vec::new();
    let mut polled = 0u32;
    let mut stopped = || {
        polled = polled.wrapping_add(1);
        polled.is_multiple_of(1024) && stop.load(Ordering::Relaxed)
    };
    // Scan one source route at a time so a settled route can be skipped wholesale
    // via its don't-look bit. A route is marked inactive only once its *entire*
    // neighbourhood (relocate, swap, and for routing: or-opt, 2-opt*, cross,
    // reverse, all originating from it) yields no improving move. Pair moves are
    // scanned from both endpoints, so a pair (src, y) is still examined when src
    // is active even if y is inactive; the redundant active/active double-scan is
    // bounded because relocate returns early while the search is still dense.
    for src in sources {
        if memory.skip(src) {
            continue;
        }
        let destination_seed = if scan_seed == 0 { 0 } else { scan_seed ^ mix64(src as u64) };
        let destinations = rotated_routes(active_routes.clone(), destination_seed);
        // --- Relocate a single item out of `src`. ---
        for src_pos in 0..state.lists[src].len() {
            let item = state.lists[src][src_pos];
            for &dst in &destinations {
                if dst == src {
                    let len = state.lists[src].len();
                    for dst_pos in 0..len {
                        if dst_pos == src_pos {
                            continue;
                        }
                        if stopped() {
                            return None;
                        }
                        let edit = Edit::MoveWithin { from: src_pos, to: dst_pos };
                        let nl = trial_list_score(per, state, src, edit, &mut scratch_a);
                        let view = EditView::new(&state.lists[src], edit);
                        // Within-list move: no item changes list, so gdelta = 0.
                        if score_one(per, state, src, &nl, &view, &mut score_scratch) < current {
                            return Some(Move::Relocate { src, src_pos, dst, dst_pos });
                        }
                    }
                } else {
                    let source_edit = Edit::Remove { pos: src_pos };
                    let na = trial_list_score(per, state, src, source_edit, &mut scratch_a);
                    let source_view = EditView::new(&state.lists[src], source_edit);
                    let gd = per.globals.delta(&state.item_list, &[(item, dst)]);
                    for dst_pos in 0..=state.lists[dst].len() {
                        if stopped() {
                            return None;
                        }
                        if use_candidates && !candidate_segment_insert(per, state, dst, dst_pos, item, item) {
                            continue;
                        }
                        let destination_edit = Edit::Insert { pos: dst_pos, item };
                        let nb = trial_list_score(per, state, dst, destination_edit, &mut scratch_b);
                        let destination_view = EditView::new(&state.lists[dst], destination_edit);
                        if score_two(
                            per,
                            state,
                            TrialList { list: src, score: &na, contents: &source_view },
                            TrialList { list: dst, score: &nb, contents: &destination_view },
                            gd,
                            &mut score_scratch,
                        ) < current
                        {
                            return Some(Move::Relocate { src, src_pos, dst, dst_pos });
                        }
                    }
                }
            }
        }
        // --- Swap one item of `src` with one item of another route. ---
        for &y in &destinations {
            if y == src {
                continue;
            }
            for xp in 0..state.lists[src].len() {
                for yp in 0..state.lists[y].len() {
                    if stopped() {
                        return None;
                    }
                    let (vx, vy) = (state.lists[src][xp], state.lists[y][yp]);
                    // A swap removes the two items from their current
                    // neighbourhoods. Its promising edges are therefore the four
                    // adjacencies it creates, not the edge between the swapped
                    // items, which does not appear in the resulting routes.
                    // Candidate pruning is disabled while infeasible so the full
                    // membership neighbourhood remains available for load repair.
                    if use_candidates && !candidate_swap(per, state, src, xp, y, yp) {
                        continue;
                    }
                    let left_edit = Edit::Replace { pos: xp, item: vy };
                    let right_edit = Edit::Replace { pos: yp, item: vx };
                    let na = trial_list_score(per, state, src, left_edit, &mut scratch_a);
                    let nb = trial_list_score(per, state, y, right_edit, &mut scratch_b);
                    let left_view = EditView::new(&state.lists[src], left_edit);
                    let right_view = EditView::new(&state.lists[y], right_edit);
                    let gd = per.globals.delta(&state.item_list, &[(vx, y), (vy, src)]);
                    if score_two(
                        per,
                        state,
                        TrialList { list: src, score: &na, contents: &left_view },
                        TrialList { list: y, score: &nb, contents: &right_view },
                        gd,
                        &mut score_scratch,
                    ) < current
                    {
                        return Some(Move::Swap { a: src, a_pos: xp, b: y, b_pos: yp });
                    }
                }
            }
        }
        // --- Routing moves originating from `src` (need edge costs). ---
        if per.has_edges {
            // Or-opt: relocate a 2..=MAX_OR_OPT segment of `src`.
            let src_len = state.lists[src].len();
            for len in 2..=MAX_OR_OPT.min(src_len) {
                for start in 0..=src_len - len {
                    let items = segment_items(&state.lists[src], start, len);
                    for &dst in &destinations {
                        if dst == src {
                            let post_len = src_len - len;
                            for dst_pos in 0..=post_len {
                                if dst_pos == start {
                                    continue;
                                }
                                if stopped() {
                                    return None;
                                }
                                let edit = Edit::SegmentMoveWithin { start, len, to: dst_pos };
                                let nl = trial_list_score(per, state, src, edit, &mut scratch_a);
                                let view = EditView::new(&state.lists[src], edit);
                                if score_one(per, state, src, &nl, &view, &mut score_scratch) < current {
                                    return Some(Move::OrOpt { src, start, len, dst, dst_pos });
                                }
                            }
                        } else {
                            let source_edit = Edit::SegmentRemove { start, len };
                            let na = trial_list_score(per, state, src, source_edit, &mut scratch_a);
                            let source_view = EditView::new(&state.lists[src], source_edit);
                            let mut overrides = [(0, 0); MAX_OR_OPT];
                            for slot in 0..len {
                                overrides[slot] = (items[slot], dst);
                            }
                            let gd = per.globals.delta(&state.item_list, &overrides[..len]);
                            for dst_pos in 0..=state.lists[dst].len() {
                                if stopped() {
                                    return None;
                                }
                                if use_candidates && !candidate_segment_insert(per, state, dst, dst_pos, items[0], items[len - 1]) {
                                    continue;
                                }
                                let destination_edit = Edit::SegmentInsert { pos: dst_pos, items, len };
                                let nb = trial_list_score(per, state, dst, destination_edit, &mut scratch_b);
                                let destination_view = EditView::new(&state.lists[dst], destination_edit);
                                if score_two(
                                    per,
                                    state,
                                    TrialList { list: src, score: &na, contents: &source_view },
                                    TrialList { list: dst, score: &nb, contents: &destination_view },
                                    gd,
                                    &mut score_scratch,
                                ) < current
                                {
                                    return Some(Move::OrOpt { src, start, len, dst, dst_pos });
                                }
                            }
                        }
                    }
                }
            }
            // 2-opt*: swap the tails of `src` and another route.
            for &y in &destinations {
                if y == src {
                    continue;
                }
                let lx = state.lists[src].len();
                let ly = state.lists[y].len();
                for cut_x in 0..=lx {
                    for cut_y in 0..=ly {
                        if cut_x == lx && cut_y == ly {
                            continue;
                        }
                        if stopped() {
                            return None;
                        }
                        if cand_cost && !candidate_two_opt_star(per, state, src, cut_x, y, cut_y) {
                            continue;
                        }
                        let left_view = ChunkView::two(&state.lists[src][..cut_x], &state.lists[y][cut_y..]);
                        let right_view = ChunkView::two(&state.lists[y][..cut_y], &state.lists[src][cut_x..]);
                        let na = trial_list_score_view(per, state, src, &left_view, None, &mut scratch_a);
                        let nb = trial_list_score_view(per, state, y, &right_view, None, &mut scratch_b);
                        overrides.clear();
                        overrides.extend(state.lists[src][cut_x..].iter().map(|&item| (item, y)));
                        overrides.extend(state.lists[y][cut_y..].iter().map(|&item| (item, src)));
                        let gd = per.globals.delta(&state.item_list, &overrides);
                        if score_two(
                            per,
                            state,
                            TrialList { list: src, score: &na, contents: &left_view },
                            TrialList { list: y, score: &nb, contents: &right_view },
                            gd,
                            &mut score_scratch,
                        ) < current
                        {
                            return Some(Move::TwoOptStar { a: src, cut_a: cut_x, b: y, cut_b: cut_y });
                        }
                    }
                }
            }
            // Cross-exchange: swap a segment of `src` with a segment of another route.
            for &y in &destinations {
                if y == src {
                    continue;
                }
                let lx = state.lists[src].len();
                let ly = state.lists[y].len();
                for len_x in 1..=MAX_OR_OPT.min(lx) {
                    for start_x in 0..=lx - len_x {
                        for len_y in 1..=MAX_OR_OPT.min(ly) {
                            if len_x == 1 && len_y == 1 {
                                continue;
                            }
                            for start_y in 0..=ly - len_y {
                                if stopped() {
                                    return None;
                                }
                                if cand_cost && !candidate_cross_exchange(per, state, (src, start_x, len_x), (y, start_y, len_y)) {
                                    continue;
                                }
                                let left_view = ChunkView::three(
                                    &state.lists[src][..start_x],
                                    &state.lists[y][start_y..start_y + len_y],
                                    &state.lists[src][start_x + len_x..],
                                );
                                let right_view = ChunkView::three(
                                    &state.lists[y][..start_y],
                                    &state.lists[src][start_x..start_x + len_x],
                                    &state.lists[y][start_y + len_y..],
                                );
                                let na = trial_list_score_view(per, state, src, &left_view, None, &mut scratch_a);
                                let nb = trial_list_score_view(per, state, y, &right_view, None, &mut scratch_b);
                                overrides.clear();
                                overrides.extend(state.lists[src][start_x..start_x + len_x].iter().map(|&item| (item, y)));
                                overrides.extend(state.lists[y][start_y..start_y + len_y].iter().map(|&item| (item, src)));
                                let gd = per.globals.delta(&state.item_list, &overrides);
                                if score_two(
                                    per,
                                    state,
                                    TrialList { list: src, score: &na, contents: &left_view },
                                    TrialList { list: y, score: &nb, contents: &right_view },
                                    gd,
                                    &mut score_scratch,
                                ) < current
                                {
                                    return Some(Move::CrossExchange {
                                        a: src,
                                        start_a: start_x,
                                        len_a: len_x,
                                        b: y,
                                        start_b: start_y,
                                        len_b: len_y,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            // 2-opt: reverse a segment within `src`.
            let len = state.lists[src].len();
            for i in 0..len {
                for j in (i + 1)..len {
                    if stopped() {
                        return None;
                    }
                    if cand_cost {
                        let Some((before, after)) = route_before_after(per, state, src, i, j + 1 - i) else {
                            continue;
                        };
                        if !candidate_edge(per, before, state.lists[src][j]) && !candidate_edge(per, state.lists[src][i], after) {
                            continue;
                        }
                    }
                    let edit = Edit::Reverse { i, j };
                    let nl = trial_list_score(per, state, src, edit, &mut scratch_a);
                    let view = EditView::new(&state.lists[src], edit);
                    if score_one(per, state, src, &nl, &view, &mut score_scratch) < current {
                        return Some(Move::Reverse { list: src, i, j });
                    }
                }
            }
        }
        memory.mark_inactive(src);
    }
    None
}

pub(super) fn apply_move(per: &PerList, state: &mut State, mv: Move, stop: &AtomicBool) -> bool {
    match mv {
        Move::Relocate { src, src_pos, dst, dst_pos } => {
            let item = state.lists[src].remove(src_pos);
            let pos = dst_pos.min(state.lists[dst].len());
            state.lists[dst].insert(pos, item);
            if !state.rescore_interruptible(per, src, stop) || !state.rescore_interruptible(per, dst, stop) {
                return false;
            }
            if src != dst {
                state.set_item_list(per, item, dst);
                state.global_viol = per.globals.total(&state.item_list);
            }
        }
        Move::OrOpt { src, start, len, dst, dst_pos } => {
            let segment: Vec<i32> = state.lists[src].drain(start..start + len).collect();
            let pos = dst_pos.min(state.lists[dst].len());
            for (offset, &item) in segment.iter().enumerate() {
                state.lists[dst].insert(pos + offset, item);
            }
            if !state.rescore_interruptible(per, src, stop) {
                return false;
            }
            if src != dst {
                if !state.rescore_interruptible(per, dst, stop) {
                    return false;
                }
                for &item in &segment {
                    state.set_item_list(per, item, dst);
                }
                state.global_viol = per.globals.total(&state.item_list);
            }
        }
        Move::TwoOptStar { a, cut_a, b, cut_b } => {
            let tail_a = state.lists[a].split_off(cut_a);
            let tail_b = state.lists[b].split_off(cut_b);
            state.lists[a].extend(tail_b.iter().copied());
            state.lists[b].extend(tail_a.iter().copied());
            if !state.rescore_interruptible(per, a, stop) || !state.rescore_interruptible(per, b, stop) {
                return false;
            }
            for item in tail_a {
                state.set_item_list(per, item, b);
            }
            for item in tail_b {
                state.set_item_list(per, item, a);
            }
            state.global_viol = per.globals.total(&state.item_list);
        }
        Move::CrossExchange { a, start_a, len_a, b, start_b, len_b } => {
            let seg_a: Vec<i32> = state.lists[a].drain(start_a..start_a + len_a).collect();
            let seg_b: Vec<i32> = state.lists[b].drain(start_b..start_b + len_b).collect();
            for (offset, &item) in seg_b.iter().enumerate() {
                state.lists[a].insert(start_a + offset, item);
            }
            for (offset, &item) in seg_a.iter().enumerate() {
                state.lists[b].insert(start_b + offset, item);
            }
            if !state.rescore_interruptible(per, a, stop) || !state.rescore_interruptible(per, b, stop) {
                return false;
            }
            for &item in &seg_a {
                state.set_item_list(per, item, b);
            }
            for &item in &seg_b {
                state.set_item_list(per, item, a);
            }
            state.global_viol = per.globals.total(&state.item_list);
        }
        Move::Swap { a, a_pos, b, b_pos } => {
            let tmp = state.lists[a][a_pos];
            state.lists[a][a_pos] = state.lists[b][b_pos];
            state.lists[b][b_pos] = tmp;
            if !state.rescore_interruptible(per, a, stop) || !state.rescore_interruptible(per, b, stop) {
                return false;
            }
            // After the swap, lists[a][a_pos] holds the item that was in b.
            state.set_item_list(per, state.lists[a][a_pos], a);
            state.set_item_list(per, state.lists[b][b_pos], b);
            state.global_viol = per.globals.total(&state.item_list);
        }
        Move::Reverse { list, i, j } => {
            state.lists[list][i..=j].reverse();
            if !state.rescore_interruptible(per, list, stop) {
                return false;
            }
            // Reversal keeps every item in the same list, so globals are unchanged.
        }
    }
    true
}

pub(super) fn shuffle(order: &mut [usize], seed: u64) {
    for i in (1..order.len()).rev() {
        let j = (mix64(seed.wrapping_add(i as u64)) % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
}
