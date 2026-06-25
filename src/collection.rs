//! General collection modeling and local search.
//!
//! A model partitions a universe of item ids among `lists` ordered list
//! variables. Objective and constraints are *reductions over a list*: an
//! aggregate (`sum` / `min` / `max` / `count`) of a lambda body evaluated at
//! each item, or at each edge of the closed tour. The lambda body is a small
//! expression IR with numbered binders (`Arg(0)` is the item or edge source,
//! `Arg(1)` the edge target), so `sum(route, i => demand[i])` and
//! `sum_edges(route, (i, j) => dist[i][j])` are just two reductions over the
//! same machinery. One engine then solves routing, sequencing, assignment, and
//! packing models, since each is a different set of reductions over the lists.
//!
//! Search is constraint-based local search: the score is the pair
//! `(violation, objective)` compared lexicographically, so feasibility is driven
//! to zero first. Moves (relocate, swap, segment reversal) keep the partition by
//! construction. The *sequence* of moves and kicks is a pure function of the
//! seed, but the search runs until the shared `stop` flag is set, so the point
//! at which it stops (and thus the best solution returned) depends on the wall
//! clock budget. Same seed plus same number of iterations gives the same answer;
//! a time budget makes the result best-effort, not bit-reproducible across
//! machines.
//!
//! Empty semantics follow Hexaly: `sum`/`count` of nothing is `0`; `min`/`max`
//! of an empty item set is *undefined* and makes the state infeasible. (`min`
//! and `max` over `Edges` are always defined, since a closed tour has at least
//! the single `start -> end` edge.)
//!
//! Arithmetic in bodies and scores saturates rather than wrapping or panicking,
//! so a pathological table value cannot crash the solver path.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::mix64;

/// How a reduction aggregates its body values over the iterable.
#[derive(Clone, Copy)]
pub enum ReduceOp {
    Sum,
    Min,
    Max,
    /// Number of steps whose body value is non-zero (so `count` with a body of
    /// `1` is the length).
    Count,
}

/// Comparison for a constraint `reduction <op> rhs`.
#[derive(Clone, Copy)]
pub enum Op {
    Le,
    Ge,
    Eq,
}

/// Index of an [`Expr`] node inside a reduction's [`ExprArena`].
#[derive(Clone, Copy)]
pub struct ExprId(pub u32);

/// One node of a lambda body. Arrays and matrices are shared by `Arc`, so the
/// same constant table reused across many reductions costs one refcount bump,
/// not a deep copy.
#[derive(Clone)]
pub enum Expr {
    Const(i64),
    /// The `k`-th lambda binder: the item for `Items`, source/target for `Edges`.
    Arg(u8),
    /// `array[index]`.
    Array(Arc<Vec<i64>>, ExprId),
    /// `matrix[row][col]`.
    Matrix(Arc<Vec<Vec<i64>>>, ExprId, ExprId),
    Add(ExprId, ExprId),
    Sub(ExprId, ExprId),
    Mul(ExprId, ExprId),
    Min(ExprId, ExprId),
    Max(ExprId, ExprId),
}

/// A flat arena of [`Expr`] nodes; the body is the last/returned [`ExprId`].
#[derive(Clone, Default)]
pub struct ExprArena {
    pub exprs: Vec<Expr>,
}

impl ExprArena {
    fn push(&mut self, e: Expr) -> ExprId {
        let id = ExprId(self.exprs.len() as u32);
        self.exprs.push(e);
        id
    }
    pub fn constant(&mut self, v: i64) -> ExprId {
        self.push(Expr::Const(v))
    }
    pub fn arg(&mut self, k: u8) -> ExprId {
        self.push(Expr::Arg(k))
    }
    pub fn array(&mut self, a: Arc<Vec<i64>>, i: ExprId) -> ExprId {
        self.push(Expr::Array(a, i))
    }
    pub fn matrix(&mut self, m: Arc<Vec<Vec<i64>>>, i: ExprId, j: ExprId) -> ExprId {
        self.push(Expr::Matrix(m, i, j))
    }
    pub fn add(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.push(Expr::Add(a, b))
    }
    pub fn sub(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.push(Expr::Sub(a, b))
    }
    pub fn mul(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.push(Expr::Mul(a, b))
    }
    pub fn min(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.push(Expr::Min(a, b))
    }
    pub fn max(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.push(Expr::Max(a, b))
    }
}

/// What a reduction iterates over. Both forms name the list they read.
#[derive(Clone)]
pub enum Iterable {
    /// Each item of the list once; the body sees the item value as `Arg(0)`.
    Items(usize),
    /// Each edge of `[start, items.., end]`; the body sees source as `Arg(0)`
    /// and target as `Arg(1)`. Used for closed-tour distance with a depot.
    Edges { list: usize, start: i32, end: i32 },
}

impl Iterable {
    /// The list variable this reduction reads.
    pub fn list(&self) -> usize {
        match self {
            Iterable::Items(l) => *l,
            Iterable::Edges { list, .. } => *list,
        }
    }
}

/// A reduction over one list: `op(iterable, args => body)`.
#[derive(Clone)]
pub struct Reduction {
    pub op: ReduceOp,
    pub iterable: Iterable,
    pub arena: ExprArena,
    pub body: ExprId,
}

/// A constraint `reduction <op> rhs`.
#[derive(Clone)]
pub struct Constraint {
    pub reduction: Reduction,
    pub op: Op,
    pub rhs: i64,
}

/// A collection model: partition `items` among `lists` ordered lists, optimise
/// the summed objective reductions, subject to the constraints. Each reduction
/// names the list variable it reads through its [`Iterable`].
#[derive(Clone)]
pub struct CollectionModel {
    pub items: Vec<i32>,
    pub lists: usize,
    pub minimize: bool,
    pub objective: Vec<Reduction>,
    pub constraints: Vec<Constraint>,
}

/// A solution: the contents of each list, plus objective and feasibility.
#[derive(Clone, Debug)]
pub struct CollectionSolution {
    pub lists: Vec<Vec<i32>>,
    pub objective: i64,
    pub feasible: bool,
}

/// Violation weight for an undefined reduction (empty `min`/`max`) or a broken
/// constraint operand, large enough to dominate any real objective.
const INFEASIBLE: i64 = 1_000_000_000;

/// Evaluate one expression node. Assumes the model was validated, so an
/// out-of-range index is unreachable; the `unwrap_or(0)` is a defensive
/// fallback rather than a silent default for user error.
fn eval_expr(arena: &[Expr], id: ExprId, args: &[i64]) -> i64 {
    match &arena[id.0 as usize] {
        Expr::Const(c) => *c,
        Expr::Arg(k) => args.get(*k as usize).copied().unwrap_or(0),
        Expr::Array(a, i) => {
            let idx = eval_expr(arena, *i, args);
            usize::try_from(idx).ok().and_then(|u| a.get(u)).copied().unwrap_or(0)
        }
        Expr::Matrix(m, i, j) => {
            let ri = eval_expr(arena, *i, args);
            let ci = eval_expr(arena, *j, args);
            usize::try_from(ri)
                .ok()
                .and_then(|r| m.get(r))
                .and_then(|row| usize::try_from(ci).ok().and_then(|c| row.get(c)))
                .copied()
                .unwrap_or(0)
        }
        Expr::Add(a, b) => eval_expr(arena, *a, args).saturating_add(eval_expr(arena, *b, args)),
        Expr::Sub(a, b) => eval_expr(arena, *a, args).saturating_sub(eval_expr(arena, *b, args)),
        Expr::Mul(a, b) => eval_expr(arena, *a, args).saturating_mul(eval_expr(arena, *b, args)),
        Expr::Min(a, b) => eval_expr(arena, *a, args).min(eval_expr(arena, *b, args)),
        Expr::Max(a, b) => eval_expr(arena, *a, args).max(eval_expr(arena, *b, args)),
    }
}

/// Evaluate a reduction over a list's contents. `None` means undefined (an
/// empty `min`/`max` over items), which the scorer treats as infeasible.
fn eval_reduction(reduction: &Reduction, contents: &[i32]) -> Option<i64> {
    let arena = &reduction.arena.exprs;
    let body = reduction.body;
    let mut sum = 0i64;
    let mut count = 0i64;
    let mut lo = i64::MAX;
    let mut hi = i64::MIN;
    let mut steps = 0u64;
    let mut fold = |v: i64| {
        sum = sum.saturating_add(v);
        if v != 0 {
            count += 1;
        }
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
        steps += 1;
    };

    match reduction.iterable {
        Iterable::Items(_) => {
            for &item in contents {
                fold(eval_expr(arena, body, &[i64::from(item)]));
            }
        }
        Iterable::Edges { start, end, .. } => {
            // The closed tour [start, items.., end] always has at least one edge
            // (start -> end when the list is empty), so edge min/max is defined.
            let n = contents.len();
            for p in 0..=n {
                let from = if p == 0 { start } else { contents[p - 1] };
                let to = if p == n { end } else { contents[p] };
                fold(eval_expr(arena, body, &[i64::from(from), i64::from(to)]));
            }
        }
    }

    match reduction.op {
        ReduceOp::Sum => Some(sum),
        ReduceOp::Count => Some(count),
        ReduceOp::Min => (steps > 0).then_some(lo),
        ReduceOp::Max => (steps > 0).then_some(hi),
    }
}

fn violation_of(value: i64, op: Op, rhs: i64) -> i64 {
    match op {
        Op::Le => value.saturating_sub(rhs).max(0),
        Op::Ge => rhs.saturating_sub(value).max(0),
        Op::Eq => value.saturating_sub(rhs).saturating_abs(),
    }
}

/// Like [`eval_expr`] but reports an out-of-range index or unbound argument
/// instead of falling back to zero. Used only by [`CollectionModel::validate`].
fn eval_expr_checked(arena: &[Expr], id: ExprId, args: &[i64]) -> Result<i64, String> {
    let node = arena.get(id.0 as usize).ok_or_else(|| format!("expr id {} out of arena range", id.0))?;
    Ok(match node {
        Expr::Const(c) => *c,
        Expr::Arg(k) => *args.get(*k as usize).ok_or_else(|| format!("lambda arg {k} is not bound by this iterable"))?,
        Expr::Array(a, i) => {
            let idx = eval_expr_checked(arena, *i, args)?;
            if idx < 0 || idx as usize >= a.len() {
                return Err(format!("array index {idx} out of range 0..{}", a.len()));
            }
            a[idx as usize]
        }
        Expr::Matrix(m, i, j) => {
            let ri = eval_expr_checked(arena, *i, args)?;
            let ci = eval_expr_checked(arena, *j, args)?;
            if ri < 0 || ri as usize >= m.len() {
                return Err(format!("matrix row index {ri} out of range 0..{}", m.len()));
            }
            let row = &m[ri as usize];
            if ci < 0 || ci as usize >= row.len() {
                return Err(format!("matrix column index {ci} out of range 0..{}", row.len()));
            }
            row[ci as usize]
        }
        Expr::Add(a, b) => eval_expr_checked(arena, *a, args)?
            .checked_add(eval_expr_checked(arena, *b, args)?)
            .ok_or_else(|| "integer overflow in a lambda body (+)".to_string())?,
        Expr::Sub(a, b) => eval_expr_checked(arena, *a, args)?
            .checked_sub(eval_expr_checked(arena, *b, args)?)
            .ok_or_else(|| "integer overflow in a lambda body (-)".to_string())?,
        Expr::Mul(a, b) => eval_expr_checked(arena, *a, args)?
            .checked_mul(eval_expr_checked(arena, *b, args)?)
            .ok_or_else(|| "integer overflow in a lambda body (*)".to_string())?,
        Expr::Min(a, b) => eval_expr_checked(arena, *a, args)?.min(eval_expr_checked(arena, *b, args)?),
        Expr::Max(a, b) => eval_expr_checked(arena, *a, args)?.max(eval_expr_checked(arena, *b, args)?),
    })
}

/// Validate a reduction's body for every value its binders can take: each item
/// for `Items`, and every item or boundary node for `Edges`.
fn validate_reduction(reduction: &Reduction, items: &[i32], lists: usize) -> Result<(), String> {
    if reduction.iterable.list() >= lists {
        return Err(format!("reduction reads list {} but only {} exist", reduction.iterable.list(), lists));
    }
    let arena = &reduction.arena.exprs;
    match reduction.iterable {
        Iterable::Items(_) => {
            for &item in items {
                eval_expr_checked(arena, reduction.body, &[i64::from(item)])?;
            }
        }
        Iterable::Edges { start, end, .. } => {
            // TODO(perf): this is O(nodes^2) per reduction. For bodies whose
            // index sub-expressions each depend on a single arg (the common
            // Matrix[Arg0][Arg1] / Array[Arg0] case) checking each node once per
            // arg position is sufficient and linear; fall back to the product
            // only for indices that mix both args.
            let mut nodes = items.to_vec();
            nodes.push(start);
            nodes.push(end);
            for &from in &nodes {
                for &to in &nodes {
                    eval_expr_checked(arena, reduction.body, &[i64::from(from), i64::from(to)])?;
                }
            }
        }
    }
    Ok(())
}

impl CollectionModel {
    /// Check the model is well-formed before solving: at least one list, every
    /// reduction reads a real list, and every body only indexes its arrays and
    /// matrices within bounds for any placement of items. Returns a
    /// human-readable error otherwise, so the engine never reads a silent zero.
    pub fn validate(&self) -> Result<(), String> {
        if self.lists == 0 {
            return Err("a collection model needs at least one list".to_string());
        }
        if self.items.is_empty() {
            // Without items, an Items body is never evaluated and so never index
            // checked; require a non-empty universe so validation is total.
            return Err("a collection model needs at least one item".to_string());
        }
        for r in &self.objective {
            validate_reduction(r, &self.items, self.lists)?;
        }
        for c in &self.constraints {
            validate_reduction(&c.reduction, &self.items, self.lists)?;
        }
        Ok(())
    }
}

/// Objective reductions and constraints grouped by the list they read, so a
/// move only rescores the lists it touches.
struct PerList {
    objective: Vec<Vec<Reduction>>,
    constraints: Vec<Vec<Constraint>>,
}

impl PerList {
    fn build(model: &CollectionModel) -> Self {
        let mut objective = vec![Vec::new(); model.lists];
        let mut constraints = vec![Vec::new(); model.lists];
        for r in &model.objective {
            objective[r.iterable.list()].push(r.clone());
        }
        for c in &model.constraints {
            constraints[c.reduction.iterable.list()].push(c.clone());
        }
        Self { objective, constraints }
    }
}

/// Cached `(violation, objective)` of one list.
#[derive(Clone, Copy)]
struct ListScore {
    violation: i64,
    objective: i64,
}

fn list_score(per: &PerList, idx: usize, contents: &[i32]) -> ListScore {
    let mut violation = 0i64;
    let mut objective = 0i64;
    for r in &per.objective[idx] {
        match eval_reduction(r, contents) {
            Some(v) => objective = objective.saturating_add(v),
            None => violation = violation.saturating_add(INFEASIBLE),
        }
    }
    for c in &per.constraints[idx] {
        match eval_reduction(&c.reduction, contents) {
            Some(v) => violation = violation.saturating_add(violation_of(v, c.op, c.rhs)),
            None => violation = violation.saturating_add(INFEASIBLE),
        }
    }
    ListScore { violation, objective }
}

/// Total score: violations dominate, then the objective in its optimisation
/// direction (so smaller is always better here).
fn total_score(minimize: bool, scores: &[ListScore]) -> (i64, i64) {
    let violation = scores.iter().fold(0i64, |acc, s| acc.saturating_add(s.violation));
    let objective = scores.iter().fold(0i64, |acc, s| acc.saturating_add(s.objective));
    (violation, if minimize { objective } else { objective.saturating_neg() })
}

/// State of the search: each list's contents and cached per-list scores.
struct State {
    lists: Vec<Vec<i32>>,
    scores: Vec<ListScore>,
}

impl State {
    fn new(model: &CollectionModel, per: &PerList, order: &[usize]) -> Self {
        let k = model.lists.max(1);
        let mut lists = vec![Vec::new(); k];
        for (rank, &item_idx) in order.iter().enumerate() {
            lists[rank % k].push(model.items[item_idx]);
        }
        let scores = (0..k).map(|idx| list_score(per, idx, &lists[idx])).collect();
        Self { lists, scores }
    }

    fn rescore(&mut self, per: &PerList, idx: usize) {
        self.scores[idx] = list_score(per, idx, &self.lists[idx]);
    }
}

/// Unsigned objective value (the search stores it signed for maximisation).
fn human_objective(minimize: bool, score: (i64, i64)) -> i64 {
    if minimize { score.1 } else { score.1.saturating_neg() }
}

/// Solve a collection model with constraint-based local search until `stop`.
/// `report` is called with the objective each time a strictly better *feasible*
/// incumbent is found, for progress output; pass `&mut |_| {}` to ignore it.
pub fn solve_collection(model: &CollectionModel, seed: u64, stop: &AtomicBool, report: &mut dyn FnMut(i64)) -> CollectionSolution {
    // Guard the search path: an invalid model would otherwise panic (bad list
    // index) or read silent zeros (out-of-range table index). Callers like the
    // Python frontend validate first to raise a precise error; this is the
    // backstop for direct Rust callers so the engine never panics or corrupts.
    if let Err(_e) = model.validate() {
        debug_assert!(false, "solve_collection called on an invalid model: {_e}");
        return CollectionSolution { lists: vec![Vec::new(); model.lists.max(1)], objective: 0, feasible: false };
    }
    let per = PerList::build(model);
    let n = model.items.len();
    let mut order: Vec<usize> = (0..n).collect();
    shuffle(&mut order, seed);
    let mut state = State::new(model, &per, &order);

    let (mut best_lists, mut best_score, mut best_feasible) = snapshot(model, &state);
    if best_feasible {
        report(human_objective(model.minimize, best_score));
    }
    // Local optima visited since the incumbent last improved. Descent moves do
    // NOT reset it (otherwise a kick that descent immediately undoes would keep
    // it at zero and the restart below would never fire). After enough fruitless
    // local optima, restart from a fresh random partition (GRASP-style); until
    // then, kick harder the longer the search has been stuck.
    const RESTART_AFTER: u64 = 25;
    let mut since_improve = 0u64;
    let mut iter = 0u64;

    while !stop.load(Ordering::Relaxed) {
        iter += 1;
        match best_improving_move(model, &per, &state) {
            Some(mv) => apply_move(&per, &mut state, mv),
            None => {
                let (lists, score, feasible) = snapshot(model, &state);
                if better(feasible, score, best_feasible, best_score) {
                    best_lists = lists;
                    best_score = score;
                    best_feasible = feasible;
                    if feasible {
                        report(human_objective(model.minimize, best_score));
                    }
                    since_improve = 0;
                } else {
                    since_improve += 1;
                }
                if since_improve >= RESTART_AFTER {
                    shuffle(&mut order, seed ^ mix64(iter));
                    state = State::new(model, &per, &order);
                    since_improve = 0;
                } else {
                    let strength = 1 + (since_improve / 5) as usize;
                    random_kick(&per, &mut state, seed ^ mix64(iter), strength);
                }
            }
        }
    }

    let (lists, score, feasible) = snapshot(model, &state);
    if better(feasible, score, best_feasible, best_score) {
        best_lists = lists;
        best_score = score;
        best_feasible = feasible;
        if feasible {
            report(human_objective(model.minimize, best_score));
        }
    }

    // Report the objective from the same score that drove the search, so the
    // returned value can never disagree with the accepted solution. When
    // infeasible the value is best-effort; `feasible` is the signal to trust.
    let objective = human_objective(model.minimize, best_score);
    CollectionSolution { lists: best_lists, objective, feasible: best_feasible }
}

fn snapshot(model: &CollectionModel, state: &State) -> (Vec<Vec<i32>>, (i64, i64), bool) {
    let score = total_score(model.minimize, &state.scores);
    (state.lists.clone(), score, score.0 == 0)
}

/// Prefer feasible over infeasible, then better lexicographic score.
fn better(feasible: bool, score: (i64, i64), best_feasible: bool, best_score: (i64, i64)) -> bool {
    match (feasible, best_feasible) {
        (true, false) => true,
        (false, true) => false,
        _ => score < best_score,
    }
}

#[derive(Clone, Copy)]
enum Move {
    Relocate { src: usize, src_pos: usize, dst: usize, dst_pos: usize },
    Swap { a: usize, a_pos: usize, b: usize, b_pos: usize },
    Reverse { list: usize, i: usize, j: usize },
}

/// Score of `state` with lists `a` and `b` replaced by trial scores.
fn score_with(minimize: bool, state: &State, a: usize, sa: ListScore, b: usize, sb: ListScore) -> (i64, i64) {
    let mut violation = 0i64;
    let mut objective = 0i64;
    for (r, s) in state.scores.iter().enumerate() {
        let s = if r == a {
            sa
        } else if r == b {
            sb
        } else {
            *s
        };
        violation = violation.saturating_add(s.violation);
        objective = objective.saturating_add(s.objective);
    }
    (violation, if minimize { objective } else { objective.saturating_neg() })
}

// TODO(perf): each candidate clones the affected list(s) and fully rescores
// them via `list_score` (re-running every reduction over the whole list). For
// large instances this should evaluate the score delta of a move directly,
// reusing scratch buffers instead of cloning, so per-iteration cost is O(1)
// in the list length rather than O(reductions * list_len).
fn best_improving_move(model: &CollectionModel, per: &PerList, state: &State) -> Option<Move> {
    let current = total_score(model.minimize, &state.scores);
    let mut best: Option<(Move, (i64, i64))> = None;
    let mut consider = |mv: Move, score: (i64, i64)| {
        if score < current && best.as_ref().is_none_or(|&(_, s)| score < s) {
            best = Some((mv, score));
        }
    };

    let k = state.lists.len();
    for src in 0..k {
        for src_pos in 0..state.lists[src].len() {
            let item = state.lists[src][src_pos];
            for dst in 0..k {
                let max_pos = if dst == src { state.lists[dst].len() - 1 } else { state.lists[dst].len() };
                for dst_pos in 0..=max_pos {
                    if dst == src && dst_pos == src_pos {
                        continue;
                    }
                    let mut s = state.lists[src].clone();
                    s.remove(src_pos);
                    if dst == src {
                        s.insert(dst_pos.min(s.len()), item);
                        let sc = list_score(per, src, &s);
                        consider(Move::Relocate { src, src_pos, dst, dst_pos }, score_with(model.minimize, state, src, sc, src, sc));
                    } else {
                        let mut d = state.lists[dst].clone();
                        d.insert(dst_pos.min(d.len()), item);
                        let ss = list_score(per, src, &s);
                        let sd = list_score(per, dst, &d);
                        consider(Move::Relocate { src, src_pos, dst, dst_pos }, score_with(model.minimize, state, src, ss, dst, sd));
                    }
                }
            }
        }
    }
    for a in 0..k {
        for b in (a + 1)..k {
            for ap in 0..state.lists[a].len() {
                for bp in 0..state.lists[b].len() {
                    let mut la = state.lists[a].clone();
                    let mut lb = state.lists[b].clone();
                    std::mem::swap(&mut la[ap], &mut lb[bp]);
                    let sa = list_score(per, a, &la);
                    let sb = list_score(per, b, &lb);
                    consider(Move::Swap { a, a_pos: ap, b, b_pos: bp }, score_with(model.minimize, state, a, sa, b, sb));
                }
            }
        }
    }
    for list in 0..k {
        let len = state.lists[list].len();
        for i in 0..len {
            for j in (i + 1)..len {
                let mut l = state.lists[list].clone();
                l[i..=j].reverse();
                let sc = list_score(per, list, &l);
                consider(Move::Reverse { list, i, j }, score_with(model.minimize, state, list, sc, list, sc));
            }
        }
    }
    best.map(|(mv, _)| mv)
}

fn apply_move(per: &PerList, state: &mut State, mv: Move) {
    match mv {
        Move::Relocate { src, src_pos, dst, dst_pos } => {
            let item = state.lists[src].remove(src_pos);
            let pos = dst_pos.min(state.lists[dst].len());
            state.lists[dst].insert(pos, item);
            state.rescore(per, src);
            state.rescore(per, dst);
        }
        Move::Swap { a, a_pos, b, b_pos } => {
            let tmp = state.lists[a][a_pos];
            state.lists[a][a_pos] = state.lists[b][b_pos];
            state.lists[b][b_pos] = tmp;
            state.rescore(per, a);
            state.rescore(per, b);
        }
        Move::Reverse { list, i, j } => {
            state.lists[list][i..=j].reverse();
            state.rescore(per, list);
        }
    }
}

fn shuffle(order: &mut [usize], seed: u64) {
    for i in (1..order.len()).rev() {
        let j = (mix64(seed.wrapping_add(i as u64)) % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
}

/// Move `strength` random items, each to a random other list, to escape a local
/// minimum. A larger strength is a bigger perturbation for a deeper basin.
fn random_kick(per: &PerList, state: &mut State, seed: u64, strength: usize) {
    if state.lists.len() < 2 {
        return;
    }
    for step in 0..strength.max(1) {
        let s = seed ^ mix64(step as u64);
        let total: usize = state.lists.iter().map(Vec::len).sum();
        if total == 0 {
            return;
        }
        let mut pick = (mix64(s) % total as u64) as usize;
        let mut src = 0;
        let mut src_pos = 0;
        for (r, l) in state.lists.iter().enumerate() {
            if pick < l.len() {
                src = r;
                src_pos = pick;
                break;
            }
            pick -= l.len();
        }
        let dst = (mix64(s ^ 0x9E37) % state.lists.len() as u64) as usize;
        if dst == src {
            continue;
        }
        let item = state.lists[src].remove(src_pos);
        state.lists[dst].push(item);
        state.rescore(per, src);
        state.rescore(per, dst);
    }
}
