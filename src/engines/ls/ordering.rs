//! Bounded primal search for permutations with nonnegative directed-distance costs.
//!
//! A recognized model has one total permutation and exact positive-part
//! definitions. Topologically ordered SCC blocks cannot increase this objective:
//! intercomponent arcs become forward and internal distances only contract.
//! Candidates still cross the orchestrator's ordinary semantic replay boundary.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::model::{AffineForm, Constraint, IntDomain, IntExpr, IntGlobalConstraint, IntVarRef, Model, Objective, Relation};

const MAX_POSITIONS: usize = 2_048;
const MAX_ARCS: usize = 32_768;
const MAX_VARIABLES: usize = 65_536;
const MAX_EXPRESSION_NODES: usize = 1_000_000;
const MAX_DOMAIN_VALUES: usize = 8_192;

#[derive(Clone, Copy)]
pub(crate) struct OrderingLimits {
    pub(crate) iterations: u64,
    pub(crate) arc_evaluations: u64,
}

impl Default for OrderingLimits {
    fn default() -> Self {
        Self { iterations: 4_000, arc_evaluations: 200_000_000 }
    }
}

pub(crate) struct OrderingSolution {
    pub(crate) values: Vec<Option<i32>>,
    pub(crate) objective: i64,
    pub(crate) iterations: u64,
    pub(crate) improvements: u64,
}

#[derive(Clone)]
pub(crate) struct OrderingPlan {
    positions: Vec<IntVarRef>,
    offset: i32,
    arcs: Vec<ArcCost>,
    incident: Vec<Vec<usize>>,
    groups: Vec<Vec<usize>>,
    blocks: Vec<(usize, usize)>,
    definitions: Vec<Definition>,
    fixed: Vec<Option<i32>>,
    constant: i64,
    estimated_bytes: u64,
}

#[derive(Clone)]
struct ArcCost {
    source: usize,
    target: usize,
    penalty: IntVarRef,
    scale: i64,
    weight: i64,
}

#[derive(Clone)]
struct Definition {
    target: IntVarRef,
    constant: i64,
    terms: Vec<(IntVarRef, i64)>,
}

#[derive(Clone, Copy)]
struct Move {
    delta: i64,
    from: usize,
    to: usize,
    swap: bool,
}

struct Work<'a> {
    stop: &'a AtomicBool,
    used: u64,
    limit: u64,
}

impl Work<'_> {
    fn tick(&mut self) -> Option<()> {
        if self.used >= self.limit || (self.used.is_multiple_of(1_024) && self.stop.load(Ordering::Acquire)) {
            return None;
        }
        self.used += 1;
        Some(())
    }

    fn stopped(&self) -> bool {
        self.used >= self.limit || self.stop.load(Ordering::Acquire)
    }
}

impl OrderingPlan {
    pub(crate) fn compile(model: &Model, stop: &AtomicBool, memory_allowance: u64) -> Option<Self> {
        if stop.load(Ordering::Acquire)
            || memory_allowance < 65_536
            || !model.sets().is_empty()
            || !model.lists().is_empty()
            || !model.intervals().is_empty()
            || model.int_vars().len() > MAX_VARIABLES
            || model.constraints().len() > MAX_VARIABLES
        {
            return None;
        }
        let [Objective::IntExpr { minimize: true, expr }] = model.objectives() else { return None };
        let mut position_refs = None;
        let mut volume = expression_volume(expr, stop)?;
        for constraint in model.constraints() {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            match constraint {
                Constraint::IntegerGlobal(IntGlobalConstraint::AllDifferent { variables, except }) => {
                    if !except.is_empty() || !(2..=MAX_POSITIONS).contains(&variables.len()) || position_refs.replace(variables).is_some() {
                        return None;
                    }
                    volume = volume.checked_add(variables.len())?;
                }
                Constraint::Intension(expression) => volume = volume.checked_add(expression_volume(expression, stop)?)?,
                Constraint::Linear { terms, relation: Relation::Eq, .. } => volume = volume.checked_add(terms.len())?,
                _ => return None,
            }
            if volume > MAX_EXPRESSION_NODES {
                return None;
            }
        }
        let positions = position_refs?;
        let mut domain_values = 0usize;
        for domain in model.int_vars() {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            if let IntDomain::Set(values) = domain {
                if values.len() > MAX_DOMAIN_VALUES {
                    return None;
                }
                domain_values = domain_values.checked_add(values.len())?;
            }
        }
        // Covers retained graph/definitions, affine normalization maps, DFS
        // stacks, candidate/best/trial permutations, and full reconstruction.
        let estimated_bytes = 65_536u64
            .saturating_add((model.int_vars().len() as u64).saturating_mul(128))
            .saturating_add((model.constraints().len() as u64).saturating_mul(256))
            .saturating_add((volume as u64).saturating_mul(128))
            .saturating_add((domain_values as u64).saturating_mul(8))
            .saturating_add((positions.len() as u64).saturating_mul(512));
        if estimated_bytes > memory_allowance {
            return None;
        }
        let n = positions.len();
        let offset = model.int_vars().get(positions[0].0)?.ceiling(i64::MIN)?;
        let end = offset.checked_add(n as i64 - 1)?;
        let offset = i32::try_from(offset).ok()?;
        let mut index = vec![None; model.int_vars().len()];
        let mut bounds = vec![None; model.int_vars().len()];
        let mut fixed = vec![None; model.int_vars().len()];
        for (i, &variable) in positions.iter().enumerate() {
            if stop.load(Ordering::Acquire) || index.get_mut(variable.0)?.replace(i).is_some() {
                return None;
            }
            let domain = &model.int_vars()[variable.0];
            if domain.ceiling(i64::MIN)? != i64::from(offset)
                || domain_max(domain)? != end
                || !covers_interval(domain, i64::from(offset), end, stop)
            {
                return None;
            }
            bounds[variable.0] = Some((i64::from(offset), end));
        }
        for (i, domain) in model.int_vars().iter().enumerate() {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            let lo = domain.ceiling(i64::MIN)?;
            if lo == domain_max(domain)? {
                fixed[i] = Some(i32::try_from(lo).ok()?);
                bounds[i] = Some((lo, lo));
            }
        }
        let mut arcs = Vec::new();
        let mut penalty_index = BTreeMap::new();
        let mut affine = Vec::new();
        for constraint in model.constraints() {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            match constraint {
                Constraint::IntegerGlobal(IntGlobalConstraint::AllDifferent { .. }) => {}
                Constraint::Intension(expression) => {
                    if let Some((penalty, source, target, scale)) = penalty_definition(expression, stop) {
                        if arcs.len() == MAX_ARCS || index.get(penalty.0)?.is_some() || penalty_index.insert(penalty, arcs.len()).is_some()
                        {
                            return None;
                        }
                        let source = (*index.get(source.0)?)?;
                        let target = (*index.get(target.0)?)?;
                        let maximum = scale.checked_mul(n as i64 - 1)?;
                        if !covers_multiples(model.int_vars().get(penalty.0)?, scale, n, stop) {
                            return None;
                        }
                        bounds[penalty.0] = Some((0, maximum));
                        arcs.push(ArcCost { source, target, penalty, scale, weight: 0 });
                    } else if let IntExpr::Eq(left, right) = expression {
                        affine.push(AffineForm::difference(left, right, stop)?);
                    } else {
                        return None;
                    }
                }
                Constraint::Linear { terms, rhs, .. } => {
                    let mut form = AffineForm { coefficients: BTreeMap::new(), constant: rhs.checked_neg()? };
                    for &(coefficient, variable) in terms {
                        if stop.load(Ordering::Acquire) {
                            return None;
                        }
                        model.int_vars().get(variable.0)?;
                        form.add_coefficient(variable, coefficient)?;
                    }
                    affine.push(form);
                }
                _ => return None,
            }
        }
        if arcs.is_empty() {
            return None;
        }
        // Only total affine auxiliary definitions are accepted. An equality
        // involving solely known variables is an extra restriction, not a
        // definition, and is deliberately rejected.
        let mut definitions = Vec::new();
        while !affine.is_empty() {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            let mut ready = None;
            for (i, form) in affine.iter().enumerate() {
                let mut unknown = None;
                let mut multiple = false;
                for (&variable, &coefficient) in &form.coefficients {
                    if stop.load(Ordering::Acquire) {
                        return None;
                    }
                    if bounds.get(variable.0)?.is_none() {
                        multiple |= unknown.replace((variable, coefficient)).is_some();
                    }
                }
                if !multiple {
                    if let Some((variable, coefficient @ (-1 | 1))) = unknown {
                        ready = Some((i, variable, coefficient));
                        break;
                    }
                }
            }
            let (i, target, coefficient) = ready?;
            let mut form = affine.remove(i);
            form.coefficients.remove(&target);
            let form = form.scale(-coefficient)?;
            let mut lo = i128::from(form.constant);
            let mut hi = lo;
            for (&variable, &coefficient) in &form.coefficients {
                let (a, b) = bounds[variable.0]?;
                let c = i128::from(coefficient);
                lo = lo.checked_add(c * i128::from(if c >= 0 { a } else { b }))?;
                hi = hi.checked_add(c * i128::from(if c >= 0 { b } else { a }))?;
            }
            let lo = i64::try_from(lo).ok()?;
            let hi = i64::try_from(hi).ok()?;
            if !covers_interval(&model.int_vars()[target.0], lo, hi, stop) {
                return None;
            }
            bounds[target.0] = Some((lo, hi));
            definitions.push(Definition { target, constant: form.constant, terms: form.coefficients.into_iter().collect() });
        }
        if bounds.iter().any(Option::is_none) {
            return None;
        }
        let mut objective = AffineForm::expression(expr, stop)?;
        for definition in definitions.iter().rev() {
            if let Some(coefficient) = objective.coefficients.remove(&definition.target) {
                objective.constant = objective.constant.checked_add(coefficient.checked_mul(definition.constant)?)?;
                for &(variable, weight) in &definition.terms {
                    if stop.load(Ordering::Acquire) {
                        return None;
                    }
                    objective.add_coefficient(variable, coefficient.checked_mul(weight)?)?;
                }
            }
        }
        let mut cost_upper = 0i128;
        for (variable, coefficient) in objective.coefficients {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            if let Some(&arc_index) = penalty_index.get(&variable) {
                if coefficient < 0 {
                    return None;
                }
                let arc = &mut arcs[arc_index];
                arc.weight = coefficient.checked_mul(arc.scale)?;
                cost_upper = cost_upper.checked_add(i128::from(arc.weight) * (n as i128 - 1))?;
            } else if let Some(value) = *fixed.get(variable.0)? {
                objective.constant = objective.constant.checked_add(coefficient.checked_mul(i64::from(value))?)?;
            } else {
                return None;
            }
        }
        i64::try_from(cost_upper).ok()?;
        i64::try_from(cost_upper.checked_add(i128::from(objective.constant))?).ok()?;
        let mut incident = vec![Vec::new(); n];
        for (i, arc) in arcs.iter().enumerate() {
            if arc.weight > 0 {
                incident[arc.source].push(i);
                if arc.target != arc.source {
                    incident[arc.target].push(i);
                }
            }
        }
        let groups = strongly_connected(n, &arcs, stop)?;
        let mut offset_in_order = 0;
        let blocks = groups
            .iter()
            .map(|group| {
                let start = offset_in_order;
                offset_in_order += group.len();
                (start, offset_in_order)
            })
            .collect();
        (!stop.load(Ordering::Acquire)).then_some(Self {
            positions: positions.clone(),
            offset,
            arcs,
            incident,
            groups,
            blocks,
            definitions,
            fixed,
            constant: objective.constant,
            estimated_bytes,
        })
    }

    pub(crate) fn estimated_bytes(&self) -> u64 {
        self.estimated_bytes
    }

    pub(crate) fn construct(&self, seed: u64, stop: &AtomicBool) -> Option<OrderingSolution> {
        self.construct_with_limits(seed, stop, OrderingLimits::default())
    }

    pub(crate) fn construct_with_limits(&self, seed: u64, stop: &AtomicBool, limits: OrderingLimits) -> Option<OrderingSolution> {
        if stop.load(Ordering::Acquire) || limits.iterations == 0 || limits.arc_evaluations == 0 {
            return None;
        }
        let mut work = Work { stop, used: 0, limit: limits.arc_evaluations };
        let mut random = seed;
        let mut order = self.initial_order(&mut random, &mut work)?;
        let mut current = self.cost(&order);
        let mut best_order = order.clone();
        let mut best_cost = current;
        let mut iterations = 0;
        let mut improvements = 0;
        let mut descents = 0u64;
        while iterations < limits.iterations && !work.stopped() && best_cost > 0 {
            let Some(movement) = self.best_move(&order, &mut random, &mut work) else { break };
            iterations += 1;
            if let Some(movement) = movement {
                apply_move(&mut order, movement);
                current += movement.delta;
                debug_assert_eq!(current, self.cost(&order));
                if current < best_cost {
                    best_cost = current;
                    best_order.clone_from(&order);
                    improvements += 1;
                }
                continue;
            }
            descents += 1;
            if descents.is_multiple_of(12) {
                let Some(restart) = self.initial_order(&mut random, &mut work) else { break };
                order = restart;
            } else {
                if !draw(&mut random).is_multiple_of(4) {
                    order.clone_from(&best_order);
                }
                let movable = self.blocks.iter().copied().filter(|&(lo, hi)| hi - lo > 1).collect::<Vec<_>>();
                if movable.is_empty() {
                    break;
                }
                for _ in 0..2 + draw(&mut random) % 6 {
                    let (lo, hi) = movable[draw(&mut random) as usize % movable.len()];
                    let from = lo + draw(&mut random) as usize % (hi - lo);
                    let mut to = lo + draw(&mut random) as usize % (hi - lo - 1);
                    if to >= from {
                        to += 1;
                    }
                    apply_move(&mut order, Move { from, to, delta: 0, swap: false });
                }
            }
            current = self.cost(&order);
            if current < best_cost {
                best_cost = current;
                best_order.clone_from(&order);
                improvements += 1;
            }
        }
        // The search deadline is soft for the already complete best candidate.
        // Reconstruction is bounded by the compile-time size envelope.
        let values = self.reconstruct(&best_order)?;
        Some(OrderingSolution { values, objective: self.constant.checked_add(best_cost)?, iterations, improvements })
    }

    fn initial_order(&self, random: &mut u64, work: &mut Work<'_>) -> Option<Vec<usize>> {
        let n = self.positions.len();
        let mut order = Vec::with_capacity(n);
        let mut remaining = vec![false; n];
        let mut outgoing = vec![0i64; n];
        let mut incoming = vec![0i64; n];
        let mut candidates = Vec::with_capacity(n);
        for group in &self.groups {
            if work.stopped() {
                return None;
            }
            for &v in group {
                remaining[v] = true;
            }
            let mut early = Vec::new();
            let mut late = Vec::new();
            for _ in 0..group.len() {
                outgoing.fill(0);
                incoming.fill(0);
                for arc in &self.arcs {
                    work.tick()?;
                    if remaining[arc.source] && remaining[arc.target] {
                        outgoing[arc.source] += arc.weight;
                        incoming[arc.target] += arc.weight;
                    }
                }
                candidates.clear();
                candidates.extend(group.iter().copied().filter(|&v| remaining[v] && outgoing[v] == 0));
                let sink = !candidates.is_empty();
                if !sink {
                    candidates.extend(group.iter().copied().filter(|&v| remaining[v] && incoming[v] == 0));
                }
                if candidates.is_empty() {
                    let highest = group.iter().copied().filter(|&v| remaining[v]).map(|v| outgoing[v] - incoming[v]).max()?;
                    candidates.extend(group.iter().copied().filter(|&v| remaining[v] && outgoing[v] - incoming[v] == highest));
                }
                candidates.sort_unstable();
                let chosen = candidates[draw(random) as usize % candidates.len()];
                remaining[chosen] = false;
                if sink {
                    late.push(chosen);
                } else {
                    early.push(chosen);
                }
            }
            order.extend(early);
            order.extend(late.into_iter().rev());
        }
        Some(order)
    }

    fn cost(&self, order: &[usize]) -> i64 {
        let positions = inverse(order);
        self.arcs.iter().map(|arc| arc.weight * (positions[arc.source] as i64 - positions[arc.target] as i64).max(0)).sum()
    }

    fn swap_delta(&self, positions: &mut [usize], u: usize, v: usize, work: &mut Work<'_>) -> Option<i64> {
        let pu = positions[u];
        let pv = positions[v];
        let mut delta = 0i64;
        for &id in &self.incident[u] {
            work.tick()?;
            delta += self.arc_swap_delta(&self.arcs[id], positions, u, v, pu, pv);
        }
        for &id in &self.incident[v] {
            work.tick()?;
            let arc = &self.arcs[id];
            if arc.source != u && arc.target != u {
                delta += self.arc_swap_delta(arc, positions, u, v, pu, pv);
            }
        }
        positions.swap(u, v);
        Some(delta)
    }

    fn arc_swap_delta(&self, arc: &ArcCost, positions: &[usize], u: usize, v: usize, pu: usize, pv: usize) -> i64 {
        let moved = |node| {
            if node == u {
                pv
            } else if node == v {
                pu
            } else {
                positions[node]
            }
        };
        let before = (positions[arc.source] as i64 - positions[arc.target] as i64).max(0);
        let after = (moved(arc.source) as i64 - moved(arc.target) as i64).max(0);
        arc.weight * (after - before)
    }

    fn best_move(&self, order: &[usize], random: &mut u64, work: &mut Work<'_>) -> Option<Option<Move>> {
        let base = inverse(order);
        let mut trial = order.to_vec();
        let mut positions = base.clone();
        let mut best = None;
        let mut ties = 0u64;
        for &(lo, hi) in &self.blocks {
            for from in lo..hi {
                if work.stopped() {
                    return None;
                }
                for right in [true, false] {
                    trial.copy_from_slice(order);
                    positions.copy_from_slice(&base);
                    let mut delta = 0;
                    let steps = if right { hi - from - 1 } else { from - lo };
                    for step in 1..=steps {
                        let to = if right { from + step } else { from - step };
                        let previous = if right { to - 1 } else { to + 1 };
                        delta += self.swap_delta(&mut positions, trial[previous], trial[to], work)?;
                        trial.swap(previous, to);
                        consider(Move { delta, from, to, swap: false }, &mut best, &mut ties, random);
                    }
                }
                for to in from + 2..hi {
                    positions.copy_from_slice(&base);
                    let delta = self.swap_delta(&mut positions, order[from], order[to], work)?;
                    consider(Move { delta, from, to, swap: true }, &mut best, &mut ties, random);
                }
            }
        }
        Some(best)
    }

    fn reconstruct(&self, order: &[usize]) -> Option<Vec<Option<i32>>> {
        let positions = inverse(order);
        let mut values = self.fixed.clone();
        for (&variable, &position) in self.positions.iter().zip(&positions) {
            values[variable.0] = Some(self.offset.checked_add(i32::try_from(position).ok()?)?);
        }
        for arc in &self.arcs {
            let distance = (positions[arc.source] as i64 - positions[arc.target] as i64).max(0);
            values[arc.penalty.0] = Some(i32::try_from(arc.scale.checked_mul(distance)?).ok()?);
        }
        for definition in &self.definitions {
            let value = definition.terms.iter().try_fold(i128::from(definition.constant), |sum, &(variable, coefficient)| {
                sum.checked_add(i128::from(coefficient) * i128::from(values[variable.0]?))
            })?;
            values[definition.target.0] = Some(i32::try_from(value).ok()?);
        }
        values.iter().all(Option::is_some).then_some(values)
    }

    #[cfg(test)]
    pub(crate) fn audit_swap_delta(&self, order: &[usize], from: usize, to: usize) -> i64 {
        let stop = AtomicBool::new(false);
        let mut work = Work { stop: &stop, used: 0, limit: u64::MAX };
        self.swap_delta(&mut inverse(order), order[from], order[to], &mut work).unwrap()
    }
}

fn consider(movement: Move, best: &mut Option<Move>, ties: &mut u64, random: &mut u64) {
    if movement.delta >= 0 || best.is_some_and(|current| movement.delta > current.delta) {
        return;
    }
    if best.is_none_or(|current| movement.delta < current.delta) {
        *best = Some(movement);
        *ties = 1;
    } else {
        *ties += 1;
        if draw(random).is_multiple_of(*ties) {
            *best = Some(movement);
        }
    }
}

fn apply_move(order: &mut [usize], movement: Move) {
    if movement.swap {
        order.swap(movement.from, movement.to);
    } else if movement.from < movement.to {
        order[movement.from..=movement.to].rotate_left(1);
    } else {
        order[movement.to..=movement.from].rotate_right(1);
    }
}

fn inverse(order: &[usize]) -> Vec<usize> {
    let mut positions = vec![0; order.len()];
    for (i, &v) in order.iter().enumerate() {
        positions[v] = i;
    }
    positions
}

fn draw(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    crate::mix64(*state)
}

fn domain_max(domain: &IntDomain) -> Option<i64> {
    match domain {
        IntDomain::Bool => Some(1),
        IntDomain::Range { lo, hi } => (lo <= hi).then_some(i64::from(*hi)),
        IntDomain::Set(values) => values.iter().max().copied().map(i64::from),
    }
}

fn covers_interval(domain: &IntDomain, lo: i64, hi: i64, stop: &AtomicBool) -> bool {
    if stop.load(Ordering::Acquire) || lo > hi || !domain.contains(lo) || !domain.contains(hi) {
        return false;
    }
    match domain {
        IntDomain::Bool | IntDomain::Range { .. } => true,
        IntDomain::Set(values) => {
            if hi - lo >= values.len() as i64 {
                return false;
            }
            let mut sorted = values.clone();
            sorted.sort_unstable();
            (lo..=hi).all(|value| !stop.load(Ordering::Acquire) && sorted.binary_search(&(value as i32)).is_ok())
        }
    }
}

fn covers_multiples(domain: &IntDomain, scale: i64, count: usize, stop: &AtomicBool) -> bool {
    let Some(maximum) = scale.checked_mul(count as i64 - 1) else { return false };
    if !domain.contains(0) || !domain.contains(maximum) {
        return false;
    }
    match domain {
        IntDomain::Bool | IntDomain::Range { .. } => true,
        IntDomain::Set(values) => {
            let mut sorted = values.clone();
            sorted.sort_unstable();
            (0..count).all(|i| !stop.load(Ordering::Acquire) && sorted.binary_search(&((i as i64 * scale) as i32)).is_ok())
        }
    }
}

fn penalty_definition(expr: &IntExpr, stop: &AtomicBool) -> Option<(IntVarRef, IntVarRef, IntVarRef, i64)> {
    let IntExpr::Eq(left, right) = expr else { return None };
    let (target, value) = match (left.as_ref(), right.as_ref()) {
        (IntExpr::Variable(target), value) | (value, IntExpr::Variable(target)) => (*target, value),
        _ => return None,
    };
    let mut scale = 1i64;
    let mut value = value;
    if let IntExpr::Mul(terms) = value {
        let mut symbolic = None;
        for term in terms {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            if let IntExpr::Constant(coefficient) = term {
                scale = scale.checked_mul(*coefficient)?;
            } else if symbolic.replace(term).is_some() {
                return None;
            }
        }
        value = symbolic?;
    }
    if scale <= 0 {
        return None;
    }
    let form = match value {
        IntExpr::Max(terms) if terms.len() == 2 => {
            let expression = match (&terms[0], &terms[1]) {
                (IntExpr::Constant(0), expression) | (expression, IntExpr::Constant(0)) => expression,
                _ => return None,
            };
            AffineForm::expression(expression, stop)?
        }
        IntExpr::IfThenElse(condition, then_value, else_value) => {
            let (expression, zero_first) = match (then_value.as_ref(), else_value.as_ref()) {
                (IntExpr::Constant(0), expression) => (expression, true),
                (expression, IntExpr::Constant(0)) => (expression, false),
                _ => return None,
            };
            let (left, right, negative) = match condition.as_ref() {
                IntExpr::Lt(left, right) | IntExpr::Le(left, right) => (left, right, true),
                IntExpr::Gt(left, right) | IntExpr::Ge(left, right) => (left, right, false),
                _ => return None,
            };
            let form = AffineForm::expression(expression, stop)?;
            let comparison = AffineForm::difference(left, right, stop)?;
            let comparison = if zero_first == negative { comparison } else { comparison.scale(-1)? };
            if form != comparison {
                return None;
            }
            form
        }
        _ => return None,
    };
    if form.constant != 0 || form.coefficients.len() != 2 {
        return None;
    }
    let source = form.coefficients.iter().find_map(|(&v, &c)| (c == 1).then_some(v))?;
    let destination = form.coefficients.iter().find_map(|(&v, &c)| (c == -1).then_some(v))?;
    Some((target, source, destination, scale))
}

fn expression_volume(expr: &IntExpr, stop: &AtomicBool) -> Option<usize> {
    fn visit(expr: &IntExpr, depth: usize, volume: &mut usize, stop: &AtomicBool) -> Option<()> {
        if stop.load(Ordering::Acquire) || depth > 256 || *volume >= MAX_EXPRESSION_NODES {
            return None;
        }
        *volume += 1;
        match expr {
            IntExpr::Constant(_) | IntExpr::Variable(_) => {}
            IntExpr::Neg(v) | IntExpr::Abs(v) | IntExpr::Not(v) => visit(v, depth + 1, volume, stop)?,
            IntExpr::Add(v) | IntExpr::Mul(v) | IntExpr::Min(v) | IntExpr::Max(v) | IntExpr::And(v) | IntExpr::Or(v) => {
                for value in v {
                    visit(value, depth + 1, volume, stop)?;
                }
            }
            IntExpr::IfThenElse(a, b, c) => {
                visit(a, depth + 1, volume, stop)?;
                visit(b, depth + 1, volume, stop)?;
                visit(c, depth + 1, volume, stop)?;
            }
            IntExpr::Sub(a, b)
            | IntExpr::Div(a, b)
            | IntExpr::Mod(a, b)
            | IntExpr::Eq(a, b)
            | IntExpr::Ne(a, b)
            | IntExpr::Lt(a, b)
            | IntExpr::Le(a, b)
            | IntExpr::Gt(a, b)
            | IntExpr::Ge(a, b)
            | IntExpr::Imp(a, b)
            | IntExpr::Iff(a, b) => {
                visit(a, depth + 1, volume, stop)?;
                visit(b, depth + 1, volume, stop)?;
            }
        }
        Some(())
    }
    let mut volume = 0;
    visit(expr, 0, &mut volume, stop)?;
    Some(volume)
}

fn strongly_connected(n: usize, arcs: &[ArcCost], stop: &AtomicBool) -> Option<Vec<Vec<usize>>> {
    let mut outgoing = vec![Vec::new(); n];
    let mut incoming = vec![Vec::new(); n];
    for arc in arcs {
        if stop.load(Ordering::Acquire) {
            return None;
        }
        if arc.weight > 0 {
            outgoing[arc.source].push(arc.target);
            incoming[arc.target].push(arc.source);
        }
    }
    let mut seen = vec![false; n];
    let mut post = Vec::with_capacity(n);
    let mut stack = Vec::new();
    for start in 0..n {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        stack.push((start, 0));
        while let Some((node, next)) = stack.last_mut() {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            if let Some(&neighbor) = outgoing[*node].get(*next) {
                *next += 1;
                if !seen[neighbor] {
                    seen[neighbor] = true;
                    stack.push((neighbor, 0));
                }
            } else {
                post.push(*node);
                stack.pop();
            }
        }
    }
    seen.fill(false);
    let mut groups = Vec::new();
    let mut pending = Vec::new();
    for &start in post.iter().rev() {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        pending.push(start);
        let mut group = Vec::new();
        while let Some(node) = pending.pop() {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            group.push(node);
            for &neighbor in &incoming[node] {
                if !seen[neighbor] {
                    seen[neighbor] = true;
                    pending.push(neighbor);
                }
            }
        }
        group.sort_unstable();
        groups.push(group);
    }
    Some(groups)
}
