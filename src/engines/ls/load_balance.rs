//! Bounded primal search for precedence-constrained assignment with derived loads.
//!
//! Assignment moves update every recognized load vector together. The plan only
//! emits complete, locally reconstructed candidates; semantic replay by the
//! orchestrator remains the publication boundary.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::model::{AffineForm, Constraint, IntDomain, IntExpr, IntGlobalConstraint, IntVarRef, Model, Objective, Relation};

const MAX_ITEMS: usize = 2_048;
const MAX_BINS: usize = 512;
const MAX_LOAD_VECTORS: usize = 8;
const MAX_EXPLICIT_DOMAIN_VALUES: usize = 4_096;

#[derive(Clone, Copy, Debug)]
pub(crate) struct LoadBalanceLimits {
    pub(crate) iterations: u64,
    pub(crate) restarts: u32,
}

impl Default for LoadBalanceLimits {
    fn default() -> Self {
        Self { iterations: 2_000, restarts: 32 }
    }
}

pub(crate) struct LoadBalanceSolution {
    pub(crate) values: Vec<Option<i32>>,
    pub(crate) objective: i64,
    pub(crate) iterations: u64,
    pub(crate) improvements: u64,
}

#[derive(Clone)]
pub(crate) struct LoadBalancePlan {
    items: Vec<IntVarRef>,
    allowed: Vec<Vec<usize>>,
    dimensions: Vec<LoadDimension>,
    objective_dimension: usize,
    objective_weights: Vec<i64>,
    objective_constant: i64,
    predecessors: Vec<Vec<(usize, i64)>>,
    successors: Vec<Vec<(usize, i64)>>,
    latest: Vec<usize>,
    fixed: Vec<Option<i32>>,
    lower_bound: i64,
    estimated_bytes: u64,
}

#[derive(Clone)]
struct LoadDimension {
    sizes: Vec<i64>,
    variables: Vec<IntVarRef>,
    domains: Vec<IntDomain>,
    minima: Vec<i64>,
    maxima: Vec<i64>,
    total: i64,
}

#[derive(Clone)]
struct Assignment {
    bins: Vec<usize>,
    loads: Vec<Vec<i64>>,
    objective: i64,
}

#[derive(Clone, Copy)]
struct Move {
    item: usize,
    destination: usize,
    swap: Option<usize>,
    delta: i64,
}

struct MoveSelection<'a> {
    tabu: &'a [u64],
    iteration: u64,
    aspiration: i64,
    random: &'a mut u64,
    best: Option<(Move, u64)>,
}

impl LoadBalancePlan {
    pub(crate) fn compile(model: &Model, stop: &AtomicBool, memory_allowance: u64) -> Option<Self> {
        if stop.load(Ordering::Acquire) || !model.sets().is_empty() || !model.lists().is_empty() || !model.intervals().is_empty() {
            return None;
        }
        let [Objective::IntExpr { minimize: true, expr }] = model.objectives() else { return None };
        let mut squares = BTreeMap::new();
        let mut objective_constant = 0;
        square_terms(expr, 1, &mut squares, &mut objective_constant, stop)?;
        if squares.is_empty() || squares.values().any(|&coefficient| coefficient <= 0) {
            return None;
        }
        let mut records = Vec::new();
        for constraint in model.constraints() {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            if let Constraint::IntegerGlobal(IntGlobalConstraint::BinLoads { items, sizes, loads }) = constraint {
                if records.len() == MAX_LOAD_VECTORS {
                    return None;
                }
                records.push((items, sizes, loads));
            }
        }
        if records.is_empty() {
            return None;
        }
        let bins = records[0].2.len();
        if records[0].0.is_empty() || records[0].0.len() > MAX_ITEMS || !(2..=MAX_BINS).contains(&bins) {
            return None;
        }
        let items = records[0].0.clone();
        // Includes scratch move state and the dense item-domain representation.
        let mut explicit_domain_bytes = 0u64;
        for domain in model.int_vars() {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            if let IntDomain::Set(values) = domain {
                if values.len() > MAX_EXPLICIT_DOMAIN_VALUES {
                    return None;
                }
                explicit_domain_bytes = explicit_domain_bytes.saturating_add((values.len() as u64).saturating_mul(4));
            }
        }
        let estimated_bytes = (model.int_vars().len() as u64)
            .saturating_mul(64)
            .saturating_add((items.len() as u64).saturating_mul(bins as u64).saturating_mul(16))
            .saturating_add((items.len() as u64).saturating_mul(256 + 16 * records.len() as u64))
            .saturating_add((model.constraints().len() as u64).saturating_mul(128))
            .saturating_add((records.len() as u64).saturating_mul(bins as u64).saturating_mul(128))
            .saturating_add(explicit_domain_bytes)
            .saturating_add(65_536);
        if estimated_bytes > memory_allowance {
            return None;
        }
        let mut item_index = vec![None; model.int_vars().len()];
        let mut represented = vec![false; model.int_vars().len()];
        let mut allowed = Vec::new();
        for (index, &variable) in items.iter().enumerate() {
            if stop.load(Ordering::Acquire) || *represented.get(variable.0)? {
                return None;
            }
            represented[variable.0] = true;
            item_index[variable.0] = Some(index);
            let domain = &model.int_vars()[variable.0];
            let values = (0..bins).filter(|&bin| domain.contains(bin as i64)).collect::<Vec<_>>();
            if values.is_empty() {
                return None;
            }
            allowed.push(values);
        }
        let mut dimensions = Vec::new();
        let mut objective_dimension = None;
        for (record_items, record_sizes, variables) in &records {
            if stop.load(Ordering::Acquire)
                || record_items.len() != items.len()
                || record_sizes.len() != items.len()
                || variables.len() != bins
            {
                return None;
            }
            let mut sizes = vec![None; items.len()];
            for (&variable, &size) in record_items.iter().zip(record_sizes.iter()) {
                let index = (*item_index.get(variable.0)?)?;
                if size < 0 || sizes[index].replace(size).is_some() {
                    return None;
                }
            }
            let sizes = sizes.into_iter().collect::<Option<Vec<_>>>()?;
            let total = sizes.iter().try_fold(0i64, |sum, &size| sum.checked_add(size))?;
            if total <= 0 || total > i64::from(i32::MAX) {
                return None;
            }
            let mut domains = Vec::new();
            let mut minima = Vec::new();
            let mut maxima = Vec::new();
            for variable in variables.iter() {
                if *represented.get(variable.0)? {
                    return None;
                }
                represented[variable.0] = true;
                let domain = model.int_vars()[variable.0].clone();
                minima.push(domain.ceiling(0)?);
                maxima.push(domain_maximum(&domain).min(total));
                domains.push(domain);
            }
            if variables.len() == squares.len()
                && variables.iter().all(|variable| squares.contains_key(variable))
                && (objective_dimension.replace(dimensions.len()).is_some() || sizes.contains(&0))
            {
                return None;
            }
            dimensions.push(LoadDimension { sizes, variables: (*variables).clone(), domains, minima, maxima, total });
        }
        let objective_dimension = objective_dimension?;
        let objective_weights = dimensions[objective_dimension].variables.iter().map(|variable| squares[variable]).collect::<Vec<_>>();
        // Check that every candidate objective, including intermediate deltas, fits i64.
        let objective_total = i128::from(dimensions[objective_dimension].total);
        let objective_maximum = i128::from(*objective_weights.iter().max()?) * objective_total * objective_total;
        i64::try_from(objective_maximum.checked_add(i128::from(objective_constant))?).ok()?;
        let mut fixed = vec![None; model.int_vars().len()];
        for (index, domain) in model.int_vars().iter().enumerate() {
            if !represented[index] {
                let minimum = domain.ceiling(i64::MIN)?;
                if minimum != domain_maximum(domain) {
                    return None;
                }
                fixed[index] = Some(i32::try_from(minimum).ok()?);
            }
        }
        let mut predecessors = vec![Vec::new(); items.len()];
        let mut successors = vec![Vec::new(); items.len()];
        for constraint in model.constraints() {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            match constraint {
                Constraint::IntegerGlobal(IntGlobalConstraint::BinLoads { .. }) => {}
                Constraint::Linear { terms, relation, rhs } => {
                    let form = AffineForm::linear(terms, *rhs)?;
                    if let Some((before, after, gap)) = precedence(&form, *relation, &item_index) {
                        predecessors[after].push((before, gap));
                        successors[before].push((after, gap));
                    } else if *relation != Relation::Eq || !derived_identity(&form, &items, &allowed, &dimensions, &fixed, stop) {
                        return None;
                    }
                }
                Constraint::Intension(expr) => {
                    let (left, right, relation) = match expr {
                        IntExpr::Lt(left, right) => (left, right, Relation::Lt),
                        IntExpr::Le(left, right) => (left, right, Relation::Le),
                        IntExpr::Gt(left, right) => (left, right, Relation::Gt),
                        IntExpr::Ge(left, right) => (left, right, Relation::Ge),
                        _ => return None,
                    };
                    let form = AffineForm::difference(left, right, stop)?;
                    let (before, after, gap) = precedence(&form, relation, &item_index)?;
                    predecessors[after].push((before, gap));
                    successors[before].push((after, gap));
                }
                _ => return None,
            }
        }
        // Propagate latest legal bins over the precedence DAG without materializing
        // transitive closure. A cycle is outside this constructive plan's envelope.
        let mut pending = successors.iter().map(Vec::len).collect::<Vec<_>>();
        let mut ready = pending.iter().enumerate().filter_map(|(item, &count)| (count == 0).then_some(item)).collect::<Vec<_>>();
        let mut latest = allowed.iter().map(|values| *values.last().unwrap()).collect::<Vec<_>>();
        let mut visited = 0;
        while let Some(item) = ready.pop() {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            visited += 1;
            for &(before, gap) in &predecessors[item] {
                let upper = (latest[item] as i64).checked_sub(gap)?;
                latest[before] = *allowed[before].iter().rev().find(|&&bin| bin as i64 <= upper && bin <= latest[before])?;
                pending[before] -= 1;
                if pending[before] == 0 {
                    ready.push(before);
                }
            }
        }
        if visited != items.len() {
            return None;
        }
        let total = dimensions[objective_dimension].total;
        let quotient = total / bins as i64;
        let remainder = total % bins as i64;
        let balance =
            i128::from(bins as i64 - remainder) * i128::from(quotient).pow(2) + i128::from(remainder) * i128::from(quotient + 1).pow(2);
        // A minimum coefficient also gives a valid, possibly weaker weighted bound.
        let lower_bound = i64::try_from(balance * i128::from(*objective_weights.iter().min()?) + i128::from(objective_constant)).ok()?;
        Some(Self {
            items,
            allowed,
            dimensions,
            objective_dimension,
            objective_weights,
            objective_constant,
            predecessors,
            successors,
            latest,
            fixed,
            lower_bound,
            estimated_bytes,
        })
    }

    pub(crate) fn estimated_bytes(&self) -> u64 {
        self.estimated_bytes
    }

    pub(crate) fn construct(
        &self,
        seed: u64,
        incumbent: Option<&[i32]>,
        stop: &AtomicBool,
        limits: LoadBalanceLimits,
    ) -> Option<LoadBalanceSolution> {
        if stop.load(Ordering::Acquire) || limits.iterations == 0 || limits.restarts == 0 {
            return None;
        }
        let mut random = crate::mix64(seed ^ 0x8e18_a84d_713f_257b);
        let mut best = incumbent.and_then(|values| {
            let bins = self.items.iter().map(|variable| usize::try_from(*values.get(variable.0)?).ok()).collect::<Option<Vec<_>>>()?;
            self.reconstruct(bins)
        });
        let mut iterations = 0;
        let mut improvements = 0;
        for restart in 0..limits.restarts {
            if stop.load(Ordering::Acquire)
                || iterations >= limits.iterations
                || best.as_ref().is_some_and(|state| state.objective == self.lower_bound)
            {
                break;
            }
            let mut current = if restart == 0 && best.is_some() {
                best.clone().unwrap()
            } else if let Some(state) = self.initial_assignment(&mut random, stop) {
                state
            } else {
                continue;
            };
            if best.as_ref().is_none_or(|state| current.objective < state.objective) {
                best = Some(current.clone());
                improvements += 1;
            }
            let mut tabu = vec![0u64; self.items.len()];
            let mut stagnant = 0;
            while iterations < limits.iterations && stagnant < 160 && current.objective > self.lower_bound {
                if stop.load(Ordering::Acquire) {
                    break;
                }
                iterations += 1;
                let aspiration = best.as_ref().map_or(i64::MAX, |state| state.objective);
                let Some(chosen) = self.best_move(&current, &tabu, iterations, aspiration, &mut random, stop) else { break };
                let source = current.bins[chosen.item];
                for (dimension, loads) in self.dimensions.iter().zip(&mut current.loads) {
                    let difference = dimension.sizes[chosen.item] - chosen.swap.map_or(0, |other| dimension.sizes[other]);
                    loads[source] -= difference;
                    loads[chosen.destination] += difference;
                }
                current.bins[chosen.item] = chosen.destination;
                if let Some(other) = chosen.swap {
                    current.bins[other] = source;
                    tabu[other] = iterations + 7 + next_random(&mut random) % 9;
                }
                tabu[chosen.item] = iterations + 7 + next_random(&mut random) % 9;
                current.objective += chosen.delta;
                stagnant += 1;
                if best.as_ref().is_none_or(|state| current.objective < state.objective) {
                    best = Some(current.clone());
                    improvements += 1;
                    stagnant = 0;
                }
            }
        }
        // Recompute all derived values after interruption as well. A soft search
        // stop must not discard an already complete feasible candidate.
        let best = self.reconstruct(best?.bins)?;
        let mut values = self.fixed.clone();
        for (&variable, &bin) in self.items.iter().zip(&best.bins) {
            values[variable.0] = Some(i32::try_from(bin).ok()?);
        }
        for (dimension, loads) in self.dimensions.iter().zip(best.loads) {
            for (&variable, value) in dimension.variables.iter().zip(loads) {
                values[variable.0] = Some(i32::try_from(value).ok()?);
            }
        }
        Some(LoadBalanceSolution { values, objective: best.objective, iterations, improvements })
    }

    fn reconstruct(&self, bins: Vec<usize>) -> Option<Assignment> {
        if bins.len() != self.items.len() {
            return None;
        }
        let mut loads = self.dimensions.iter().map(|dimension| vec![0i64; dimension.variables.len()]).collect::<Vec<_>>();
        for (item, &bin) in bins.iter().enumerate() {
            if !self.allowed[item].contains(&bin)
                || !self.predecessors[item].iter().all(|&(before, gap)| bins[before] as i64 + gap <= bin as i64)
            {
                return None;
            }
            for (dimension, loads) in self.dimensions.iter().zip(&mut loads) {
                loads[bin] += dimension.sizes[item];
            }
        }
        if self
            .dimensions
            .iter()
            .zip(&loads)
            .any(|(dimension, loads)| dimension.domains.iter().zip(loads).any(|(domain, &load)| !domain.contains(load)))
        {
            return None;
        }
        let objective = self.objective(&loads)?;
        Some(Assignment { bins, loads, objective })
    }

    fn objective(&self, loads: &[Vec<i64>]) -> Option<i64> {
        let value = loads[self.objective_dimension]
            .iter()
            .zip(&self.objective_weights)
            .fold(i128::from(self.objective_constant), |sum, (&load, &weight)| sum + i128::from(weight) * i128::from(load).pow(2));
        i64::try_from(value).ok()
    }

    fn initial_assignment(&self, random: &mut u64, stop: &AtomicBool) -> Option<Assignment> {
        let bins_count = self.objective_weights.len();
        let mut assignment = vec![usize::MAX; self.items.len()];
        let mut remaining = self.dimensions.iter().map(|dimension| dimension.total).collect::<Vec<_>>();
        let mut suffix_minimum = self.dimensions.iter().map(|dimension| dimension.minima.iter().sum::<i64>()).collect::<Vec<_>>();
        for bin in 0..bins_count {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            let mut loads = vec![0; self.dimensions.len()];
            let targets = remaining.iter().map(|&total| total as f64 / (bins_count - bin) as f64).collect::<Vec<_>>();
            for (dimension, suffix) in self.dimensions.iter().zip(&mut suffix_minimum) {
                *suffix -= dimension.minima[bin];
            }
            let mut ready = Vec::new();
            for item in 0..self.items.len() {
                if stop.load(Ordering::Acquire) {
                    return None;
                }
                if assignment[item] == usize::MAX
                    && self.allowed[item].contains(&bin)
                    && self.predecessors[item]
                        .iter()
                        .all(|&(before, gap)| assignment[before] != usize::MAX && assignment[before] as i64 + gap <= bin as i64)
                {
                    ready.push(item);
                }
            }
            if (0..self.items.len()).any(|item| assignment[item] == usize::MAX && self.latest[item] < bin) {
                return None;
            }
            let mut selected = vec![false; ready.len()];
            loop {
                if stop.load(Ordering::Acquire) {
                    return None;
                }
                let minima_met = self.dimensions.iter().zip(&loads).all(|(dimension, &load)| dimension.domains[bin].contains(load));
                let baseline =
                    loads.iter().zip(&targets).map(|(&load, &target)| ((load as f64 - target) / target.max(1.0)).powi(2)).sum::<f64>();
                let mut choice = None;
                for (position, &item) in ready.iter().enumerate() {
                    if position.is_multiple_of(64) && stop.load(Ordering::Acquire) {
                        return None;
                    }
                    if selected[position]
                        || self.dimensions.iter().enumerate().any(|(d, dimension)| {
                            loads[d] + dimension.sizes[item] > dimension.maxima[bin]
                                || remaining[d] - dimension.sizes[item] < suffix_minimum[d]
                        })
                    {
                        continue;
                    }
                    let mandatory = self.latest[item] == bin;
                    let distance = self
                        .dimensions
                        .iter()
                        .enumerate()
                        .map(|(d, dimension)| ((loads[d] as f64 + dimension.sizes[item] as f64 - targets[d]) / targets[d].max(1.0)).powi(2))
                        .sum::<f64>();
                    // Among similarly balanced choices, advance courses with little
                    // remaining precedence slack and vary ties across restarts.
                    let urgency = 0.08 / (self.latest[item] - bin + 1) as f64;
                    let jitter = (next_random(random) % 1024) as f64 * 0.000_01;
                    let score = distance - urgency + jitter - if mandatory { 1_000.0 } else { 0.0 };
                    if minima_met && !mandatory && distance >= baseline {
                        continue;
                    }
                    if choice.is_none_or(|(_, old_score)| score < old_score) {
                        choice = Some((position, score));
                    }
                }
                let Some((position, _)) = choice else { break };
                let item = ready[position];
                selected[position] = true;
                assignment[item] = bin;
                for (d, dimension) in self.dimensions.iter().enumerate() {
                    loads[d] += dimension.sizes[item];
                    remaining[d] -= dimension.sizes[item];
                }
            }
            if self.dimensions.iter().enumerate().any(|(d, dimension)| !dimension.domains[bin].contains(loads[d])) {
                return None;
            }
        }
        self.reconstruct(assignment)
    }

    fn best_move(
        &self,
        state: &Assignment,
        tabu: &[u64],
        iteration: u64,
        aspiration: i64,
        random: &mut u64,
        stop: &AtomicBool,
    ) -> Option<Move> {
        let mut selection = MoveSelection { tabu, iteration, aspiration, random, best: None };
        for item in 0..self.items.len() {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            for &destination in &self.allowed[item] {
                if destination == state.bins[item] {
                    continue;
                }
                let movement = Move { item, destination, swap: None, delta: 0 };
                self.consider_move(state, movement, &mut selection);
            }
            for other in item + 1..self.items.len() {
                if other.is_multiple_of(64) && stop.load(Ordering::Acquire) {
                    return None;
                }
                let destination = state.bins[other];
                if destination == state.bins[item]
                    || !self.allowed[item].contains(&destination)
                    || !self.allowed[other].contains(&state.bins[item])
                {
                    continue;
                }
                let movement = Move { item, destination, swap: Some(other), delta: 0 };
                self.consider_move(state, movement, &mut selection);
            }
        }
        selection.best.map(|(movement, _)| movement)
    }

    fn consider_move(&self, state: &Assignment, mut movement: Move, selection: &mut MoveSelection<'_>) {
        let source = state.bins[movement.item];
        let destination = movement.destination;
        let objective = &self.dimensions[self.objective_dimension];
        let difference = objective.sizes[movement.item] - movement.swap.map_or(0, |other| objective.sizes[other]);
        let loads = &state.loads[self.objective_dimension];
        let delta = i128::from(self.objective_weights[source])
            * (i128::from(loads[source] - difference).pow(2) - i128::from(loads[source]).pow(2))
            + i128::from(self.objective_weights[destination])
                * (i128::from(loads[destination] + difference).pow(2) - i128::from(loads[destination]).pow(2));
        let Ok(delta) = i64::try_from(delta) else { return };
        if selection.best.as_ref().is_some_and(|(best, _)| delta > best.delta)
            || ((selection.tabu[movement.item] > selection.iteration
                || movement.swap.is_some_and(|other| selection.tabu[other] > selection.iteration))
                && i128::from(state.objective) + i128::from(delta) >= i128::from(selection.aspiration))
        {
            return;
        }
        let moved_bin = |item: usize| {
            if item == movement.item {
                destination
            } else if Some(item) == movement.swap {
                source
            } else {
                state.bins[item]
            }
        };
        for item in std::iter::once(movement.item).chain(movement.swap) {
            let bin = moved_bin(item) as i64;
            if self.predecessors[item].iter().any(|&(before, gap)| moved_bin(before) as i64 + gap > bin)
                || self.successors[item].iter().any(|&(after, gap)| bin + gap > moved_bin(after) as i64)
            {
                return;
            }
        }
        for (dimension, loads) in self.dimensions.iter().zip(&state.loads) {
            let difference = dimension.sizes[movement.item] - movement.swap.map_or(0, |other| dimension.sizes[other]);
            if !dimension.domains[source].contains(loads[source] - difference)
                || !dimension.domains[destination].contains(loads[destination] + difference)
            {
                return;
            }
        }
        movement.delta = delta;
        let tie = next_random(selection.random);
        if selection.best.as_ref().is_none_or(|(best, old_tie)| delta < best.delta || tie < *old_tie) {
            selection.best = Some((movement, tie));
        }
    }
}

fn next_random(state: &mut u64) -> u64 {
    *state = crate::mix64(state.wrapping_add(0x9e37_79b9_7f4a_7c15));
    *state
}

fn domain_maximum(domain: &IntDomain) -> i64 {
    match domain {
        IntDomain::Bool => 1,
        IntDomain::Range { hi, .. } => i64::from(*hi),
        IntDomain::Set(values) => values.iter().copied().max().map_or(i64::MIN, i64::from),
    }
}

fn square_terms(expr: &IntExpr, scale: i64, squares: &mut BTreeMap<IntVarRef, i64>, constant: &mut i64, stop: &AtomicBool) -> Option<()> {
    if stop.load(Ordering::Acquire) {
        return None;
    }
    match expr {
        IntExpr::Constant(value) => *constant = constant.checked_add(scale.checked_mul(*value)?)?,
        IntExpr::Add(terms) => {
            for term in terms {
                square_terms(term, scale, squares, constant, stop)?;
            }
        }
        IntExpr::Mul(terms) => {
            let mut coefficient = scale;
            let mut variables = Vec::new();
            for term in terms {
                match term {
                    IntExpr::Constant(value) => coefficient = coefficient.checked_mul(*value)?,
                    IntExpr::Variable(variable) => variables.push(*variable),
                    _ => return None,
                }
            }
            let [first, second] = variables.as_slice() else { return None };
            if first != second {
                return None;
            }
            let old = squares.get(first).copied().unwrap_or(0);
            squares.insert(*first, old.checked_add(coefficient)?);
        }
        _ => return None,
    }
    Some(())
}

fn precedence(form: &AffineForm, relation: Relation, item_index: &[Option<usize>]) -> Option<(usize, usize, i64)> {
    let mut entries = form.coefficients.iter();
    let (&first, &a) = entries.next()?;
    let (&second, &b) = entries.next()?;
    if entries.next().is_some() || !matches!(a, -1 | 1) || b != -a {
        return None;
    }
    let (positive, negative) = if a > 0 { (first, second) } else { (second, first) };
    let (before, after, gap) = match relation {
        Relation::Le => (positive, negative, form.constant),
        Relation::Lt => (positive, negative, form.constant.checked_add(1)?),
        Relation::Ge => (negative, positive, form.constant.checked_neg()?),
        Relation::Gt => (negative, positive, form.constant.checked_neg()?.checked_add(1)?),
        _ => return None,
    };
    if gap < 0 {
        return None;
    }
    Some(((*item_index.get(before.0)?)?, (*item_index.get(after.0)?)?, gap))
}

/// Prove an equality by substituting all load definitions. Each item's
/// contribution must be constant over its actual allowed bin values.
fn derived_identity(
    form: &AffineForm,
    items: &[IntVarRef],
    allowed: &[Vec<usize>],
    dimensions: &[LoadDimension],
    fixed: &[Option<i32>],
    stop: &AtomicBool,
) -> bool {
    let mut constant = i128::from(form.constant);
    for (&variable, &coefficient) in &form.coefficients {
        if stop.load(Ordering::Acquire) {
            return false;
        }
        if let Some(value) = fixed.get(variable.0).copied().flatten() {
            constant += i128::from(coefficient) * i128::from(value);
        }
    }
    for (item, variable) in items.iter().enumerate() {
        if stop.load(Ordering::Acquire) {
            return false;
        }
        let contribution = |bin: usize| {
            i128::from(form.coefficients.get(variable).copied().unwrap_or(0)) * bin as i128
                + dimensions
                    .iter()
                    .map(|dimension| {
                        i128::from(dimension.sizes[item])
                            * i128::from(form.coefficients.get(&dimension.variables[bin]).copied().unwrap_or(0))
                    })
                    .sum::<i128>()
        };
        let value = contribution(allowed[item][0]);
        if allowed[item].iter().any(|&bin| contribution(bin) != value) {
            return false;
        }
        constant += value;
    }
    constant == 0
}
