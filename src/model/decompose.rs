//! Proven-independent semantic decomposition helpers.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};

use super::{
    list, BoolLiteral, Constraint, ConstraintRef, IntExpr, IntGlobalConstraint, IntVarRef, IntervalMode, IntervalModeRef, IntervalVarRef,
    ListVarRef, Model, ModelObject, Objective, SetVarRef,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IndependentFamily {
    IntegerSet,
    Lists,
    Intervals,
}

#[derive(Clone)]
pub(crate) struct IndependentComponent {
    pub family: IndependentFamily,
    pub model: Model,
    /// Original to component-local semantic indices whose values this
    /// component owns at merge.
    pub integers: Vec<(usize, usize)>,
    pub sets: Vec<(usize, usize)>,
    pub lists: Vec<(usize, usize)>,
    pub intervals: Vec<(usize, usize)>,
    /// Original to component-local interval-mode indices.
    pub interval_modes: Vec<(usize, usize)>,
    /// Component-local objective tier to original semantic tier mapping.
    pub objective_tiers: Vec<usize>,
    /// Original object to component-local object mapping for metadata.
    pub objects: BTreeMap<ModelObject, ModelObject>,
}

pub(crate) enum IndependentDecomposition {
    Interrupted,
    NotApplicable,
    Components(Vec<IndependentComponent>),
}

impl Model {
    /// Split a model along family boundaries and connected components of each
    /// family's semantic dependency graph. Every objective tier must belong
    /// entirely to one component. A tier spanning several components keeps
    /// that family on its non-decomposed solve path.
    pub(crate) fn independent_family_components_interruptible(&self, stop: &AtomicBool) -> IndependentDecomposition {
        if interrupted(stop) {
            return IndependentDecomposition::Interrupted;
        }
        // Integer/set models are the common CP path. They can only be split by
        // the dependency graph below, so building and cloning three temporary
        // family models first would be pure overhead.
        if self.lists.is_empty() && self.intervals.is_empty() {
            return match self.integer_connected_components_interruptible(stop) {
                Ok(components) => finish_decomposition(stop, components),
                Err(()) => IndependentDecomposition::Interrupted,
            };
        }
        let families = match self.family_components_interruptible(stop) {
            Ok(families) => families,
            Err(()) => return IndependentDecomposition::Interrupted,
        };
        if families.len() > 1 {
            let mut refined = Vec::new();
            for family in families {
                if interrupted(stop) {
                    return IndependentDecomposition::Interrupted;
                }
                match refine_family_component_interruptible(family, stop) {
                    Ok(mut components) => refined.append(&mut components),
                    Err(()) => return IndependentDecomposition::Interrupted,
                }
            }
            return finish_decomposition(stop, Some(refined));
        }
        let components = if !self.lists.is_empty() {
            self.list_connected_components_interruptible(stop)
        } else if !self.intervals.is_empty() {
            self.interval_connected_components_interruptible(stop)
        } else {
            self.integer_connected_components_interruptible(stop)
        };
        match components {
            Ok(components) => finish_decomposition(stop, components),
            Err(()) => IndependentDecomposition::Interrupted,
        }
    }

    fn family_components_interruptible(&self, stop: &AtomicBool) -> Result<Vec<IndependentComponent>, ()> {
        check_stop(stop)?;

        let mut integer = Model::new();
        integer.int_vars = clone_slice_interruptible(&self.int_vars, stop)?;
        integer.sets = clone_slice_interruptible(&self.sets, stop)?;
        let mut integer_constraints = Vec::new();
        let mut integer_objectives = Vec::new();

        let mut lists = Model::new();
        lists.lists = clone_slice_interruptible(&self.lists, stop)?;
        let mut list_constraints = Vec::new();
        let mut list_objectives = Vec::new();

        let mut intervals = Model::new();
        intervals.intervals = clone_slice_interruptible(&self.intervals, stop)?;
        intervals.interval_modes = clone_slice_interruptible(&self.interval_modes, stop)?;
        let mut interval_constraints = Vec::new();
        let mut interval_objectives = Vec::new();

        for (index, constraint) in self.constraints.iter().enumerate() {
            check_stop(stop)?;
            let (target, mapping) = match constraint {
                Constraint::Intension(_)
                | Constraint::Selected { .. }
                | Constraint::Linear { .. }
                | Constraint::Clause(_)
                | Constraint::IntegerGlobal(_)
                | Constraint::SetSubset { .. }
                | Constraint::SetDisjoint { .. }
                | Constraint::SetCardinality { .. } => (&mut integer, &mut integer_constraints),
                Constraint::ListPartition { .. }
                | Constraint::ListPartitionWithCoverage { .. }
                | Constraint::SameList { .. }
                | Constraint::ItemPrecedence { .. }
                | Constraint::CollectionGlobal(_)
                | Constraint::ListLength { .. }
                | Constraint::ListItemSum { .. }
                | Constraint::ListReduction(_) => (&mut lists, &mut list_constraints),
                Constraint::IntervalPrecedence { .. }
                | Constraint::IntervalAlternative { .. }
                | Constraint::IntervalEndpointRelation { .. }
                | Constraint::IntervalResource(_) => (&mut intervals, &mut interval_constraints),
            };
            let local = target.add_constraint(constraint.clone());
            check_stop(stop)?;
            mapping.push((index, local.0));
        }

        for (index, objective) in self.objectives.iter().enumerate() {
            check_stop(stop)?;
            let (target, mapping) = match objective {
                Objective::IntExpr { .. } => (&mut integer, &mut integer_objectives),
                Objective::ListTerms { .. } => (&mut lists, &mut list_objectives),
                Objective::Makespan { .. } => (&mut intervals, &mut interval_objectives),
            };
            let local = target.add_objective(objective.clone());
            check_stop(stop)?;
            mapping.push((index, local.0));
        }

        let mut components = Vec::new();
        if !integer.int_vars.is_empty() || !integer.sets.is_empty() || !integer.constraints.is_empty() || !integer.objectives.is_empty() {
            components.push(component_interruptible(
                IndependentFamily::IntegerSet,
                integer,
                identity_mappings_interruptible(self.int_vars.len(), stop)?,
                identity_mappings_interruptible(self.sets.len(), stop)?,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                integer_constraints,
                integer_objectives,
                stop,
            )?);
        }
        if !lists.lists.is_empty() || !lists.constraints.is_empty() || !lists.objectives.is_empty() {
            components.push(component_interruptible(
                IndependentFamily::Lists,
                lists,
                Vec::new(),
                Vec::new(),
                identity_mappings_interruptible(self.lists.len(), stop)?,
                Vec::new(),
                Vec::new(),
                list_constraints,
                list_objectives,
                stop,
            )?);
        }
        if !intervals.intervals.is_empty() || !intervals.constraints.is_empty() || !intervals.objectives.is_empty() {
            components.push(component_interruptible(
                IndependentFamily::Intervals,
                intervals,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                identity_mappings_interruptible(self.intervals.len(), stop)?,
                identity_mappings_interruptible(self.interval_modes.len(), stop)?,
                interval_constraints,
                interval_objectives,
                stop,
            )?);
        }
        check_stop(stop)?;
        Ok(components)
    }

    fn integer_connected_components_interruptible(&self, stop: &AtomicBool) -> Result<Option<Vec<IndependentComponent>>, ()> {
        check_stop(stop)?;
        let variable_count = self.int_vars.len() + self.sets.len();
        if variable_count < 2 {
            return Ok(None);
        }
        let mut union = UnionFind::new_interruptible(variable_count, stop)?;
        let mut scopes = Vec::with_capacity(self.constraints.len());
        for constraint in &self.constraints {
            check_stop(stop)?;
            let scope = integer_scope_interruptible(constraint, self.int_vars.len(), stop)?;
            if scope.is_empty() {
                return Ok(None);
            }
            scopes.push(scope);
        }
        for scope in &scopes {
            check_stop(stop)?;
            if let Some((&first, rest)) = scope.split_first() {
                for &node in rest {
                    check_stop(stop)?;
                    union.join(first, node, stop)?;
                }
            }
        }

        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for node in 0..variable_count {
            check_stop(stop)?;
            let root = union.root(node, stop)?;
            groups.entry(root).or_default().push(node);
        }
        if groups.len() < 2 {
            return Ok(None);
        }

        let mut grouped_constraints: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (index, scope) in scopes.iter().enumerate() {
            check_stop(stop)?;
            let root = union.root(scope[0], stop)?;
            grouped_constraints.entry(root).or_default().push(index);
        }

        let mut grouped_objectives: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (index, objective) in self.objectives.iter().enumerate() {
            check_stop(stop)?;
            let Objective::IntExpr { expr, .. } = objective else {
                return Ok(None);
            };
            let mut scope = BTreeSet::new();
            insert_expr_scope(expr, &mut scope, stop)?;
            let Some(&first) = scope.iter().next() else {
                // A constant tier has no unique owner among several connected
                // components. Keep it on the single-engine path.
                return Ok(None);
            };
            let root = union.root(first, stop)?;
            for &node in scope.iter().skip(1) {
                check_stop(stop)?;
                if union.root(node, stop)? != root {
                    // The objective couples otherwise independent components.
                    return Ok(None);
                }
            }
            grouped_objectives.entry(root).or_default().push(index);
        }

        let mut components = Vec::with_capacity(groups.len());
        let mut integer_map = vec![None; self.int_vars.len()];
        let mut set_map = vec![None; self.sets.len()];
        for (root, nodes) in groups {
            check_stop(stop)?;
            let mut integers = Vec::new();
            let mut sets = Vec::new();
            let mut model = Model::new();
            for node in nodes {
                check_stop(stop)?;
                if node < self.int_vars.len() {
                    let local = model.int_vars.len();
                    model.int_vars.push(self.int_vars[node].clone());
                    integer_map[node] = Some(local);
                    integers.push((node, local));
                } else {
                    let original = node - self.int_vars.len();
                    let local = model.sets.len();
                    model.sets.push(self.sets[original].clone());
                    set_map[original] = Some(local);
                    sets.push((original, local));
                }
                check_stop(stop)?;
            }
            let mut constraints = Vec::new();
            for original in grouped_constraints.remove(&root).unwrap_or_default() {
                check_stop(stop)?;
                let Some(constraint) = remap_integer_constraint_interruptible(&self.constraints[original], &integer_map, &set_map, stop)?
                else {
                    return Ok(None);
                };
                let local = model.add_constraint(constraint);
                constraints.push((original, local.0));
            }
            let mut objectives = Vec::new();
            for original in grouped_objectives.remove(&root).unwrap_or_default() {
                check_stop(stop)?;
                let Objective::IntExpr { minimize, expr } = &self.objectives[original] else {
                    return Ok(None);
                };
                let Some(expr) = remap_int_expr_interruptible(expr, &integer_map, stop)? else {
                    return Ok(None);
                };
                let objective = Objective::IntExpr { minimize: *minimize, expr };
                let local = model.add_objective(objective);
                objectives.push((original, local.0));
            }
            for &(original, _) in &integers {
                integer_map[original] = None;
            }
            for &(original, _) in &sets {
                set_map[original] = None;
            }
            components.push(component_interruptible(
                IndependentFamily::IntegerSet,
                model,
                integers,
                sets,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                constraints,
                objectives,
                stop,
            )?);
        }
        check_stop(stop)?;
        Ok(Some(components))
    }

    fn list_connected_components_interruptible(&self, stop: &AtomicBool) -> Result<Option<Vec<IndependentComponent>>, ()> {
        check_stop(stop)?;
        let variable_count = self.lists.len();
        if variable_count < 2 {
            return Ok(None);
        }
        let mut union = UnionFind::new_interruptible(variable_count, stop)?;
        let mut scopes = Vec::with_capacity(self.constraints.len());
        for constraint in &self.constraints {
            check_stop(stop)?;
            let scope = list_scope_interruptible(constraint, variable_count, stop)?;
            if scope.is_empty() || scope.iter().any(|&node| node >= variable_count) {
                return Ok(None);
            }
            join_scope_interruptible(&mut union, &scope, stop)?;
            scopes.push(scope);
        }
        let groups = dependency_groups_interruptible(&mut union, variable_count, stop)?;
        if groups.len() < 2 {
            return Ok(None);
        }

        let mut grouped_constraints: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (index, scope) in scopes.iter().enumerate() {
            check_stop(stop)?;
            grouped_constraints.entry(union.root(scope[0], stop)?).or_default().push(index);
        }
        let mut grouped_objectives: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (index, objective) in self.objectives.iter().enumerate() {
            check_stop(stop)?;
            let Objective::ListTerms { .. } = objective else {
                return Ok(None);
            };
            let scope = list_objective_scope_interruptible(objective, stop)?;
            let Some(&first) = scope.first() else {
                // A constant list tier has no unique component owner.
                return Ok(None);
            };
            if scope.iter().any(|&node| node >= variable_count) {
                return Ok(None);
            }
            let root = union.root(first, stop)?;
            for &node in scope.iter().skip(1) {
                check_stop(stop)?;
                if union.root(node, stop)? != root {
                    return Ok(None);
                }
            }
            grouped_objectives.entry(root).or_default().push(index);
        }

        let mut components = Vec::with_capacity(groups.len());
        let mut list_map = vec![None; variable_count];
        for (root, nodes) in groups {
            check_stop(stop)?;
            let mut lists = Vec::with_capacity(nodes.len());
            let mut model = Model::new();
            for original in nodes {
                check_stop(stop)?;
                let local = model.lists.len();
                model.lists.push(self.lists[original].clone());
                list_map[original] = Some(local);
                lists.push((original, local));
                check_stop(stop)?;
            }
            let mut constraints = Vec::new();
            for original in grouped_constraints.remove(&root).unwrap_or_default() {
                check_stop(stop)?;
                let Some(constraint) = remap_list_constraint_interruptible(&self.constraints[original], &list_map, stop)? else {
                    return Ok(None);
                };
                let local = model.add_constraint(constraint);
                constraints.push((original, local.0));
            }
            let mut objectives = Vec::new();
            for original in grouped_objectives.remove(&root).unwrap_or_default() {
                check_stop(stop)?;
                let Some(objective) = remap_list_objective_interruptible(&self.objectives[original], &list_map, stop)? else {
                    return Ok(None);
                };
                let local = model.add_objective(objective);
                objectives.push((original, local.0));
            }
            for &(original, _) in &lists {
                list_map[original] = None;
            }
            components.push(component_interruptible(
                IndependentFamily::Lists,
                model,
                Vec::new(),
                Vec::new(),
                lists,
                Vec::new(),
                Vec::new(),
                constraints,
                objectives,
                stop,
            )?);
        }
        check_stop(stop)?;
        Ok(Some(components))
    }

    fn interval_connected_components_interruptible(&self, stop: &AtomicBool) -> Result<Option<Vec<IndependentComponent>>, ()> {
        check_stop(stop)?;
        let variable_count = self.intervals.len();
        if variable_count < 2 {
            return Ok(None);
        }
        let mut union = UnionFind::new_interruptible(variable_count, stop)?;
        let mut scopes = Vec::with_capacity(self.constraints.len());
        for constraint in &self.constraints {
            check_stop(stop)?;
            let scope = interval_scope_interruptible(constraint, variable_count, stop)?;
            if scope.is_empty() || scope.iter().any(|&node| node >= variable_count) {
                return Ok(None);
            }
            join_scope_interruptible(&mut union, &scope, stop)?;
            scopes.push(scope);
        }
        let groups = dependency_groups_interruptible(&mut union, variable_count, stop)?;
        if groups.len() < 2 {
            return Ok(None);
        }

        let mut grouped_constraints: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (index, scope) in scopes.iter().enumerate() {
            check_stop(stop)?;
            grouped_constraints.entry(union.root(scope[0], stop)?).or_default().push(index);
        }
        let mut grouped_objectives: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (index, objective) in self.objectives.iter().enumerate() {
            check_stop(stop)?;
            let Objective::Makespan { intervals, .. } = objective else {
                return Ok(None);
            };
            let mut scope = BTreeSet::new();
            for interval in intervals {
                check_stop(stop)?;
                scope.insert(interval.0);
            }
            let Some(&first) = scope.iter().next() else {
                return Ok(None);
            };
            if scope.iter().any(|&node| node >= variable_count) {
                return Ok(None);
            }
            let root = union.root(first, stop)?;
            for &node in scope.iter().skip(1) {
                check_stop(stop)?;
                if union.root(node, stop)? != root {
                    return Ok(None);
                }
            }
            grouped_objectives.entry(root).or_default().push(index);
        }

        let mut modes_by_interval = vec![Vec::new(); variable_count];
        for (mode, declaration) in self.interval_modes.iter().enumerate() {
            check_stop(stop)?;
            let Some(modes) = modes_by_interval.get_mut(declaration.interval.0) else {
                return Ok(None);
            };
            modes.push(mode);
        }
        let mut components = Vec::with_capacity(groups.len());
        let mut interval_map = vec![None; variable_count];
        let mut mode_map = vec![None; self.interval_modes.len()];
        for (root, nodes) in groups {
            check_stop(stop)?;
            let mut intervals = Vec::with_capacity(nodes.len());
            let mut model = Model::new();
            for original in nodes {
                check_stop(stop)?;
                let local = model.intervals.len();
                let mut declaration = self.intervals[original].clone();
                declaration.modes = Vec::new();
                model.intervals.push(declaration);
                interval_map[original] = Some(local);
                intervals.push((original, local));
                check_stop(stop)?;
            }

            let mut owned_modes = Vec::new();
            for &(original, _) in &intervals {
                check_stop(stop)?;
                owned_modes.extend_from_slice(&modes_by_interval[original]);
            }
            owned_modes.sort_unstable();
            let mut interval_modes = Vec::with_capacity(owned_modes.len());
            for original in owned_modes {
                check_stop(stop)?;
                let mode = &self.interval_modes[original];
                let Some(owner) = interval_map[mode.interval.0] else {
                    return Ok(None);
                };
                let local = model.interval_modes.len();
                model.interval_modes.push(IntervalMode { interval: IntervalVarRef(owner), ..*mode });
                mode_map[original] = Some(local);
                interval_modes.push((original, local));
                check_stop(stop)?;
            }
            for &(original, local) in &intervals {
                check_stop(stop)?;
                let mut modes = Vec::with_capacity(self.intervals[original].modes.len());
                for mode in &self.intervals[original].modes {
                    check_stop(stop)?;
                    let Some(local_mode) = mode_map.get(mode.0).copied().flatten() else {
                        return Ok(None);
                    };
                    modes.push(IntervalModeRef(local_mode));
                }
                model.intervals[local].modes = modes;
            }

            let mut constraints = Vec::new();
            for original in grouped_constraints.remove(&root).unwrap_or_default() {
                check_stop(stop)?;
                let Some(constraint) = remap_interval_constraint_interruptible(&self.constraints[original], &interval_map, stop)? else {
                    return Ok(None);
                };
                let local = model.add_constraint(constraint);
                constraints.push((original, local.0));
            }
            let mut objectives = Vec::new();
            for original in grouped_objectives.remove(&root).unwrap_or_default() {
                check_stop(stop)?;
                let Some(objective) = remap_interval_objective_interruptible(&self.objectives[original], &interval_map, stop)? else {
                    return Ok(None);
                };
                let local = model.add_objective(objective);
                objectives.push((original, local.0));
            }
            for &(original, _) in &intervals {
                interval_map[original] = None;
            }
            for &(original, _) in &interval_modes {
                mode_map[original] = None;
            }
            components.push(component_interruptible(
                IndependentFamily::Intervals,
                model,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                intervals,
                interval_modes,
                constraints,
                objectives,
                stop,
            )?);
        }
        check_stop(stop)?;
        Ok(Some(components))
    }
}

fn refine_family_component_interruptible(component: IndependentComponent, stop: &AtomicBool) -> Result<Vec<IndependentComponent>, ()> {
    check_stop(stop)?;
    let nested = match component.family {
        IndependentFamily::IntegerSet => component.model.integer_connected_components_interruptible(stop)?,
        IndependentFamily::Lists => component.model.list_connected_components_interruptible(stop)?,
        IndependentFamily::Intervals => component.model.interval_connected_components_interruptible(stop)?,
    };
    let Some(nested) = nested else {
        return Ok(vec![component]);
    };
    let parent_indices = ParentIndexMappings {
        integers: original_by_local_interruptible(&component.integers, stop)?,
        sets: original_by_local_interruptible(&component.sets, stop)?,
        lists: original_by_local_interruptible(&component.lists, stop)?,
        intervals: original_by_local_interruptible(&component.intervals, stop)?,
        interval_modes: original_by_local_interruptible(&component.interval_modes, stop)?,
        objects: reverse_object_mappings_interruptible(&component.objects, stop)?,
    };
    let mut refined = Vec::with_capacity(nested.len());
    for nested in nested {
        check_stop(stop)?;
        refined.push(compose_component_mappings_interruptible(&component, &parent_indices, nested, stop)?);
    }
    check_stop(stop)?;
    Ok(refined)
}

fn compose_component_mappings_interruptible(
    parent: &IndependentComponent,
    parent_indices: &ParentIndexMappings,
    mut child: IndependentComponent,
    stop: &AtomicBool,
) -> Result<IndependentComponent, ()> {
    child.integers = compose_index_mappings_interruptible(&parent_indices.integers, &child.integers, stop)?;
    child.sets = compose_index_mappings_interruptible(&parent_indices.sets, &child.sets, stop)?;
    child.lists = compose_index_mappings_interruptible(&parent_indices.lists, &child.lists, stop)?;
    child.intervals = compose_index_mappings_interruptible(&parent_indices.intervals, &child.intervals, stop)?;
    child.interval_modes = compose_index_mappings_interruptible(&parent_indices.interval_modes, &child.interval_modes, stop)?;

    let mut objective_tiers = Vec::with_capacity(child.objective_tiers.len());
    for &parent_local in &child.objective_tiers {
        check_stop(stop)?;
        let Some(&original) = parent.objective_tiers.get(parent_local) else {
            return Err(());
        };
        objective_tiers.push(original);
    }
    child.objective_tiers = objective_tiers;

    let mut objects = BTreeMap::new();
    for (&parent_local, &child_local) in &child.objects {
        check_stop(stop)?;
        let Some(&original) = parent_indices.objects.get(&parent_local) else {
            return Err(());
        };
        objects.insert(original, child_local);
    }
    child.objects = objects;
    check_stop(stop)?;
    Ok(child)
}

fn compose_index_mappings_interruptible(
    original_by_local: &[usize],
    child: &[(usize, usize)],
    stop: &AtomicBool,
) -> Result<Vec<(usize, usize)>, ()> {
    let mut composed = Vec::with_capacity(child.len());
    for &(parent_local, child_local) in child {
        check_stop(stop)?;
        let Some(&original) = original_by_local.get(parent_local) else {
            return Err(());
        };
        composed.push((original, child_local));
    }
    check_stop(stop)?;
    Ok(composed)
}

struct ParentIndexMappings {
    integers: Vec<usize>,
    sets: Vec<usize>,
    lists: Vec<usize>,
    intervals: Vec<usize>,
    interval_modes: Vec<usize>,
    objects: BTreeMap<ModelObject, ModelObject>,
}

fn original_by_local_interruptible(mappings: &[(usize, usize)], stop: &AtomicBool) -> Result<Vec<usize>, ()> {
    let mut originals = vec![usize::MAX; mappings.len()];
    for &(original, local) in mappings {
        check_stop(stop)?;
        let Some(slot) = originals.get_mut(local) else {
            return Err(());
        };
        if std::mem::replace(slot, original) != usize::MAX {
            return Err(());
        }
    }
    if originals.contains(&usize::MAX) {
        return Err(());
    }
    check_stop(stop)?;
    Ok(originals)
}

fn reverse_object_mappings_interruptible(
    mappings: &BTreeMap<ModelObject, ModelObject>,
    stop: &AtomicBool,
) -> Result<BTreeMap<ModelObject, ModelObject>, ()> {
    let mut reversed = BTreeMap::new();
    for (&original, &local) in mappings {
        check_stop(stop)?;
        if reversed.insert(local, original).is_some() {
            return Err(());
        }
    }
    check_stop(stop)?;
    Ok(reversed)
}

fn finish_decomposition(stop: &AtomicBool, components: Option<Vec<IndependentComponent>>) -> IndependentDecomposition {
    if interrupted(stop) {
        IndependentDecomposition::Interrupted
    } else if let Some(components) = components {
        IndependentDecomposition::Components(components)
    } else {
        IndependentDecomposition::NotApplicable
    }
}

fn interrupted(stop: &AtomicBool) -> bool {
    stop.load(Ordering::Acquire)
}

fn check_stop(stop: &AtomicBool) -> Result<(), ()> {
    if interrupted(stop) {
        Err(())
    } else {
        Ok(())
    }
}

fn clone_slice_interruptible<T: Clone>(values: &[T], stop: &AtomicBool) -> Result<Vec<T>, ()> {
    let mut copied = Vec::with_capacity(values.len());
    for value in values {
        check_stop(stop)?;
        copied.push(value.clone());
        check_stop(stop)?;
    }
    Ok(copied)
}

fn identity_mappings_interruptible(length: usize, stop: &AtomicBool) -> Result<Vec<(usize, usize)>, ()> {
    let mut indices = Vec::with_capacity(length);
    for index in 0..length {
        check_stop(stop)?;
        indices.push((index, index));
    }
    Ok(indices)
}

#[allow(clippy::too_many_arguments)]
fn component_interruptible(
    family: IndependentFamily,
    model: Model,
    integers: Vec<(usize, usize)>,
    sets: Vec<(usize, usize)>,
    lists: Vec<(usize, usize)>,
    intervals: Vec<(usize, usize)>,
    interval_modes: Vec<(usize, usize)>,
    constraints: Vec<(usize, usize)>,
    objectives: Vec<(usize, usize)>,
    stop: &AtomicBool,
) -> Result<IndependentComponent, ()> {
    check_stop(stop)?;
    let mut objects = BTreeMap::new();
    for &(original, local) in &integers {
        check_stop(stop)?;
        objects.insert(ModelObject::IntVar(IntVarRef(original)), ModelObject::IntVar(IntVarRef(local)));
    }
    for &(original, local) in &sets {
        check_stop(stop)?;
        objects.insert(ModelObject::SetVar(SetVarRef(original)), ModelObject::SetVar(SetVarRef(local)));
    }
    for &(original, local) in &lists {
        check_stop(stop)?;
        objects.insert(ModelObject::ListVar(ListVarRef(original)), ModelObject::ListVar(ListVarRef(local)));
    }
    for &(original, local) in &intervals {
        check_stop(stop)?;
        objects.insert(ModelObject::IntervalVar(IntervalVarRef(original)), ModelObject::IntervalVar(IntervalVarRef(local)));
    }
    for &(original, local) in &interval_modes {
        check_stop(stop)?;
        objects.insert(ModelObject::IntervalMode(IntervalModeRef(original)), ModelObject::IntervalMode(IntervalModeRef(local)));
    }
    for (original, local) in constraints {
        check_stop(stop)?;
        objects.insert(ModelObject::Constraint(ConstraintRef(original)), ModelObject::Constraint(ConstraintRef(local)));
    }
    let mut objective_tiers = vec![usize::MAX; objectives.len()];
    for (original, local) in objectives {
        check_stop(stop)?;
        let Some(tier) = objective_tiers.get_mut(local) else {
            // Component construction owns both indices, so this can only be a
            // programming error rather than malformed user input.
            return Err(());
        };
        *tier = original;
        objects.insert(ModelObject::Objective(super::ObjectiveRef(original)), ModelObject::Objective(super::ObjectiveRef(local)));
    }
    if objective_tiers.contains(&usize::MAX) {
        return Err(());
    }
    check_stop(stop)?;
    Ok(IndependentComponent { family, model, integers, sets, lists, intervals, interval_modes, objective_tiers, objects })
}

fn join_scope_interruptible(union: &mut UnionFind, scope: &[usize], stop: &AtomicBool) -> Result<(), ()> {
    if let Some((&first, rest)) = scope.split_first() {
        for &node in rest {
            check_stop(stop)?;
            union.join(first, node, stop)?;
        }
    }
    check_stop(stop)
}

fn dependency_groups_interruptible(
    union: &mut UnionFind,
    variable_count: usize,
    stop: &AtomicBool,
) -> Result<BTreeMap<usize, Vec<usize>>, ()> {
    let mut groups = BTreeMap::new();
    for node in 0..variable_count {
        check_stop(stop)?;
        groups.entry(union.root(node, stop)?).or_insert_with(Vec::new).push(node);
    }
    check_stop(stop)?;
    Ok(groups)
}

fn list_scope_interruptible(constraint: &Constraint, list_count: usize, stop: &AtomicBool) -> Result<Vec<usize>, ()> {
    let mut scope = BTreeSet::new();
    match constraint {
        Constraint::ListPartition { lists, .. }
        | Constraint::ListPartitionWithCoverage { lists, .. }
        | Constraint::SameList { lists, .. }
        | Constraint::ItemPrecedence { lists, .. } => {
            for list in lists {
                check_stop(stop)?;
                scope.insert(list.0);
            }
        }
        Constraint::CollectionGlobal(_) => {
            for list in 0..list_count {
                check_stop(stop)?;
                scope.insert(list);
            }
        }
        Constraint::ListLength { list, .. } | Constraint::ListItemSum { list, .. } => {
            scope.insert(list.0);
        }
        Constraint::ListReduction(constraint) => {
            scope.insert(constraint.reduction.iterable.list());
        }
        Constraint::IntervalPrecedence { .. }
        | Constraint::IntervalAlternative { .. }
        | Constraint::IntervalEndpointRelation { .. }
        | Constraint::IntervalResource(_)
        | Constraint::Intension(_)
        | Constraint::Selected { .. }
        | Constraint::Linear { .. }
        | Constraint::Clause(_)
        | Constraint::IntegerGlobal(_)
        | Constraint::SetSubset { .. }
        | Constraint::SetDisjoint { .. }
        | Constraint::SetCardinality { .. } => {}
    }
    check_stop(stop)?;
    Ok(scope.into_iter().collect())
}

fn list_objective_scope_interruptible(objective: &Objective, stop: &AtomicBool) -> Result<Vec<usize>, ()> {
    let Objective::ListTerms { terms, max_terms, .. } = objective else {
        return Ok(Vec::new());
    };
    let mut scope = BTreeSet::new();
    for reduction in terms {
        check_stop(stop)?;
        scope.insert(reduction.iterable.list());
    }
    for reduction in max_terms.iter().flat_map(|terms| terms.iter()).flat_map(|term| term.groups.iter()).flat_map(|group| group.iter()) {
        check_stop(stop)?;
        scope.insert(reduction.iterable.list());
    }
    check_stop(stop)?;
    Ok(scope.into_iter().collect())
}

fn interval_scope_interruptible(constraint: &Constraint, interval_count: usize, stop: &AtomicBool) -> Result<Vec<usize>, ()> {
    let mut scope = BTreeSet::new();
    match constraint {
        Constraint::IntervalPrecedence { before, after } => {
            scope.insert(before.0);
            scope.insert(after.0);
        }
        Constraint::IntervalAlternative { master, members } => {
            scope.insert(master.0);
            for member in members {
                check_stop(stop)?;
                scope.insert(member.0);
            }
        }
        Constraint::IntervalEndpointRelation { left, right, .. } => {
            scope.insert(left.0);
            scope.insert(right.0);
        }
        Constraint::IntervalResource(list::Resource::NoOverlap(intervals)) => {
            for &interval in intervals {
                check_stop(stop)?;
                scope.insert(interval);
            }
        }
        Constraint::IntervalResource(list::Resource::MachineNoOverlap) => {
            // The resource is implicit over the complete mode arena. Keeping
            // it in one component is conservative and gives its metadata one
            // unambiguous owner.
            for interval in 0..interval_count {
                check_stop(stop)?;
                scope.insert(interval);
            }
        }
        Constraint::IntervalResource(list::Resource::Cumulative { demands, .. }) => {
            for &(interval, _) in demands {
                check_stop(stop)?;
                scope.insert(interval);
            }
        }
        Constraint::ListPartition { .. }
        | Constraint::ListPartitionWithCoverage { .. }
        | Constraint::SameList { .. }
        | Constraint::ItemPrecedence { .. }
        | Constraint::CollectionGlobal(_)
        | Constraint::ListLength { .. }
        | Constraint::ListItemSum { .. }
        | Constraint::ListReduction(_)
        | Constraint::Intension(_)
        | Constraint::Selected { .. }
        | Constraint::Linear { .. }
        | Constraint::Clause(_)
        | Constraint::IntegerGlobal(_)
        | Constraint::SetSubset { .. }
        | Constraint::SetDisjoint { .. }
        | Constraint::SetCardinality { .. } => {}
    }
    check_stop(stop)?;
    Ok(scope.into_iter().collect())
}

fn remap_integer_constraint_interruptible(
    constraint: &Constraint,
    integers: &[Option<usize>],
    sets: &[Option<usize>],
    stop: &AtomicBool,
) -> Result<Option<Constraint>, ()> {
    check_stop(stop)?;
    let mapped = match constraint {
        Constraint::Intension(expression) => {
            let Some(expression) = remap_int_expr_interruptible(expression, integers, stop)? else {
                return Ok(None);
            };
            Constraint::Intension(expression)
        }
        Constraint::Selected { selector, constraint } => {
            let Some(selector) = remap_int_ref(*selector, integers) else {
                return Ok(None);
            };
            let Some(constraint) = remap_integer_constraint_interruptible(constraint, integers, sets, stop)? else {
                return Ok(None);
            };
            Constraint::Selected { selector, constraint: Box::new(constraint) }
        }
        Constraint::Linear { terms, relation, rhs } => {
            let mut mapped = Vec::with_capacity(terms.len());
            for &(coefficient, variable) in terms {
                check_stop(stop)?;
                let Some(variable) = remap_int_ref(variable, integers) else {
                    return Ok(None);
                };
                mapped.push((coefficient, variable));
            }
            Constraint::Linear { terms: mapped, relation: *relation, rhs: *rhs }
        }
        Constraint::Clause(literals) => {
            let mut mapped = Vec::with_capacity(literals.len());
            for literal in literals {
                check_stop(stop)?;
                let Some(variable) = remap_int_ref(literal.variable, integers) else {
                    return Ok(None);
                };
                mapped.push(BoolLiteral { variable, positive: literal.positive });
            }
            Constraint::Clause(mapped)
        }
        Constraint::IntegerGlobal(global) => {
            let Some(global) = remap_int_global_interruptible(global, integers, stop)? else {
                return Ok(None);
            };
            Constraint::IntegerGlobal(global)
        }
        Constraint::SetSubset { subset, superset } => {
            let (Some(subset), Some(superset)) = (remap_set_ref(*subset, sets), remap_set_ref(*superset, sets)) else {
                return Ok(None);
            };
            Constraint::SetSubset { subset, superset }
        }
        Constraint::SetDisjoint { left, right } => {
            let (Some(left), Some(right)) = (remap_set_ref(*left, sets), remap_set_ref(*right, sets)) else {
                return Ok(None);
            };
            Constraint::SetDisjoint { left, right }
        }
        Constraint::SetCardinality { set, min, max } => {
            let Some(set) = remap_set_ref(*set, sets) else {
                return Ok(None);
            };
            Constraint::SetCardinality { set, min: *min, max: *max }
        }
        Constraint::ListPartition { .. }
        | Constraint::ListPartitionWithCoverage { .. }
        | Constraint::SameList { .. }
        | Constraint::ItemPrecedence { .. }
        | Constraint::CollectionGlobal(_)
        | Constraint::ListLength { .. }
        | Constraint::ListItemSum { .. }
        | Constraint::ListReduction(_)
        | Constraint::IntervalPrecedence { .. }
        | Constraint::IntervalAlternative { .. }
        | Constraint::IntervalEndpointRelation { .. }
        | Constraint::IntervalResource(_) => return Ok(None),
    };
    check_stop(stop)?;
    Ok(Some(mapped))
}

fn remap_list_constraint_interruptible(
    constraint: &Constraint,
    lists: &[Option<usize>],
    stop: &AtomicBool,
) -> Result<Option<Constraint>, ()> {
    check_stop(stop)?;
    let mapped = match constraint {
        Constraint::ListPartition { lists: references, items } => Constraint::ListPartition {
            lists: match remap_list_refs_interruptible(references, lists, stop)? {
                Some(references) => references,
                None => return Ok(None),
            },
            items: clone_slice_interruptible(items, stop)?,
        },
        Constraint::ListPartitionWithCoverage { lists: references, items, coverage } => Constraint::ListPartitionWithCoverage {
            lists: match remap_list_refs_interruptible(references, lists, stop)? {
                Some(references) => references,
                None => return Ok(None),
            },
            items: clone_slice_interruptible(items, stop)?,
            coverage: *coverage,
        },
        Constraint::SameList { lists: references, a, b } => Constraint::SameList {
            lists: match remap_list_refs_interruptible(references, lists, stop)? {
                Some(references) => references,
                None => return Ok(None),
            },
            a: *a,
            b: *b,
        },
        Constraint::ItemPrecedence { lists: references, before, after } => Constraint::ItemPrecedence {
            lists: match remap_list_refs_interruptible(references, lists, stop)? {
                Some(references) => references,
                None => return Ok(None),
            },
            before: *before,
            after: *after,
        },
        Constraint::CollectionGlobal(global) => Constraint::CollectionGlobal(global.clone()),
        Constraint::ListLength { list, min, max } => {
            let Some(list) = remap_list_ref(*list, lists) else {
                return Ok(None);
            };
            Constraint::ListLength { list, min: *min, max: *max }
        }
        Constraint::ListItemSum { list, weights, min, max } => {
            let Some(list) = remap_list_ref(*list, lists) else {
                return Ok(None);
            };
            Constraint::ListItemSum { list, weights: clone_slice_interruptible(weights, stop)?, min: *min, max: *max }
        }
        Constraint::ListReduction(constraint) => {
            let Some(reduction) = remap_list_reduction_interruptible(&constraint.reduction, lists, stop)? else {
                return Ok(None);
            };
            Constraint::ListReduction(list::Constraint { reduction, op: constraint.op, rhs: constraint.rhs })
        }
        Constraint::IntervalPrecedence { .. }
        | Constraint::IntervalAlternative { .. }
        | Constraint::IntervalEndpointRelation { .. }
        | Constraint::IntervalResource(_)
        | Constraint::Intension(_)
        | Constraint::Selected { .. }
        | Constraint::Linear { .. }
        | Constraint::Clause(_)
        | Constraint::IntegerGlobal(_)
        | Constraint::SetSubset { .. }
        | Constraint::SetDisjoint { .. }
        | Constraint::SetCardinality { .. } => return Ok(None),
    };
    check_stop(stop)?;
    Ok(Some(mapped))
}

fn remap_list_objective_interruptible(objective: &Objective, lists: &[Option<usize>], stop: &AtomicBool) -> Result<Option<Objective>, ()> {
    let Objective::ListTerms { minimize, terms, max_terms } = objective else {
        return Ok(None);
    };
    let mut mapped_terms = Vec::with_capacity(terms.len());
    for reduction in terms {
        check_stop(stop)?;
        let Some(reduction) = remap_list_reduction_interruptible(reduction, lists, stop)? else {
            return Ok(None);
        };
        mapped_terms.push(reduction);
    }
    let mapped_max_terms = if let Some(max_terms) = max_terms {
        let mut mapped = Vec::with_capacity(max_terms.len());
        for max_term in max_terms {
            check_stop(stop)?;
            let mut groups = Vec::with_capacity(max_term.groups.len());
            for group in &max_term.groups {
                check_stop(stop)?;
                let mut mapped_group = Vec::with_capacity(group.len());
                for reduction in group {
                    check_stop(stop)?;
                    let Some(reduction) = remap_list_reduction_interruptible(reduction, lists, stop)? else {
                        return Ok(None);
                    };
                    mapped_group.push(reduction);
                }
                groups.push(mapped_group);
            }
            mapped.push(list::MaxTerm { groups, coeff: max_term.coeff });
        }
        Some(mapped)
    } else {
        None
    };
    check_stop(stop)?;
    Ok(Some(Objective::ListTerms { minimize: *minimize, terms: mapped_terms, max_terms: mapped_max_terms }))
}

fn remap_list_reduction_interruptible(
    reduction: &list::Reduction,
    lists: &[Option<usize>],
    stop: &AtomicBool,
) -> Result<Option<list::Reduction>, ()> {
    check_stop(stop)?;
    let original = reduction.iterable.list();
    let Some(local) = lists.get(original).copied().flatten() else {
        return Ok(None);
    };
    let iterable = match &reduction.iterable {
        list::Iterable::Items(_) => list::Iterable::Items(local),
        list::Iterable::SetItems(_) => list::Iterable::SetItems(local),
        list::Iterable::Edges { start, end, .. } => list::Iterable::Edges { list: local, start: *start, end: *end },
        list::Iterable::Pairs(_) => list::Iterable::Pairs(local),
        list::Iterable::Scan { init, boundary, step, end, .. } => {
            list::Iterable::Scan { list: local, init: *init, boundary: *boundary, step: *step, end: *end }
        }
        list::Iterable::Windows { size, inner, .. } => list::Iterable::Windows { list: local, size: *size, inner: *inner },
    };
    let arena = reduction.arena.clone();
    check_stop(stop)?;
    Ok(Some(list::Reduction { op: reduction.op, iterable, arena, body: reduction.body, coeff: reduction.coeff }))
}

fn remap_interval_constraint_interruptible(
    constraint: &Constraint,
    intervals: &[Option<usize>],
    stop: &AtomicBool,
) -> Result<Option<Constraint>, ()> {
    check_stop(stop)?;
    let mapped = match constraint {
        Constraint::IntervalPrecedence { before, after } => {
            let (Some(before), Some(after)) = (remap_interval_ref(*before, intervals), remap_interval_ref(*after, intervals)) else {
                return Ok(None);
            };
            Constraint::IntervalPrecedence { before, after }
        }
        Constraint::IntervalAlternative { master, members } => {
            let Some(master) = remap_interval_ref(*master, intervals) else {
                return Ok(None);
            };
            let Some(members) = remap_interval_refs_interruptible(members, intervals, stop)? else {
                return Ok(None);
            };
            Constraint::IntervalAlternative { master, members }
        }
        Constraint::IntervalEndpointRelation { left, left_endpoint, relation, right, right_endpoint, offset } => {
            let (Some(left), Some(right)) = (remap_interval_ref(*left, intervals), remap_interval_ref(*right, intervals)) else {
                return Ok(None);
            };
            Constraint::IntervalEndpointRelation {
                left,
                left_endpoint: *left_endpoint,
                relation: *relation,
                right,
                right_endpoint: *right_endpoint,
                offset: *offset,
            }
        }
        Constraint::IntervalResource(resource) => {
            let resource = match resource {
                list::Resource::NoOverlap(references) => {
                    let Some(references) = remap_interval_indices_interruptible(references, intervals, stop)? else {
                        return Ok(None);
                    };
                    list::Resource::NoOverlap(references)
                }
                list::Resource::MachineNoOverlap => list::Resource::MachineNoOverlap,
                list::Resource::Cumulative { demands, capacity } => {
                    let mut mapped = Vec::with_capacity(demands.len());
                    for &(interval, demand) in demands {
                        check_stop(stop)?;
                        let Some(interval) = intervals.get(interval).copied().flatten() else {
                            return Ok(None);
                        };
                        mapped.push((interval, demand));
                    }
                    list::Resource::Cumulative { demands: mapped, capacity: *capacity }
                }
            };
            Constraint::IntervalResource(resource)
        }
        Constraint::ListPartition { .. }
        | Constraint::ListPartitionWithCoverage { .. }
        | Constraint::SameList { .. }
        | Constraint::ItemPrecedence { .. }
        | Constraint::CollectionGlobal(_)
        | Constraint::ListLength { .. }
        | Constraint::ListItemSum { .. }
        | Constraint::ListReduction(_)
        | Constraint::Intension(_)
        | Constraint::Selected { .. }
        | Constraint::Linear { .. }
        | Constraint::Clause(_)
        | Constraint::IntegerGlobal(_)
        | Constraint::SetSubset { .. }
        | Constraint::SetDisjoint { .. }
        | Constraint::SetCardinality { .. } => return Ok(None),
    };
    check_stop(stop)?;
    Ok(Some(mapped))
}

fn remap_interval_objective_interruptible(
    objective: &Objective,
    intervals: &[Option<usize>],
    stop: &AtomicBool,
) -> Result<Option<Objective>, ()> {
    let Objective::Makespan { minimize, intervals: references } = objective else {
        return Ok(None);
    };
    let Some(references) = remap_interval_refs_interruptible(references, intervals, stop)? else {
        return Ok(None);
    };
    Ok(Some(Objective::Makespan { minimize: *minimize, intervals: references }))
}

fn remap_int_ref(reference: IntVarRef, mapping: &[Option<usize>]) -> Option<IntVarRef> {
    mapping.get(reference.0).copied().flatten().map(IntVarRef)
}

fn remap_set_ref(reference: SetVarRef, mapping: &[Option<usize>]) -> Option<SetVarRef> {
    mapping.get(reference.0).copied().flatten().map(SetVarRef)
}

fn remap_list_ref(reference: ListVarRef, mapping: &[Option<usize>]) -> Option<ListVarRef> {
    mapping.get(reference.0).copied().flatten().map(ListVarRef)
}

fn remap_interval_ref(reference: IntervalVarRef, mapping: &[Option<usize>]) -> Option<IntervalVarRef> {
    mapping.get(reference.0).copied().flatten().map(IntervalVarRef)
}

fn remap_list_refs_interruptible(
    references: &[ListVarRef],
    mapping: &[Option<usize>],
    stop: &AtomicBool,
) -> Result<Option<Vec<ListVarRef>>, ()> {
    let mut mapped = Vec::with_capacity(references.len());
    for &reference in references {
        check_stop(stop)?;
        let Some(reference) = remap_list_ref(reference, mapping) else {
            return Ok(None);
        };
        mapped.push(reference);
    }
    check_stop(stop)?;
    Ok(Some(mapped))
}

fn remap_interval_refs_interruptible(
    references: &[IntervalVarRef],
    mapping: &[Option<usize>],
    stop: &AtomicBool,
) -> Result<Option<Vec<IntervalVarRef>>, ()> {
    let mut mapped = Vec::with_capacity(references.len());
    for &reference in references {
        check_stop(stop)?;
        let Some(reference) = remap_interval_ref(reference, mapping) else {
            return Ok(None);
        };
        mapped.push(reference);
    }
    check_stop(stop)?;
    Ok(Some(mapped))
}

fn remap_interval_indices_interruptible(
    references: &[usize],
    mapping: &[Option<usize>],
    stop: &AtomicBool,
) -> Result<Option<Vec<usize>>, ()> {
    let mut mapped = Vec::with_capacity(references.len());
    for &reference in references {
        check_stop(stop)?;
        let Some(reference) = mapping.get(reference).copied().flatten() else {
            return Ok(None);
        };
        mapped.push(reference);
    }
    check_stop(stop)?;
    Ok(Some(mapped))
}

fn remap_int_expr_interruptible(expression: &IntExpr, mapping: &[Option<usize>], stop: &AtomicBool) -> Result<Option<IntExpr>, ()> {
    check_stop(stop)?;
    let mapped = match expression {
        IntExpr::Constant(value) => IntExpr::Constant(*value),
        IntExpr::Variable(variable) => {
            let Some(variable) = remap_int_ref(*variable, mapping) else {
                return Ok(None);
            };
            IntExpr::Variable(variable)
        }
        IntExpr::Neg(value) => IntExpr::Neg(Box::new(match remap_int_expr_interruptible(value, mapping, stop)? {
            Some(value) => value,
            None => return Ok(None),
        })),
        IntExpr::Abs(value) => IntExpr::Abs(Box::new(match remap_int_expr_interruptible(value, mapping, stop)? {
            Some(value) => value,
            None => return Ok(None),
        })),
        IntExpr::Not(value) => IntExpr::Not(Box::new(match remap_int_expr_interruptible(value, mapping, stop)? {
            Some(value) => value,
            None => return Ok(None),
        })),
        IntExpr::Add(values) => IntExpr::Add(match remap_int_exprs_interruptible(values, mapping, stop)? {
            Some(values) => values,
            None => return Ok(None),
        }),
        IntExpr::Mul(values) => IntExpr::Mul(match remap_int_exprs_interruptible(values, mapping, stop)? {
            Some(values) => values,
            None => return Ok(None),
        }),
        IntExpr::Min(values) => IntExpr::Min(match remap_int_exprs_interruptible(values, mapping, stop)? {
            Some(values) => values,
            None => return Ok(None),
        }),
        IntExpr::Max(values) => IntExpr::Max(match remap_int_exprs_interruptible(values, mapping, stop)? {
            Some(values) => values,
            None => return Ok(None),
        }),
        IntExpr::And(values) => IntExpr::And(match remap_int_exprs_interruptible(values, mapping, stop)? {
            Some(values) => values,
            None => return Ok(None),
        }),
        IntExpr::Or(values) => IntExpr::Or(match remap_int_exprs_interruptible(values, mapping, stop)? {
            Some(values) => values,
            None => return Ok(None),
        }),
        IntExpr::Sub(left, right) => IntExpr::Sub(
            Box::new(match remap_int_expr_interruptible(left, mapping, stop)? {
                Some(value) => value,
                None => return Ok(None),
            }),
            Box::new(match remap_int_expr_interruptible(right, mapping, stop)? {
                Some(value) => value,
                None => return Ok(None),
            }),
        ),
        IntExpr::Div(left, right) => IntExpr::Div(
            Box::new(match remap_int_expr_interruptible(left, mapping, stop)? {
                Some(value) => value,
                None => return Ok(None),
            }),
            Box::new(match remap_int_expr_interruptible(right, mapping, stop)? {
                Some(value) => value,
                None => return Ok(None),
            }),
        ),
        IntExpr::Mod(left, right) => IntExpr::Mod(
            Box::new(match remap_int_expr_interruptible(left, mapping, stop)? {
                Some(value) => value,
                None => return Ok(None),
            }),
            Box::new(match remap_int_expr_interruptible(right, mapping, stop)? {
                Some(value) => value,
                None => return Ok(None),
            }),
        ),
        IntExpr::Eq(left, right) => IntExpr::Eq(
            Box::new(match remap_int_expr_interruptible(left, mapping, stop)? {
                Some(value) => value,
                None => return Ok(None),
            }),
            Box::new(match remap_int_expr_interruptible(right, mapping, stop)? {
                Some(value) => value,
                None => return Ok(None),
            }),
        ),
        IntExpr::Ne(left, right) => IntExpr::Ne(
            Box::new(match remap_int_expr_interruptible(left, mapping, stop)? {
                Some(value) => value,
                None => return Ok(None),
            }),
            Box::new(match remap_int_expr_interruptible(right, mapping, stop)? {
                Some(value) => value,
                None => return Ok(None),
            }),
        ),
        IntExpr::Lt(left, right) => IntExpr::Lt(
            Box::new(match remap_int_expr_interruptible(left, mapping, stop)? {
                Some(value) => value,
                None => return Ok(None),
            }),
            Box::new(match remap_int_expr_interruptible(right, mapping, stop)? {
                Some(value) => value,
                None => return Ok(None),
            }),
        ),
        IntExpr::Le(left, right) => IntExpr::Le(
            Box::new(match remap_int_expr_interruptible(left, mapping, stop)? {
                Some(value) => value,
                None => return Ok(None),
            }),
            Box::new(match remap_int_expr_interruptible(right, mapping, stop)? {
                Some(value) => value,
                None => return Ok(None),
            }),
        ),
        IntExpr::Gt(left, right) => IntExpr::Gt(
            Box::new(match remap_int_expr_interruptible(left, mapping, stop)? {
                Some(value) => value,
                None => return Ok(None),
            }),
            Box::new(match remap_int_expr_interruptible(right, mapping, stop)? {
                Some(value) => value,
                None => return Ok(None),
            }),
        ),
        IntExpr::Ge(left, right) => IntExpr::Ge(
            Box::new(match remap_int_expr_interruptible(left, mapping, stop)? {
                Some(value) => value,
                None => return Ok(None),
            }),
            Box::new(match remap_int_expr_interruptible(right, mapping, stop)? {
                Some(value) => value,
                None => return Ok(None),
            }),
        ),
        IntExpr::Imp(left, right) => IntExpr::Imp(
            Box::new(match remap_int_expr_interruptible(left, mapping, stop)? {
                Some(value) => value,
                None => return Ok(None),
            }),
            Box::new(match remap_int_expr_interruptible(right, mapping, stop)? {
                Some(value) => value,
                None => return Ok(None),
            }),
        ),
        IntExpr::Iff(left, right) => IntExpr::Iff(
            Box::new(match remap_int_expr_interruptible(left, mapping, stop)? {
                Some(value) => value,
                None => return Ok(None),
            }),
            Box::new(match remap_int_expr_interruptible(right, mapping, stop)? {
                Some(value) => value,
                None => return Ok(None),
            }),
        ),
        IntExpr::IfThenElse(condition, then_value, else_value) => IntExpr::IfThenElse(
            Box::new(match remap_int_expr_interruptible(condition, mapping, stop)? {
                Some(value) => value,
                None => return Ok(None),
            }),
            Box::new(match remap_int_expr_interruptible(then_value, mapping, stop)? {
                Some(value) => value,
                None => return Ok(None),
            }),
            Box::new(match remap_int_expr_interruptible(else_value, mapping, stop)? {
                Some(value) => value,
                None => return Ok(None),
            }),
        ),
    };
    check_stop(stop)?;
    Ok(Some(mapped))
}

fn remap_int_exprs_interruptible(
    expressions: &[IntExpr],
    mapping: &[Option<usize>],
    stop: &AtomicBool,
) -> Result<Option<Vec<IntExpr>>, ()> {
    let mut mapped = Vec::with_capacity(expressions.len());
    for expression in expressions {
        check_stop(stop)?;
        let Some(expression) = remap_int_expr_interruptible(expression, mapping, stop)? else {
            return Ok(None);
        };
        mapped.push(expression);
    }
    check_stop(stop)?;
    Ok(Some(mapped))
}

fn remap_int_refs_interruptible(
    references: &[IntVarRef],
    mapping: &[Option<usize>],
    stop: &AtomicBool,
) -> Result<Option<Vec<IntVarRef>>, ()> {
    let mut mapped = Vec::with_capacity(references.len());
    for &reference in references {
        check_stop(stop)?;
        let Some(reference) = remap_int_ref(reference, mapping) else {
            return Ok(None);
        };
        mapped.push(reference);
    }
    check_stop(stop)?;
    Ok(Some(mapped))
}

fn remap_int_global_interruptible(
    global: &IntGlobalConstraint,
    mapping: &[Option<usize>],
    stop: &AtomicBool,
) -> Result<Option<IntGlobalConstraint>, ()> {
    use IntGlobalConstraint as Global;

    check_stop(stop)?;
    let mapped = match global {
        Global::AllDifferent { variables, except } => Global::AllDifferent {
            variables: match remap_int_refs_interruptible(variables, mapping, stop)? {
                Some(variables) => variables,
                None => return Ok(None),
            },
            except: clone_slice_interruptible(except, stop)?,
        },
        Global::AllEqual(variables) => Global::AllEqual(match remap_int_refs_interruptible(variables, mapping, stop)? {
            Some(variables) => variables,
            None => return Ok(None),
        }),
        Global::Ordered { variables, relation } => Global::Ordered {
            variables: match remap_int_refs_interruptible(variables, mapping, stop)? {
                Some(variables) => variables,
                None => return Ok(None),
            },
            relation: *relation,
        },
        Global::Instantiation { variables, values } => Global::Instantiation {
            variables: match remap_int_refs_interruptible(variables, mapping, stop)? {
                Some(variables) => variables,
                None => return Ok(None),
            },
            values: clone_slice_interruptible(values, stop)?,
        },
        Global::Minimum { target, variables } => {
            let Some(target) = remap_int_ref(*target, mapping) else {
                return Ok(None);
            };
            Global::Minimum {
                target,
                variables: match remap_int_refs_interruptible(variables, mapping, stop)? {
                    Some(variables) => variables,
                    None => return Ok(None),
                },
            }
        }
        Global::Maximum { target, variables } => {
            let Some(target) = remap_int_ref(*target, mapping) else {
                return Ok(None);
            };
            Global::Maximum {
                target,
                variables: match remap_int_refs_interruptible(variables, mapping, stop)? {
                    Some(variables) => variables,
                    None => return Ok(None),
                },
            }
        }
        Global::Element { array, index, value } => {
            let (Some(index), Some(value)) = (remap_int_ref(*index, mapping), remap_int_ref(*value, mapping)) else {
                return Ok(None);
            };
            Global::Element {
                array: match remap_int_refs_interruptible(array, mapping, stop)? {
                    Some(array) => array,
                    None => return Ok(None),
                },
                index,
                value,
            }
        }
        Global::ElementConst { array, index, value } => {
            let (Some(index), Some(value)) = (remap_int_ref(*index, mapping), remap_int_ref(*value, mapping)) else {
                return Ok(None);
            };
            Global::ElementConst { array: clone_slice_interruptible(array, stop)?, index, value }
        }
        Global::Count { variables, value, relation, count } => Global::Count {
            variables: match remap_int_refs_interruptible(variables, mapping, stop)? {
                Some(variables) => variables,
                None => return Ok(None),
            },
            value: *value,
            relation: *relation,
            count: *count,
        },
        Global::Cardinality { variables, values, lower, upper, closed } => Global::Cardinality {
            variables: match remap_int_refs_interruptible(variables, mapping, stop)? {
                Some(variables) => variables,
                None => return Ok(None),
            },
            values: clone_slice_interruptible(values, stop)?,
            lower: clone_slice_interruptible(lower, stop)?,
            upper: clone_slice_interruptible(upper, stop)?,
            closed: *closed,
        },
        Global::NValues { variables, relation, count } => Global::NValues {
            variables: match remap_int_refs_interruptible(variables, mapping, stop)? {
                Some(variables) => variables,
                None => return Ok(None),
            },
            relation: *relation,
            count: *count,
        },
        Global::Table { variables, tuples, positive } => Global::Table {
            variables: match remap_int_refs_interruptible(variables, mapping, stop)? {
                Some(variables) => variables,
                None => return Ok(None),
            },
            tuples: clone_nested_slice_interruptible(tuples, stop)?,
            positive: *positive,
        },
        Global::Regular { variables, automaton } => Global::Regular {
            variables: match remap_int_refs_interruptible(variables, mapping, stop)? {
                Some(variables) => variables,
                None => return Ok(None),
            },
            automaton: super::Automaton {
                states: automaton.states,
                start: automaton.start,
                accepting: clone_slice_interruptible(&automaton.accepting, stop)?,
                transitions: clone_slice_interruptible(&automaton.transitions, stop)?,
            },
        },
        Global::Mdd { variables, mdd } => Global::Mdd {
            variables: match remap_int_refs_interruptible(variables, mapping, stop)? {
                Some(variables) => variables,
                None => return Ok(None),
            },
            mdd: super::Mdd {
                layers: clone_nested_slice_interruptible(&mdd.layers, stop)?,
                nodes_per_layer: clone_slice_interruptible(&mdd.nodes_per_layer, stop)?,
            },
        },
        Global::Lex { left, right, strict } => Global::Lex {
            left: match remap_int_refs_interruptible(left, mapping, stop)? {
                Some(left) => left,
                None => return Ok(None),
            },
            right: match remap_int_refs_interruptible(right, mapping, stop)? {
                Some(right) => right,
                None => return Ok(None),
            },
            strict: *strict,
        },
        Global::LexChain { rows, strict } => {
            let mut mapped_rows = Vec::with_capacity(rows.len());
            for row in rows {
                check_stop(stop)?;
                let Some(row) = remap_int_refs_interruptible(row, mapping, stop)? else {
                    return Ok(None);
                };
                mapped_rows.push(row);
            }
            Global::LexChain { rows: mapped_rows, strict: *strict }
        }
        Global::Channel { left, right } => Global::Channel {
            left: match remap_int_refs_interruptible(left, mapping, stop)? {
                Some(left) => left,
                None => return Ok(None),
            },
            right: match remap_int_refs_interruptible(right, mapping, stop)? {
                Some(right) => right,
                None => return Ok(None),
            },
        },
        Global::Circuit { successors, cutset } => Global::Circuit {
            successors: match remap_int_refs_interruptible(successors, mapping, stop)? {
                Some(successors) => successors,
                None => return Ok(None),
            },
            cutset: *cutset,
        },
        Global::NoOverlap { starts, durations } => Global::NoOverlap {
            starts: match remap_int_refs_interruptible(starts, mapping, stop)? {
                Some(starts) => starts,
                None => return Ok(None),
            },
            durations: clone_slice_interruptible(durations, stop)?,
        },
        Global::OptionalNoOverlap { starts, durations, presences } => {
            let mut mapped_presences = Vec::with_capacity(presences.len());
            for presence in presences {
                check_stop(stop)?;
                let presence = match presence {
                    Some(presence) => {
                        let Some(presence) = remap_int_ref(*presence, mapping) else {
                            return Ok(None);
                        };
                        Some(presence)
                    }
                    None => None,
                };
                mapped_presences.push(presence);
            }
            Global::OptionalNoOverlap {
                starts: match remap_int_refs_interruptible(starts, mapping, stop)? {
                    Some(starts) => starts,
                    None => return Ok(None),
                },
                durations: clone_slice_interruptible(durations, stop)?,
                presences: mapped_presences,
            }
        }
        Global::AlternativeChannel { shared_start, starts, durations, presences } => {
            let Some(shared_start) = remap_int_ref(*shared_start, mapping) else {
                return Ok(None);
            };
            Global::AlternativeChannel {
                shared_start,
                starts: match remap_int_refs_interruptible(starts, mapping, stop)? {
                    Some(starts) => starts,
                    None => return Ok(None),
                },
                durations: clone_slice_interruptible(durations, stop)?,
                presences: match remap_int_refs_interruptible(presences, mapping, stop)? {
                    Some(presences) => presences,
                    None => return Ok(None),
                },
            }
        }
        Global::Cumulative { starts, durations, demands, capacity } => Global::Cumulative {
            starts: match remap_int_refs_interruptible(starts, mapping, stop)? {
                Some(starts) => starts,
                None => return Ok(None),
            },
            durations: clone_slice_interruptible(durations, stop)?,
            demands: clone_slice_interruptible(demands, stop)?,
            capacity: *capacity,
        },
        Global::CumulativeVar { starts, durations, demands, capacity } => {
            let Some(capacity) = remap_int_ref(*capacity, mapping) else {
                return Ok(None);
            };
            Global::CumulativeVar {
                starts: match remap_int_refs_interruptible(starts, mapping, stop)? {
                    Some(starts) => starts,
                    None => return Ok(None),
                },
                durations: match remap_int_refs_interruptible(durations, mapping, stop)? {
                    Some(durations) => durations,
                    None => return Ok(None),
                },
                demands: match remap_int_refs_interruptible(demands, mapping, stop)? {
                    Some(demands) => demands,
                    None => return Ok(None),
                },
                capacity,
            }
        }
        Global::BinPacking { items, sizes, capacities } => Global::BinPacking {
            items: match remap_int_refs_interruptible(items, mapping, stop)? {
                Some(items) => items,
                None => return Ok(None),
            },
            sizes: clone_slice_interruptible(sizes, stop)?,
            capacities: clone_slice_interruptible(capacities, stop)?,
        },
        Global::BinLoads { items, sizes, loads } => Global::BinLoads {
            items: match remap_int_refs_interruptible(items, mapping, stop)? {
                Some(items) => items,
                None => return Ok(None),
            },
            sizes: clone_slice_interruptible(sizes, stop)?,
            loads: match remap_int_refs_interruptible(loads, mapping, stop)? {
                Some(loads) => loads,
                None => return Ok(None),
            },
        },
        Global::Knapsack { variables, weights, profits, weight_relation, weight_limit, profit_relation, profit_limit } => {
            Global::Knapsack {
                variables: match remap_int_refs_interruptible(variables, mapping, stop)? {
                    Some(variables) => variables,
                    None => return Ok(None),
                },
                weights: clone_slice_interruptible(weights, stop)?,
                profits: clone_slice_interruptible(profits, stop)?,
                weight_relation: *weight_relation,
                weight_limit: *weight_limit,
                profit_relation: *profit_relation,
                profit_limit: *profit_limit,
            }
        }
        Global::ValuePrecedence { variables, values, covered } => Global::ValuePrecedence {
            variables: match remap_int_refs_interruptible(variables, mapping, stop)? {
                Some(variables) => variables,
                None => return Ok(None),
            },
            values: clone_slice_interruptible(values, stop)?,
            covered: *covered,
        },
    };
    check_stop(stop)?;
    Ok(Some(mapped))
}

fn clone_nested_slice_interruptible<T: Clone>(values: &[Vec<T>], stop: &AtomicBool) -> Result<Vec<Vec<T>>, ()> {
    let mut copied = Vec::with_capacity(values.len());
    for value in values {
        check_stop(stop)?;
        copied.push(clone_slice_interruptible(value, stop)?);
    }
    check_stop(stop)?;
    Ok(copied)
}

fn integer_scope_interruptible(constraint: &Constraint, integer_count: usize, stop: &AtomicBool) -> Result<Vec<usize>, ()> {
    let mut scope = BTreeSet::new();
    let mut pending = vec![constraint];
    while let Some(constraint) = pending.pop() {
        check_stop(stop)?;
        match constraint {
            Constraint::Intension(expression) => insert_expr_scope(expression, &mut scope, stop)?,
            Constraint::Selected { selector, constraint } => {
                scope.insert(selector.0);
                pending.push(constraint);
            }
            Constraint::Linear { terms, .. } => {
                for (_, variable) in terms {
                    check_stop(stop)?;
                    scope.insert(variable.0);
                }
            }
            Constraint::Clause(literals) => {
                for literal in literals {
                    check_stop(stop)?;
                    scope.insert(literal.variable.0);
                }
            }
            Constraint::IntegerGlobal(global) => insert_global_scope(global, &mut scope, stop)?,
            Constraint::SetSubset { subset, superset } => {
                scope.insert(integer_count + subset.0);
                scope.insert(integer_count + superset.0);
            }
            Constraint::SetDisjoint { left, right } => {
                scope.insert(integer_count + left.0);
                scope.insert(integer_count + right.0);
            }
            Constraint::SetCardinality { set, .. } => {
                scope.insert(integer_count + set.0);
            }
            Constraint::ListPartition { .. }
            | Constraint::ListPartitionWithCoverage { .. }
            | Constraint::SameList { .. }
            | Constraint::ItemPrecedence { .. }
            | Constraint::CollectionGlobal(_)
            | Constraint::ListLength { .. }
            | Constraint::ListItemSum { .. }
            | Constraint::ListReduction(_)
            | Constraint::IntervalPrecedence { .. }
            | Constraint::IntervalAlternative { .. }
            | Constraint::IntervalEndpointRelation { .. }
            | Constraint::IntervalResource(_) => {}
        }
    }
    check_stop(stop)?;
    let mut nodes = Vec::with_capacity(scope.len());
    for node in scope {
        check_stop(stop)?;
        nodes.push(node);
    }
    check_stop(stop)?;
    Ok(nodes)
}

fn insert_expr_scope(expression: &IntExpr, scope: &mut BTreeSet<usize>, stop: &AtomicBool) -> Result<(), ()> {
    let mut pending = vec![expression];
    while let Some(expression) = pending.pop() {
        check_stop(stop)?;
        match expression {
            IntExpr::Constant(_) => {}
            IntExpr::Variable(variable) => {
                scope.insert(variable.0);
            }
            IntExpr::Neg(value) | IntExpr::Abs(value) | IntExpr::Not(value) => pending.push(value),
            IntExpr::Add(values)
            | IntExpr::Mul(values)
            | IntExpr::Min(values)
            | IntExpr::Max(values)
            | IntExpr::And(values)
            | IntExpr::Or(values) => {
                for value in values {
                    check_stop(stop)?;
                    pending.push(value);
                }
            }
            IntExpr::Sub(left, right)
            | IntExpr::Div(left, right)
            | IntExpr::Mod(left, right)
            | IntExpr::Eq(left, right)
            | IntExpr::Ne(left, right)
            | IntExpr::Lt(left, right)
            | IntExpr::Le(left, right)
            | IntExpr::Gt(left, right)
            | IntExpr::Ge(left, right)
            | IntExpr::Imp(left, right)
            | IntExpr::Iff(left, right) => {
                pending.push(left);
                pending.push(right);
            }
            IntExpr::IfThenElse(condition, then_value, else_value) => {
                pending.push(condition);
                pending.push(then_value);
                pending.push(else_value);
            }
        }
    }
    check_stop(stop)
}

fn insert_global_scope(global: &IntGlobalConstraint, scope: &mut BTreeSet<usize>, stop: &AtomicBool) -> Result<(), ()> {
    match global {
        IntGlobalConstraint::AllDifferent { variables, .. }
        | IntGlobalConstraint::AllEqual(variables)
        | IntGlobalConstraint::Ordered { variables, .. }
        | IntGlobalConstraint::Instantiation { variables, .. }
        | IntGlobalConstraint::Count { variables, .. }
        | IntGlobalConstraint::Cardinality { variables, .. }
        | IntGlobalConstraint::NValues { variables, .. }
        | IntGlobalConstraint::Table { variables, .. }
        | IntGlobalConstraint::Regular { variables, .. }
        | IntGlobalConstraint::Mdd { variables, .. }
        | IntGlobalConstraint::Circuit { successors: variables, .. }
        | IntGlobalConstraint::ValuePrecedence { variables, .. }
        | IntGlobalConstraint::NoOverlap { starts: variables, .. }
        | IntGlobalConstraint::Cumulative { starts: variables, .. }
        | IntGlobalConstraint::BinPacking { items: variables, .. }
        | IntGlobalConstraint::Knapsack { variables, .. } => insert_ints(variables, scope, stop),
        IntGlobalConstraint::Minimum { target, variables } | IntGlobalConstraint::Maximum { target, variables } => {
            scope.insert(target.0);
            insert_ints(variables, scope, stop)
        }
        IntGlobalConstraint::Element { array, index, value } => {
            insert_ints(array, scope, stop)?;
            scope.insert(index.0);
            scope.insert(value.0);
            check_stop(stop)
        }
        IntGlobalConstraint::ElementConst { index, value, .. } => {
            scope.insert(index.0);
            scope.insert(value.0);
            check_stop(stop)
        }
        IntGlobalConstraint::Lex { left, right, .. } | IntGlobalConstraint::Channel { left, right } => {
            insert_ints(left, scope, stop)?;
            insert_ints(right, scope, stop)
        }
        IntGlobalConstraint::LexChain { rows, .. } => {
            for row in rows {
                insert_ints(row, scope, stop)?;
            }
            check_stop(stop)
        }
        IntGlobalConstraint::OptionalNoOverlap { starts, presences, .. } => {
            insert_ints(starts, scope, stop)?;
            for presence in presences.iter().flatten() {
                check_stop(stop)?;
                scope.insert(presence.0);
            }
            check_stop(stop)
        }
        IntGlobalConstraint::AlternativeChannel { shared_start, starts, presences, .. } => {
            scope.insert(shared_start.0);
            insert_ints(starts, scope, stop)?;
            insert_ints(presences, scope, stop)
        }
        IntGlobalConstraint::CumulativeVar { starts, durations, demands, capacity } => {
            insert_ints(starts, scope, stop)?;
            insert_ints(durations, scope, stop)?;
            insert_ints(demands, scope, stop)?;
            scope.insert(capacity.0);
            check_stop(stop)
        }
        IntGlobalConstraint::BinLoads { items, loads, .. } => {
            insert_ints(items, scope, stop)?;
            insert_ints(loads, scope, stop)
        }
    }
}

fn insert_ints(variables: &[super::IntVarRef], scope: &mut BTreeSet<usize>, stop: &AtomicBool) -> Result<(), ()> {
    for variable in variables {
        check_stop(stop)?;
        scope.insert(variable.0);
    }
    check_stop(stop)
}

struct UnionFind {
    parents: Vec<usize>,
}

impl UnionFind {
    fn new_interruptible(size: usize, stop: &AtomicBool) -> Result<Self, ()> {
        let mut parents = Vec::with_capacity(size);
        for node in 0..size {
            check_stop(stop)?;
            parents.push(node);
        }
        Ok(Self { parents })
    }

    fn root(&mut self, node: usize, stop: &AtomicBool) -> Result<usize, ()> {
        let mut root = node;
        loop {
            check_stop(stop)?;
            let parent = self.parents[root];
            if parent == root {
                break;
            }
            root = parent;
        }
        let mut current = node;
        while self.parents[current] != root {
            check_stop(stop)?;
            let parent = self.parents[current];
            self.parents[current] = root;
            current = parent;
        }
        Ok(root)
    }

    fn join(&mut self, left: usize, right: usize, stop: &AtomicBool) -> Result<(), ()> {
        let left = self.root(left, stop)?;
        let right = self.root(right, stop)?;
        if left != right {
            self.parents[right] = left;
        }
        check_stop(stop)
    }
}
