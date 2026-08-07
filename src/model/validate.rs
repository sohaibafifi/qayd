use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};

use super::{list, Constraint, IntGlobalConstraint, Model, ModelObject, Objective, PartitionCoverage, Relation};

impl Model {
    /// Validate semantic references and declaration invariants without lowering
    /// to any engine representation.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let stop = AtomicBool::new(false);
        match self.validate_interruptible(&stop) {
            Ok(true) => Ok(()),
            Ok(false) => unreachable!("a private false cancellation flag cannot interrupt validation"),
            Err(errors) => Err(errors),
        }
    }

    /// Validate while cooperatively sharing the solve cancellation flag.
    /// `Ok(false)` means validation was interrupted before completion.
    pub(crate) fn validate_interruptible(&self, stop: &AtomicBool) -> Result<bool, Vec<String>> {
        if interrupted(stop) {
            return Ok(false);
        }

        let mut errors = Vec::new();

        for (index, domain) in self.int_vars.iter().enumerate() {
            if interrupted(stop) {
                return Ok(false);
            }
            if let Err(error) = domain.validate() {
                errors.push(format!("integer variable {index}: {error}"));
            }
            if let super::IntDomain::Set(values) = domain {
                let Some(duplicate) = contains_duplicate(values, stop) else {
                    return Ok(false);
                };
                if duplicate {
                    errors.push(format!("integer variable {index}: explicit domain contains duplicate values"));
                }
            }
        }

        // Retain the sets built for duplicate detection so constraint reference
        // checks do not repeatedly scan large list universes.
        let mut list_universes = Vec::with_capacity(self.lists.len());
        for (index, list) in self.lists.iter().enumerate() {
            if interrupted(stop) {
                return Ok(false);
            }
            let Some((universe, duplicate)) = collect_unique(&list.universe, stop) else {
                return Ok(false);
            };
            if duplicate {
                errors.push(format!("list variable {index}: universe contains duplicate items"));
            }
            list_universes.push(universe);
        }

        for (index, set) in self.sets.iter().enumerate() {
            if interrupted(stop) {
                return Ok(false);
            }
            let mut missing = false;
            for value in &set.required {
                if interrupted(stop) {
                    return Ok(false);
                }
                if !set.possible.contains(value) {
                    missing = true;
                    break;
                }
            }
            if missing {
                errors.push(format!("set variable {index}: lower bound is not a subset of its upper bound"));
            }
        }

        for (index, interval) in self.intervals.iter().enumerate() {
            if interrupted(stop) {
                return Ok(false);
            }
            if interval.start_min > interval.start_max {
                errors.push(format!("interval {index}: start lower bound exceeds its upper bound"));
            }
            if interval.duration < 0 {
                errors.push(format!("interval {index}: duration must be non-negative"));
            }
            for mode in &interval.modes {
                if interrupted(stop) {
                    return Ok(false);
                }
                match self.interval_modes.get(mode.0) {
                    Some(declaration) if declaration.interval.0 == index => {}
                    Some(_) => errors.push(format!("interval {index}: mode {} belongs to another interval", mode.0)),
                    None => errors.push(format!("interval {index}: references unknown mode {}", mode.0)),
                }
            }
        }
        for (index, mode) in self.interval_modes.iter().enumerate() {
            if interrupted(stop) {
                return Ok(false);
            }
            if mode.interval.0 >= self.intervals.len() {
                errors.push(format!("interval mode {index}: owner is unknown"));
            }
            if mode.duration < 0 {
                errors.push(format!("interval mode {index}: duration must be non-negative"));
            }
            if let Some((start_min, start_max)) = mode.start_window {
                if start_min > start_max {
                    errors.push(format!("interval mode {index}: start lower bound exceeds its upper bound"));
                }
            }
        }

        for (index, constraint) in self.constraints.iter().enumerate() {
            if interrupted(stop) || !validate_constraint(self, &list_universes, constraint, index, &mut errors, stop) {
                return Ok(false);
            }
        }
        for (index, objective) in self.objectives.iter().enumerate() {
            if interrupted(stop) {
                return Ok(false);
            }
            let context = format!("objective {index}");
            match objective {
                Objective::IntExpr { expr, .. } => {
                    if !validate_expr(self, expr, &context, &mut errors, stop) {
                        return Ok(false);
                    }
                }
                Objective::ListTerms { terms, max_terms, .. } => {
                    for (term, reduction) in terms.iter().enumerate() {
                        if !validate_list_reduction(self, reduction, &format!("{context}, term {term}"), &mut errors, stop) {
                            return Ok(false);
                        }
                    }
                    for (max_term, component) in max_terms.iter().flatten().enumerate() {
                        for (group, reductions) in component.groups.iter().enumerate() {
                            for (term, reduction) in reductions.iter().enumerate() {
                                let reduction_context = format!("{context}, max term {max_term}, group {group}, term {term}");
                                if !validate_list_reduction(self, reduction, &reduction_context, &mut errors, stop) {
                                    return Ok(false);
                                }
                            }
                        }
                    }
                }
                Objective::Makespan { intervals, .. } => {
                    if intervals.is_empty() {
                        errors.push(format!("objective {index}: makespan interval set is empty"));
                    }
                    for interval in intervals {
                        if interrupted(stop) {
                            return Ok(false);
                        }
                        check_interval(self, interval.0, &context, &mut errors);
                    }
                }
            }
        }

        if interrupted(stop) {
            Ok(false)
        } else if errors.is_empty() {
            Ok(true)
        } else {
            Err(errors)
        }
    }

    pub(crate) fn contains_object(&self, object: ModelObject) -> bool {
        match object {
            ModelObject::IntVar(reference) => reference.0 < self.int_vars.len(),
            ModelObject::SetVar(reference) => reference.0 < self.sets.len(),
            ModelObject::ListVar(reference) => reference.0 < self.lists.len(),
            ModelObject::IntervalVar(reference) => reference.0 < self.intervals.len(),
            ModelObject::IntervalMode(reference) => reference.0 < self.interval_modes.len(),
            ModelObject::Constraint(reference) => reference.0 < self.constraints.len(),
            ModelObject::Objective(reference) => reference.0 < self.objectives.len(),
        }
    }
}

fn interrupted(stop: &AtomicBool) -> bool {
    stop.load(Ordering::Acquire)
}

fn collect_unique(values: &[i32], stop: &AtomicBool) -> Option<(BTreeSet<i32>, bool)> {
    let mut unique = BTreeSet::new();
    let mut duplicate = false;
    for &value in values {
        if interrupted(stop) {
            return None;
        }
        duplicate |= !unique.insert(value);
    }
    (!interrupted(stop)).then_some((unique, duplicate))
}

fn contains_duplicate(values: &[i32], stop: &AtomicBool) -> Option<bool> {
    collect_unique(values, stop).map(|(_, duplicate)| duplicate)
}

fn validate_constraint(
    model: &Model,
    list_universes: &[BTreeSet<i32>],
    constraint: &Constraint,
    index: usize,
    errors: &mut Vec<String>,
    stop: &AtomicBool,
) -> bool {
    if interrupted(stop) {
        return false;
    }
    let context = format!("constraint {index}");
    match constraint {
        Constraint::ListPartition { lists, items }
        | Constraint::ListPartitionWithCoverage { lists, items, coverage: PartitionCoverage::Exact }
        | Constraint::ListPartitionWithCoverage { lists, items, coverage: PartitionCoverage::Partial } => {
            if lists.is_empty() {
                errors.push(format!("{context}: partition has no lists"));
            }
            for list in lists {
                if interrupted(stop) {
                    return false;
                }
                check_list(model, list.0, &context, errors);
                if let Some(universe) = list_universes.get(list.0) {
                    let mut outside = false;
                    for item in items {
                        if interrupted(stop) {
                            return false;
                        }
                        if !universe.contains(item) {
                            outside = true;
                            break;
                        }
                    }
                    if outside {
                        errors.push(format!("{context}: partition item is outside a list universe"));
                    }
                }
            }
            let Some(duplicate) = contains_duplicate(items, stop) else {
                return false;
            };
            if duplicate {
                errors.push(format!("{context}: partition contains duplicate items"));
            }
        }
        Constraint::SameList { lists, a, b } => {
            if !validate_list_items(model, list_universes, lists, &[*a, *b], &context, errors, stop) {
                return false;
            }
        }
        Constraint::ItemPrecedence { lists, before, after } => {
            if !validate_list_items(model, list_universes, lists, &[*before, *after], &context, errors, stop) {
                return false;
            }
        }
        Constraint::CollectionGlobal(global) => {
            if !validate_collection_global(model, list_universes, global, &context, errors, stop) {
                return false;
            }
        }
        Constraint::ListReduction(constraint) => {
            if !validate_list_reduction(model, &constraint.reduction, &context, errors, stop) {
                return false;
            }
        }
        Constraint::ListLength { list, min, max } => {
            check_list(model, list.0, &context, errors);
            if min > max {
                errors.push(format!("{context}: list length lower bound exceeds its upper bound"));
            }
        }
        Constraint::ListItemSum { list, weights, min, max } => {
            check_list(model, list.0, &context, errors);
            if min > max {
                errors.push(format!("{context}: item-sum lower bound exceeds its upper bound"));
            }
            let mut items = BTreeSet::new();
            let mut duplicate = false;
            for (item, _) in weights {
                if interrupted(stop) {
                    return false;
                }
                duplicate |= !items.insert(*item);
            }
            if duplicate {
                errors.push(format!("{context}: item-sum contains duplicate item weights"));
            }
        }
        Constraint::IntervalPrecedence { before, after } => {
            check_interval(model, before.0, &context, errors);
            check_interval(model, after.0, &context, errors);
        }
        Constraint::IntervalAlternative { master, members } => {
            check_interval(model, master.0, &context, errors);
            if members.is_empty() {
                errors.push(format!("{context}: interval alternative has no members"));
            }
            for member in members {
                if interrupted(stop) {
                    return false;
                }
                check_interval(model, member.0, &context, errors);
            }
        }
        Constraint::IntervalEndpointRelation { left, relation, right, .. } => {
            check_interval(model, left.0, &context, errors);
            check_interval(model, right.0, &context, errors);
            validate_order_relation(*relation, &context, errors);
        }
        Constraint::IntervalResource(resource) => match resource {
            super::list::Resource::NoOverlap(intervals) => {
                for interval in intervals {
                    if interrupted(stop) {
                        return false;
                    }
                    check_interval(model, *interval, &context, errors);
                }
            }
            super::list::Resource::MachineNoOverlap => {}
            super::list::Resource::Cumulative { demands, capacity } => {
                for (interval, _) in demands {
                    if interrupted(stop) {
                        return false;
                    }
                    check_interval(model, *interval, &context, errors);
                }
                if *capacity < 0 {
                    errors.push(format!("{context}: cumulative capacity must be non-negative"));
                }
            }
        },
        Constraint::Intension(expr) => {
            if !validate_expr(model, expr, &context, errors, stop) {
                return false;
            }
        }
        Constraint::Selected { selector, constraint } => {
            check_int(model, selector.0, &context, errors);
            if model.int_vars.get(selector.0).is_some_and(|domain| !matches!(domain, super::IntDomain::Bool)) {
                errors.push(format!("{context}: selector is not Boolean"));
            }
            if matches!(constraint.as_ref(), Constraint::Selected { .. }) {
                errors.push(format!("{context}: nested selectors are not supported"));
            }
            if !matches!(
                constraint.as_ref(),
                Constraint::Intension(_) | Constraint::Linear { .. } | Constraint::Clause(_) | Constraint::IntegerGlobal(_)
            ) {
                errors.push(format!("{context}: selectors currently support integer constraints only"));
            }
            if !validate_constraint(model, list_universes, constraint, index, errors, stop) {
                return false;
            }
        }
        Constraint::Linear { terms, .. } => {
            for (_, variable) in terms {
                if interrupted(stop) {
                    return false;
                }
                check_int(model, variable.0, &context, errors);
            }
        }
        Constraint::Clause(literals) => {
            for literal in literals {
                if interrupted(stop) {
                    return false;
                }
                check_int(model, literal.variable.0, &context, errors);
                if model.int_vars.get(literal.variable.0).is_some_and(|domain| !matches!(domain, super::IntDomain::Bool)) {
                    errors.push(format!("{context}: clause references a non-Boolean variable"));
                }
            }
        }
        Constraint::IntegerGlobal(global) => {
            if !validate_global(model, global, &context, errors, stop) {
                return false;
            }
        }
        Constraint::SetSubset { subset, superset } => {
            check_set(model, subset.0, &context, errors);
            check_set(model, superset.0, &context, errors);
        }
        Constraint::SetDisjoint { left, right } => {
            check_set(model, left.0, &context, errors);
            check_set(model, right.0, &context, errors);
        }
        Constraint::SetCardinality { set, min, max } => {
            check_set(model, set.0, &context, errors);
            if min > max {
                errors.push(format!("{context}: set cardinality lower bound exceeds its upper bound"));
            }
        }
    }
    !interrupted(stop)
}

fn validate_collection_global(
    model: &Model,
    list_universes: &[BTreeSet<i32>],
    global: &list::GlobalConstraint,
    context: &str,
    errors: &mut Vec<String>,
    stop: &AtomicBool,
) -> bool {
    if model.lists.is_empty() {
        errors.push(format!("{context}: collection-global constraint has no list variables"));
        return !interrupted(stop);
    }

    let mut validate_item = |item: i32| {
        for (list, universe) in list_universes.iter().enumerate() {
            if interrupted(stop) {
                return false;
            }
            if !universe.contains(&item) {
                errors.push(format!("{context}: collection-global item {item} is outside list variable {list} universe"));
                break;
            }
        }
        true
    };

    match global {
        list::GlobalConstraint::ListLe { before, after } => validate_item(*before) && validate_item(*after),
        list::GlobalConstraint::SameList { a, b }
        | list::GlobalConstraint::DifferentList { a, b }
        | list::GlobalConstraint::ListDistance { a, b, .. } => validate_item(*a) && validate_item(*b),
        list::GlobalConstraint::AllSameList { items } | list::GlobalConstraint::AllDifferentLists { items } => {
            for &item in items.iter() {
                if !validate_item(item) {
                    return false;
                }
            }
            !interrupted(stop)
        }
    }
}

fn validate_list_reduction(model: &Model, reduction: &list::Reduction, context: &str, errors: &mut Vec<String>, stop: &AtomicBool) -> bool {
    if interrupted(stop) {
        return false;
    }

    check_list(model, reduction.iterable.list(), context, errors);
    let arena = &reduction.arena.exprs;
    let mut references_valid = validate_list_expr_id(arena.len(), reduction.body, "reduction body", context, errors);

    match reduction.iterable {
        list::Iterable::Scan { step, .. } => {
            references_valid &= validate_list_expr_id(arena.len(), step, "scan step", context, errors);
        }
        list::Iterable::Windows { inner, .. } => {
            references_valid &= validate_list_expr_id(arena.len(), inner, "window inner expression", context, errors);
        }
        list::Iterable::Items(_) | list::Iterable::SetItems(_) | list::Iterable::Edges { .. } | list::Iterable::Pairs(_) => {}
    }

    for (expression_index, expression) in arena.iter().enumerate() {
        if interrupted(stop) {
            return false;
        }
        let mut child_index = 0;
        while let Some(child) = list_expr_child(expression, child_index) {
            if interrupted(stop) {
                return false;
            }
            references_valid &=
                validate_list_expr_id(arena.len(), child, &format!("expression {expression_index} child {child_index}"), context, errors);
            child_index += 1;
        }
    }

    if references_valid && !validate_list_expr_arena_acyclic(arena, context, errors, stop) {
        return false;
    }
    if references_valid && !validate_reduction_bindings(reduction, context, errors, stop) {
        return false;
    }
    !interrupted(stop)
}

fn validate_list_expr_id(arena_len: usize, expression: list::ExprId, role: &str, context: &str, errors: &mut Vec<String>) -> bool {
    if expression.0 as usize >= arena_len {
        errors.push(format!("{context}: {role} references expression {} outside an arena of length {arena_len}", expression.0));
        false
    } else {
        true
    }
}

fn list_expr_child(expression: &list::Expr, index: usize) -> Option<list::ExprId> {
    use list::Expr;

    match expression {
        Expr::Const(_) | Expr::Arg(_) => None,
        Expr::Array(_, child) | Expr::Pow(child, _) | Expr::Abs(child) | Expr::PiecewiseLinear { input: child, .. } => {
            (index == 0).then_some(*child)
        }
        Expr::Matrix(_, left, right)
        | Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Mod(left, right)
        | Expr::MulScaled(left, right, _)
        | Expr::DivScaled(left, right, _)
        | Expr::Min(left, right)
        | Expr::Max(left, right)
        | Expr::Div(left, right)
        | Expr::Lt(left, right)
        | Expr::Le(left, right)
        | Expr::Eq(left, right)
        | Expr::Ne(left, right) => match index {
            0 => Some(*left),
            1 => Some(*right),
            _ => None,
        },
        Expr::IfThenElse(condition, then_value, otherwise) => match index {
            0 => Some(*condition),
            1 => Some(*then_value),
            2 => Some(*otherwise),
            _ => None,
        },
        Expr::External { args, .. } => args.get(index).copied(),
    }
}

fn validate_list_expr_arena_acyclic(arena: &[list::Expr], context: &str, errors: &mut Vec<String>, stop: &AtomicBool) -> bool {
    let mut state = vec![0u8; arena.len()];
    for root in 0..arena.len() {
        if interrupted(stop) {
            return false;
        }
        if state[root] != 0 {
            continue;
        }
        state[root] = 1;
        let mut pending = vec![(root, 0usize)];
        while let Some(&(node, next_child)) = pending.last() {
            if interrupted(stop) {
                return false;
            }
            let Some(child) = list_expr_child(&arena[node], next_child) else {
                state[node] = 2;
                pending.pop();
                continue;
            };
            pending.last_mut().expect("the expression traversal stack is non-empty").1 += 1;
            let child = child.0 as usize;
            match state[child] {
                0 => {
                    state[child] = 1;
                    pending.push((child, 0));
                }
                1 => {
                    errors.push(format!("{context}: expression arena contains a cycle through expression {child}"));
                    return true;
                }
                _ => {}
            }
        }
    }
    true
}

fn validate_reduction_bindings(reduction: &list::Reduction, context: &str, errors: &mut Vec<String>, stop: &AtomicBool) -> bool {
    let body_arity = match reduction.iterable {
        list::Iterable::Items(_) | list::Iterable::SetItems(_) => 1,
        list::Iterable::Edges { .. } | list::Iterable::Windows { .. } => 2,
        list::Iterable::Pairs(_) => 4,
        list::Iterable::Scan { .. } => 3,
    };
    if !validate_list_expr_root_bindings(&reduction.arena.exprs, reduction.body, body_arity, "reduction body", context, errors, stop) {
        return false;
    }
    match reduction.iterable {
        list::Iterable::Scan { step, .. } => {
            validate_list_expr_root_bindings(&reduction.arena.exprs, step, 3, "scan step", context, errors, stop)
        }
        list::Iterable::Windows { inner, .. } => {
            validate_list_expr_root_bindings(&reduction.arena.exprs, inner, 1, "window inner expression", context, errors, stop)
        }
        list::Iterable::Items(_) | list::Iterable::SetItems(_) | list::Iterable::Edges { .. } | list::Iterable::Pairs(_) => {
            !interrupted(stop)
        }
    }
}

fn validate_list_expr_root_bindings(
    arena: &[list::Expr],
    root: list::ExprId,
    arity: u8,
    role: &str,
    context: &str,
    errors: &mut Vec<String>,
    stop: &AtomicBool,
) -> bool {
    let mut seen = vec![false; arena.len()];
    let mut pending = vec![root.0 as usize];
    while let Some(expression_index) = pending.pop() {
        if interrupted(stop) {
            return false;
        }
        if std::mem::replace(&mut seen[expression_index], true) {
            continue;
        }
        let expression = &arena[expression_index];
        if let list::Expr::Arg(argument) = expression {
            if *argument >= arity {
                errors.push(format!("{context}: {role} uses lambda argument {argument}, but its iterable binds only {arity}"));
            }
        }
        let mut child_index = 0;
        while let Some(child) = list_expr_child(expression, child_index) {
            pending.push(child.0 as usize);
            child_index += 1;
        }
    }
    !interrupted(stop)
}

fn validate_expr(model: &Model, expression: &super::IntExpr, context: &str, errors: &mut Vec<String>, stop: &AtomicBool) -> bool {
    let mut pending = vec![expression];
    while let Some(expr) = pending.pop() {
        if interrupted(stop) {
            return false;
        }
        match expr {
            super::IntExpr::Constant(_) => {}
            super::IntExpr::Variable(variable) => check_int(model, variable.0, context, errors),
            super::IntExpr::Neg(value) | super::IntExpr::Abs(value) | super::IntExpr::Not(value) => pending.push(value),
            super::IntExpr::Add(values)
            | super::IntExpr::Mul(values)
            | super::IntExpr::Min(values)
            | super::IntExpr::Max(values)
            | super::IntExpr::And(values)
            | super::IntExpr::Or(values) => {
                for value in values {
                    if interrupted(stop) {
                        return false;
                    }
                    pending.push(value);
                }
            }
            super::IntExpr::Sub(left, right)
            | super::IntExpr::Div(left, right)
            | super::IntExpr::Mod(left, right)
            | super::IntExpr::Eq(left, right)
            | super::IntExpr::Ne(left, right)
            | super::IntExpr::Lt(left, right)
            | super::IntExpr::Le(left, right)
            | super::IntExpr::Gt(left, right)
            | super::IntExpr::Ge(left, right)
            | super::IntExpr::Imp(left, right)
            | super::IntExpr::Iff(left, right) => {
                pending.push(left);
                pending.push(right);
            }
            super::IntExpr::IfThenElse(condition, then_value, else_value) => {
                pending.push(condition);
                pending.push(then_value);
                pending.push(else_value);
            }
        }
    }
    !interrupted(stop)
}

fn validate_global(model: &Model, global: &IntGlobalConstraint, context: &str, errors: &mut Vec<String>, stop: &AtomicBool) -> bool {
    if !validate_global_variables(model, global, context, errors, stop) {
        return false;
    }
    match global {
        IntGlobalConstraint::Ordered { relation, .. } => validate_order_relation(*relation, context, errors),
        IntGlobalConstraint::Instantiation { variables, values } if variables.len() != values.len() => {
            errors.push(format!("{context}: instantiation variables and values differ in length"));
        }
        IntGlobalConstraint::Minimum { variables, .. } | IntGlobalConstraint::Maximum { variables, .. } if variables.is_empty() => {
            errors.push(format!("{context}: extremum has no input variables"));
        }
        IntGlobalConstraint::Element { array, .. } if array.is_empty() => errors.push(format!("{context}: element array is empty")),
        IntGlobalConstraint::ElementConst { array, .. } if array.is_empty() => {
            errors.push(format!("{context}: constant element array is empty"));
        }
        IntGlobalConstraint::Cardinality { values, lower, upper, .. } if values.len() != lower.len() || values.len() != upper.len() => {
            errors.push(format!("{context}: cardinality values and bounds differ in length"));
        }
        IntGlobalConstraint::Table { variables, tuples, .. } => {
            let mut wrong_arity = false;
            for tuple in tuples {
                if interrupted(stop) {
                    return false;
                }
                if tuple.len() != variables.len() {
                    wrong_arity = true;
                    break;
                }
            }
            if wrong_arity {
                errors.push(format!("{context}: table contains a tuple with the wrong arity"));
            }
        }
        IntGlobalConstraint::Regular { variables, automaton } => {
            if automaton.states == 0 || automaton.start >= automaton.states {
                errors.push(format!("{context}: automaton has an invalid start state"));
            }
            let mut unknown_state = false;
            let mut accepts_start = false;
            for state in &automaton.accepting {
                if interrupted(stop) {
                    return false;
                }
                accepts_start |= *state == automaton.start;
                if *state >= automaton.states {
                    unknown_state = true;
                }
            }
            if !unknown_state {
                for (from, _, to) in &automaton.transitions {
                    if interrupted(stop) {
                        return false;
                    }
                    if *from >= automaton.states || *to >= automaton.states {
                        unknown_state = true;
                        break;
                    }
                }
            }
            if unknown_state {
                errors.push(format!("{context}: automaton references an unknown state"));
            }
            if variables.is_empty() && !accepts_start {
                errors.push(format!("{context}: empty regular sequence is not accepted"));
            }
        }
        IntGlobalConstraint::Mdd { variables, mdd }
            if mdd.layers.len() != variables.len() || mdd.nodes_per_layer.len() != variables.len() + 1 =>
        {
            errors.push(format!("{context}: MDD layer shape does not match its variables"));
        }
        IntGlobalConstraint::Lex { left, right, .. } | IntGlobalConstraint::Channel { left, right } if left.len() != right.len() => {
            errors.push(format!("{context}: paired arrays differ in length"));
        }
        IntGlobalConstraint::LexChain { rows, .. } => {
            let mut wrong_length = false;
            for pair in rows.windows(2) {
                if interrupted(stop) {
                    return false;
                }
                if pair[0].len() != pair[1].len() {
                    wrong_length = true;
                    break;
                }
            }
            if wrong_length {
                errors.push(format!("{context}: lex chain rows differ in length"));
            }
        }
        IntGlobalConstraint::NoOverlap { starts, durations } => {
            if starts.len() != durations.len() {
                errors.push(format!("{context}: no-overlap starts and durations differ in length"));
            }
        }
        IntGlobalConstraint::OptionalNoOverlap { starts, durations, presences } => {
            if starts.len() != durations.len() || starts.len() != presences.len() {
                errors.push(format!("{context}: optional no-overlap arrays differ in length"));
            }
            if !validate_interval_durations(durations, context, errors, stop) {
                return false;
            }
            for presence in presences.iter().flatten() {
                if interrupted(stop) {
                    return false;
                }
                validate_bool(model, *presence, context, "interval presence", errors);
            }
        }
        IntGlobalConstraint::AlternativeChannel { starts, durations, presences, .. } => {
            if starts.is_empty() {
                errors.push(format!("{context}: alternative has no members"));
            }
            if starts.len() != durations.len() || starts.len() != presences.len() {
                errors.push(format!("{context}: alternative arrays differ in length"));
            }
            if !validate_interval_durations(durations, context, errors, stop) {
                return false;
            }
            for presence in presences {
                if interrupted(stop) {
                    return false;
                }
                validate_bool(model, *presence, context, "alternative presence", errors);
            }
        }
        IntGlobalConstraint::Cumulative { starts, durations, demands, .. }
            if starts.len() != durations.len() || starts.len() != demands.len() =>
        {
            errors.push(format!("{context}: cumulative arrays differ in length"));
        }
        IntGlobalConstraint::CumulativeVar { starts, durations, demands, .. }
            if starts.len() != durations.len() || starts.len() != demands.len() =>
        {
            errors.push(format!("{context}: variable cumulative arrays differ in length"));
        }
        IntGlobalConstraint::BinPacking { items, sizes, .. } | IntGlobalConstraint::BinLoads { items, sizes, .. }
            if items.len() != sizes.len() =>
        {
            errors.push(format!("{context}: bin-packing items and sizes differ in length"));
        }
        IntGlobalConstraint::Knapsack { variables, weights, profits, .. }
            if variables.len() != weights.len() || variables.len() != profits.len() =>
        {
            errors.push(format!("{context}: knapsack arrays differ in length"));
        }
        _ => {}
    }
    !interrupted(stop)
}

fn validate_global_variables(
    model: &Model,
    global: &IntGlobalConstraint,
    context: &str,
    errors: &mut Vec<String>,
    stop: &AtomicBool,
) -> bool {
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
        | IntGlobalConstraint::Knapsack { variables, .. } => validate_ints(model, variables, context, errors, stop),
        IntGlobalConstraint::Minimum { target, variables } | IntGlobalConstraint::Maximum { target, variables } => {
            validate_ints(model, std::slice::from_ref(target), context, errors, stop)
                && validate_ints(model, variables, context, errors, stop)
        }
        IntGlobalConstraint::Element { array, index, value } => {
            validate_ints(model, array, context, errors, stop) && validate_ints(model, &[*index, *value], context, errors, stop)
        }
        IntGlobalConstraint::ElementConst { index, value, .. } => validate_ints(model, &[*index, *value], context, errors, stop),
        IntGlobalConstraint::Lex { left, right, .. } | IntGlobalConstraint::Channel { left, right } => {
            validate_ints(model, left, context, errors, stop) && validate_ints(model, right, context, errors, stop)
        }
        IntGlobalConstraint::LexChain { rows, .. } => {
            for row in rows {
                if !validate_ints(model, row, context, errors, stop) {
                    return false;
                }
            }
            !interrupted(stop)
        }
        IntGlobalConstraint::OptionalNoOverlap { starts, presences, .. } => {
            if !validate_ints(model, starts, context, errors, stop) {
                return false;
            }
            for presence in presences.iter().flatten() {
                if interrupted(stop) {
                    return false;
                }
                check_int(model, presence.0, context, errors);
            }
            !interrupted(stop)
        }
        IntGlobalConstraint::AlternativeChannel { shared_start, starts, presences, .. } => {
            validate_ints(model, std::slice::from_ref(shared_start), context, errors, stop)
                && validate_ints(model, starts, context, errors, stop)
                && validate_ints(model, presences, context, errors, stop)
        }
        IntGlobalConstraint::CumulativeVar { starts, durations, demands, capacity } => {
            validate_ints(model, starts, context, errors, stop)
                && validate_ints(model, durations, context, errors, stop)
                && validate_ints(model, demands, context, errors, stop)
                && validate_ints(model, std::slice::from_ref(capacity), context, errors, stop)
        }
        IntGlobalConstraint::BinLoads { items, loads, .. } => {
            validate_ints(model, items, context, errors, stop) && validate_ints(model, loads, context, errors, stop)
        }
    }
}

fn validate_ints(model: &Model, variables: &[super::IntVarRef], context: &str, errors: &mut Vec<String>, stop: &AtomicBool) -> bool {
    for variable in variables {
        if interrupted(stop) {
            return false;
        }
        check_int(model, variable.0, context, errors);
    }
    !interrupted(stop)
}

fn validate_order_relation(relation: Relation, context: &str, errors: &mut Vec<String>) {
    if matches!(relation, Relation::Eq | Relation::Ne) {
        errors.push(format!("{context}: endpoint or ordered relation must be an inequality"));
    }
}

fn validate_interval_durations(durations: &[i64], context: &str, errors: &mut Vec<String>, stop: &AtomicBool) -> bool {
    let mut negative = false;
    let mut too_large = false;
    for duration in durations {
        if interrupted(stop) {
            return false;
        }
        negative |= *duration < 0;
        too_large |= *duration > i64::from(i32::MAX);
    }
    if negative {
        errors.push(format!("{context}: interval duration must be non-negative"));
    }
    if too_large {
        errors.push(format!("{context}: interval duration exceeds the CP representation"));
    }
    !interrupted(stop)
}

fn validate_bool(model: &Model, variable: super::IntVarRef, context: &str, role: &str, errors: &mut Vec<String>) {
    if model.int_vars.get(variable.0).is_some_and(|domain| !matches!(domain, super::IntDomain::Bool)) {
        errors.push(format!("{context}: {role} is not Boolean"));
    }
}

fn validate_list_items(
    model: &Model,
    list_universes: &[BTreeSet<i32>],
    lists: &[super::ListVarRef],
    items: &[i32],
    context: &str,
    errors: &mut Vec<String>,
    stop: &AtomicBool,
) -> bool {
    for list in lists {
        if interrupted(stop) {
            return false;
        }
        check_list(model, list.0, context, errors);
        if let Some(universe) = list_universes.get(list.0) {
            let mut outside = false;
            for item in items {
                if interrupted(stop) {
                    return false;
                }
                if !universe.contains(item) {
                    outside = true;
                    break;
                }
            }
            if outside {
                errors.push(format!("{context}: item is outside a referenced list universe"));
            }
        }
    }
    !interrupted(stop)
}

fn check_int(model: &Model, index: usize, context: &str, errors: &mut Vec<String>) {
    if index >= model.int_vars.len() {
        errors.push(format!("{context}: references unknown integer variable {index}"));
    }
}

fn check_set(model: &Model, index: usize, context: &str, errors: &mut Vec<String>) {
    if index >= model.sets.len() {
        errors.push(format!("{context}: references unknown set variable {index}"));
    }
}

fn check_list(model: &Model, index: usize, context: &str, errors: &mut Vec<String>) {
    if index >= model.lists.len() {
        errors.push(format!("{context}: references unknown list variable {index}"));
    }
}

fn check_interval(model: &Model, index: usize, context: &str, errors: &mut Vec<String>) {
    if index >= model.intervals.len() {
        errors.push(format!("{context}: references unknown interval variable {index}"));
    }
}
