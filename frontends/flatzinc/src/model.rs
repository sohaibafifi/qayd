//! The in-memory model: symbol tables plus the `post_*` helpers that translate
//! FlatZinc predicates into the canonical semantic model.

use std::collections::HashMap;

use qayd::model::{
    Automaton, Constraint, ConstraintRef, IntExpr, IntGlobalConstraint, IntVarRef, Model as SemanticModel, ModelObject, ModelPackage,
    Objective as SemanticObjective, ObjectiveRef, Relation, SourceRange,
};

use crate::parse::parse_int_set;
use crate::text::{bracket_items, require, split_args};

pub(crate) const UNBOUNDED_LO: i32 = -1_000_000;
pub(crate) const UNBOUNDED_HI: i32 = 1_000_000;

type Expr = IntExpr;

/// Small constructor facade that keeps the predicate lowering readable while
/// producing only semantic expressions.
pub(crate) mod expr {
    use super::{Expr, IntExpr, IntVarRef};

    pub(crate) fn int(value: i64) -> Expr {
        IntExpr::Constant(value)
    }

    pub(crate) fn var(variable: IntVarRef) -> Expr {
        IntExpr::Variable(variable)
    }

    pub(crate) fn add(values: Vec<Expr>) -> Expr {
        IntExpr::Add(values)
    }

    pub(crate) fn sub(left: Expr, right: Expr) -> Expr {
        IntExpr::Sub(Box::new(left), Box::new(right))
    }

    pub(crate) fn mul(values: Vec<Expr>) -> Expr {
        IntExpr::Mul(values)
    }

    pub(crate) fn div(left: Expr, right: Expr) -> Expr {
        IntExpr::Div(Box::new(left), Box::new(right))
    }

    pub(crate) fn rem(left: Expr, right: Expr) -> Expr {
        IntExpr::Mod(Box::new(left), Box::new(right))
    }

    pub(crate) fn min_of(values: Vec<Expr>) -> Expr {
        IntExpr::Min(values)
    }

    pub(crate) fn max_of(values: Vec<Expr>) -> Expr {
        IntExpr::Max(values)
    }

    pub(crate) fn abs(value: Expr) -> Expr {
        IntExpr::Abs(Box::new(value))
    }

    pub(crate) fn eq(left: Expr, right: Expr) -> Expr {
        IntExpr::Eq(Box::new(left), Box::new(right))
    }

    pub(crate) fn ne(left: Expr, right: Expr) -> Expr {
        IntExpr::Ne(Box::new(left), Box::new(right))
    }

    pub(crate) fn le(left: Expr, right: Expr) -> Expr {
        IntExpr::Le(Box::new(left), Box::new(right))
    }

    pub(crate) fn lt(left: Expr, right: Expr) -> Expr {
        IntExpr::Lt(Box::new(left), Box::new(right))
    }

    pub(crate) fn ge(left: Expr, right: Expr) -> Expr {
        IntExpr::Ge(Box::new(left), Box::new(right))
    }

    pub(crate) fn gt(left: Expr, right: Expr) -> Expr {
        IntExpr::Gt(Box::new(left), Box::new(right))
    }

    pub(crate) fn and(values: Vec<Expr>) -> Expr {
        IntExpr::And(values)
    }

    pub(crate) fn or(values: Vec<Expr>) -> Expr {
        IntExpr::Or(values)
    }

    pub(crate) fn not(value: Expr) -> Expr {
        IntExpr::Not(Box::new(value))
    }

    pub(crate) fn imp(left: Expr, right: Expr) -> Expr {
        IntExpr::Imp(Box::new(left), Box::new(right))
    }

    pub(crate) fn iff(left: Expr, right: Expr) -> Expr {
        IntExpr::Iff(Box::new(left), Box::new(right))
    }
}

#[derive(Clone)]
struct Dfa {
    n_states: usize,
    start: usize,
    accept: Vec<usize>,
    transitions: Vec<(usize, i32, usize)>,
}

fn intension(model: &mut SemanticModel, expression: Expr) {
    model.add_constraint(Constraint::Intension(expression));
}

fn linear(model: &mut SemanticModel, coefficients: &[i64], variables: &[IntVarRef], relation: Relation, rhs: i64) {
    debug_assert_eq!(coefficients.len(), variables.len());
    model.add_constraint(Constraint::Linear {
        terms: coefficients.iter().copied().zip(variables.iter().copied()).collect(),
        relation,
        rhs,
    });
}

fn global(model: &mut SemanticModel, constraint: IntGlobalConstraint) {
    model.add_constraint(Constraint::IntegerGlobal(constraint));
}

fn count(model: &mut SemanticModel, variables: &[IntVarRef], value: i32, relation: Relation, count: i64) {
    global(model, IntGlobalConstraint::Count { variables: variables.to_vec(), value, relation, count });
}

fn cardinality(model: &mut SemanticModel, variables: &[IntVarRef], values: &[i32], lower: &[i64], upper: &[i64], closed: bool) {
    global(
        model,
        IntGlobalConstraint::Cardinality {
            variables: variables.to_vec(),
            values: values.to_vec(),
            lower: lower.to_vec(),
            upper: upper.to_vec(),
            closed,
        },
    );
}

fn all_different(model: &mut SemanticModel, variables: &[IntVarRef]) {
    global(model, IntGlobalConstraint::AllDifferent { variables: variables.to_vec(), except: Vec::new() });
}

fn all_equal(model: &mut SemanticModel, variables: &[IntVarRef]) {
    global(model, IntGlobalConstraint::AllEqual(variables.to_vec()));
}

fn precedence(model: &mut SemanticModel, variables: &[IntVarRef], values: &[i32]) {
    global(model, IntGlobalConstraint::ValuePrecedence { variables: variables.to_vec(), values: values.to_vec(), covered: false });
}

fn circuit(model: &mut SemanticModel, successors: &[IntVarRef]) {
    global(model, IntGlobalConstraint::Circuit { successors: successors.to_vec(), cutset: false });
}

fn element(model: &mut SemanticModel, array: &[IntVarRef], index: IntVarRef, value: IntVarRef) {
    global(model, IntGlobalConstraint::Element { array: array.to_vec(), index, value });
}

fn minimum(model: &mut SemanticModel, target: IntVarRef, variables: &[IntVarRef]) {
    global(model, IntGlobalConstraint::Minimum { target, variables: variables.to_vec() });
}

fn maximum(model: &mut SemanticModel, target: IntVarRef, variables: &[IntVarRef]) {
    global(model, IntGlobalConstraint::Maximum { target, variables: variables.to_vec() });
}

fn ordered(model: &mut SemanticModel, variables: &[IntVarRef], relation: Relation) {
    global(model, IntGlobalConstraint::Ordered { variables: variables.to_vec(), relation });
}

fn cumulative_var(model: &mut SemanticModel, starts: &[IntVarRef], durations: &[IntVarRef], demands: &[IntVarRef], capacity: IntVarRef) {
    global(
        model,
        IntGlobalConstraint::CumulativeVar { starts: starts.to_vec(), durations: durations.to_vec(), demands: demands.to_vec(), capacity },
    );
}

fn no_overlap(model: &mut SemanticModel, starts: &[IntVarRef], durations: &[i64]) {
    global(model, IntGlobalConstraint::NoOverlap { starts: starts.to_vec(), durations: durations.to_vec() });
}

fn extension(model: &mut SemanticModel, variables: &[IntVarRef], tuples: &[Vec<i32>], positive: bool) {
    global(model, IntGlobalConstraint::Table { variables: variables.to_vec(), tuples: tuples.to_vec().into(), positive });
}

fn regular(model: &mut SemanticModel, variables: &[IntVarRef], dfa: Dfa) {
    global(
        model,
        IntGlobalConstraint::Regular {
            variables: variables.to_vec(),
            automaton: Automaton { states: dfa.n_states, start: dfa.start, accepting: dfa.accept, transitions: dfa.transitions },
        },
    );
}

fn lex(model: &mut SemanticModel, left: &[IntVarRef], right: &[IntVarRef], strict: bool) {
    global(model, IntGlobalConstraint::Lex { left: left.to_vec(), right: right.to_vec(), strict });
}

/// How a Boolean reifies a base constraint.
#[derive(Clone, Copy)]
pub(crate) enum Reif {
    /// `r <-> base` (`*_reif`).
    Iff,
    /// `r -> base` (`*_imp`).
    Imp,
}

/// A parsed variable domain.
pub(crate) enum FznDomain {
    Bool,
    Range(i32, i32),
    Set(Vec<i32>),
}

/// An item the model marks for output (`:: output_var` / `:: output_array`).
/// Used by the MiniZinc output protocol.
pub(crate) enum Output {
    /// `name = <value>;`
    Var { name: String, var: IntVarRef, is_bool: bool },
    /// `name = arrayNd(<dims>, [<values>]);`
    Array { name: String, dims: Vec<(i32, i32)>, vars: Vec<IntVarRef>, is_bool: bool },
}

/// Symbol tables and the canonical package under construction.
pub(crate) struct Model {
    package: ModelPackage,
    pub(crate) vars: HashMap<String, IntVarRef>,
    pub(crate) arrays: HashMap<String, Vec<IntVarRef>>,
    pub(crate) ints: HashMap<String, i32>,
    pub(crate) int_arrays: HashMap<String, Vec<i32>>,
    /// Set variables, represented by their characteristic vector: a `0/1`
    /// membership variable for each value in the set's universe.
    pub(crate) set_vars: HashMap<String, HashMap<i32, IntVarRef>>,
    constants: HashMap<i32, IntVarRef>,
    /// Items annotated for output, in declaration order.
    outputs: Vec<Output>,
    /// Objective plus its frontend-owned metadata, materialized at package handoff.
    objective: Option<PendingObjective>,
}

struct PendingObjective {
    minimize: bool,
    variable: IntVarRef,
    item: usize,
    annotations: Vec<String>,
    source: SourceRange,
}

/// Semantic object counts at the start of one FlatZinc item.
pub(crate) struct MetadataMark {
    int_vars: usize,
    constraints: usize,
    objectives: usize,
    outputs: usize,
}

impl Model {
    pub(crate) fn new() -> Self {
        Self {
            package: ModelPackage::new(SemanticModel::new()),
            vars: HashMap::new(),
            arrays: HashMap::new(),
            ints: HashMap::new(),
            int_arrays: HashMap::new(),
            set_vars: HashMap::new(),
            constants: HashMap::new(),
            outputs: Vec::new(),
            objective: None,
        }
    }

    pub(crate) fn new_var(&mut self, name: String, domain: &FznDomain) -> IntVarRef {
        let var = match domain {
            FznDomain::Bool => self.package.model.bool_var(),
            FznDomain::Range(lo, hi) => self.package.model.int_range(*lo, *hi),
            FznDomain::Set(values) => self.package.model.int_set(values.clone()),
        };
        self.vars.insert(name.clone(), var);
        let object = ModelObject::IntVar(var);
        self.package.metadata.names.insert(object, name.clone());
        self.package.metadata.frontend_ids.insert(("flatzinc".to_string(), name), object);
        var
    }

    pub(crate) fn constant(&mut self, value: i32) -> IntVarRef {
        if let Some(&variable) = self.constants.get(&value) {
            return variable;
        }
        let variable = self.package.model.int_set(vec![value]);
        self.constants.insert(value, variable);
        variable
    }

    pub(crate) fn new_aux_bool(&mut self) -> IntVarRef {
        self.package.model.bool_var()
    }

    pub(crate) fn add_output(&mut self, output: Output) {
        let variables: &[IntVarRef] = match &output {
            Output::Var { var, .. } => std::slice::from_ref(var),
            Output::Array { vars, .. } => vars,
        };
        for &variable in variables {
            let object = ModelObject::IntVar(variable);
            if !self.package.metadata.outputs.contains(&object) {
                self.package.metadata.outputs.push(object);
            }
        }
        self.outputs.push(output);
    }

    pub(crate) fn post_expr(&mut self, expression: Expr) {
        intension(&mut self.package.model, expression);
    }

    pub(crate) fn post_raw_linear(&mut self, coefficients: &[i64], variables: &[IntVarRef], relation: Relation, rhs: i64) {
        linear(&mut self.package.model, coefficients, variables, relation, rhs);
    }

    pub(crate) fn post_all_different(&mut self, variables: &[IntVarRef]) {
        all_different(&mut self.package.model, variables);
    }

    pub(crate) fn post_all_equal(&mut self, variables: &[IntVarRef]) {
        all_equal(&mut self.package.model, variables);
    }

    pub(crate) fn post_precedence(&mut self, variables: &[IntVarRef], values: &[i32]) {
        precedence(&mut self.package.model, variables, values);
    }

    pub(crate) fn set_objective(
        &mut self,
        minimize: bool,
        variable: IntVarRef,
        item: usize,
        annotations: Vec<String>,
        source: SourceRange,
    ) {
        self.objective = Some(PendingObjective { minimize, variable, item, annotations, source });
    }

    pub(crate) fn into_package(mut self) -> (ModelPackage, Vec<Output>) {
        if let Some(objective) = self.objective {
            let reference = self
                .package
                .model
                .add_objective(SemanticObjective::IntExpr { minimize: objective.minimize, expr: IntExpr::Variable(objective.variable) });
            let object = ModelObject::Objective(reference);
            self.package.metadata.sources.entry(object).or_default().push(objective.source);
            Self::insert_object_annotations(&mut self.package, object, objective.item, &objective.annotations);
        }
        (self.package, self.outputs)
    }

    /// Snapshot semantic object counts before parsing one source item.
    pub(crate) fn metadata_mark(&self) -> MetadataMark {
        MetadataMark {
            int_vars: self.package.model.int_vars().len(),
            constraints: self.package.model.constraints().len(),
            objectives: self.package.model.objectives().len(),
            outputs: self.outputs.len(),
        }
    }

    /// Retain an item's opaque FlatZinc annotations and attach its source range
    /// to every semantic object introduced while lowering that item.
    pub(crate) fn record_item_metadata(&mut self, item: usize, mark: MetadataMark, annotations: &[String], source: SourceRange) {
        for (index, annotation) in annotations.iter().enumerate() {
            self.package.metadata.annotations.insert(format!("flatzinc.item.{item}.annotation.{index}"), annotation.clone());
        }

        let mut objects = (mark.int_vars..self.package.model.int_vars().len())
            .map(|index| ModelObject::IntVar(IntVarRef(index)))
            .chain((mark.constraints..self.package.model.constraints().len()).map(|index| ModelObject::Constraint(ConstraintRef(index))))
            .chain((mark.objectives..self.package.model.objectives().len()).map(|index| ModelObject::Objective(ObjectiveRef(index))))
            .collect::<Vec<_>>();
        for output in &self.outputs[mark.outputs..] {
            let variables: &[IntVarRef] = match output {
                Output::Var { var, .. } => std::slice::from_ref(var),
                Output::Array { vars, .. } => vars,
            };
            for &variable in variables {
                let object = ModelObject::IntVar(variable);
                if !objects.contains(&object) {
                    objects.push(object);
                }
            }
        }

        if objects.is_empty() {
            self.package
                .metadata
                .annotations
                .insert(format!("flatzinc.item.{item}.source"), format!("{}:{}..{}", source.source, source.start, source.end));
            return;
        }
        for object in objects {
            self.package.metadata.sources.entry(object).or_default().push(source.clone());
            Self::insert_object_annotations(&mut self.package, object, item, annotations);
        }
    }

    fn insert_object_annotations(package: &mut ModelPackage, object: ModelObject, item: usize, annotations: &[String]) {
        let object_annotations = package.metadata.object_annotations.entry(object).or_default();
        for (index, annotation) in annotations.iter().enumerate() {
            object_annotations.insert(format!("flatzinc.item.{item}.annotation.{index}"), annotation.clone());
        }
    }

    pub(crate) fn int_atom(&self, s: &str) -> Result<i32, String> {
        let s = s.trim();
        match s {
            "true" => Ok(1),
            "false" => Ok(0),
            _ => s.parse().or_else(|_| self.ints.get(s).copied().ok_or_else(|| format!("unknown integer `{s}`"))),
        }
    }

    pub(crate) fn var_atom(&mut self, s: &str) -> Result<IntVarRef, String> {
        let s = s.trim();
        if let Ok(value) = self.int_atom(s) {
            return Ok(self.constant(value));
        }
        self.vars.get(s).copied().ok_or_else(|| format!("unknown variable `{s}`"))
    }

    pub(crate) fn int_list(&self, s: &str) -> Result<Vec<i32>, String> {
        let s = s.trim();
        if let Some(items) = bracket_items(s) {
            return items.iter().map(|x| self.int_atom(x)).collect();
        }
        self.int_arrays.get(s).cloned().ok_or_else(|| format!("unknown integer array `{s}`"))
    }

    pub(crate) fn var_list(&mut self, s: &str) -> Result<Vec<IntVarRef>, String> {
        let s = s.trim();
        if let Some(items) = bracket_items(s) {
            return items.iter().map(|x| self.var_atom(x)).collect();
        }
        if let Some(vars) = self.arrays.get(s) {
            return Ok(vars.clone());
        }
        // A parameter (constant) array used where a variable array is expected:
        // materialise each value as a fixed variable.
        if let Some(values) = self.int_arrays.get(s).cloned() {
            return Ok(values.iter().map(|&v| self.constant(v)).collect());
        }
        Ok(vec![self.var_atom(s)?])
    }

    pub(crate) fn post_cmp(&mut self, a: &str, b: &str, rel: Relation) -> Result<(), String> {
        let x = self.var_atom(a)?;
        let y = self.var_atom(b)?;
        linear(&mut self.package.model, &[1, -1], &[x, y], rel, 0);
        Ok(())
    }

    pub(crate) fn post_linear(&mut self, args: &[String], rel: Relation) -> Result<(), String> {
        require(args.len() == 3, "linear predicate expects 3 arguments")?;
        let coeffs = self.int_list(&args[0])?.into_iter().map(i64::from).collect::<Vec<_>>();
        let mut vars = self.var_list(&args[1])?;
        require(coeffs.len() == vars.len(), "linear coeffs/vars length mismatch")?;
        let rhs = if let Ok(k) = self.int_atom(&args[2]) {
            k as i64
        } else {
            vars.push(self.var_atom(&args[2])?);
            0
        };
        let mut coeffs = coeffs;
        if coeffs.len() + 1 == vars.len() {
            coeffs.push(-1);
        }
        linear(&mut self.package.model, &coeffs, &vars, rel, rhs);
        Ok(())
    }

    pub(crate) fn post_element(&mut self, args: &[String], var_array: bool) -> Result<(), String> {
        require(args.len() == 3, "element predicate expects 3 arguments")?;
        let index = self.var_atom(&args[0])?;
        let array =
            if var_array { self.var_list(&args[1])? } else { self.int_list(&args[1])?.into_iter().map(|v| self.constant(v)).collect() };
        let value = self.var_atom(&args[2])?;
        let zero = self.package.model.int_range(0, array.len() as i32 - 1);
        linear(&mut self.package.model, &[1, -1], &[index, zero], Relation::Eq, 1);
        element(&mut self.package.model, &array, zero, value);
        Ok(())
    }

    pub(crate) fn post_gecode_element(&mut self, args: &[String]) -> Result<(), String> {
        require(args.len() == 4, "gecode_int_element expects 4 arguments")?;
        let index = self.var_atom(&args[0])?;
        let offset = self.int_atom(&args[1])?;
        let array = self.var_list(&args[2])?;
        let value = self.var_atom(&args[3])?;
        let zero = self.package.model.int_range(0, array.len() as i32 - 1);
        linear(&mut self.package.model, &[1, -1], &[index, zero], Relation::Eq, offset as i64);
        element(&mut self.package.model, &array, zero, value);
        Ok(())
    }

    pub(crate) fn post_count(&mut self, args: &[String], rel: Relation) -> Result<(), String> {
        require(args.len() == 3, "count predicate expects 3 arguments")?;
        let vars = self.var_list(&args[0])?;
        // Fast path: a constant value and target use the normalized count global.
        if let (Ok(value), Ok(target)) = (self.int_atom(&args[1]), self.int_atom(&args[2])) {
            count(&mut self.package.model, &vars, value, rel, target as i64);
            return Ok(());
        }
        // General form `#{ i : vars[i] == value } <rel> target` with a variable
        // value and/or target, decomposed through `intension`.
        let value = self.atom_expr(&args[1])?;
        let target = self.atom_expr(&args[2])?;
        let occ: Vec<Expr> = vars.iter().map(|&v| expr::eq(expr::var(v), value.clone())).collect();
        intension(&mut self.package.model, cmp_expr(rel, expr::add(occ), target));
        Ok(())
    }

    pub(crate) fn post_extremum(&mut self, args: &[String], is_min: bool) -> Result<(), String> {
        require(args.len() == 2, "array extremum predicate expects 2 arguments")?;
        let target = self.var_atom(&args[0])?;
        let vars = self.var_list(&args[1])?;
        if is_min {
            minimum(&mut self.package.model, target, &vars);
        } else {
            maximum(&mut self.package.model, target, &vars);
        }
        Ok(())
    }

    pub(crate) fn post_bool_clause(&mut self, args: &[String]) -> Result<(), String> {
        require(args.len() == 2, "bool_clause expects 2 arguments")?;
        let mut terms = Vec::new();
        for var in self.var_list(&args[0])? {
            terms.push(expr::eq(expr::var(var), expr::int(1)));
        }
        for var in self.var_list(&args[1])? {
            terms.push(expr::eq(expr::var(var), expr::int(0)));
        }
        intension(&mut self.package.model, expr::or(terms));
        Ok(())
    }

    /// An atom (`int`/`bool` literal or variable) as an [`Expr`].
    pub(crate) fn atom_expr(&mut self, s: &str) -> Result<Expr, String> {
        let s = s.trim();
        if let Ok(k) = self.int_atom(s) {
            Ok(expr::int(k as i64))
        } else {
            Ok(expr::var(self.var_atom(s)?))
        }
    }

    pub(crate) fn two_atoms(&mut self, a: &str, b: &str) -> Result<(Expr, Expr), String> {
        Ok((self.atom_expr(a)?, self.atom_expr(b)?))
    }

    /// A variable list as `Var` expressions.
    pub(crate) fn var_exprs(&mut self, s: &str) -> Result<Vec<Expr>, String> {
        Ok(self.var_list(s)?.into_iter().map(expr::var).collect())
    }

    /// The weighted sum `sum_i coeffs[i] * vars[i]` as an [`Expr`].
    pub(crate) fn lin_sum_expr(&mut self, coeffs_s: &str, vars_s: &str) -> Result<Expr, String> {
        let coeffs = self.int_list(coeffs_s)?;
        let vars = self.var_list(vars_s)?;
        require(coeffs.len() == vars.len(), "linear coeffs/vars length mismatch")?;
        let terms = coeffs.iter().zip(&vars).map(|(&a, &v)| expr::mul(vec![expr::int(a as i64), expr::var(v)])).collect();
        Ok(expr::add(terms))
    }

    /// Post `r <-> base` or `r -> base`.
    pub(crate) fn post_reif(&mut self, base: Expr, r_s: &str, mode: Reif) -> Result<(), String> {
        let r = self.atom_expr(r_s)?;
        let formula = match mode {
            Reif::Iff => expr::iff(r, base),
            Reif::Imp => expr::imp(r, base),
        };
        intension(&mut self.package.model, formula);
        Ok(())
    }

    /// Post `target = value`.
    pub(crate) fn post_def(&mut self, target_s: &str, value: Expr) -> Result<(), String> {
        let t = self.atom_expr(target_s)?;
        intension(&mut self.package.model, expr::eq(t, value));
        Ok(())
    }

    /// `int_<cmp>_reif` / `int_<cmp>_imp`: reify `a <cmp> b`.
    pub(crate) fn post_int_cmp_reif(&mut self, args: &[String], rel: Relation, mode: Reif) -> Result<(), String> {
        require(args.len() == 3, "reified comparison expects 3 arguments")?;
        let (a, b) = self.two_atoms(&args[0], &args[1])?;
        self.post_reif(cmp_expr(rel, a, b), &args[2], mode)
    }

    /// `int_lin_<cmp>_reif` / `int_lin_<cmp>_imp`: reify `sum a_i x_i <cmp> c`.
    pub(crate) fn post_lin_reif(&mut self, args: &[String], rel: Relation, mode: Reif) -> Result<(), String> {
        require(args.len() == 4, "reified linear expects 4 arguments")?;
        let sum = self.lin_sum_expr(&args[0], &args[1])?;
        let c = expr::int(self.int_atom(&args[2])? as i64);
        self.post_reif(cmp_expr(rel, sum, c), &args[3], mode)
    }

    /// Membership of `x` in a FlatZinc set literal (`lo..hi` or `{...}`).
    pub(crate) fn set_member_expr(&self, x: Expr, set_s: &str) -> Result<Expr, String> {
        let s = set_s.trim();
        if let Some(inner) = s.strip_prefix('{').and_then(|r| r.strip_suffix('}')) {
            let mut terms = Vec::new();
            for item in split_args(inner) {
                if item.trim().is_empty() {
                    continue;
                }
                terms.push(expr::eq(x.clone(), expr::int(self.int_atom(&item)? as i64)));
            }
            require(!terms.is_empty(), "empty set in set_in")?;
            Ok(expr::or(terms))
        } else if let Some((lo, hi)) = s.split_once("..") {
            let lo = self.int_atom(lo)?;
            let hi = self.int_atom(hi)?;
            Ok(expr::and(vec![expr::ge(x.clone(), expr::int(lo as i64)), expr::le(x, expr::int(hi as i64))]))
        } else {
            Ok(expr::eq(x, expr::int(self.int_atom(s)? as i64)))
        }
    }

    /// `set_in` (and reified variants), over either a set *variable* or a
    /// constant set literal.
    pub(crate) fn post_set_in(&mut self, args: &[String], mode: Option<Reif>) -> Result<(), String> {
        require(args.len() >= 2, "set_in expects at least 2 arguments")?;
        let set_s = args[1].trim();
        // Set variable: look the element up in the characteristic vector.
        if self.set_vars.contains_key(set_s) {
            let e = self.int_atom(&args[0])?; // membership of a constant element
            let mem = self.set_vars[set_s].get(&e).copied();
            let r = args.get(2).map(String::as_str);
            return self.post_set_var_in(mem, r, mode);
        }
        // Constant set literal.
        let x = self.atom_expr(&args[0])?;
        let member = self.set_member_expr(x, set_s)?;
        match mode {
            None => {
                intension(&mut self.package.model, member);
                Ok(())
            }
            Some(m) => {
                require(args.len() == 3, "reified set_in expects 3 arguments")?;
                self.post_reif(member, &args[2], m)
            }
        }
    }

    /// Membership of a constant element in a set variable. `mem` is its `0/1`
    /// membership variable, or `None` when the element is outside the universe
    /// (membership is then constantly false).
    pub(crate) fn post_set_var_in(&mut self, mem: Option<IntVarRef>, r: Option<&str>, mode: Option<Reif>) -> Result<(), String> {
        let base = match mem {
            Some(v) => expr::var(v),
            None => expr::int(0),
        };
        match mode {
            None => {
                intension(&mut self.package.model, base);
                Ok(())
            }
            Some(m) => {
                let r = r.ok_or("reified set_in expects a reification variable")?;
                self.post_reif(base, r, m)
            }
        }
    }

    /// `int_pow` / `gecode_int_pow`: `z = x^y` for a constant exponent `y >= 0`.
    pub(crate) fn post_int_pow(&mut self, args: &[String]) -> Result<(), String> {
        require(args.len() == 3, "int_pow expects 3 arguments")?;
        let base = self.atom_expr(&args[0])?;
        let exp = self.int_atom(&args[1])?;
        require(exp >= 0, "int_pow: negative exponent unsupported")?;
        let factors = if exp == 0 { vec![expr::int(1)] } else { (0..exp).map(|_| base.clone()).collect() };
        self.post_def(&args[2], expr::mul(factors))
    }

    /// `gecode_table_int(x, t)`: `x` is a row of the flattened tuple table `t`.
    pub(crate) fn post_table(&mut self, args: &[String]) -> Result<(), String> {
        require(args.len() == 2, "table expects 2 arguments")?;
        let vars = self.var_list(&args[0])?;
        require(!vars.is_empty(), "table over empty scope")?;
        let flat = self.int_list(&args[1])?;
        let arity = vars.len();
        require(flat.len() % arity == 0, "table tuple length not a multiple of arity")?;
        let tuples: Vec<Vec<i32>> = flat.chunks(arity).map(<[i32]>::to_vec).collect();
        extension(&mut self.package.model, &vars, &tuples, true);
        Ok(())
    }

    /// `gecode_regular(x, Q, S, d, q0, F)` -> layered-DFA `regular`.
    pub(crate) fn post_regular(&mut self, args: &[String]) -> Result<(), String> {
        require(args.len() == 6, "regular expects 6 arguments")?;
        let vars = self.var_list(&args[0])?;
        let q = self.int_atom(&args[1])? as usize;
        let s = self.int_atom(&args[2])? as usize;
        let d = self.int_list(&args[3])?;
        let q0 = self.int_atom(&args[4])? as usize;
        let accept = parse_int_set(self, &args[5])?;
        require(d.len() == q * s, "regular: transition table size mismatch")?;
        require(q0 >= 1 && q0 <= q, "regular: start state out of range")?;
        // FlatZinc states are 1..Q, symbols 1..S, row-major `d`; `0` is the dead state.
        let mut transitions = Vec::new();
        for st in 1..=q {
            for sym in 1..=s {
                let nxt = d[(st - 1) * s + (sym - 1)];
                if nxt >= 1 && nxt as usize <= q {
                    transitions.push((st - 1, sym as i32, nxt as usize - 1));
                }
            }
        }
        let dfa = Dfa { n_states: q, start: q0 - 1, accept: accept.iter().map(|&f| f as usize - 1).collect(), transitions };
        regular(&mut self.package.model, &vars, dfa);
        Ok(())
    }

    /// `gecode_circuit(offset, x)`: Hamiltonian circuit with `offset`-based successors.
    pub(crate) fn post_circuit(&mut self, args: &[String]) -> Result<(), String> {
        require(args.len() == 2, "circuit expects 2 arguments")?;
        let offset = self.int_atom(&args[0])?;
        let succ = self.var_list(&args[1])?;
        if offset == 0 {
            circuit(&mut self.package.model, &succ);
        } else {
            let n = succ.len();
            let shifted: Vec<IntVarRef> = (0..n).map(|_| self.package.model.int_range(0, n as i32 - 1)).collect();
            for k in 0..n {
                linear(&mut self.package.model, &[1, -1], &[succ[k], shifted[k]], Relation::Eq, offset as i64);
            }
            circuit(&mut self.package.model, &shifted);
        }
        Ok(())
    }

    /// `gecode_cumulatives(s, d, r, b)` -> `cumulative` with variable rows.
    pub(crate) fn post_cumulatives(&mut self, args: &[String]) -> Result<(), String> {
        require(args.len() == 4, "cumulatives expects 4 arguments")?;
        let starts = self.var_list(&args[0])?;
        let durations = self.var_list(&args[1])?;
        let heights = self.var_list(&args[2])?;
        let capacity = self.var_atom(&args[3])?;
        cumulative_var(&mut self.package.model, &starts, &durations, &heights, capacity);
        Ok(())
    }

    /// `gecode_schedule_unary(x, p)` -> `no_overlap` with constant durations.
    pub(crate) fn post_schedule_unary(&mut self, args: &[String]) -> Result<(), String> {
        require(args.len() == 2, "schedule_unary expects 2 arguments")?;
        let starts = self.var_list(&args[0])?;
        // Constant durations use the normalized no-overlap global.
        if let Ok(durations) = self.int_list(&args[1]) {
            let durations: Vec<i64> = durations.into_iter().map(i64::from).collect();
            require(starts.len() == durations.len(), "schedule_unary length mismatch")?;
            no_overlap(&mut self.package.model, &starts, &durations);
            return Ok(());
        }
        // Variable durations: pairwise ordering decomposition,
        // `s_i + d_i <= s_j \/ s_j + d_j <= s_i`.
        let durations = self.var_list(&args[1])?;
        require(starts.len() == durations.len(), "schedule_unary length mismatch")?;
        for i in 0..starts.len() {
            for j in i + 1..starts.len() {
                let i_first = expr::le(expr::add(vec![expr::var(starts[i]), expr::var(durations[i])]), expr::var(starts[j]));
                let j_first = expr::le(expr::add(vec![expr::var(starts[j]), expr::var(durations[j])]), expr::var(starts[i]));
                intension(&mut self.package.model, expr::or(vec![i_first, j_first]));
            }
        }
        Ok(())
    }

    /// `gecode_global_cardinality(x, cover, counts)` with variable occurrence counts.
    pub(crate) fn post_gcc_counts(&mut self, args: &[String], closed: bool) -> Result<(), String> {
        require(args.len() == 3, "global_cardinality expects 3 arguments")?;
        let vars = self.var_list(&args[0])?;
        let cover = self.int_list(&args[1])?;
        let counts = self.var_list(&args[2])?;
        require(cover.len() == counts.len(), "global_cardinality cover/counts mismatch")?;
        for (j, &val) in cover.iter().enumerate() {
            let terms: Vec<Expr> = vars.iter().map(|&v| expr::eq(expr::var(v), expr::int(val as i64))).collect();
            intension(&mut self.package.model, expr::eq(expr::add(terms), expr::var(counts[j])));
        }
        if closed {
            // Closed form: every variable must take a value from `cover`.
            for &v in &vars {
                let member = cover.iter().map(|&c| expr::eq(expr::var(v), expr::int(c as i64))).collect();
                intension(&mut self.package.model, expr::or(member));
            }
        }
        Ok(())
    }

    /// `fzn_global_cardinality_low_up[_closed](x, cover, lbound, ubound)`.
    pub(crate) fn post_gcc_low_up(&mut self, args: &[String], closed: bool) -> Result<(), String> {
        require(args.len() == 4, "global_cardinality_low_up expects 4 arguments")?;
        let vars = self.var_list(&args[0])?;
        let cover = self.int_list(&args[1])?;
        let low: Vec<i64> = self.int_list(&args[2])?.into_iter().map(i64::from).collect();
        let up: Vec<i64> = self.int_list(&args[3])?.into_iter().map(i64::from).collect();
        cardinality(&mut self.package.model, &vars, &cover, &low, &up, closed);
        Ok(())
    }

    /// `gecode_bin_packing_load(l, bin, w, minIndex)` or the standard 3-argument
    /// `fzn_bin_packing_load(l, bin, w)` (bins 1-based): per-bin load equals the
    /// sum of the weights of the items assigned to it.
    pub(crate) fn post_bin_packing_load(&mut self, args: &[String]) -> Result<(), String> {
        require(args.len() == 3 || args.len() == 4, "bin_packing_load expects 3 or 4 arguments")?;
        let load = self.var_list(&args[0])?;
        let bin = self.var_list(&args[1])?;
        let weight = self.int_list(&args[2])?;
        let min_index = if args.len() == 4 { self.int_atom(&args[3])? } else { 1 };
        require(bin.len() == weight.len(), "bin_packing_load bin/weight mismatch")?;
        for (j, &l) in load.iter().enumerate() {
            let target = min_index + j as i32;
            let terms: Vec<Expr> = bin
                .iter()
                .zip(&weight)
                .map(|(&b, &w)| expr::mul(vec![expr::int(w as i64), expr::eq(expr::var(b), expr::int(target as i64))]))
                .collect();
            intension(&mut self.package.model, expr::eq(expr::add(terms), expr::var(l)));
        }
        Ok(())
    }

    /// `gecode_maximum_arg_int_offset(x, offset, i)`: `i` is `offset` plus the
    /// index of the first maximum of `x`.
    pub(crate) fn post_arg_max(&mut self, args: &[String]) -> Result<(), String> {
        require(args.len() == 3, "maximum_arg expects 3 arguments")?;
        let xs = self.var_list(&args[0])?;
        require(!xs.is_empty(), "maximum_arg over empty array")?;
        let offset = self.int_atom(&args[1])?;
        let idx = self.var_atom(&args[2])?;
        let n = xs.len();
        // m = max(xs); pos = idx - offset is a 0-based index with xs[pos] = m.
        let m = self.package.model.int_range(UNBOUNDED_LO, UNBOUNDED_HI);
        maximum(&mut self.package.model, m, &xs);
        let pos = self.package.model.int_range(0, n as i32 - 1);
        linear(&mut self.package.model, &[1, -1], &[idx, pos], Relation::Eq, offset as i64);
        element(&mut self.package.model, &xs, pos, m);
        // First maximum: any earlier element must be strictly smaller.
        for (j, &xj) in xs.iter().enumerate() {
            intension(
                &mut self.package.model,
                expr::imp(expr::gt(expr::var(pos), expr::int(j as i64)), expr::ne(expr::var(xj), expr::var(m))),
            );
        }
        Ok(())
    }

    /// `array_int_lq` / `array_int_lt` (and bool variants) -> lexicographic order.
    pub(crate) fn post_array_lex(&mut self, args: &[String], strict: bool) -> Result<(), String> {
        require(args.len() == 2, "array lex expects 2 arguments")?;
        let x = self.var_list(&args[0])?;
        let y = self.var_list(&args[1])?;
        require(x.len() == y.len(), "array lex length mismatch")?;
        lex(&mut self.package.model, &x, &y, strict);
        Ok(())
    }

    /// `fzn_increasing_*` / `fzn_decreasing_*`.
    pub(crate) fn post_ordered(&mut self, args: &[String], rel: Relation) -> Result<(), String> {
        require(args.len() == 1, "increasing/decreasing expects 1 argument")?;
        let vars = self.var_list(&args[0])?;
        if vars.len() > 1 {
            ordered(&mut self.package.model, &vars, rel);
        }
        Ok(())
    }

    /// `fzn_member_int(x, y)`: `y` occurs in the array `x`.
    pub(crate) fn post_member(&mut self, args: &[String]) -> Result<(), String> {
        require(args.len() == 2, "member expects 2 arguments")?;
        let xs = self.var_list(&args[0])?;
        require(!xs.is_empty(), "member over empty array")?;
        let y = self.atom_expr(&args[1])?;
        let terms = xs.iter().map(|&x| expr::eq(expr::var(x), y.clone())).collect();
        intension(&mut self.package.model, expr::or(terms));
        Ok(())
    }

    /// `bool_sum_eq/le/ge(x, c)`: sum of Booleans `<rel>` `c` (constant or variable).
    pub(crate) fn post_bool_sum(&mut self, args: &[String], rel: Relation) -> Result<(), String> {
        require(args.len() == 2, "bool_sum expects 2 arguments")?;
        let mut vars = self.var_list(&args[0])?;
        let mut coeffs = vec![1i64; vars.len()];
        let rhs = if let Ok(k) = self.int_atom(&args[1]) {
            i64::from(k)
        } else {
            vars.push(self.var_atom(&args[1])?);
            coeffs.push(-1);
            0
        };
        linear(&mut self.package.model, &coeffs, &vars, rel, rhs);
        Ok(())
    }

    /// `chuffed_connected(from, to, ns, es)`: the subgraph induced by the
    /// selected nodes `ns` and edges `es` is connected. Decomposed with BFS
    /// levels: one selected root at level 0; every other selected node needs a
    /// selected edge to a neighbour exactly one level below.
    pub(crate) fn post_connected(&mut self, args: &[String]) -> Result<(), String> {
        require(args.len() == 4, "connected expects 4 arguments")?;
        let from = self.int_list(&args[0])?;
        let to = self.int_list(&args[1])?;
        let ns = self.var_list(&args[2])?;
        let es = self.var_list(&args[3])?;
        require(from.len() == to.len(), "connected: from/to length mismatch")?;
        require(from.len() == es.len(), "connected: edge arrays length mismatch")?;
        let n = ns.len();
        let node = |id: i32| -> Result<usize, String> {
            let i = id - 1; // FlatZinc node ids are 1-based
            if i >= 0 && (i as usize) < n {
                Ok(i as usize)
            } else {
                Err(format!("connected: node id `{id}` out of range"))
            }
        };
        // A selected edge selects both its endpoints.
        for (e, &sel) in es.iter().enumerate() {
            let (u, v) = (node(from[e])?, node(to[e])?);
            intension(&mut self.package.model, expr::imp(expr::var(sel), expr::var(ns[u])));
            intension(&mut self.package.model, expr::imp(expr::var(sel), expr::var(ns[v])));
        }
        if n == 0 {
            return Ok(());
        }
        // Levels, root indicators, and "any node selected". The semantic CP
        // compiler includes every declared integer variable in its search.
        let level: Vec<IntVarRef> = (0..n).map(|_| self.package.model.int_range(0, n as i32 - 1)).collect();
        let root: Vec<IntVarRef> = (0..n).map(|_| self.package.model.int_range(0, 1)).collect();
        let any = self.package.model.int_range(0, 1);
        intension(&mut self.package.model, expr::iff(expr::var(any), expr::or(ns.iter().map(|&v| expr::var(v)).collect())));
        // Exactly one root when any node is selected, none otherwise.
        let mut coeffs = vec![1i64; n];
        let mut vars = root.clone();
        coeffs.push(-1);
        vars.push(any);
        linear(&mut self.package.model, &coeffs, &vars, Relation::Eq, 0);
        // Incident (edge, neighbour) pairs per node.
        let mut incident: Vec<Vec<(IntVarRef, usize)>> = vec![Vec::new(); n];
        for (e, &sel) in es.iter().enumerate() {
            let (u, v) = (node(from[e])?, node(to[e])?);
            incident[u].push((sel, v));
            incident[v].push((sel, u));
        }
        for v in 0..n {
            // A root is a selected node at level 0.
            intension(
                &mut self.package.model,
                expr::imp(expr::var(root[v]), expr::and(vec![expr::var(ns[v]), expr::eq(expr::var(level[v]), expr::int(0))])),
            );
            // A selected non-root node has a selected edge to a parent one level below.
            let parents: Vec<Expr> = incident[v]
                .iter()
                .map(|&(e, u)| expr::and(vec![expr::var(e), expr::eq(expr::var(level[u]), expr::sub(expr::var(level[v]), expr::int(1)))]))
                .collect();
            intension(
                &mut self.package.model,
                expr::imp(expr::and(vec![expr::var(ns[v]), expr::not(expr::var(root[v]))]), expr::or(parents)),
            );
        }
        Ok(())
    }

    /// `array_bool_and` / `array_bool_or`: reify the conjunction/disjunction.
    pub(crate) fn post_bool_nary(&mut self, args: &[String], conj: bool) -> Result<(), String> {
        require(args.len() == 2, "array bool op expects 2 arguments")?;
        let terms = self.var_exprs(&args[0])?;
        let base = if conj { expr::and(terms) } else { expr::or(terms) };
        self.post_reif(base, &args[1], Reif::Iff)
    }

    /// `bool_clause_reif(pos, neg, r)`: `r <-> (OR pos) | (OR !neg)`.
    pub(crate) fn post_bool_clause_reif(&mut self, args: &[String]) -> Result<(), String> {
        require(args.len() == 3, "bool_clause_reif expects 3 arguments")?;
        let mut terms = Vec::new();
        for v in self.var_list(&args[0])? {
            terms.push(expr::var(v));
        }
        for v in self.var_list(&args[1])? {
            terms.push(expr::not(expr::var(v)));
        }
        let base = if terms.is_empty() { expr::int(0) } else { expr::or(terms) };
        self.post_reif(base, &args[2], Reif::Iff)
    }
}

/// Build the relational [`Expr`] for `a <rel> b`.
pub(crate) fn cmp_expr(rel: Relation, a: Expr, b: Expr) -> Expr {
    match rel {
        Relation::Eq => expr::eq(a, b),
        Relation::Ne => expr::ne(a, b),
        Relation::Le => expr::le(a, b),
        Relation::Lt => expr::lt(a, b),
        Relation::Ge => expr::ge(a, b),
        Relation::Gt => expr::gt(a, b),
    }
}
