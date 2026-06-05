//! The in-memory model: symbol tables plus the `post_*` helpers that translate
//! FlatZinc predicates into `qayd` propagators.

use std::collections::HashMap;

use qayd::constraints::count::{cardinality, count};
use qayd::constraints::graph::circuit;
use qayd::constraints::intension::intension;
use qayd::constraints::lex::lex;
use qayd::constraints::linear::{linear, Relation};
use qayd::constraints::primitives::{element, maximum, minimum, ordered};
use qayd::constraints::scheduling::{cumulative_var, no_overlap};
use qayd::constraints::table::{extension, regular, Dfa};
use qayd::expr::{self, Expr};
use qayd::{Solver, VarId};

use crate::parse::parse_int_set;
use crate::text::{bracket_items, require, split_args};

pub(crate) const UNBOUNDED_LO: i32 = -1_000_000;
pub(crate) const UNBOUNDED_HI: i32 = 1_000_000;

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
    Range(i32, i32),
    Set(Vec<i32>),
}

/// Symbol tables and the solver under construction.
pub(crate) struct Model {
    pub(crate) solver: Solver,
    pub(crate) vars: HashMap<String, VarId>,
    pub(crate) arrays: HashMap<String, Vec<VarId>>,
    pub(crate) ints: HashMap<String, i32>,
    pub(crate) int_arrays: HashMap<String, Vec<i32>>,
    /// Set variables, represented by their characteristic vector: a `0/1`
    /// membership variable for each value in the set's universe.
    pub(crate) set_vars: HashMap<String, HashMap<i32, VarId>>,
    pub(crate) search: Vec<VarId>,
    pub(crate) names: Vec<(String, VarId)>,
    /// `(minimizing, objective var)`.
    pub(crate) objective: Option<(bool, VarId)>,
}

impl Model {
    pub(crate) fn new() -> Self {
        Self {
            solver: Solver::new(),
            vars: HashMap::new(),
            arrays: HashMap::new(),
            ints: HashMap::new(),
            int_arrays: HashMap::new(),
            set_vars: HashMap::new(),
            search: Vec::new(),
            names: Vec::new(),
            objective: None,
        }
    }

    pub(crate) fn new_var(&mut self, name: String, domain: &FznDomain) -> VarId {
        let var = match domain {
            FznDomain::Range(lo, hi) => self.solver.new_var_range(*lo, *hi),
            FznDomain::Set(values) => self.solver.new_var_set(values),
        };
        self.vars.insert(name.clone(), var);
        self.search.push(var);
        self.names.push((name, var));
        var
    }

    pub(crate) fn constant(&mut self, value: i32) -> VarId {
        self.solver.new_var_set(&[value])
    }

    pub(crate) fn int_atom(&self, s: &str) -> Result<i32, String> {
        let s = s.trim();
        match s {
            "true" => Ok(1),
            "false" => Ok(0),
            _ => s.parse().or_else(|_| self.ints.get(s).copied().ok_or_else(|| format!("unknown integer `{s}`"))),
        }
    }

    pub(crate) fn var_atom(&mut self, s: &str) -> Result<VarId, String> {
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

    pub(crate) fn var_list(&mut self, s: &str) -> Result<Vec<VarId>, String> {
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
        linear(&mut self.solver, &[1, -1], &[x, y], rel, 0);
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
        linear(&mut self.solver, &coeffs, &vars, rel, rhs);
        Ok(())
    }

    pub(crate) fn post_element(&mut self, args: &[String], var_array: bool) -> Result<(), String> {
        require(args.len() == 3, "element predicate expects 3 arguments")?;
        let index = self.var_atom(&args[0])?;
        let array =
            if var_array { self.var_list(&args[1])? } else { self.int_list(&args[1])?.into_iter().map(|v| self.constant(v)).collect() };
        let value = self.var_atom(&args[2])?;
        let zero = self.solver.new_var_range(0, array.len() as i32 - 1);
        linear(&mut self.solver, &[1, -1], &[index, zero], Relation::Eq, 1);
        element(&mut self.solver, &array, zero, value);
        Ok(())
    }

    pub(crate) fn post_gecode_element(&mut self, args: &[String]) -> Result<(), String> {
        require(args.len() == 4, "gecode_int_element expects 4 arguments")?;
        let index = self.var_atom(&args[0])?;
        let offset = self.int_atom(&args[1])?;
        let array = self.var_list(&args[2])?;
        let value = self.var_atom(&args[3])?;
        let zero = self.solver.new_var_range(0, array.len() as i32 - 1);
        linear(&mut self.solver, &[1, -1], &[index, zero], Relation::Eq, offset as i64);
        element(&mut self.solver, &array, zero, value);
        Ok(())
    }

    pub(crate) fn post_count(&mut self, args: &[String], rel: Relation) -> Result<(), String> {
        require(args.len() == 3, "count predicate expects 3 arguments")?;
        let vars = self.var_list(&args[0])?;
        // Fast path: a constant value and target use the dedicated propagator.
        if let (Ok(value), Ok(target)) = (self.int_atom(&args[1]), self.int_atom(&args[2])) {
            count(&mut self.solver, &vars, value, rel, target as i64);
            return Ok(());
        }
        // General form `#{ i : vars[i] == value } <rel> target` with a variable
        // value and/or target, decomposed through `intension`.
        let value = self.atom_expr(&args[1])?;
        let target = self.atom_expr(&args[2])?;
        let occ: Vec<Expr> = vars.iter().map(|&v| expr::eq(expr::var(v), value.clone())).collect();
        intension(&mut self.solver, cmp_expr(rel, expr::add(occ), target));
        Ok(())
    }

    pub(crate) fn post_extremum(&mut self, args: &[String], is_min: bool) -> Result<(), String> {
        require(args.len() == 2, "array extremum predicate expects 2 arguments")?;
        let target = self.var_atom(&args[0])?;
        let vars = self.var_list(&args[1])?;
        if is_min {
            minimum(&mut self.solver, target, &vars);
        } else {
            maximum(&mut self.solver, target, &vars);
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
        intension(&mut self.solver, expr::or(terms));
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
        intension(&mut self.solver, formula);
        Ok(())
    }

    /// Post `target = value`.
    pub(crate) fn post_def(&mut self, target_s: &str, value: Expr) -> Result<(), String> {
        let t = self.atom_expr(target_s)?;
        intension(&mut self.solver, expr::eq(t, value));
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
                intension(&mut self.solver, member);
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
    pub(crate) fn post_set_var_in(&mut self, mem: Option<VarId>, r: Option<&str>, mode: Option<Reif>) -> Result<(), String> {
        let base = match mem {
            Some(v) => expr::var(v),
            None => expr::int(0),
        };
        match mode {
            None => {
                intension(&mut self.solver, base);
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
        extension(&mut self.solver, &vars, &tuples, true);
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
        regular(&mut self.solver, &vars, dfa);
        Ok(())
    }

    /// `gecode_circuit(offset, x)`: Hamiltonian circuit with `offset`-based successors.
    pub(crate) fn post_circuit(&mut self, args: &[String]) -> Result<(), String> {
        require(args.len() == 2, "circuit expects 2 arguments")?;
        let offset = self.int_atom(&args[0])?;
        let succ = self.var_list(&args[1])?;
        if offset == 0 {
            circuit(&mut self.solver, &succ);
        } else {
            let n = succ.len();
            let shifted: Vec<VarId> = (0..n).map(|_| self.solver.new_var_range(0, n as i32 - 1)).collect();
            for k in 0..n {
                linear(&mut self.solver, &[1, -1], &[succ[k], shifted[k]], Relation::Eq, offset as i64);
            }
            circuit(&mut self.solver, &shifted);
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
        cumulative_var(&mut self.solver, &starts, &durations, &heights, capacity);
        Ok(())
    }

    /// `gecode_schedule_unary(x, p)` -> `no_overlap` with constant durations.
    pub(crate) fn post_schedule_unary(&mut self, args: &[String]) -> Result<(), String> {
        require(args.len() == 2, "schedule_unary expects 2 arguments")?;
        let starts = self.var_list(&args[0])?;
        let durations: Vec<i64> = self.int_list(&args[1])?.into_iter().map(i64::from).collect();
        require(starts.len() == durations.len(), "schedule_unary length mismatch")?;
        no_overlap(&mut self.solver, &starts, &durations);
        Ok(())
    }

    /// `gecode_global_cardinality(x, cover, counts)` with variable occurrence counts.
    pub(crate) fn post_gcc_counts(&mut self, args: &[String]) -> Result<(), String> {
        require(args.len() == 3, "global_cardinality expects 3 arguments")?;
        let vars = self.var_list(&args[0])?;
        let cover = self.int_list(&args[1])?;
        let counts = self.var_list(&args[2])?;
        require(cover.len() == counts.len(), "global_cardinality cover/counts mismatch")?;
        for (j, &val) in cover.iter().enumerate() {
            let terms: Vec<Expr> = vars.iter().map(|&v| expr::eq(expr::var(v), expr::int(val as i64))).collect();
            intension(&mut self.solver, expr::eq(expr::add(terms), expr::var(counts[j])));
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
        cardinality(&mut self.solver, &vars, &cover, &low, &up, closed);
        Ok(())
    }

    /// `gecode_bin_packing_load(l, bin, w, minIndex)`: per-bin load equals the
    /// sum of the weights of the items assigned to it.
    pub(crate) fn post_bin_packing_load(&mut self, args: &[String]) -> Result<(), String> {
        require(args.len() == 4, "bin_packing_load expects 4 arguments")?;
        let load = self.var_list(&args[0])?;
        let bin = self.var_list(&args[1])?;
        let weight = self.int_list(&args[2])?;
        let min_index = self.int_atom(&args[3])?;
        require(bin.len() == weight.len(), "bin_packing_load bin/weight mismatch")?;
        for (j, &l) in load.iter().enumerate() {
            let target = min_index + j as i32;
            let terms: Vec<Expr> = bin
                .iter()
                .zip(&weight)
                .map(|(&b, &w)| expr::mul(vec![expr::int(w as i64), expr::eq(expr::var(b), expr::int(target as i64))]))
                .collect();
            intension(&mut self.solver, expr::eq(expr::add(terms), expr::var(l)));
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
        let m = self.solver.new_var_range(UNBOUNDED_LO, UNBOUNDED_HI);
        maximum(&mut self.solver, m, &xs);
        let pos = self.solver.new_var_range(0, n as i32 - 1);
        linear(&mut self.solver, &[1, -1], &[idx, pos], Relation::Eq, offset as i64);
        element(&mut self.solver, &xs, pos, m);
        // First maximum: any earlier element must be strictly smaller.
        for (j, &xj) in xs.iter().enumerate() {
            intension(&mut self.solver, expr::imp(expr::gt(expr::var(pos), expr::int(j as i64)), expr::ne(expr::var(xj), expr::var(m))));
        }
        Ok(())
    }

    /// `array_int_lq` / `array_int_lt` (and bool variants) -> lexicographic order.
    pub(crate) fn post_array_lex(&mut self, args: &[String], strict: bool) -> Result<(), String> {
        require(args.len() == 2, "array lex expects 2 arguments")?;
        let x = self.var_list(&args[0])?;
        let y = self.var_list(&args[1])?;
        require(x.len() == y.len(), "array lex length mismatch")?;
        lex(&mut self.solver, &x, &y, strict);
        Ok(())
    }

    /// `fzn_increasing_*` / `fzn_decreasing_*`.
    pub(crate) fn post_ordered(&mut self, args: &[String], rel: Relation) -> Result<(), String> {
        require(args.len() == 1, "increasing/decreasing expects 1 argument")?;
        let vars = self.var_list(&args[0])?;
        if vars.len() > 1 {
            ordered(&mut self.solver, &vars, rel);
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
