use std::collections::HashMap;

use qayd::constraints::count::count;
use qayd::constraints::intension::intension;
use qayd::constraints::linear::{linear, Relation};
use qayd::constraints::primitives::{all_different, element};
use qayd::expr;
use qayd::{first_solution, maximize, minimize, Solver, VarId};

const UNBOUNDED_LO: i32 = -1_000_000;
const UNBOUNDED_HI: i32 = 1_000_000;

struct Model {
    solver: Solver,
    vars: HashMap<String, VarId>,
    arrays: HashMap<String, Vec<VarId>>,
    ints: HashMap<String, i32>,
    int_arrays: HashMap<String, Vec<i32>>,
    search: Vec<VarId>,
    names: Vec<(String, VarId)>,
    objective: Option<(bool, VarId)>,
}

impl Model {
    fn new() -> Self {
        Self {
            solver: Solver::new(),
            vars: HashMap::new(),
            arrays: HashMap::new(),
            ints: HashMap::new(),
            int_arrays: HashMap::new(),
            search: Vec::new(),
            names: Vec::new(),
            objective: None,
        }
    }

    fn new_var(&mut self, name: String, lo: i32, hi: i32) -> VarId {
        let var = self.solver.new_var_range(lo, hi);
        self.vars.insert(name.clone(), var);
        self.search.push(var);
        self.names.push((name, var));
        var
    }

    fn constant(&mut self, value: i32) -> VarId {
        self.solver.new_var_set(&[value])
    }

    fn int_atom(&self, s: &str) -> Result<i32, String> {
        let s = s.trim();
        match s {
            "true" => Ok(1),
            "false" => Ok(0),
            _ => s.parse().or_else(|_| self.ints.get(s).copied().ok_or_else(|| format!("unknown integer `{s}`"))),
        }
    }

    fn var_atom(&mut self, s: &str) -> Result<VarId, String> {
        let s = s.trim();
        if let Ok(value) = self.int_atom(s) {
            return Ok(self.constant(value));
        }
        self.vars.get(s).copied().ok_or_else(|| format!("unknown variable `{s}`"))
    }

    fn int_list(&self, s: &str) -> Result<Vec<i32>, String> {
        let s = s.trim();
        if let Some(items) = bracket_items(s) {
            return items.iter().map(|x| self.int_atom(x)).collect();
        }
        self.int_arrays.get(s).cloned().ok_or_else(|| format!("unknown integer array `{s}`"))
    }

    fn var_list(&mut self, s: &str) -> Result<Vec<VarId>, String> {
        let s = s.trim();
        if let Some(items) = bracket_items(s) {
            return items.iter().map(|x| self.var_atom(x)).collect();
        }
        if let Some(vars) = self.arrays.get(s) {
            return Ok(vars.clone());
        }
        Ok(vec![self.var_atom(s)?])
    }

    fn post_cmp(&mut self, a: &str, b: &str, rel: Relation) -> Result<(), String> {
        let x = self.var_atom(a)?;
        let y = self.var_atom(b)?;
        linear(&mut self.solver, &[1, -1], &[x, y], rel, 0);
        Ok(())
    }

    fn post_linear(&mut self, args: &[String], rel: Relation) -> Result<(), String> {
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

    fn post_element(&mut self, args: &[String], var_array: bool) -> Result<(), String> {
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

    fn post_gecode_element(&mut self, args: &[String]) -> Result<(), String> {
        require(args.len() == 4, "gecode_int_element expects 4 arguments")?;
        let index = self.var_atom(&args[0])?;
        let offset = self.int_atom(&args[1])?;
        let array = self.int_list(&args[2])?.into_iter().map(|v| self.constant(v)).collect::<Vec<_>>();
        let value = self.var_atom(&args[3])?;
        let zero = self.solver.new_var_range(0, array.len() as i32 - 1);
        linear(&mut self.solver, &[1, -1], &[index, zero], Relation::Eq, offset as i64);
        element(&mut self.solver, &array, zero, value);
        Ok(())
    }

    fn post_count(&mut self, args: &[String], rel: Relation) -> Result<(), String> {
        require(args.len() == 3, "count predicate expects 3 arguments")?;
        let vars = self.var_list(&args[0])?;
        let value = self.int_atom(&args[1])?;
        let target = self.int_atom(&args[2])?;
        count(&mut self.solver, &vars, value, rel, target as i64);
        Ok(())
    }

    fn post_bool_clause(&mut self, args: &[String]) -> Result<(), String> {
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
}

fn require(ok: bool, msg: &str) -> Result<(), String> {
    if ok {
        Ok(())
    } else {
        Err(msg.to_string())
    }
}

fn strip_annotations(s: &str) -> &str {
    s.split("::").next().unwrap_or(s).trim()
}

fn strip_leading_solve_annotations(mut s: &str) -> &str {
    loop {
        s = s.trim_start();
        let Some(rest) = s.strip_prefix("::") else { return s };
        s = rest.trim_start();
        let mut depth = 0i32;
        let mut end = s.len();
        for (i, ch) in s.char_indices() {
            match ch {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                _ if ch.is_whitespace() && depth == 0 => {
                    end = i;
                    break;
                }
                _ => {}
            }
        }
        s = &s[end..];
    }
}

fn bracket_items(s: &str) -> Option<Vec<String>> {
    let s = s.trim();
    let inner = s.strip_prefix('[')?.strip_suffix(']')?;
    if inner.trim().is_empty() {
        return Some(Vec::new());
    }
    Some(split_args(inner))
}

fn split_args(s: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, ch) in s.char_indices() {
        match ch {
            '[' | '(' => depth += 1,
            ']' | ')' => depth -= 1,
            ',' if depth == 0 => {
                args.push(s[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    args.push(s[start..].trim().to_string());
    args
}

fn parse_bound(model: &Model, s: &str) -> Result<(i32, i32), String> {
    let s = s.trim();
    if s == "bool" {
        return Ok((0, 1));
    }
    if s == "int" {
        return Ok((UNBOUNDED_LO, UNBOUNDED_HI));
    }
    let (lo, hi) = s.split_once("..").ok_or_else(|| format!("unsupported domain `{s}`"))?;
    Ok((model.int_atom(lo)?, model.int_atom(hi)?))
}

fn array_len(model: &Model, left: &str) -> Result<usize, String> {
    let spec = left.split_once('[').and_then(|(_, r)| r.split_once(']')).map(|(x, _)| x).ok_or("array without index set")?;
    require(!spec.contains(','), "only one-dimensional arrays are supported")?;
    let (lo, hi) = spec.split_once("..").ok_or("array index must be an interval")?;
    Ok((model.int_atom(hi)? - model.int_atom(lo)? + 1) as usize)
}

fn clean_name(s: &str) -> String {
    strip_annotations(s).split('=').next().unwrap_or(s).trim().to_string()
}

fn parse_decl(model: &mut Model, stmt: &str) -> Result<(), String> {
    if stmt.starts_with("array ") {
        let (left, right) = stmt.split_once(':').ok_or("bad array declaration")?;
        let name = clean_name(right);
        if let Some((_, value)) = right.split_once('=') {
            if left.contains(" of var ") {
                let vars = model.var_list(strip_annotations(value))?;
                model.arrays.insert(name, vars);
            } else {
                let values = model.int_list(strip_annotations(value))?;
                model.int_arrays.insert(name, values);
            }
            return Ok(());
        }
        let n = array_len(model, left)?;
        let domain = left.rsplit_once(" of var ").map(|(_, d)| d).ok_or("only var arrays without assignment are supported")?;
        let (lo, hi) = parse_bound(model, domain)?;
        let vars = (1..=n).map(|i| model.new_var(format!("{name}[{i}]"), lo, hi)).collect();
        model.arrays.insert(name, vars);
        return Ok(());
    }
    let (left, right) = stmt.split_once(':').ok_or("bad declaration")?;
    if left.trim() == "int" || left.trim() == "bool" {
        let (name, value) = right.split_once('=').ok_or("constant declaration without value")?;
        model.ints.insert(clean_name(name), model.int_atom(strip_annotations(value))?);
        return Ok(());
    }
    let domain = left.trim().strip_prefix("var ").ok_or("unsupported declaration")?;
    let name = clean_name(right);
    let (lo, hi) = parse_bound(model, domain)?;
    let var = model.new_var(name, lo, hi);
    if let Some((_, value)) = right.split_once('=') {
        let fixed = model.var_atom(strip_annotations(value))?;
        linear(&mut model.solver, &[1, -1], &[var, fixed], Relation::Eq, 0);
    }
    Ok(())
}

fn parse_constraint(model: &mut Model, stmt: &str) -> Result<(), String> {
    let body = strip_annotations(stmt.strip_prefix("constraint").unwrap()).trim();
    let open = body.find('(').ok_or("constraint without arguments")?;
    let name = body[..open].trim();
    let args = split_args(body[open + 1..].trim_end_matches(')'));
    match name {
        "int_eq" => model.post_cmp(&args[0], &args[1], Relation::Eq),
        "int_ne" => model.post_cmp(&args[0], &args[1], Relation::Ne),
        "int_le" => model.post_cmp(&args[0], &args[1], Relation::Le),
        "int_lt" => model.post_cmp(&args[0], &args[1], Relation::Lt),
        "int_ge" => model.post_cmp(&args[0], &args[1], Relation::Ge),
        "int_gt" => model.post_cmp(&args[0], &args[1], Relation::Gt),
        "int_lin_eq" => model.post_linear(&args, Relation::Eq),
        "int_lin_ne" => model.post_linear(&args, Relation::Ne),
        "int_lin_le" => model.post_linear(&args, Relation::Le),
        "int_lin_lt" => model.post_linear(&args, Relation::Lt),
        "int_lin_ge" => model.post_linear(&args, Relation::Ge),
        "int_lin_gt" => model.post_linear(&args, Relation::Gt),
        "all_different_int" | "fzn_all_different_int" => {
            let vars = model.var_list(&args[0])?;
            all_different(&mut model.solver, &vars);
            Ok(())
        }
        "array_int_element" => model.post_element(&args, false),
        "array_var_int_element" => model.post_element(&args, true),
        "gecode_int_element" => model.post_gecode_element(&args),
        "fzn_count_eq" => model.post_count(&args, Relation::Eq),
        "fzn_count_ne" | "fzn_count_neq" => model.post_count(&args, Relation::Ne),
        "fzn_count_le" | "fzn_count_leq" => model.post_count(&args, Relation::Le),
        "fzn_count_lt" => model.post_count(&args, Relation::Lt),
        "fzn_count_ge" | "fzn_count_geq" => model.post_count(&args, Relation::Ge),
        "fzn_count_gt" => model.post_count(&args, Relation::Gt),
        "bool_clause" => model.post_bool_clause(&args),
        other => Err(format!("unsupported FlatZinc predicate `{other}`")),
    }
}

fn parse_solve(model: &mut Model, stmt: &str) -> Result<(), String> {
    let stmt = strip_annotations(strip_leading_solve_annotations(stmt.strip_prefix("solve").unwrap())).trim();
    if stmt == "satisfy" {
        return Ok(());
    }
    if let Some(obj) = stmt.strip_prefix("minimize ") {
        model.objective = Some((true, model.var_atom(obj)?));
        return Ok(());
    }
    if let Some(obj) = stmt.strip_prefix("maximize ") {
        model.objective = Some((false, model.var_atom(obj)?));
        return Ok(());
    }
    Err(format!("unsupported solve item `{stmt}`"))
}

fn parse(input: &str) -> Result<Model, String> {
    let mut model = Model::new();
    let mut text = String::new();
    for line in input.lines() {
        text.push_str(line.split('%').next().unwrap_or(""));
        text.push('\n');
    }
    for raw in text.split(';') {
        let stmt = raw.trim();
        if stmt.is_empty() || stmt.starts_with("predicate ") || stmt.starts_with("output ") {
            continue;
        }
        if stmt.starts_with("constraint ") {
            parse_constraint(&mut model, stmt)?;
        } else if stmt.starts_with("solve ") {
            parse_solve(&mut model, stmt)?;
        } else {
            parse_decl(&mut model, stmt)?;
        }
    }
    Ok(model)
}

fn print_solution(model: &Model, assignment: &[i32]) {
    for (name, var) in &model.names {
        if let Some(pos) = model.search.iter().position(|&v| v == *var) {
            println!("v {name} = {}", assignment[pos]);
        }
    }
}

fn solve(mut model: Model) {
    match model.objective {
        Some((true, obj)) => match minimize(&mut model.solver, &model.search, obj) {
            Some((solution, value)) => {
                println!("o {value}");
                println!("s OPTIMUM FOUND");
                print_solution(&model, &solution);
            }
            None => println!("s UNSATISFIABLE"),
        },
        Some((false, obj)) => match maximize(&mut model.solver, &model.search, obj) {
            Some((solution, value)) => {
                println!("o {value}");
                println!("s OPTIMUM FOUND");
                print_solution(&model, &solution);
            }
            None => println!("s UNSATISFIABLE"),
        },
        None => match first_solution(&mut model.solver, &model.search) {
            Some(solution) => {
                println!("s SATISFIABLE");
                print_solution(&model, &solution);
            }
            None => println!("s UNSATISFIABLE"),
        },
    }
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: qayd-fzn <model.fzn>");
        std::process::exit(1);
    });
    let input = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    match parse(&input) {
        Ok(model) => solve(model),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    }
}
