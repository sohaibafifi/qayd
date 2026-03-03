//! Sub-parsers shared by the XCSP3 builder: domains, variable references,
//! intension expressions (functional notation), and conditions.

use std::collections::HashMap;

use crate::constraints::linear::Relation;
use crate::expr::{self, Expr};
use crate::ids::VarId;

/// A multi-dimensional array of variables, flattened row-major.
pub struct ArrayInfo {
    /// Dimension sizes (row-major).
    pub dims: Vec<usize>,
    /// Element variables in row-major order.
    pub flat: Vec<VarId>,
}

/// Maps XCSP identifiers to solver variables: scalars and (multi-dim) arrays.
#[derive(Default)]
pub struct SymTab {
    /// Scalar variables, by id.
    pub scalars: HashMap<String, VarId>,
    /// Array variables, by array id.
    pub arrays: HashMap<String, ArrayInfo>,
}

/// One bracket of an array reference.
enum Idx {
    One(usize),
    Range(usize, usize),
    All,
}

/// Row-major strides for the given dimensions.
fn strides(dims: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; dims.len()];
    for d in (0..dims.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * dims[d + 1];
    }
    s
}

/// Split `x[0][]` into (`x`, `[One(0), All]`).
fn parse_brackets(tok: &str) -> Result<(String, Vec<Idx>), String> {
    let open = tok
        .find('[')
        .ok_or_else(|| format!("not an array ref `{tok}`"))?;
    let name = tok[..open].to_string();
    let mut specs = Vec::new();
    let mut cur = &tok[open..];
    while !cur.is_empty() {
        if !cur.starts_with('[') {
            return Err(format!("bad reference `{tok}`"));
        }
        let close = cur
            .find(']')
            .ok_or_else(|| format!("bad reference `{tok}`"))?;
        let inside = cur[1..close].trim();
        let idx = if inside.is_empty() {
            Idx::All
        } else if let Some((a, b)) = inside.split_once("..") {
            Idx::Range(
                a.trim().parse().map_err(|_| "bad index")?,
                b.trim().parse().map_err(|_| "bad index")?,
            )
        } else {
            Idx::One(inside.parse().map_err(|_| "bad index")?)
        };
        specs.push(idx);
        cur = &cur[close + 1..];
    }
    Ok((name, specs))
}

/// The flat (row-major) indices selected by `specs` over `dims`.
fn flat_indices(dims: &[usize], specs: &[Idx]) -> Result<Vec<usize>, String> {
    if specs.len() != dims.len() {
        return Err(format!("expected {} indices, got {}", dims.len(), specs.len()));
    }
    let st = strides(dims);
    let mut lo = Vec::with_capacity(specs.len());
    let mut sizes = Vec::with_capacity(specs.len());
    for (d, spec) in specs.iter().enumerate() {
        let (l, h) = match *spec {
            Idx::One(k) => (k, k),
            Idx::All => (0, dims[d].saturating_sub(1)),
            Idx::Range(a, b) => (a, b),
        };
        if h >= dims[d] || l > h {
            return Err("array index out of range".to_string());
        }
        lo.push(l);
        sizes.push(h - l + 1);
    }
    let total: usize = sizes.iter().product();
    let mut out = Vec::with_capacity(total);
    for t in 0..total {
        let mut rem = t;
        let mut flat = 0;
        for d in (0..specs.len()).rev() {
            let c = lo[d] + rem % sizes[d];
            rem /= sizes[d];
            flat += c * st[d];
        }
        out.push(flat);
    }
    Ok(out)
}

/// Expand a bracketed reference into the variables it selects (row-major).
fn expand(info: &ArrayInfo, specs: &[Idx]) -> Result<Vec<VarId>, String> {
    Ok(flat_indices(&info.dims, specs)?
        .into_iter()
        .map(|i| info.flat[i])
        .collect())
}

/// Flat indices selected by a reference token (e.g. `a[3]`, `a[1..4]`) over an
/// array of the given dimensions. Used while *declaring* an array (before it is
/// in the symbol table) to apply per-element `<domain for="...">` specs.
pub fn expand_indices(dims: &[usize], tok: &str) -> Result<Vec<usize>, String> {
    let (_name, specs) = parse_brackets(tok)?;
    flat_indices(dims, &specs)
}

impl SymTab {
    /// Resolve a single reference token like `x`, `a[3]`, `m[2][1]`.
    pub fn resolve_one(&self, tok: &str) -> Result<VarId, String> {
        if tok.contains('[') {
            let (name, specs) = parse_brackets(tok)?;
            let info = self
                .arrays
                .get(&name)
                .ok_or_else(|| format!("unknown array `{name}`"))?;
            let vs = expand(info, &specs)?;
            if vs.len() != 1 {
                return Err(format!("`{tok}` does not denote a single variable"));
            }
            Ok(vs[0])
        } else {
            self.scalars
                .get(tok)
                .copied()
                .ok_or_else(|| format!("unknown variable `{tok}`"))
        }
    }
}

/// Parse a domain spec like `0..8` or `1 3 5..7` into inclusive intervals.
pub fn parse_intervals(spec: &str) -> Result<Vec<(i32, i32)>, String> {
    let mut out = Vec::new();
    for tok in spec.split_whitespace() {
        if let Some((a, b)) = tok.split_once("..") {
            let lo = a.trim().parse().map_err(|_| format!("bad bound `{a}`"))?;
            let hi = b.trim().parse().map_err(|_| format!("bad bound `{b}`"))?;
            out.push((lo, hi));
        } else {
            let v = tok.parse().map_err(|_| format!("bad value `{tok}`"))?;
            out.push((v, v));
        }
    }
    if out.is_empty() {
        return Err("empty domain".to_string());
    }
    Ok(out)
}

/// Expand intervals to an explicit, sorted, de-duplicated value list.
pub fn interval_values(intervals: &[(i32, i32)]) -> Vec<i32> {
    let mut v: Vec<i32> = intervals.iter().flat_map(|&(lo, hi)| lo..=hi).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Expand a whitespace-separated reference list (`x[][] y z[1..3] m[2][]`) into
/// the underlying variables, in row-major order.
pub fn parse_var_refs(s: &str, sym: &SymTab) -> Result<Vec<VarId>, String> {
    let mut out = Vec::new();
    for tok in s.split_whitespace() {
        if tok.contains('[') {
            let (name, specs) = parse_brackets(tok)?;
            let info = sym
                .arrays
                .get(&name)
                .ok_or_else(|| format!("unknown array `{name}`"))?;
            out.extend(expand(info, &specs)?);
        } else if let Some(&v) = sym.scalars.get(tok) {
            out.push(v);
        } else if let Some(info) = sym.arrays.get(tok) {
            out.extend(info.flat.iter().copied());
        } else {
            return Err(format!("unknown reference `{tok}`"));
        }
    }
    Ok(out)
}

// --- intension expression parser (functional notation) ---

#[derive(Debug, PartialEq, Eq)]
enum Tok {
    LParen,
    RParen,
    Comma,
    Atom(String),
}

fn tokenize(s: &str) -> Result<Vec<Tok>, String> {
    let mut toks = Vec::new();
    let mut atom = String::new();
    let flush = |atom: &mut String, toks: &mut Vec<Tok>| {
        if !atom.is_empty() {
            toks.push(Tok::Atom(std::mem::take(atom)));
        }
    };
    for ch in s.chars() {
        match ch {
            '(' => {
                flush(&mut atom, &mut toks);
                toks.push(Tok::LParen);
            }
            ')' => {
                flush(&mut atom, &mut toks);
                toks.push(Tok::RParen);
            }
            ',' => {
                flush(&mut atom, &mut toks);
                toks.push(Tok::Comma);
            }
            c if c.is_whitespace() => flush(&mut atom, &mut toks),
            c => atom.push(c),
        }
    }
    flush(&mut atom, &mut toks);
    Ok(toks)
}

/// Parse an intension expression in functional notation (`eq(add(x,y),10)`).
pub fn parse_expr(s: &str, sym: &SymTab) -> Result<Expr, String> {
    let toks = tokenize(s)?;
    let mut pos = 0;
    let e = parse_node(&toks, &mut pos, sym)?;
    if pos != toks.len() {
        return Err("trailing tokens in expression".to_string());
    }
    Ok(e)
}

fn parse_node(toks: &[Tok], pos: &mut usize, sym: &SymTab) -> Result<Expr, String> {
    let tok = toks.get(*pos).ok_or("unexpected end of expression")?;
    *pos += 1;
    let name = match tok {
        Tok::Atom(a) => a.clone(),
        _ => return Err("expected atom".to_string()),
    };
    if toks.get(*pos) == Some(&Tok::LParen) {
        *pos += 1;
        let mut args = Vec::new();
        if toks.get(*pos) != Some(&Tok::RParen) {
            loop {
                args.push(parse_node(toks, pos, sym)?);
                match toks.get(*pos) {
                    Some(Tok::Comma) => *pos += 1,
                    Some(Tok::RParen) => break,
                    _ => return Err("expected , or )".to_string()),
                }
            }
        }
        *pos += 1; // consume ')'
        build_op(&name, args)
    } else {
        leaf(&name, sym)
    }
}

fn leaf(name: &str, sym: &SymTab) -> Result<Expr, String> {
    if let Ok(n) = name.parse::<i64>() {
        Ok(expr::int(n))
    } else {
        Ok(expr::var(sym.resolve_one(name)?))
    }
}

fn two(mut args: Vec<Expr>, name: &str) -> Result<(Expr, Expr), String> {
    if args.len() != 2 {
        return Err(format!("`{name}` expects 2 arguments"));
    }
    let b = args.pop().unwrap();
    let a = args.pop().unwrap();
    Ok((a, b))
}

fn one(mut args: Vec<Expr>, name: &str) -> Result<Expr, String> {
    if args.len() != 1 {
        return Err(format!("`{name}` expects 1 argument"));
    }
    Ok(args.pop().unwrap())
}

fn build_op(name: &str, args: Vec<Expr>) -> Result<Expr, String> {
    Ok(match name {
        "add" => expr::add(args),
        "mul" => expr::mul(args),
        "min" => expr::min_of(args),
        "max" => expr::max_of(args),
        "and" => expr::and(args),
        "or" => expr::or(args),
        "neg" => expr::neg(one(args, name)?),
        "abs" => expr::abs(one(args, name)?),
        "not" => expr::not(one(args, name)?),
        "sub" => {
            let (a, b) = two(args, name)?;
            expr::sub(a, b)
        }
        "div" => {
            let (a, b) = two(args, name)?;
            expr::div(a, b)
        }
        "mod" => {
            let (a, b) = two(args, name)?;
            expr::rem(a, b)
        }
        "dist" => {
            let (a, b) = two(args, name)?;
            expr::abs(expr::sub(a, b))
        }
        "eq" => {
            let (a, b) = two(args, name)?;
            expr::eq(a, b)
        }
        "ne" => {
            let (a, b) = two(args, name)?;
            expr::ne(a, b)
        }
        "lt" => {
            let (a, b) = two(args, name)?;
            expr::lt(a, b)
        }
        "le" => {
            let (a, b) = two(args, name)?;
            expr::le(a, b)
        }
        "gt" => {
            let (a, b) = two(args, name)?;
            expr::gt(a, b)
        }
        "ge" => {
            let (a, b) = two(args, name)?;
            expr::ge(a, b)
        }
        "imp" => {
            let (a, b) = two(args, name)?;
            expr::imp(a, b)
        }
        "iff" => {
            let (a, b) = two(args, name)?;
            expr::iff(a, b)
        }
        "xor" => {
            let (a, b) = two(args, name)?;
            expr::not(expr::iff(a, b))
        }
        "if" => {
            if args.len() != 3 {
                return Err("`if` expects 3 arguments".to_string());
            }
            let mut it = args.into_iter();
            expr::ite(it.next().unwrap(), it.next().unwrap(), it.next().unwrap())
        }
        other => return Err(format!("unsupported operator `{other}`")),
    })
}

// --- conditions: (op, operand) ---

/// The right-hand side of a condition: a constant or a variable.
pub enum Operand {
    /// A constant operand.
    Const(i64),
    /// A variable operand.
    Var(VarId),
}

/// A parsed `(op, operand)` condition.
pub struct Condition {
    /// The comparison operator.
    pub rel: Relation,
    /// The right-hand side.
    pub operand: Operand,
}

/// Parse a condition like `(le,10)` or `(eq,y)`.
pub fn parse_condition(s: &str, sym: &SymTab) -> Result<Condition, String> {
    let inner = s
        .trim()
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| format!("bad condition `{s}`"))?;
    let (op, rhs) = inner
        .split_once(',')
        .ok_or_else(|| format!("bad condition `{s}`"))?;
    let rel = parse_rel(op.trim())?;
    let rhs = rhs.trim();
    let operand = if let Ok(n) = rhs.parse::<i64>() {
        Operand::Const(n)
    } else {
        Operand::Var(sym.resolve_one(rhs)?)
    };
    Ok(Condition { rel, operand })
}

/// Parse a relation keyword (`lt`, `le`, `ge`, `gt`, `eq`, `ne`).
pub fn parse_rel(op: &str) -> Result<Relation, String> {
    Ok(match op {
        "lt" => Relation::Lt,
        "le" => Relation::Le,
        "ge" => Relation::Ge,
        "gt" => Relation::Gt,
        "eq" => Relation::Eq,
        "ne" => Relation::Ne,
        other => return Err(format!("unsupported operator `{other}`")),
    })
}
