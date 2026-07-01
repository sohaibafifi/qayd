//! Parser for the (restricted) OPB format used by the Pseudo-Boolean
//! Competition.
//!
//! Grammar (restricted / linear subset):
//!
//! ```text
//! <file>       ::= <comment>* [<objective>] <constraint>*
//! <comment>    ::= '*' ... end-of-line
//! <objective>  ::= ('min:' | 'max:') <term>* ';'
//! <constraint> ::= <term>* <op> <int> ';'
//! <op>         ::= '>=' | '<=' | '=' | '>' | '<'
//! <term>       ::= <int> <literal>            (linear)
//!                | <int> <literal> <literal>+ (non-linear product, flagged)
//! <literal>    ::= 'x'<pos-int> | '~x'<pos-int>
//! ```
//!
//! A leading header comment `* #variable= N #constraint= M` is recognised but
//! not required; the variable count is otherwise inferred from the highest
//! `x<i>` index seen.

/// Comparison operator of a linear constraint (subset of OPB operators).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// `>=`
    Ge,
    /// `<=`
    Le,
    /// `=`
    Eq,
    /// `>`
    Gt,
    /// `<`
    Lt,
}

/// One `coeff * literal` term. `var` is the 1-based OPB index; `negated` marks
/// a `~x` literal (value `1 - x`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Term {
    /// Integer coefficient.
    pub coeff: i64,
    /// 1-based variable index.
    pub var: usize,
    /// `true` for a `~x` literal.
    pub negated: bool,
}

/// One linear constraint `sum(terms) op rhs`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constraint {
    /// Left-hand-side terms.
    pub terms: Vec<Term>,
    /// Comparison operator.
    pub op: Op,
    /// Right-hand-side integer.
    pub rhs: i64,
}

/// A parsed OPB instance.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Opb {
    /// Number of variables (from header or inferred).
    pub n_vars: usize,
    /// Objective terms and whether it is a maximisation. `None` = pure
    /// satisfaction instance.
    pub objective: Option<Objective>,
    /// Linear constraints.
    pub constraints: Vec<Constraint>,
    /// Set when a non-linear (product) term was seen — the caller should report
    /// `s UNSUPPORTED`.
    pub nonlinear: bool,
}

/// Parsed objective.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Objective {
    /// `true` for `max:`, `false` for `min:`.
    pub maximize: bool,
    /// Objective terms.
    pub terms: Vec<Term>,
}

/// Parse error with a 1-based line number.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    /// 1-based source line.
    pub line: usize,
    /// Human-readable message.
    pub msg: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.msg)
    }
}

/// Parse an OPB instance from source text.
pub fn parse(input: &str) -> Result<Opb, ParseError> {
    let mut opb = Opb::default();
    let mut header_vars: Option<usize> = None;
    let mut max_var = 0usize;

    for (lineno, raw) in input.lines().enumerate() {
        let line = lineno + 1;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix('*') {
            // Header comment may carry `#variable= N`.
            if header_vars.is_none() {
                if let Some(n) = parse_header_vars(rest) {
                    header_vars = Some(n);
                }
            }
            continue;
        }

        let body = trimmed
            .strip_suffix(';')
            .map(str::trim)
            .ok_or_else(|| err(line, "statement must end with ';'"))?;

        if let Some(rest) = strip_objective_keyword(body) {
            if opb.objective.is_some() {
                return Err(err(line, "multiple objective lines"));
            }
            let (terms, nonlinear) = parse_terms(rest, line, &mut max_var)?;
            opb.nonlinear |= nonlinear;
            opb.objective = Some(Objective { maximize: rest.starts_with("max"), terms });
        } else {
            let (constraint, nonlinear) = parse_constraint(body, line, &mut max_var)?;
            opb.nonlinear |= nonlinear;
            opb.constraints.push(constraint);
        }
    }

    opb.n_vars = header_vars.unwrap_or(max_var).max(max_var);
    Ok(opb)
}

fn strip_objective_keyword(body: &str) -> Option<&str> {
    for kw in ["min:", "max:"] {
        if let Some(rest) = body.strip_prefix(kw) {
            return Some(rest.trim());
        }
    }
    None
}

/// Parse one constraint. Returns the constraint and whether it contained a
/// non-linear (product) term, which the caller flags at the model level.
fn parse_constraint(body: &str, line: usize, max_var: &mut usize) -> Result<(Constraint, bool), ParseError> {
    // Find the operator token (`>=`, `<=`, `=`, `>`, `<`).
    let (lhs, op, rhs_str) = split_on_op(body).ok_or_else(|| err(line, "missing comparison operator"))?;
    let (terms, nonlinear) = parse_terms(lhs, line, max_var)?;
    let rhs = parse_int(rhs_str.trim(), line)?;
    Ok((Constraint { terms, op, rhs }, nonlinear))
}

/// Split a constraint body into `(lhs, op, rhs)`. Operators are matched longest
/// first so `>=`/`<=` win over `>`/`<`.
fn split_on_op(body: &str) -> Option<(&str, Op, &str)> {
    for (tok, op) in [(">=", Op::Ge), ("<=", Op::Le), ("=", Op::Eq), (">", Op::Gt), ("<", Op::Lt)] {
        if let Some(idx) = body.find(tok) {
            let lhs = &body[..idx];
            let rhs = &body[idx + tok.len()..];
            return Some((lhs, op, rhs));
        }
    }
    None
}

/// Parse a whitespace-separated term list. Returns the terms and whether any
/// non-linear (product) term was encountered.
fn parse_terms(s: &str, line: usize, max_var: &mut usize) -> Result<(Vec<Term>, bool), ParseError> {
    let mut terms = Vec::new();
    let mut nonlinear = false;
    let mut tokens = s.split_whitespace().peekable();

    while let Some(tok) = tokens.next() {
        let coeff = parse_int(tok, line)?;
        // Read one literal.
        let lit = tokens.next().ok_or_else(|| err(line, "coefficient without literal"))?;
        let (var, negated) = parse_literal(lit, line, max_var)?;
        // A product term has extra literals before the next coefficient.
        while let Some(next) = tokens.peek() {
            if is_literal(next) {
                let extra = tokens.next().unwrap();
                let _ = parse_literal(extra, line, max_var)?;
                nonlinear = true;
            } else {
                break;
            }
        }
        terms.push(Term { coeff, var, negated });
    }
    Ok((terms, nonlinear))
}

fn is_literal(tok: &str) -> bool {
    let body = tok.strip_prefix('~').unwrap_or(tok);
    body.strip_prefix('x').is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
}

fn parse_literal(tok: &str, line: usize, max_var: &mut usize) -> Result<(usize, bool), ParseError> {
    let (negated, body) = match tok.strip_prefix('~') {
        Some(rest) => (true, rest),
        None => (false, tok),
    };
    let digits = body.strip_prefix('x').ok_or_else(|| err(line, format!("expected literal, got '{tok}'")))?;
    let idx: usize = digits.parse().map_err(|_| err(line, format!("bad variable index '{tok}'")))?;
    if idx == 0 {
        return Err(err(line, "variable index must be >= 1"));
    }
    *max_var = (*max_var).max(idx);
    Ok((idx, negated))
}

fn parse_int(tok: &str, line: usize) -> Result<i64, ParseError> {
    let tok = tok.strip_prefix('+').unwrap_or(tok);
    tok.parse::<i64>().map_err(|_| err(line, format!("expected integer, got '{tok}'")))
}

fn parse_header_vars(rest: &str) -> Option<usize> {
    // Look for a `#variable= N` field anywhere in the header comment.
    let mut toks = rest.split_whitespace().peekable();
    while let Some(tok) = toks.next() {
        if let Some(inline) = tok.strip_prefix("#variable=") {
            if let Ok(n) = inline.trim().parse::<usize>() {
                return Some(n);
            }
            // `#variable=` then a separate number token.
            if let Some(next) = toks.peek() {
                if let Ok(n) = next.parse::<usize>() {
                    return Some(n);
                }
            }
        } else if tok == "#variable=" {
            if let Some(next) = toks.next() {
                if let Ok(n) = next.parse::<usize>() {
                    return Some(n);
                }
            }
        }
    }
    None
}

fn err(line: usize, msg: impl Into<String>) -> ParseError {
    ParseError { line, msg: msg.into() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_satisfaction_instance() {
        let src = "* #variable= 3 #constraint= 2\n+1 x1 +1 x2 +1 x3 >= 2 ;\n-1 x1 +1 x3 = 0 ;\n";
        let opb = parse(src).unwrap();
        assert_eq!(opb.n_vars, 3);
        assert!(opb.objective.is_none());
        assert_eq!(opb.constraints.len(), 2);
        assert_eq!(opb.constraints[0].op, Op::Ge);
        assert_eq!(opb.constraints[0].rhs, 2);
        assert!(!opb.nonlinear);
    }

    #[test]
    fn parses_objective_and_negated_literals() {
        let src = "min: +2 x1 +3 ~x2 ;\n+1 x1 +1 x2 <= 1 ;\n";
        let opb = parse(src).unwrap();
        let obj = opb.objective.unwrap();
        assert!(!obj.maximize);
        assert_eq!(obj.terms.len(), 2);
        assert_eq!(obj.terms[1], Term { coeff: 3, var: 2, negated: true });
    }

    #[test]
    fn infers_variable_count_without_header() {
        let opb = parse("+1 x5 >= 1 ;\n").unwrap();
        assert_eq!(opb.n_vars, 5);
    }

    #[test]
    fn flags_nonlinear_products() {
        let opb = parse("+1 x1 x2 >= 1 ;\n").unwrap();
        assert!(opb.nonlinear);
    }

    #[test]
    fn rejects_missing_semicolon() {
        assert!(parse("+1 x1 >= 1\n").is_err());
    }

    #[test]
    fn rejects_missing_operator() {
        assert!(parse("+1 x1 1 ;\n").is_err());
    }
}
