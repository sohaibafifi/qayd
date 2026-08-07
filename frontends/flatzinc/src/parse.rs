//! The FlatZinc reader: domains, declarations, constraints, and the solve item.

use std::collections::HashMap;

use qayd::model::{IntVarRef, Relation, SourceRange};

use crate::model::{expr, FznDomain, Model, Output, Reif, UNBOUNDED_HI, UNBOUNDED_LO};
use crate::text::{clean_name, require, split_args, strip_leading_solve_annotations};

const MAX_EXPLICIT_DOMAIN_VALUES: usize = 100_000;

/// Parse a whole FlatZinc model: strip comments, then dispatch each `;`-terminated
/// statement to a declaration, constraint, or solve handler.
pub(crate) fn parse(input: &str) -> Result<Model, String> {
    let mut model = Model::new();
    let text = strip_comments_preserving_offsets(input);
    let mut item_start = 0;
    let mut item = 0;
    for item_end in statement_ends(&text).into_iter().chain(std::iter::once(text.len())) {
        let raw = &text[item_start..item_end.saturating_sub(usize::from(item_end > item_start && text.as_bytes()[item_end - 1] == b';'))];
        let leading = raw.len() - raw.trim_start().len();
        let trailing = raw.trim_end().len();
        let stmt = raw.trim();
        let source = SourceRange {
            source: "flatzinc".to_string(),
            start: item_start + leading,
            end: item_start + trailing + usize::from(item_end > item_start && text.as_bytes()[item_end - 1] == b';'),
        };
        item_start = item_end;
        if stmt.is_empty() {
            continue;
        }
        let mark = model.metadata_mark();
        let annotations = if stmt.starts_with("constraint ") {
            let (body, parsed_annotations) = split_trailing_annotations(stmt)?;
            parse_constraint(&mut model, &body)?;
            parsed_annotations
        } else if stmt.starts_with("solve ") {
            parse_solve(&mut model, stmt, item, source.clone())?
        } else if stmt.starts_with("predicate ") || stmt.starts_with("output ") {
            let (_, parsed_annotations) = split_trailing_annotations(stmt)?;
            parsed_annotations
        } else {
            let (body, parsed_annotations) = split_declaration_annotations(stmt)?;
            parse_decl(&mut model, &body, &parsed_annotations)?;
            parsed_annotations
        };
        model.record_item_metadata(item, mark, &annotations, source);
        item += 1;
    }
    Ok(model)
}

fn statement_ends(text: &str) -> Vec<usize> {
    let mut ends = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    for (index, byte) in text.bytes().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
        } else if byte == b'"' {
            in_string = true;
        } else if byte == b';' {
            ends.push(index + 1);
        }
    }
    ends
}

/// Replace comments with spaces so statement byte offsets still refer to the
/// original FlatZinc source. A percent sign inside a string is not a comment.
fn strip_comments_preserving_offsets(input: &str) -> String {
    let mut bytes = input.as_bytes().to_vec();
    let mut in_string = false;
    let mut escaped = false;
    let mut in_comment = false;
    for byte in &mut bytes {
        if in_comment {
            if *byte == b'\n' {
                in_comment = false;
            } else {
                *byte = b' ';
            }
            continue;
        }
        if in_string {
            if escaped {
                escaped = false;
            } else if *byte == b'\\' {
                escaped = true;
            } else if *byte == b'"' {
                in_string = false;
            }
        } else if *byte == b'"' {
            in_string = true;
        } else if *byte == b'%' {
            in_comment = true;
            *byte = b' ';
        }
    }
    String::from_utf8(bytes).expect("replacing comment bytes with ASCII preserves UTF-8")
}

/// Split top-level `:: annotation` tails while retaining each annotation as
/// deterministic opaque FlatZinc text.
fn split_trailing_annotations(stmt: &str) -> Result<(String, Vec<String>), String> {
    let mut separators = Vec::new();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    let bytes = stmt.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            index += 1;
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b':' if depth == 0 && bytes.get(index + 1) == Some(&b':') => {
                separators.push(index);
                index += 1;
            }
            _ => {}
        }
        index += 1;
    }
    if separators.is_empty() {
        return Ok((stmt.trim().to_string(), Vec::new()));
    }
    let body = stmt[..separators[0]].trim().to_string();
    let mut annotations = Vec::new();
    for (position, start) in separators.iter().copied().enumerate() {
        let end = separators.get(position + 1).copied().unwrap_or(stmt.len());
        let annotation = stmt[start + 2..end].trim();
        validate_annotation(annotation)?;
        annotations.push(annotation.to_string());
    }
    Ok((body, annotations))
}

/// Declaration annotations may precede an initializer (`x :: ann = value`) or
/// follow it (`x = value :: ann`). Restore the initializer to the annotation-
/// free declaration while retaining the annotation term itself.
fn split_declaration_annotations(stmt: &str) -> Result<(String, Vec<String>), String> {
    let (mut body, raw_annotations) = split_trailing_annotations(stmt)?;
    let mut annotations = Vec::with_capacity(raw_annotations.len());
    let mut initializer = None;
    for raw in raw_annotations {
        if let Some(equals) = top_level_equals(&raw) {
            if initializer.is_some() {
                return Err(format!("Unsupported malformed FlatZinc annotation `{raw}`"));
            }
            let annotation = raw[..equals].trim();
            validate_annotation(annotation)?;
            annotations.push(annotation.to_string());
            initializer = Some(raw[equals + 1..].trim().to_string());
        } else {
            annotations.push(raw);
        }
    }
    if let Some(initializer) = initializer {
        body.push_str(" = ");
        body.push_str(&initializer);
    }
    Ok((body, annotations))
}

fn top_level_equals(text: &str) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for (index, character) in text.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                in_string = false;
            }
            continue;
        }
        match character {
            '"' => in_string = true,
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            '=' if depth == 0 => return Some(index),
            _ => {}
        }
    }
    None
}

fn validate_annotation(annotation: &str) -> Result<(), String> {
    if annotation.is_empty() {
        return Err("Unsupported empty FlatZinc annotation".to_string());
    }
    let mut stack = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    for character in annotation.chars() {
        if in_string {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                in_string = false;
            }
            continue;
        }
        match character {
            '"' => in_string = true,
            '(' | '[' | '{' => stack.push(character),
            ')' if stack.pop() != Some('(') => return Err(format!("Unsupported malformed FlatZinc annotation `{annotation}`")),
            ']' if stack.pop() != Some('[') => return Err(format!("Unsupported malformed FlatZinc annotation `{annotation}`")),
            '}' if stack.pop() != Some('{') => return Err(format!("Unsupported malformed FlatZinc annotation `{annotation}`")),
            _ => {}
        }
    }
    if in_string || !stack.is_empty() {
        Err(format!("Unsupported malformed FlatZinc annotation `{annotation}`"))
    } else {
        Ok(())
    }
}

/// Enumerate the members of a FlatZinc set literal (`lo..hi` or `{...}`).
pub(crate) fn parse_int_set(model: &Model, s: &str) -> Result<Vec<i32>, String> {
    let s = s.trim();
    let inner = s.strip_prefix('{').and_then(|r| r.strip_suffix('}')).unwrap_or(s);
    let mut values = Vec::new();
    for item in split_args(inner) {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        if let Some((lo, hi)) = item.split_once("..") {
            values.extend(model.int_atom(lo)?..=model.int_atom(hi)?);
        } else {
            values.push(model.int_atom(item)?);
        }
    }
    values.sort_unstable();
    values.dedup();
    Ok(values)
}

fn parse_domain(model: &Model, s: &str) -> Result<FznDomain, String> {
    let s = s.trim();
    if s == "bool" {
        return Ok(FznDomain::Bool);
    }
    if s == "int" {
        return Ok(FznDomain::Range(UNBOUNDED_LO, UNBOUNDED_HI));
    }
    if let Some(inner) = s.strip_prefix('{').and_then(|rest| rest.strip_suffix('}')) {
        return parse_set_domain(model, inner);
    }
    let (lo, hi) = s.split_once("..").ok_or_else(|| format!("unsupported domain `{s}`"))?;
    Ok(FznDomain::Range(model.int_atom(lo)?, model.int_atom(hi)?))
}

fn parse_set_domain(model: &Model, s: &str) -> Result<FznDomain, String> {
    let mut values = Vec::new();
    for item in split_args(s) {
        if item.is_empty() {
            continue;
        }
        if let Some((lo, hi)) = item.split_once("..") {
            let lo = model.int_atom(lo)?;
            let hi = model.int_atom(hi)?;
            require(lo <= hi, "set domain interval has lower bound above upper bound")?;
            let len = i64::from(hi) - i64::from(lo) + 1;
            require(len as usize <= MAX_EXPLICIT_DOMAIN_VALUES.saturating_sub(values.len()), "explicit set domain is too large")?;
            values.extend(lo..=hi);
        } else {
            values.push(model.int_atom(&item)?);
        }
    }
    require(!values.is_empty(), "empty set domain")?;
    values.sort_unstable();
    values.dedup();
    if values.len() == (i64::from(*values.last().unwrap()) - i64::from(values[0]) + 1) as usize {
        Ok(FznDomain::Range(values[0], *values.last().unwrap()))
    } else {
        Ok(FznDomain::Set(values))
    }
}

fn array_len(model: &Model, left: &str) -> Result<usize, String> {
    let spec = left.split_once('[').and_then(|(_, r)| r.split_once(']')).map(|(x, _)| x).ok_or("array without index set")?;
    require(!spec.contains(','), "only one-dimensional arrays are supported")?;
    let (lo, hi) = spec.split_once("..").ok_or("array index must be an interval")?;
    Ok((model.int_atom(hi)? - model.int_atom(lo)? + 1) as usize)
}

/// Dimension ranges of an `:: output_array([1..a, 1..b, ...])` annotation, if present.
fn output_array_dims(model: &Model, annotations: &[String]) -> Result<Option<Vec<(i32, i32)>>, String> {
    let Some(annotation) = annotations.iter().find(|annotation| annotation_name(annotation) == "output_array") else {
        return Ok(None);
    };
    let inner = annotation
        .split_once('[')
        .and_then(|(_, rest)| rest.rsplit_once(']'))
        .map(|(inner, _)| inner)
        .ok_or("Unsupported malformed output_array annotation")?;
    let mut dims = Vec::new();
    for item in split_args(inner) {
        let (lo, hi) = item.split_once("..").ok_or("output_array dimension must be an interval")?;
        dims.push((model.int_atom(lo)?, model.int_atom(hi)?));
    }
    Ok(Some(dims))
}

fn annotation_name(annotation: &str) -> &str {
    annotation.split_once('(').map_or(annotation, |(name, _)| name).trim()
}

fn parse_decl(model: &mut Model, stmt: &str, annotations: &[String]) -> Result<(), String> {
    if stmt.starts_with("array ") {
        let (left, right) = stmt.split_once(':').ok_or("bad array declaration")?;
        let name = clean_name(right);
        let is_bool = left.contains(" of bool") || left.contains(" of var bool");
        let dims = output_array_dims(model, annotations)?;
        if let Some((_, value)) = right.split_once('=') {
            if left.contains(" of var ") {
                let vars = model.var_list(value)?;
                if let Some(dims) = dims {
                    model.add_output(Output::Array { name: name.clone(), dims, vars: vars.clone(), is_bool });
                }
                model.arrays.insert(name, vars);
            } else {
                let values = model.int_list(value)?;
                model.int_arrays.insert(name, values);
            }
            return Ok(());
        }
        let n = array_len(model, left)?;
        let domain = left.rsplit_once(" of var ").map(|(_, d)| d).ok_or("only var arrays without assignment are supported")?;
        let domain = parse_domain(model, domain)?;
        let vars: Vec<IntVarRef> = (1..=n).map(|i| model.new_var(format!("{name}[{i}]"), &domain)).collect();
        if let Some(dims) = dims {
            model.add_output(Output::Array { name: name.clone(), dims, vars: vars.clone(), is_bool });
        }
        model.arrays.insert(name, vars);
        return Ok(());
    }
    let (left, right) = stmt.split_once(':').ok_or("bad declaration")?;
    if left.trim() == "int" || left.trim() == "bool" {
        let (name, value) = right.split_once('=').ok_or("constant declaration without value")?;
        model.ints.insert(clean_name(name), model.int_atom(value)?);
        return Ok(());
    }
    let domain = left.trim().strip_prefix("var ").ok_or("unsupported declaration")?;
    let name = clean_name(right);
    // Set variable: store a 0/1 membership variable per value of the universe.
    if let Some(set_dom) = domain.strip_prefix("set of ") {
        let values = parse_int_set(model, set_dom)?;
        require(!values.is_empty(), "set variable with empty universe")?;
        let members: HashMap<i32, IntVarRef> = values.iter().map(|&v| (v, model.new_aux_bool())).collect();
        model.set_vars.insert(name, members);
        return Ok(());
    }
    let is_bool = domain.trim() == "bool";
    let domain = parse_domain(model, domain)?;
    let var = model.new_var(name.clone(), &domain);
    if annotations.iter().any(|annotation| annotation_name(annotation) == "output_var") {
        model.add_output(Output::Var { name, var, is_bool });
    }
    if let Some((_, value)) = right.split_once('=') {
        let fixed = model.var_atom(value)?;
        model.post_raw_linear(&[1, -1], &[var, fixed], Relation::Eq, 0);
    }
    Ok(())
}

/// The `i`th argument of a constraint item, or a typed error (instead of a
/// panic) when the item supplies fewer arguments than its predicate needs.
fn arg<'a>(name: &str, args: &'a [String], i: usize) -> Result<&'a str, String> {
    args.get(i).map(String::as_str).ok_or_else(|| format!("{name} expects at least {} arguments", i + 1))
}

fn parse_constraint(model: &mut Model, stmt: &str) -> Result<(), String> {
    let body = stmt.strip_prefix("constraint").unwrap().trim();
    let open = body.find('(').ok_or("constraint without arguments")?;
    let name = body[..open].trim();
    let args = split_args(body[open + 1..].trim_end_matches(')'));
    match name {
        "int_eq" => model.post_cmp(arg(name, &args, 0)?, arg(name, &args, 1)?, Relation::Eq),
        "int_ne" => model.post_cmp(arg(name, &args, 0)?, arg(name, &args, 1)?, Relation::Ne),
        "int_le" => model.post_cmp(arg(name, &args, 0)?, arg(name, &args, 1)?, Relation::Le),
        "int_lt" => model.post_cmp(arg(name, &args, 0)?, arg(name, &args, 1)?, Relation::Lt),
        "int_ge" => model.post_cmp(arg(name, &args, 0)?, arg(name, &args, 1)?, Relation::Ge),
        "int_gt" => model.post_cmp(arg(name, &args, 0)?, arg(name, &args, 1)?, Relation::Gt),
        "int_lin_eq" => model.post_linear(&args, Relation::Eq),
        "int_lin_ne" => model.post_linear(&args, Relation::Ne),
        "int_lin_le" => model.post_linear(&args, Relation::Le),
        "int_lin_lt" => model.post_linear(&args, Relation::Lt),
        "int_lin_ge" => model.post_linear(&args, Relation::Ge),
        "int_lin_gt" => model.post_linear(&args, Relation::Gt),
        "all_different_int" | "fzn_all_different_int" => {
            let vars = model.var_list(&args[0])?;
            model.post_all_different(&vars);
            Ok(())
        }
        "array_int_element" | "array_bool_element" => model.post_element(&args, false),
        "array_var_int_element" | "array_var_bool_element" => model.post_element(&args, true),
        "gecode_int_element" => model.post_gecode_element(&args),
        "array_int_minimum" => model.post_extremum(&args, true),
        "array_int_maximum" => model.post_extremum(&args, false),
        "fzn_count_eq" => model.post_count(&args, Relation::Eq),
        "fzn_count_ne" | "fzn_count_neq" => model.post_count(&args, Relation::Ne),
        "fzn_count_le" | "fzn_count_leq" => model.post_count(&args, Relation::Le),
        "fzn_count_lt" => model.post_count(&args, Relation::Lt),
        "fzn_count_ge" | "fzn_count_geq" => model.post_count(&args, Relation::Ge),
        "fzn_count_gt" => model.post_count(&args, Relation::Gt),

        // Reified / half-reified comparisons (bool reifs share the int form: 0/1).
        "int_eq_reif" | "bool_eq_reif" => model.post_int_cmp_reif(&args, Relation::Eq, Reif::Iff),
        "int_ne_reif" | "bool_ne_reif" | "bool_xor_reif" => model.post_int_cmp_reif(&args, Relation::Ne, Reif::Iff),
        "int_le_reif" | "bool_le_reif" => model.post_int_cmp_reif(&args, Relation::Le, Reif::Iff),
        "int_lt_reif" | "bool_lt_reif" => model.post_int_cmp_reif(&args, Relation::Lt, Reif::Iff),
        "int_ge_reif" => model.post_int_cmp_reif(&args, Relation::Ge, Reif::Iff),
        "int_gt_reif" => model.post_int_cmp_reif(&args, Relation::Gt, Reif::Iff),
        "int_eq_imp" | "bool_eq_imp" => model.post_int_cmp_reif(&args, Relation::Eq, Reif::Imp),
        "int_ne_imp" | "bool_ne_imp" => model.post_int_cmp_reif(&args, Relation::Ne, Reif::Imp),
        "int_le_imp" | "bool_le_imp" => model.post_int_cmp_reif(&args, Relation::Le, Reif::Imp),
        "int_lt_imp" | "bool_lt_imp" => model.post_int_cmp_reif(&args, Relation::Lt, Reif::Imp),
        "int_ge_imp" => model.post_int_cmp_reif(&args, Relation::Ge, Reif::Imp),
        "int_gt_imp" => model.post_int_cmp_reif(&args, Relation::Gt, Reif::Imp),

        // Reified / half-reified linear relations.
        "int_lin_eq_reif" => model.post_lin_reif(&args, Relation::Eq, Reif::Iff),
        "int_lin_ne_reif" => model.post_lin_reif(&args, Relation::Ne, Reif::Iff),
        "int_lin_le_reif" => model.post_lin_reif(&args, Relation::Le, Reif::Iff),
        "int_lin_lt_reif" => model.post_lin_reif(&args, Relation::Lt, Reif::Iff),
        "int_lin_ge_reif" => model.post_lin_reif(&args, Relation::Ge, Reif::Iff),
        "int_lin_gt_reif" => model.post_lin_reif(&args, Relation::Gt, Reif::Iff),
        "int_lin_eq_imp" => model.post_lin_reif(&args, Relation::Eq, Reif::Imp),
        "int_lin_ne_imp" => model.post_lin_reif(&args, Relation::Ne, Reif::Imp),
        "int_lin_le_imp" => model.post_lin_reif(&args, Relation::Le, Reif::Imp),
        "int_lin_lt_imp" => model.post_lin_reif(&args, Relation::Lt, Reif::Imp),
        "int_lin_ge_imp" => model.post_lin_reif(&args, Relation::Ge, Reif::Imp),
        "int_lin_gt_imp" => model.post_lin_reif(&args, Relation::Gt, Reif::Imp),

        // Boolean channelling and logic.
        "bool2int" | "bool_eq" => model.post_cmp(arg(name, &args, 0)?, arg(name, &args, 1)?, Relation::Eq),
        "bool_le" => model.post_cmp(arg(name, &args, 0)?, arg(name, &args, 1)?, Relation::Le),
        "bool_lt" => model.post_cmp(arg(name, &args, 0)?, arg(name, &args, 1)?, Relation::Lt),
        "bool_ge" => model.post_cmp(arg(name, &args, 0)?, arg(name, &args, 1)?, Relation::Ge),
        "bool_gt" => model.post_cmp(arg(name, &args, 0)?, arg(name, &args, 1)?, Relation::Gt),
        "bool_not" => {
            let x = model.var_atom(arg(name, &args, 0)?)?;
            let y = model.var_atom(arg(name, &args, 1)?)?;
            model.post_raw_linear(&[1, 1], &[x, y], Relation::Eq, 1);
            Ok(())
        }
        "bool_and" => {
            let (a, b) = model.two_atoms(arg(name, &args, 0)?, arg(name, &args, 1)?)?;
            model.post_reif(expr::and(vec![a, b]), arg(name, &args, 2)?, Reif::Iff)
        }
        "bool_or" => {
            let (a, b) = model.two_atoms(arg(name, &args, 0)?, arg(name, &args, 1)?)?;
            model.post_reif(expr::or(vec![a, b]), arg(name, &args, 2)?, Reif::Iff)
        }
        "bool_xor" => {
            let (a, b) = model.two_atoms(arg(name, &args, 0)?, arg(name, &args, 1)?)?;
            if args.len() >= 3 {
                model.post_reif(expr::ne(a, b), &args[2], Reif::Iff)
            } else {
                model.post_expr(expr::ne(a, b));
                Ok(())
            }
        }
        "bool_imp" => {
            let b = model.atom_expr(arg(name, &args, 1)?)?;
            model.post_reif(b, arg(name, &args, 0)?, Reif::Imp)
        }
        "array_bool_and" => model.post_bool_nary(&args, true),
        "array_bool_or" => model.post_bool_nary(&args, false),
        "bool_clause" => model.post_bool_clause(&args),
        "bool_clause_reif" => model.post_bool_clause_reif(&args),

        // Set membership.
        "set_in" => model.post_set_in(&args, None),
        "set_in_reif" => model.post_set_in(&args, Some(Reif::Iff)),
        "set_in_imp" => model.post_set_in(&args, Some(Reif::Imp)),

        // Functional integer arithmetic.
        "int_times" => {
            let (a, b) = model.two_atoms(arg(name, &args, 0)?, arg(name, &args, 1)?)?;
            model.post_def(arg(name, &args, 2)?, expr::mul(vec![a, b]))
        }
        "int_plus" => {
            let (a, b) = model.two_atoms(arg(name, &args, 0)?, arg(name, &args, 1)?)?;
            model.post_def(arg(name, &args, 2)?, expr::add(vec![a, b]))
        }
        "int_minus" => {
            let (a, b) = model.two_atoms(arg(name, &args, 0)?, arg(name, &args, 1)?)?;
            model.post_def(arg(name, &args, 2)?, expr::sub(a, b))
        }
        "int_max" => {
            let (a, b) = model.two_atoms(arg(name, &args, 0)?, arg(name, &args, 1)?)?;
            model.post_def(arg(name, &args, 2)?, expr::max_of(vec![a, b]))
        }
        "int_min" => {
            let (a, b) = model.two_atoms(arg(name, &args, 0)?, arg(name, &args, 1)?)?;
            model.post_def(arg(name, &args, 2)?, expr::min_of(vec![a, b]))
        }
        "int_div" => {
            let (a, b) = model.two_atoms(arg(name, &args, 0)?, arg(name, &args, 1)?)?;
            model.post_def(arg(name, &args, 2)?, expr::div(a, b))
        }
        "int_mod" => {
            let (a, b) = model.two_atoms(arg(name, &args, 0)?, arg(name, &args, 1)?)?;
            model.post_def(arg(name, &args, 2)?, expr::rem(a, b))
        }
        "int_abs" => {
            let a = model.atom_expr(arg(name, &args, 0)?)?;
            model.post_def(arg(name, &args, 1)?, expr::abs(a))
        }
        "int_pow" | "gecode_int_pow" => model.post_int_pow(&args),

        // Boolean sums (Chuffed flavour; the target may be a variable).
        "bool_sum_eq" => model.post_bool_sum(&args, Relation::Eq),
        "bool_sum_le" => model.post_bool_sum(&args, Relation::Le),
        "bool_sum_lt" => model.post_bool_sum(&args, Relation::Lt),
        "bool_sum_ge" => model.post_bool_sum(&args, Relation::Ge),
        "bool_sum_gt" => model.post_bool_sum(&args, Relation::Gt),
        "bool_lin_eq" => model.post_linear(&args, Relation::Eq),
        "bool_lin_le" => model.post_linear(&args, Relation::Le),

        // Globals mapped onto normalized semantic constraints or decompositions.
        "gecode_table_int" | "chuffed_table_int" | "fzn_table_int" | "table_int" | "qayd_table_int" => model.post_table(&args),
        "gecode_bool_element" => model.post_gecode_element(&args),
        "gecode_regular" | "chuffed_regular" | "fzn_regular" | "qayd_regular" => model.post_regular(&args),
        "gecode_circuit" => model.post_circuit(&args),
        "fzn_circuit" => {
            require(args.len() == 1, "fzn_circuit expects 1 argument")?;
            // Standard form: successors are 1-based over the index set.
            model.post_circuit(&["1".to_string(), args[0].clone()])
        }
        "gecode_cumulatives" | "chuffed_cumulative_vars" | "fzn_cumulative" => model.post_cumulatives(&args),
        "gecode_schedule_unary" | "chuffed_disjunctive_strict" | "fzn_disjunctive" | "fzn_disjunctive_strict" => {
            model.post_schedule_unary(&args)
        }
        "gecode_global_cardinality" | "fzn_global_cardinality" => model.post_gcc_counts(&args, false),
        "fzn_global_cardinality_closed" => model.post_gcc_counts(&args, true),
        "fzn_global_cardinality_low_up" => model.post_gcc_low_up(&args, false),
        "fzn_global_cardinality_low_up_closed" => model.post_gcc_low_up(&args, true),
        "gecode_bin_packing_load" | "fzn_bin_packing_load" => model.post_bin_packing_load(&args),
        "gecode_maximum_arg_int_offset" => model.post_arg_max(&args),
        "gecode_precede" | "fzn_int_precede" => {
            let list = model.var_list(arg(name, &args, 0)?)?;
            let s = model.int_atom(arg(name, &args, 1)?)?;
            let t = model.int_atom(arg(name, &args, 2)?)?;
            model.post_precedence(&list, &[s, t]);
            Ok(())
        }
        "chuffed_value_precede" | "fzn_value_precede_int" => {
            // Same as `gecode_precede`, with the values first: `(s, t, x)`.
            let s = model.int_atom(arg(name, &args, 0)?)?;
            let t = model.int_atom(arg(name, &args, 1)?)?;
            let list = model.var_list(arg(name, &args, 2)?)?;
            model.post_precedence(&list, &[s, t]);
            Ok(())
        }
        "chuffed_connected" | "fzn_connected" => model.post_connected(&args),
        "fzn_member_int" => model.post_member(&args),
        "fzn_all_equal_int" => {
            let vars = model.var_list(&args[0])?;
            if vars.len() > 1 {
                model.post_all_equal(&vars);
            }
            Ok(())
        }
        "fzn_maximum_int" => model.post_extremum(&args, false),
        "fzn_minimum_int" => model.post_extremum(&args, true),
        "array_int_lq" | "array_bool_lq" | "fzn_lex_lesseq_int" | "fzn_lex_lesseq_bool" => model.post_array_lex(&args, false),
        "array_int_lt" | "array_bool_lt" | "fzn_lex_less_int" | "fzn_lex_less_bool" => model.post_array_lex(&args, true),
        "fzn_increasing_int" | "fzn_increasing_bool" => model.post_ordered(&args, Relation::Le),
        "fzn_decreasing_int" | "fzn_decreasing_bool" => model.post_ordered(&args, Relation::Ge),

        other => Err(format!("unsupported FlatZinc predicate `{other}`")),
    }
}

fn parse_solve(model: &mut Model, stmt: &str, item: usize, source: SourceRange) -> Result<Vec<String>, String> {
    let solve_body = stmt.strip_prefix("solve").unwrap();
    let (leading_annotations, goal) = leading_solve_annotations(solve_body)?;
    debug_assert_eq!(strip_leading_solve_annotations(solve_body).trim(), goal);
    let (goal, trailing_annotations) = split_trailing_annotations(goal)?;
    let annotations = leading_annotations.into_iter().chain(trailing_annotations).collect::<Vec<_>>();
    if goal == "satisfy" {
        return Ok(annotations);
    }
    if let Some(obj) = goal.strip_prefix("minimize ") {
        let variable = model.var_atom(obj)?;
        model.set_objective(true, variable, item, annotations.clone(), source);
        return Ok(annotations);
    }
    if let Some(obj) = goal.strip_prefix("maximize ") {
        let variable = model.var_atom(obj)?;
        model.set_objective(false, variable, item, annotations.clone(), source);
        return Ok(annotations);
    }
    Err(format!("unsupported solve item `{goal}`"))
}

/// Parse the annotation terms between `solve` and its goal. Unlike a trailing
/// annotation, each leading term ends at top-level whitespace.
fn leading_solve_annotations(mut text: &str) -> Result<(Vec<String>, &str), String> {
    let mut annotations = Vec::new();
    loop {
        text = text.trim_start();
        let Some(rest) = text.strip_prefix("::") else {
            return Ok((annotations, text.trim()));
        };
        let annotation = rest.trim_start();
        let mut depth = 0i32;
        let mut in_string = false;
        let mut escaped = false;
        let mut end = annotation.len();
        for (index, character) in annotation.char_indices() {
            if in_string {
                if escaped {
                    escaped = false;
                } else if character == '\\' {
                    escaped = true;
                } else if character == '"' {
                    in_string = false;
                }
                continue;
            }
            match character {
                '"' => in_string = true,
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                _ if character.is_whitespace() && depth == 0 => {
                    end = index;
                    break;
                }
                _ => {}
            }
        }
        let value = annotation[..end].trim();
        validate_annotation(value)?;
        annotations.push(value.to_string());
        text = &annotation[end..];
    }
}
