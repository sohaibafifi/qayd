//! Pure text helpers for the line-oriented FlatZinc reader: argument splitting,
//! bracket/annotation stripping. No solver state.

/// Return `Err(msg)` unless `ok`.
pub(crate) fn require(ok: bool, msg: &str) -> Result<(), String> {
    if ok {
        Ok(())
    } else {
        Err(msg.to_string())
    }
}

/// Drop a trailing `:: annotation ...` tail, keeping the part before it.
pub(crate) fn strip_annotations(s: &str) -> &str {
    s.split("::").next().unwrap_or(s).trim()
}

/// Skip the leading `:: ann` annotations of a `solve` item, returning the goal.
pub(crate) fn strip_leading_solve_annotations(mut s: &str) -> &str {
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

/// Split the contents of a `[...]` literal into its items, or `None` if `s` is
/// not bracketed.
pub(crate) fn bracket_items(s: &str) -> Option<Vec<String>> {
    let s = s.trim();
    let inner = s.strip_prefix('[')?.strip_suffix(']')?;
    if inner.trim().is_empty() {
        return Some(Vec::new());
    }
    Some(split_args(inner))
}

/// Split `s` on top-level commas (respecting `[] () {}` nesting).
pub(crate) fn split_args(s: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, ch) in s.char_indices() {
        match ch {
            '[' | '(' | '{' => depth += 1,
            ']' | ')' | '}' => depth -= 1,
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

/// The declared name of a statement's right-hand side (before any `=` or `::`).
pub(crate) fn clean_name(s: &str) -> String {
    strip_annotations(s).split('=').next().unwrap_or(s).trim().to_string()
}
