//! Brute-force satisfaction oracle for testing.
//!
//! Enumerate the full Cartesian product of small per-variable domains and keep
//! the assignments a predicate accepts. The solver's solution set must match
//! this exactly — the single most valuable correctness tool in the project.

/// Every assignment in the Cartesian product of `domains` that satisfies `pred`.
pub fn solutions(domains: &[Vec<i32>], mut pred: impl FnMut(&[i32]) -> bool) -> Vec<Vec<i32>> {
    let mut out = Vec::new();
    let mut assign = vec![0i32; domains.len()];
    enumerate(domains, 0, &mut assign, &mut pred, &mut out);
    out
}

/// Count the assignments in the Cartesian product of `domains` satisfying `pred`.
pub fn count(domains: &[Vec<i32>], mut pred: impl FnMut(&[i32]) -> bool) -> usize {
    let mut n = 0usize;
    let mut assign = vec![0i32; domains.len()];
    count_rec(domains, 0, &mut assign, &mut pred, &mut n);
    n
}

fn enumerate(
    domains: &[Vec<i32>],
    i: usize,
    assign: &mut [i32],
    pred: &mut impl FnMut(&[i32]) -> bool,
    out: &mut Vec<Vec<i32>>,
) {
    if i == domains.len() {
        if pred(assign) {
            out.push(assign.to_vec());
        }
        return;
    }
    for &v in &domains[i] {
        assign[i] = v;
        enumerate(domains, i + 1, assign, pred, out);
    }
}

fn count_rec(
    domains: &[Vec<i32>],
    i: usize,
    assign: &mut [i32],
    pred: &mut impl FnMut(&[i32]) -> bool,
    n: &mut usize,
) {
    if i == domains.len() {
        if pred(assign) {
            *n += 1;
        }
        return;
    }
    for &v in &domains[i] {
        assign[i] = v;
        count_rec(domains, i + 1, assign, pred, n);
    }
}
