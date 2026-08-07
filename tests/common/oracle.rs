pub fn solutions(domains: &[Vec<i32>], mut pred: impl FnMut(&[i32]) -> bool) -> Vec<Vec<i32>> {
    let mut out = Vec::new();
    let mut assign = vec![0i32; domains.len()];
    visit(domains, 0, &mut assign, &mut |assign| {
        if pred(assign) {
            out.push(assign.to_vec());
        }
    });
    out
}

#[allow(dead_code)]
pub fn count(domains: &[Vec<i32>], mut pred: impl FnMut(&[i32]) -> bool) -> usize {
    let mut n = 0usize;
    let mut assign = vec![0i32; domains.len()];
    visit(domains, 0, &mut assign, &mut |assign| {
        n += usize::from(pred(assign));
    });
    n
}

fn visit(domains: &[Vec<i32>], i: usize, assign: &mut [i32], f: &mut impl FnMut(&[i32])) {
    if i == domains.len() {
        f(assign);
        return;
    }
    for &v in &domains[i] {
        assign[i] = v;
        visit(domains, i + 1, assign, f);
    }
}
