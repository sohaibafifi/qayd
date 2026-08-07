//! Unordered finite-set declarations.

use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SetVarRef(pub usize);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SetDecl {
    pub required: BTreeSet<i32>,
    pub possible: BTreeSet<i32>,
}

impl SetDecl {
    pub fn new(required: impl IntoIterator<Item = i32>, possible: impl IntoIterator<Item = i32>) -> Result<Self, String> {
        let required = required.into_iter().collect::<BTreeSet<_>>();
        let possible = possible.into_iter().collect::<BTreeSet<_>>();
        if !required.is_subset(&possible) {
            return Err("set lower bound must be a subset of its upper bound".to_string());
        }
        Ok(Self { required, possible })
    }
}
