//! Integer variable declarations.

/// Reference to an integer variable declaration inside [`Model`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct IntVarRef(pub usize);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Relation {
    Eq,
    Ne,
    Le,
    Lt,
    Ge,
    Gt,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BoolLiteral {
    pub variable: IntVarRef,
    pub positive: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Automaton {
    pub states: usize,
    pub start: usize,
    pub accepting: Vec<usize>,
    pub transitions: Vec<(usize, i32, usize)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MddArc {
    pub from: usize,
    pub value: i32,
    pub to: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mdd {
    pub layers: Vec<Vec<MddArc>>,
    pub nodes_per_layer: Vec<usize>,
}

/// Normalized global constraints over semantic integer variables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntGlobalConstraint {
    AllDifferent {
        variables: Vec<IntVarRef>,
        except: Vec<i32>,
    },
    AllEqual(Vec<IntVarRef>),
    Ordered {
        variables: Vec<IntVarRef>,
        relation: Relation,
    },
    Instantiation {
        variables: Vec<IntVarRef>,
        values: Vec<i32>,
    },
    Minimum {
        target: IntVarRef,
        variables: Vec<IntVarRef>,
    },
    Maximum {
        target: IntVarRef,
        variables: Vec<IntVarRef>,
    },
    Element {
        array: Vec<IntVarRef>,
        index: IntVarRef,
        value: IntVarRef,
    },
    ElementConst {
        array: Vec<i32>,
        index: IntVarRef,
        value: IntVarRef,
    },
    Count {
        variables: Vec<IntVarRef>,
        value: i32,
        relation: Relation,
        count: i64,
    },
    Cardinality {
        variables: Vec<IntVarRef>,
        values: Vec<i32>,
        lower: Vec<i64>,
        upper: Vec<i64>,
        closed: bool,
    },
    NValues {
        variables: Vec<IntVarRef>,
        relation: Relation,
        count: i64,
    },
    Table {
        variables: Vec<IntVarRef>,
        tuples: Vec<Vec<i32>>,
        positive: bool,
    },
    Regular {
        variables: Vec<IntVarRef>,
        automaton: Automaton,
    },
    Mdd {
        variables: Vec<IntVarRef>,
        mdd: Mdd,
    },
    Lex {
        left: Vec<IntVarRef>,
        right: Vec<IntVarRef>,
        strict: bool,
    },
    LexChain {
        rows: Vec<Vec<IntVarRef>>,
        strict: bool,
    },
    Channel {
        left: Vec<IntVarRef>,
        right: Vec<IntVarRef>,
    },
    Circuit {
        successors: Vec<IntVarRef>,
        cutset: bool,
    },
    NoOverlap {
        starts: Vec<IntVarRef>,
        durations: Vec<i64>,
    },
    /// Unary-resource no-overlap over fixed-duration intervals whose
    /// presences are semantic Boolean variables. `None` denotes a mandatory
    /// interval.
    OptionalNoOverlap {
        starts: Vec<IntVarRef>,
        durations: Vec<i64>,
        presences: Vec<Option<IntVarRef>>,
    },
    /// Exactly one optional member is present and its start equals the shared
    /// start of the alternative.
    AlternativeChannel {
        shared_start: IntVarRef,
        starts: Vec<IntVarRef>,
        durations: Vec<i64>,
        presences: Vec<IntVarRef>,
    },
    Cumulative {
        starts: Vec<IntVarRef>,
        durations: Vec<i64>,
        demands: Vec<i64>,
        capacity: i64,
    },
    CumulativeVar {
        starts: Vec<IntVarRef>,
        durations: Vec<IntVarRef>,
        demands: Vec<IntVarRef>,
        capacity: IntVarRef,
    },
    BinPacking {
        items: Vec<IntVarRef>,
        sizes: Vec<i64>,
        capacities: Vec<i64>,
    },
    BinLoads {
        items: Vec<IntVarRef>,
        sizes: Vec<i64>,
        loads: Vec<IntVarRef>,
    },
    Knapsack {
        variables: Vec<IntVarRef>,
        weights: Vec<i64>,
        profits: Vec<i64>,
        weight_relation: Relation,
        weight_limit: i64,
        profit_relation: Relation,
        profit_limit: i64,
    },
    ValuePrecedence {
        variables: Vec<IntVarRef>,
        values: Vec<i32>,
        covered: bool,
    },
}

impl IntGlobalConstraint {
    pub fn variables(&self, output: &mut Vec<IntVarRef>) {
        match self {
            Self::AllDifferent { variables, .. }
            | Self::AllEqual(variables)
            | Self::Ordered { variables, .. }
            | Self::Instantiation { variables, .. }
            | Self::Count { variables, .. }
            | Self::Cardinality { variables, .. }
            | Self::NValues { variables, .. }
            | Self::Table { variables, .. }
            | Self::Regular { variables, .. }
            | Self::Mdd { variables, .. }
            | Self::Circuit { successors: variables, .. }
            | Self::ValuePrecedence { variables, .. } => output.extend_from_slice(variables),
            Self::Minimum { target, variables } | Self::Maximum { target, variables } => {
                output.push(*target);
                output.extend_from_slice(variables);
            }
            Self::Element { array, index, value } => {
                output.extend_from_slice(array);
                output.extend([*index, *value]);
            }
            Self::ElementConst { index, value, .. } => output.extend([*index, *value]),
            Self::Lex { left, right, .. } | Self::Channel { left, right } => {
                output.extend_from_slice(left);
                output.extend_from_slice(right);
            }
            Self::LexChain { rows, .. } => output.extend(rows.iter().flatten().copied()),
            Self::NoOverlap { starts, .. } | Self::Cumulative { starts, .. } => output.extend_from_slice(starts),
            Self::OptionalNoOverlap { starts, presences, .. } => {
                output.extend_from_slice(starts);
                output.extend(presences.iter().flatten().copied());
            }
            Self::AlternativeChannel { shared_start, starts, presences, .. } => {
                output.push(*shared_start);
                output.extend_from_slice(starts);
                output.extend_from_slice(presences);
            }
            Self::CumulativeVar { starts, durations, demands, capacity } => {
                output.extend_from_slice(starts);
                output.extend_from_slice(durations);
                output.extend_from_slice(demands);
                output.push(*capacity);
            }
            Self::BinPacking { items, .. } => output.extend_from_slice(items),
            Self::BinLoads { items, loads, .. } => {
                output.extend_from_slice(items);
                output.extend_from_slice(loads);
            }
            Self::Knapsack { variables, .. } => output.extend_from_slice(variables),
        }
    }
}

/// Integer variable domain declaration.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum IntDomain {
    /// Boolean domain `{0, 1}`.
    Bool,
    /// Contiguous integer range.
    Range { lo: i32, hi: i32 },
    /// Explicit finite set.
    Set(Vec<i32>),
}

impl IntDomain {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Bool => Ok(()),
            Self::Range { lo, hi } if lo <= hi => Ok(()),
            Self::Range { .. } => Err("integer range lower bound exceeds its upper bound".to_string()),
            Self::Set(values) if values.is_empty() => Err("explicit integer domain is empty".to_string()),
            Self::Set(_) => Ok(()),
        }
    }

    pub fn contains(&self, value: i64) -> bool {
        i32::try_from(value).ok().is_some_and(|value| match self {
            Self::Bool => matches!(value, 0 | 1),
            Self::Range { lo, hi } => (*lo..=*hi).contains(&value),
            Self::Set(values) => values.contains(&value),
        })
    }

    /// Least domain value greater than or equal to `lower_bound`.
    pub(crate) fn ceiling(&self, lower_bound: i64) -> Option<i64> {
        match self {
            Self::Bool => (lower_bound <= 1).then_some(lower_bound.max(0)),
            Self::Range { lo, hi } => {
                let value = lower_bound.max(i64::from(*lo));
                (value <= i64::from(*hi)).then_some(value)
            }
            Self::Set(values) => values.iter().copied().map(i64::from).filter(|&value| value >= lower_bound).min(),
        }
    }
}
