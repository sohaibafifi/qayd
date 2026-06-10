//! Large-neighborhood search policy shared by solving front-ends.

use crate::ids::VarId;
use crate::store::Solver;

const BASE_CONFLICT_BUDGET: u64 = 10_000;
const MIN_CONFLICT_BUDGET: u64 = 2_000;
const MAX_CONFLICT_BUDGET: u64 = 50_000;
const INITIAL_RELAX_PERCENT: u32 = 20;
const MIN_RELAX_PERCENT: u32 = 5;
const MAX_RELAX_PERCENT: u32 = 60;
const RELAX_STEP: u32 = 5;

use crate::mix64;

pub(crate) struct LnsState {
    relax_percent: u32,
    stagnant: u32,
}

impl LnsState {
    pub(crate) fn new(seed: u64) -> Self {
        let jitter = (mix64(seed) % 11) as i32 - 5;
        let relax_percent = (INITIAL_RELAX_PERCENT as i32 + jitter).clamp(MIN_RELAX_PERCENT as i32, MAX_RELAX_PERCENT as i32);
        Self { relax_percent: relax_percent as u32, stagnant: 0 }
    }

    pub(crate) fn relax_percent(&self) -> u32 {
        self.relax_percent
    }

    pub(crate) fn budget(&self) -> u64 {
        (BASE_CONFLICT_BUDGET * self.relax_percent as u64 / INITIAL_RELAX_PERCENT as u64).clamp(MIN_CONFLICT_BUDGET, MAX_CONFLICT_BUDGET)
    }

    pub(crate) fn record(&mut self, improved: bool, complete: bool) {
        if improved {
            self.stagnant = 0;
            self.relax_percent = self.relax_percent.saturating_sub(RELAX_STEP).max(MIN_RELAX_PERCENT);
        } else if complete {
            self.stagnant = self.stagnant.saturating_add(1);
            let step = if self.stagnant >= 3 { 2 * RELAX_STEP } else { RELAX_STEP };
            self.relax_percent = self.relax_percent.saturating_add(step).min(MAX_RELAX_PERCENT);
        } else {
            self.stagnant = self.stagnant.saturating_add(1);
            self.relax_percent = self.relax_percent.saturating_sub(RELAX_STEP).max(MIN_RELAX_PERCENT);
        }
    }
}

pub(crate) fn fix_neighborhood(
    solver: &mut Solver,
    vars: &[VarId],
    solution: &[i32],
    candidates: &[usize],
    seed: u64,
    relax_percent: u32,
) -> bool {
    let pivot_pos = mix64(seed) as usize % candidates.len();
    candidates.iter().copied().enumerate().all(|(pos, i)| {
        let relaxed = pos == pivot_pos || mix64(seed ^ i as u64) % 100 < relax_percent as u64;
        relaxed || solver.store.fix(vars[i], solution[i]).is_ok()
    })
}
