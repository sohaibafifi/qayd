use qayd::engines::ls::lists::{
    audit_bounded_alns, audit_bounded_structural_work, audit_bounded_worst_removal, audit_elite_archive,
    audit_incremental_macro_accounting, audit_interrupted_attempt_accounting, audit_macro_operators, audit_path_relink,
    audit_path_relink_interruption_accounting, audit_path_relink_large_partition, audit_relink_bound, audit_route_elimination_budget_1000,
    audit_routing_scale_operator_exploration, audit_size_safe_routing_compound_budget, audit_skipped_operator_adaptation,
    audit_timing_independent_operator_learning, audit_unbounded_alns_compatibility, audit_unproductive_deterministic_cost_balance,
};

#[test]
fn destroy_repair_budget_has_distinct_terminal_outcomes() {
    assert!(audit_bounded_alns());
}

#[test]
fn alns_adaptation_uses_deterministic_cost_not_timing() {
    assert!(audit_timing_independent_operator_learning());
}

#[test]
fn zero_reward_learning_limits_expensive_operator_projected_cost() {
    assert!(audit_unproductive_deterministic_cost_balance());
}

#[test]
fn interrupted_slices_account_for_both_selected_operators() {
    assert!(audit_interrupted_attempt_accounting());
}

#[test]
fn skipped_stages_do_not_change_adaptive_weights() {
    assert!(audit_skipped_operator_adaptation());
}

#[test]
fn worst_removal_is_bounded_deterministic_and_transactional() {
    assert!(audit_bounded_worst_removal());
}

#[test]
fn realistic_routing_work_keeps_operator_weights_resolved_and_explored() {
    assert!(audit_routing_scale_operator_exploration());
}

#[test]
fn bounded_alns_accounts_for_structural_work_and_interrupts_promptly() {
    assert!(audit_bounded_structural_work());
}

#[test]
fn generic_unbounded_alns_keeps_its_legacy_search_shape() {
    assert!(audit_unbounded_alns_compatibility());
}

#[test]
fn route_elimination_repairs_a_typical_route_at_one_thousand_customers() {
    assert!(audit_route_elimination_budget_1000());
}

#[test]
fn compound_routing_neighborhoods_find_their_causal_improvements() {
    assert!(audit_macro_operators());
}

#[test]
fn macro_scans_reserve_work_and_rebuild_only_a_winner() {
    assert!(audit_incremental_macro_accounting());
}

#[test]
fn compound_budget_retains_materialization_headroom_above_the_exploration_cap() {
    assert!(audit_size_safe_routing_compound_budget());
}

#[test]
fn elite_archive_is_stable_diverse_and_keeps_the_best() {
    assert!(audit_elite_archive());
}

#[test]
fn path_relinking_has_a_hard_customer_scaled_step_bound() {
    assert_eq!(audit_relink_bound(3), 1);
    assert_eq!(audit_relink_bound(20), 5);
    assert_eq!(audit_relink_bound(10_000), 32);
}

#[test]
fn path_relinking_moves_toward_the_target_and_returns_the_best_prefix() {
    assert!(audit_path_relink());
}

#[test]
fn interrupted_relink_clones_and_rebuilds_do_not_publish_incomplete_work() {
    assert!(audit_path_relink_interruption_accounting());
}

#[test]
fn large_terminal_relink_check_is_single_pass_and_budgeted() {
    assert!(audit_path_relink_large_partition());
}
