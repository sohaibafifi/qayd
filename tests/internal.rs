//! Crate-private solver tests, assembled independently from production sources.
//! Run with `cargo test --test internal`.

extern crate self as qayd;

// The library event names are public API; this private embedding must preserve them.
#[allow(clippy::enum_variant_names)]
#[path = "../src/lib.rs"]
mod implementation;

pub(crate) use engines::ls::lists::audit_exhausted_neighborhood_learning;
pub(crate) use implementation::*;

#[path = "internal/orchestrator/architecture_consolidation.rs"]
mod architecture_consolidation;
#[path = "internal/orchestrator/backend_phase3.rs"]
mod backend_phase3;
#[path = "internal/model/collection.rs"]
mod collection;
#[path = "internal/orchestrator/collection_memory.rs"]
mod collection_memory;
#[path = "internal/orchestrator/cp_compile_interruption.rs"]
mod cp_compile_interruption;
#[path = "internal/scheduling/disjunctive_schedule.rs"]
mod disjunctive_schedule;
#[path = "internal/constraints/divisibility_compaction.rs"]
mod divisibility_compaction;
#[path = "internal/routing/exact_cover_warm_start.rs"]
mod exact_cover_warm_start;
#[path = "internal/inference/inference_runtime.rs"]
mod inference_runtime;

#[path = "internal/inference/deferred_runtime.rs"]
mod deferred_runtime;
#[path = "internal/constraints/integer_disequality_clause.rs"]
mod integer_disequality_clause_tests;
#[path = "internal/lcg/lcg_atom_layout.rs"]
mod lcg_atom_layout;
#[path = "internal/lcg/lcg_conflict_budget.rs"]
mod lcg_conflict_budget;
#[path = "internal/lcg/lcg_guarded_sum.rs"]
mod lcg_guarded_sum;
#[path = "internal/lcg/lcg_objective_branching.rs"]
mod lcg_objective_branching;
#[path = "internal/lcg/lcg_research.rs"]
mod lcg_research;
#[path = "internal/lcg/lcg_root_probe_budget.rs"]
mod lcg_root_probe_budget;
#[path = "internal/lcg/lcg_sync_batch.rs"]
mod lcg_sync_batch;
#[path = "internal/lists/list_alns_phase4.rs"]
mod list_alns_phase4;
#[path = "internal/lists/list_exact_interruption.rs"]
mod list_exact_interruption;
#[path = "internal/lists/list_incremental_phase2.rs"]
mod list_incremental_phase2;
#[path = "internal/lists/list_portfolio_phase5.rs"]
mod list_portfolio_phase5;
#[path = "internal/metrics/list_search_metrics.rs"]
mod list_search_metrics;
#[path = "internal/routing/load_balance.rs"]
mod load_balance_tests;
#[path = "internal/search/ls_cop.rs"]
mod ls_cop;
#[path = "internal/search/ls_element_functional.rs"]
mod ls_element_functional_tests;
#[path = "internal/search/ls_objective_scope.rs"]
mod ls_objective_scope_tests;
#[path = "internal/search/ls_penalty_warm_start.rs"]
mod ls_penalty_warm_start_tests;
#[path = "internal/search/ls_unary_domains.rs"]
mod ls_unary_domains_tests;
#[path = "internal/search/ls_violation_oracle.rs"]
mod ls_violation_oracle;
#[path = "internal/limits/mem_limit.rs"]
mod mem_limit;
#[path = "internal/model/model_architecture.rs"]
mod model_architecture;
#[path = "internal/orchestrator/orchestrator_architecture.rs"]
mod orchestrator_architecture;
#[path = "internal/orchestrator/orchestrator_budget.rs"]
mod orchestrator_budget;
#[path = "internal/routing/ordering_warm_start.rs"]
mod ordering_warm_start;
#[path = "internal/metrics/phase4_quality_bench.rs"]
mod phase4_quality_bench;
#[path = "internal/metrics/phase5_portfolio_bench.rs"]
mod phase5_portfolio_bench;
#[path = "internal/routing/phase6_dual.rs"]
mod phase6_dual;
#[path = "internal/model/phase7.rs"]
mod phase7;
#[path = "internal/orchestrator/phase9_phase10.rs"]
mod phase9_phase10;
#[path = "internal/constraints/positive_difference.rs"]
mod positive_difference_tests;
#[path = "internal/constraints/prefix_set_compaction.rs"]
mod prefix_set_compaction;
#[path = "internal/scheduling/resource_schedule.rs"]
mod resource_schedule;
#[path = "internal/routing/routing_alns_elite.rs"]
mod routing_alns_elite;
#[path = "internal/routing/routing_neighborhoods.rs"]
mod routing_neighborhoods;
#[path = "internal/routing/routing_search.rs"]
mod routing_search;
#[path = "internal/sat/sat_preprocess_interrupt.rs"]
mod sat_preprocess_interrupt;
#[path = "internal/scheduling/scenario_schedule.rs"]
mod scenario_schedule;
#[path = "internal/scheduling/schedule_cancellation.rs"]
mod schedule_cancellation;
#[path = "internal/scheduling/schedule_compile_contract.rs"]
mod schedule_compile_contract;
#[path = "internal/scheduling/schedule_elite.rs"]
mod schedule_elite;
#[path = "internal/scheduling/schedule_interruption.rs"]
mod schedule_interruption;
#[path = "internal/scheduling/schedule_iteration_limit.rs"]
mod schedule_iteration_limit;
#[path = "internal/scheduling/schedule_lns.rs"]
mod schedule_lns;
#[path = "internal/scheduling/schedule_ls_interruption.rs"]
mod schedule_ls_interruption;
#[path = "internal/scheduling/schedule_mode_identity.rs"]
mod schedule_mode_identity;
#[path = "internal/scheduling/schedule_portfolio_exchange.rs"]
mod schedule_portfolio_exchange;
#[path = "internal/scheduling/schedule_relink.rs"]
mod schedule_relink;
#[path = "internal/scheduling/schedule_state.rs"]
mod schedule_state;
#[path = "internal/scheduling/schedule_tsab_adaptive.rs"]
mod schedule_tsab_adaptive;
#[path = "internal/scheduling/scheduling_kernel.rs"]
mod scheduling_kernel;
#[path = "internal/orchestrator/semantic_preparation_interruption.rs"]
mod semantic_preparation_interruption;
#[path = "internal/constraints/sum_squares.rs"]
mod sum_squares_tests;
#[path = "internal/orchestrator/verification_boundaries.rs"]
mod verification_boundaries;
#[path = "internal/routing/vrptw_search.rs"]
mod vrptw_search;

#[path = "internal/constraints/unary_edge_finding.rs"]
mod unary_edge_finding;

#[path = "internal/lcg/lcg_nogoods.rs"]
mod lcg_nogoods_tests;

#[path = "internal/constraints/cumulative_profile.rs"]
mod cumulative_profile_tests;

#[path = "internal/metrics/cumulative_metrics.rs"]
mod cumulative_metrics;

#[path = "internal/certificates/certificate_hash.rs"]
mod certificate_hash_tests;

#[path = "internal/certificates/certificate_producer.rs"]
mod certificate_producer_tests;

#[path = "internal/certificates/certificate_runtime.rs"]
mod certificate_runtime_tests;

#[path = "internal/constraints/graph_constraints.rs"]
mod graph_constraints;
#[path = "internal/lists/list_exact_engine.rs"]
mod list_exact_engine;
#[path = "internal/routing/routing_engine.rs"]
mod routing_engine;
#[path = "internal/sat/sat_proof.rs"]
mod sat_proof;
