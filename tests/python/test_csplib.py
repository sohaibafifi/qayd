"""Correctness checks for the CSPLib collection."""

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

cp = pytest.importorskip("qayd")

from examples.python.csplib.catalog import (
    ALL_PROBLEM_IDS,
    IMPLEMENTATIONS,
    coverage_counts,
    normalize_problem_id,
)
from examples.python.csplib.problems import (
    prob001_car_sequencing,
    prob003_quasigroup,
    prob005_labs,
    prob006_golomb_ruler,
    prob007_all_interval,
    prob010_social_golfers,
    prob012_nonogram,
    prob014_battleships,
    prob015_schur,
    prob017_ramsey,
    prob018_water_buckets,
    prob019_magic_square,
    prob023_magic_hexagon,
    prob024_langford,
    prob027_alien_tiles,
    prob028_bibd,
    prob029_prime_queen,
    prob032_still_life,
    prob034_warehouse_location,
    prob036_error_correcting_codes,
    prob044_steiner_triples,
    prob049_number_partitioning,
    prob050_diamond_free,
    prob052_extremal_graphs,
    prob053_graceful_graph,
    prob054_n_queens,
    prob055_efpa,
    prob056_sonet,
    prob057_killer_sudoku,
    prob063_combinatorial_auction,
    prob067_quasigroup_completion,
    prob074_maximum_clique,
    prob076_costas_array,
    prob079_queens_completion,
    prob080_blocked_queens,
    prob081_black_hole,
    prob083_transshipment,
    prob133_knapsack,
)


def solve(model, *, time_limit=10):
    return model.solve(engine="exact", time_limit=time_limit, seed=0, threads=1)


def test_catalog_covers_the_current_97_problem_ids():
    assert len(ALL_PROBLEM_IDS) == 97
    assert len(set(ALL_PROBLEM_IDS)) == 97
    assert ALL_PROBLEM_IDS[:2] == ("prob001", "prob002")
    assert ALL_PROBLEM_IDS[-6:] == (
        "prob110",
        "prob115",
        "prob116",
        "prob131",
        "prob132",
        "prob133",
    )
    assert normalize_problem_id("7") == "prob007"
    assert normalize_problem_id("prob054") == "prob054"
    assert coverage_counts() == (97, 0, 97)
    assert set(IMPLEMENTATIONS) == set(ALL_PROBLEM_IDS)
    assert IMPLEMENTATIONS["prob003"].status == "complete"
    with pytest.raises(ValueError):
        normalize_problem_id("prob999")


def test_prob003_qg3_order_four():
    built = prob003_quasigroup.build_model(4, law=3)
    solution = solve(built.model)
    assert solution.is_sat()
    table = prob003_quasigroup.decode(built, solution)
    prob003_quasigroup.validate(table, law=3)


@pytest.mark.parametrize("law", [1, 2])
def test_prob003_qg1_and_qg2_order_four(law):
    built = prob003_quasigroup.build_model(4, law=law)
    solution = solve(built.model)
    assert solution.is_sat()
    table = prob003_quasigroup.decode(built, solution)
    prob003_quasigroup.validate(table, law=law)


def test_prob001_official_car_sequencing_sample():
    instance = prob001_car_sequencing.parse_instance(
        prob001_car_sequencing.SAMPLE_INSTANCE
    )
    built = prob001_car_sequencing.build_model(instance)
    solution = solve(built.model)
    sequence = prob001_car_sequencing.decode(built, solution)
    prob001_car_sequencing.validate(sequence, instance)


def test_prob005_open_and_periodic_labs():
    open_model = prob005_labs.build_model(5)
    open_solution = solve(open_model.model)
    assert open_solution.status == "OPTIMAL"
    assert open_solution.objective == 2
    open_sequence = prob005_labs.decode(open_model, open_solution)
    prob005_labs.validate(
        open_sequence, periodic=False, objective=open_solution.objective
    )

    periodic_model = prob005_labs.build_model(4, periodic=True)
    periodic_solution = solve(periodic_model.model)
    periodic_sequence = prob005_labs.decode(periodic_model, periodic_solution)
    prob005_labs.validate(
        periodic_sequence, periodic=True, objective=periodic_solution.objective
    )


def test_prob006_five_mark_optimum_is_eleven():
    built = prob006_golomb_ruler.build_model(5)
    solution = solve(built.model)
    assert solution.status == "OPTIMAL"
    assert solution.objective == 11
    ruler = prob006_golomb_ruler.decode(built, solution)
    prob006_golomb_ruler.validate(ruler)


def test_prob007_all_interval_size_eight():
    built = prob007_all_interval.build_model(8)
    solution = solve(built.model)
    series, intervals = prob007_all_interval.decode(built, solution)
    prob007_all_interval.validate(series, intervals)


def test_prob010_social_golfers_round_robin_pairs():
    built = prob010_social_golfers.build_model(3, 2, 5)
    solution = solve(built.model)
    schedule = prob010_social_golfers.decode(built, solution)
    prob010_social_golfers.validate(schedule, groups=3, group_size=2)


def test_prob012_cross_nonogram():
    clues = [[1], [3], [5], [3], [1]]
    built = prob012_nonogram.build_model(clues, clues)
    solution = solve(built.model)
    grid = prob012_nonogram.decode(built, solution)
    prob012_nonogram.validate(grid, clues, clues)


def test_prob014_battleships_default_instance():
    instance = prob014_battleships.parse_instance(
        json.dumps(prob014_battleships.DEFAULT_INSTANCE)
    )
    built = prob014_battleships.build_model(instance)
    solution = solve(built.model)
    ships = prob014_battleships.decode(built, solution)
    prob014_battleships.validate(built, ships)
    assert prob014_battleships.render(built, ships).count("#") == sum(instance.fleet)


def test_prob015_schur_boundary_for_three_boxes():
    feasible = prob015_schur.build_model(13)
    feasible_solution = solve(feasible.model)
    assignment = prob015_schur.decode(feasible, feasible_solution)
    prob015_schur.validate(assignment, boxes=3)

    infeasible = prob015_schur.build_model(14)
    assert solve(infeasible.model).status == "UNSATISFIABLE"


def test_prob017_ramsey_three_three_boundary():
    feasible = prob017_ramsey.build_two_colour_model(5, red_clique=3, blue_clique=3)
    feasible_solution = solve(feasible.model)
    colouring = prob017_ramsey.decode(feasible, feasible_solution)
    prob017_ramsey.validate(feasible, colouring)

    infeasible = prob017_ramsey.build_two_colour_model(6, red_clique=3, blue_clique=3)
    assert solve(infeasible.model).status == "UNSATISFIABLE"


def test_prob017_multicolour_triangle_variant():
    built = prob017_ramsey.build_triangle_model(5, colours=3)
    solution = solve(built.model)
    colouring = prob017_ramsey.decode(built, solution)
    prob017_ramsey.validate(built, colouring)


def test_prob018_water_bucket_minimum_is_seven_transfers():
    built = prob018_water_buckets.build_model(
        (8, 5, 3), (8, 0, 0), (4, 4, 0), max_steps=8
    )
    solution = solve(built.model)
    assert solution.status == "OPTIMAL"
    assert solution.objective == 7
    states, moved = prob018_water_buckets.decode(built, solution)
    prob018_water_buckets.validate(built, states, moved, solution.objective)


def test_prob019_order_three_magic_square():
    built = prob019_magic_square.build_magic_square(3)
    solution = solve(built.model)
    square = prob019_magic_square.decode_magic_square(built, solution)
    prob019_magic_square.validate_magic_square(square)


def test_prob019_length_four_magic_sequence():
    built = prob019_magic_square.build_magic_sequence(4)
    solution = solve(built.model)
    sequence = prob019_magic_square.decode_magic_sequence(built, solution)
    prob019_magic_square.validate_magic_sequence(sequence)


def test_prob023_order_three_magic_hexagon():
    built = prob023_magic_hexagon.build_model(3)
    solution = solve(built.model)
    values_by_coordinate = prob023_magic_hexagon.decode(built, solution)
    prob023_magic_hexagon.validate(values_by_coordinate, order=3)
    assert built.magic_sum == 38

    impossible = prob023_magic_hexagon.build_model(2)
    assert solve(impossible.model).status == "UNSATISFIABLE"


def test_prob024_langford_feasible_and_infeasible_orders():
    feasible = prob024_langford.build_model(4)
    feasible_solution = solve(feasible.model)
    sequence = prob024_langford.decode(feasible, feasible_solution)
    prob024_langford.validate(sequence, pair_count=4)

    infeasible = prob024_langford.build_model(5)
    assert solve(infeasible.model).status == "UNSATISFIABLE"


def test_prob027_alien_tiles_goal_and_hardest_small_board():
    goal = ((1, 1), (1, 0))
    built = prob027_alien_tiles.build_model(goal, colours=3)
    solution = solve(built.model)
    clicks = prob027_alien_tiles.decode(built, solution)
    prob027_alien_tiles.validate(built, clicks, solution.objective)
    assert solution.objective == 1

    hardest = prob027_alien_tiles.find_hardest_goal(2, 2)
    assert hardest.minimum_clicks == 4


def test_prob028_fano_plane_bibd():
    built = prob028_bibd.build_model(7, 7, 3, 3, 1)
    solution = solve(built.model)
    matrix = prob028_bibd.decode(built, solution)
    prob028_bibd.validate(matrix, built.parameters)


def test_prob029_five_by_five_prime_queen_has_no_free_primes():
    built = prob029_prime_queen.build_model(5)
    solution = solve(built.model)
    assert solution.status == "OPTIMAL"
    assert solution.objective == 0
    locations, queen = prob029_prime_queen.decode(built, solution)
    prob029_prime_queen.validate(built, locations, queen, solution.objective)


def test_prob032_three_by_three_still_life_optimum():
    built = prob032_still_life.build_model(3)
    solution = solve(built.model)
    assert solution.status == "OPTIMAL"
    assert solution.objective == 6
    grid = prob032_still_life.decode(built, solution)
    prob032_still_life.validate(grid)


def test_prob034_official_warehouse_sample_optimum():
    instance = prob034_warehouse_location.parse_instance(
        json.dumps(prob034_warehouse_location.DEFAULT_INSTANCE)
    )
    built = prob034_warehouse_location.build_model(instance)
    solution = solve(built.model)
    assert solution.status == "OPTIMAL"
    assert solution.objective == 383
    opened, assignments = prob034_warehouse_location.decode(built, solution)
    prob034_warehouse_location.validate(built, opened, assignments, solution.objective)


def test_prob036_hamming_and_lee_codes():
    for metric, alphabet in (("hamming", 2), ("lee", 4)):
        built = prob036_error_correcting_codes.build_model(
            4,
            3,
            alphabet,
            2,
            metric=metric,
        )
        solution = solve(built.model)
        codewords = prob036_error_correcting_codes.decode(built, solution)
        prob036_error_correcting_codes.validate(built, codewords)


def test_prob044_order_seven_steiner_system():
    built = prob044_steiner_triples.build_model(7)
    solution = solve(built.model)
    triples = prob044_steiner_triples.decode(built, solution)
    prob044_steiner_triples.validate(triples, order=7)

    impossible = prob044_steiner_triples.build_model(6)
    assert solve(impossible.model).status == "UNSATISFIABLE"


def test_prob049_partition_of_one_through_eight():
    built = prob049_number_partitioning.build_model(8, highest_power=2)
    solution = solve(built.model)
    first, second = prob049_number_partitioning.decode(built, solution)
    prob049_number_partitioning.validate(first, second, size=8, highest_power=2)


def test_prob050_order_ten_unique_degree_sequence():
    built = prob050_diamond_free.build_model(10)
    solution = solve(built.model)
    degrees, selected_edges = prob050_diamond_free.decode(built, solution)
    prob050_diamond_free.validate(built, degrees, selected_edges)
    assert degrees == [6, 6, 3, 3, 3, 3, 3, 3, 3, 3]
    assert prob050_diamond_free.enumerate_degree_sequences(10) == [tuple(degrees)]


def test_prob052_order_five_triangle_free_extremal_graph():
    built = prob052_extremal_graphs.build_model(5, 3)
    solution = solve(built.model)
    assert solution.status == "OPTIMAL"
    assert solution.objective == 6
    selected_edges = prob052_extremal_graphs.decode(built, solution)
    prob052_extremal_graphs.validate(built, selected_edges, solution.objective)
    assert prob052_extremal_graphs.count_non_isomorphic_extremal(5, 3, 6) == 1


def test_prob053_graceful_path():
    edges = prob053_graceful_graph.graph_edges("path", 6)
    built = prob053_graceful_graph.build_model(6, edges)
    solution = solve(built.model)
    labels = prob053_graceful_graph.decode(built, solution)
    prob053_graceful_graph.validate(labels, built.edges)


def test_prob054_eight_queens():
    built = prob054_n_queens.build_model(8)
    solution = solve(built.model)
    rows = prob054_n_queens.decode(built, solution)
    prob054_n_queens.validate(rows)
    rendered = prob054_n_queens.render(rows)
    assert rendered.count("Q") == 8


def test_prob055_official_small_efpa_parameters():
    built = prob055_efpa.build_model(5, 3, 2, 4)
    solution = solve(built.model)
    codewords = prob055_efpa.decode(built, solution)
    prob055_efpa.validate(built, codewords)


def test_prob056_sonet_default_optimum_is_six_adms():
    instance = prob056_sonet.parse_instance(json.dumps(prob056_sonet.DEFAULT_INSTANCE))
    built = prob056_sonet.build_model(instance)
    solution = solve(built.model)
    assert solution.status == "OPTIMAL"
    assert solution.objective == 6
    rings, flows = prob056_sonet.decode(built, solution)
    prob056_sonet.validate(built, rings, flows, solution.objective)


def test_prob057_generalized_four_by_four_killer_sudoku():
    size, cages = prob057_killer_sudoku.parse_instance(
        json.dumps(prob057_killer_sudoku.DEFAULT_INSTANCE)
    )
    built = prob057_killer_sudoku.build_model(size, cages)
    solution = solve(built.model)
    grid = prob057_killer_sudoku.decode(built, solution)
    prob057_killer_sudoku.validate(built, grid)


def test_prob063_combinatorial_auction_default_optimum():
    instance = prob063_combinatorial_auction.parse_instance(
        json.dumps(prob063_combinatorial_auction.DEFAULT_INSTANCE)
    )
    built = prob063_combinatorial_auction.build_model(instance)
    solution = solve(built.model)
    assert solution.status == "OPTIMAL"
    assert solution.objective == 21
    accepted = prob063_combinatorial_auction.decode(built, solution)
    prob063_combinatorial_auction.validate(built, accepted, solution.objective)


def test_prob067_quasigroup_completion_preserves_clues():
    clues = prob067_quasigroup_completion.parse_grid("1,.,.,4;.,.,2,.;3,.,1,.;.,3,.,.")
    built = prob067_quasigroup_completion.build_model(clues)
    solution = solve(built.model)
    table = prob067_quasigroup_completion.decode(built, solution)
    prob067_quasigroup_completion.validate(table, clues)


def test_prob074_dimacs_sample_has_maximum_clique_three():
    graph = prob074_maximum_clique.parse_dimacs(
        prob074_maximum_clique.SAMPLE_DIMACS.splitlines()
    )
    built = prob074_maximum_clique.build_model(graph)
    solution = solve(built.model)
    assert solution.status == "OPTIMAL"
    assert solution.objective == 3
    clique = prob074_maximum_clique.decode(built, solution)
    prob074_maximum_clique.validate(clique, graph)


def test_prob076_order_six_costas_array():
    built = prob076_costas_array.build_model(6)
    solution = solve(built.model)
    permutation = prob076_costas_array.decode(built, solution)
    prob076_costas_array.validate(permutation)


def test_prob079_completion_and_excluded_diagonals():
    completion = prob079_queens_completion.build_completion_model(
        8, preplaced=((0, 0),)
    )
    completion_solution = solve(completion.model)
    completion_queens = prob079_queens_completion.decode(
        completion, completion_solution
    )
    prob079_queens_completion.validate(completion, completion_queens)

    excluded = prob079_queens_completion.build_excluded_diagonals_model(
        8,
        excluded_sums=(0,),
        excluded_differences=(0,),
    )
    excluded_solution = solve(excluded.model)
    excluded_queens = prob079_queens_completion.decode(excluded, excluded_solution)
    prob079_queens_completion.validate(excluded, excluded_queens)


def test_prob080_blocked_queens_avoids_blocked_squares():
    blocked = frozenset({(0, 0), (7, 7)})
    built = prob080_blocked_queens.build_model(8, blocked=blocked)
    solution = solve(built.model)
    rows = prob080_blocked_queens.decode(built, solution)
    prob080_blocked_queens.validate(rows, blocked=blocked)


def test_prob081_black_hole_default_deal():
    instance = prob081_black_hole.parse_instance(
        json.dumps(prob081_black_hole.DEFAULT_INSTANCE)
    )
    built = prob081_black_hole.build_model(instance)
    solution = solve(built.model)
    play_order = prob081_black_hole.decode(built, solution)
    prob081_black_hole.validate(built, play_order)


def test_prob083_transshipment_default_optimum():
    instance = prob083_transshipment.parse_instance(
        json.dumps(prob083_transshipment.DEFAULT_INSTANCE)
    )
    built = prob083_transshipment.build_model(instance)
    solution = solve(built.model)
    assert solution.status == "OPTIMAL"
    assert solution.objective == 23
    inbound, outbound = prob083_transshipment.decode(built, solution)
    prob083_transshipment.validate(built, inbound, outbound, solution.objective)


def test_prob133_csplib_sample_optimum_is_eighty():
    instance = prob133_knapsack.parse_essence_parameter(prob133_knapsack.SAMPLE_ESSENCE)
    built = prob133_knapsack.build_model(instance)
    solution = solve(built.model)
    assert solution.status == "OPTIMAL"
    assert solution.objective == 80
    selected = prob133_knapsack.decode(built, solution)
    prob133_knapsack.validate(built, selected, solution.objective)


@pytest.mark.parametrize("problem_id", ALL_PROBLEM_IDS)
def test_every_registered_cli_default_instance(problem_id, capsys):
    implementation = IMPLEMENTATIONS[problem_id]
    module = importlib.import_module(implementation.module)
    assert module.main(["--time-limit", "30"]) == 0
    output = capsys.readouterr().out
    assert problem_id in output
    assert "status=" in output
