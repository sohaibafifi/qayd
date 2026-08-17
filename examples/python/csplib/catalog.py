"""Canonical CSPLib inventory and local implementation registry."""

from __future__ import annotations

from dataclasses import dataclass

ALL_PROBLEM_IDS = tuple(f"prob{number:03d}" for number in range(1, 92)) + (
    "prob110",
    "prob115",
    "prob116",
    "prob131",
    "prob132",
    "prob133",
)


@dataclass(frozen=True)
class Implementation:
    """One runnable local implementation and its current coverage level."""

    title: str
    module: str
    status: str = "complete"
    scope: str = ""


IMPLEMENTATIONS = {
    "prob001": Implementation(
        "Car Sequencing",
        "examples.python.csplib.problems.prob001_car_sequencing",
    ),
    "prob003": Implementation(
        "Quasigroup Existence",
        "examples.python.csplib.problems.prob003_quasigroup",
        scope="Latin squares with QG1 through QG7 laws",
    ),
    "prob005": Implementation(
        "Low Autocorrelation Binary Sequences",
        "examples.python.csplib.problems.prob005_labs",
    ),
    "prob006": Implementation(
        "Golomb rulers",
        "examples.python.csplib.problems.prob006_golomb_ruler",
    ),
    "prob007": Implementation(
        "All-Interval Series",
        "examples.python.csplib.problems.prob007_all_interval",
    ),
    "prob010": Implementation(
        "Social Golfers Problem",
        "examples.python.csplib.problems.prob010_social_golfers",
    ),
    "prob012": Implementation(
        "Nonogram",
        "examples.python.csplib.problems.prob012_nonogram",
    ),
    "prob014": Implementation(
        "Solitaire Battleships",
        "examples.python.csplib.problems.prob014_battleships",
    ),
    "prob015": Implementation(
        "Schur's Lemma",
        "examples.python.csplib.problems.prob015_schur",
    ),
    "prob017": Implementation(
        "Ramsey Numbers",
        "examples.python.csplib.problems.prob017_ramsey",
    ),
    "prob018": Implementation(
        "Water Bucket Problem",
        "examples.python.csplib.problems.prob018_water_buckets",
    ),
    "prob019": Implementation(
        "Magic Squares and Sequences",
        "examples.python.csplib.problems.prob019_magic_square",
    ),
    "prob024": Implementation(
        "Langford's number problem",
        "examples.python.csplib.problems.prob024_langford",
    ),
    "prob023": Implementation(
        "Magic Hexagon",
        "examples.python.csplib.problems.prob023_magic_hexagon",
    ),
    "prob027": Implementation(
        "Alien Tiles Problem",
        "examples.python.csplib.problems.prob027_alien_tiles",
    ),
    "prob028": Implementation(
        "Balanced Incomplete Block Designs",
        "examples.python.csplib.problems.prob028_bibd",
    ),
    "prob029": Implementation(
        "Prime queen attacking problem",
        "examples.python.csplib.problems.prob029_prime_queen",
    ),
    "prob032": Implementation(
        "Maximum density still life",
        "examples.python.csplib.problems.prob032_still_life",
    ),
    "prob034": Implementation(
        "Warehouse Location Problem",
        "examples.python.csplib.problems.prob034_warehouse_location",
    ),
    "prob036": Implementation(
        "Fixed Length Error Correcting Codes",
        "examples.python.csplib.problems.prob036_error_correcting_codes",
    ),
    "prob044": Implementation(
        "Steiner triple systems",
        "examples.python.csplib.problems.prob044_steiner_triples",
    ),
    "prob049": Implementation(
        "Number Partitioning",
        "examples.python.csplib.problems.prob049_number_partitioning",
    ),
    "prob050": Implementation(
        "Diamond-free Degree Sequences",
        "examples.python.csplib.problems.prob050_diamond_free",
    ),
    "prob052": Implementation(
        "Extremal Graphs with Small Girth",
        "examples.python.csplib.problems.prob052_extremal_graphs",
    ),
    "prob053": Implementation(
        "Graceful Graphs",
        "examples.python.csplib.problems.prob053_graceful_graph",
    ),
    "prob054": Implementation(
        "N-Queens",
        "examples.python.csplib.problems.prob054_n_queens",
    ),
    "prob055": Implementation(
        "Equidistant Frequency Permutation Arrays",
        "examples.python.csplib.problems.prob055_efpa",
    ),
    "prob056": Implementation(
        "Synchronous Optical Networking Problem",
        "examples.python.csplib.problems.prob056_sonet",
    ),
    "prob057": Implementation(
        "Killer Sudoku",
        "examples.python.csplib.problems.prob057_killer_sudoku",
    ),
    "prob063": Implementation(
        "Winner Determination Problem",
        "examples.python.csplib.problems.prob063_combinatorial_auction",
    ),
    "prob067": Implementation(
        "Quasigroup Completion",
        "examples.python.csplib.problems.prob067_quasigroup_completion",
    ),
    "prob074": Implementation(
        "Maximum Clique",
        "examples.python.csplib.problems.prob074_maximum_clique",
    ),
    "prob076": Implementation(
        "Costas Arrays",
        "examples.python.csplib.problems.prob076_costas_array",
    ),
    "prob079": Implementation(
        "n-Queens Completion and Excluded Diagonals",
        "examples.python.csplib.problems.prob079_queens_completion",
    ),
    "prob080": Implementation(
        "Blocked n-Queens Problem",
        "examples.python.csplib.problems.prob080_blocked_queens",
    ),
    "prob081": Implementation(
        "Black Hole",
        "examples.python.csplib.problems.prob081_black_hole",
    ),
    "prob083": Implementation(
        "Transshipment problem",
        "examples.python.csplib.problems.prob083_transshipment",
    ),
    "prob133": Implementation(
        "Knapsack Problem",
        "examples.python.csplib.problems.prob133_knapsack",
    ),
}

_ADDITIONAL_IMPLEMENTATIONS = {
    "prob002": ("Template Design", "prob002_template_design"),
    "prob004": ("Mystery Shopper", "prob004_mystery_shopper"),
    "prob008": ("Vessel Loading", "prob008_vessel_loading"),
    "prob009": ("Perfect Square Placement", "prob009_perfect_square"),
    "prob011": ("ACC Basketball Schedule", "prob011_acc_basketball"),
    "prob013": ("Progressive Party", "prob013_progressive_party"),
    "prob016": ("Traffic Lights", "prob016_traffic_lights"),
    "prob020": ("Darts Tournament", "prob020_darts"),
    "prob021": ("Crossfigures", "prob021_crossfigures"),
    "prob022": ("Bus Driver Scheduling", "prob022_bus_driver"),
    "prob025": ("Lam's Projective-Plane Problem", "prob025_lams_problem"),
    "prob026": ("Sports Tournament Scheduling", "prob026_sports_tournament"),
    "prob030": ("Balanced Academic Curriculum", "prob030_bacp"),
    "prob031": ("Rack Configuration", "prob031_rack_configuration"),
    "prob033": ("Word Design for DNA Computing", "prob033_dna_word_design"),
    "prob035": ("Molnar's Determinant Problem", "prob035_molnar"),
    "prob037": ("Peg Solitaire", "prob037_peg_solitaire"),
    "prob038": ("Steel Mill Slab Design", "prob038_steel_mill"),
    "prob039": ("Rehearsal Scheduling", "prob039_rehearsal"),
    "prob040": ("Multi-Level Distribution", "prob040_distribution"),
    "prob041": ("The N-Fractions Puzzle", "prob041_n_fractions"),
    "prob042": ("Diagnosis of Digital Circuits", "prob042_diagnosis"),
    "prob043": ("Differential Diagnosis", "prob043_differential_diagnosis"),
    "prob045": ("Covering Arrays", "prob045_covering_array"),
    "prob046": ("Meeting Scheduling", "prob046_meeting_scheduling"),
    "prob047": ("Supply Chain Coordination", "prob047_supply_chain"),
    "prob048": ("Minimum Energy Broadcast", "prob048_minimum_energy_broadcast"),
    "prob051": ("Tank Allocation", "prob051_tank_allocation"),
    "prob058": ("Discrete Lot Sizing", "prob058_discrete_lot_sizing"),
    "prob059": ("Energy-Cost Aware Scheduling", "prob059_energy_scheduling"),
    "prob060": ("Ridesharing", "prob060_ridesharing"),
    "prob061": ("Resource-Constrained Project Scheduling", "prob061_rcpsp"),
    "prob062": ("Interview Assignment", "prob062_interview_assignment"),
    "prob064": ("Generalized Balanced Academic Curriculum", "prob064_generalized_bacp"),
    "prob065": ("Optimal Financial Portfolio Design", "prob065_portfolio_design"),
    "prob066": (
        "Distance-Based Constrained Clustering",
        "prob066_constrained_clustering",
    ),
    "prob068": ("Travelling Tournament with Predefined Venues", "prob068_ttppv"),
    "prob069": ("Balanced Nursing Workload", "prob069_nursing_workload"),
    "prob070": ("Distributed Channel Assignment", "prob070_channel_assignment"),
    "prob071": ("Network Design", "prob071_network_design"),
    "prob072": ("Target Tracking", "prob072_target_tracking"),
    "prob073": ("Test Scheduling", "prob073_test_scheduling"),
    "prob075": ("Product Matrix Travelling Salesman", "prob075_product_matrix_tsp"),
    "prob077": (
        "Stochastic Assignment and Scheduling",
        "prob077_stochastic_scheduling",
    ),
    "prob078": ("Train Traffic Rescheduling", "prob078_train_rescheduling"),
    "prob082": ("Patient Transportation", "prob082_patient_transportation"),
    "prob084": ("Hadamard Legendre Pairs", "prob084_hadamard"),
    "prob085": ("Bookshelves", "prob085_bookshelves"),
    "prob086": ("Capacitated Vehicle Routing", "prob086_cvrp"),
    "prob087": ("Rotating Rostering", "prob087_rotating_roster"),
    "prob088": ("Plotting", "prob088_plotting"),
    "prob089": ("Medical Appointment Scheduling", "prob089_medical_appointment"),
    "prob090": ("WordPress Cloud Deployment", "prob090_wordpress_deployment"),
    "prob091": ("Medical Appointment Sequence Scheduling", "prob091_medical_sequence"),
    "prob110": ("Peaceably Co-Existing Armies of Queens", "prob110_peaceable_queens"),
    "prob115": ("Tail Assignment", "prob115_tail_assignment"),
    "prob116": ("Vellino's Coloured-Bin Problem", "prob116_vellino"),
    "prob131": ("Production Line Sequencing", "prob131_production_line"),
    "prob132": ("Layout Problem", "prob132_layout"),
}

IMPLEMENTATIONS.update(
    {
        problem_id: Implementation(title, f"examples.python.csplib.problems.{module}")
        for problem_id, (title, module) in _ADDITIONAL_IMPLEMENTATIONS.items()
    }
)


def normalize_problem_id(value: str | int) -> str:
    """Return a canonical ``probNNN`` identifier and reject unknown IDs."""

    raw = str(value).strip().lower()
    if raw.startswith("prob"):
        raw = raw[4:]
    if not raw.isdigit():
        raise ValueError(f"invalid CSPLib problem identifier: {value!r}")
    problem_id = f"prob{int(raw):03d}"
    if problem_id not in ALL_PROBLEM_IDS:
        raise ValueError(f"unknown CSPLib problem identifier: {problem_id}")
    return problem_id


def coverage_counts() -> tuple[int, int, int]:
    """Return complete, partial, and total counts."""

    complete = sum(item.status == "complete" for item in IMPLEMENTATIONS.values())
    partial = sum(item.status == "partial" for item in IMPLEMENTATIONS.values())
    return complete, partial, len(ALL_PROBLEM_IDS)
