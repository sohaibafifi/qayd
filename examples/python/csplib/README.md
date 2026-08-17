# CSPLib models with Qayd

This directory contains Qayd/Python models based on the
[CSPLib numerical catalog](https://www.csplib.org/Problems/). Models can be used
as command-line examples or imported as regular Python modules.

## Run

From the repository root:

```bash
uv run python -m examples.python.csplib list
uv run python -m examples.python.csplib prob007 --size 12
uv run python -m examples.python.csplib 54 --size 8
```

Every problem module exposes a small library API as well as its command line:

- `build_model(...)` creates a Qayd model and named variable handles.
- `decode(...)` converts a `qayd.Solution` into problem-domain values.
- `validate(...)` independently replays the CSPLib constraints.
- `main(...)` provides a reproducible CLI with time, seed, thread, and engine
  controls.

Problem 019 has two builders because CSPLib groups magic squares and magic
sequences under the same identifier.

## Models

Use `uv run python -m examples.python.csplib <ID> --help` for the arguments and
instance format accepted by a model. The module name in the last column can be
imported below `examples.python.csplib.problems`.

| ID | Model | Python module |
| --- | --- | --- |
| prob001 | Car Sequencing | `prob001_car_sequencing` |
| prob002 | Template Design | `prob002_template_design` |
| prob003 | Quasigroup Existence | `prob003_quasigroup` |
| prob004 | Mystery Shopper | `prob004_mystery_shopper` |
| prob005 | Low Autocorrelation Binary Sequences | `prob005_labs` |
| prob006 | Golomb rulers | `prob006_golomb_ruler` |
| prob007 | All-Interval Series | `prob007_all_interval` |
| prob008 | Vessel Loading | `prob008_vessel_loading` |
| prob009 | Perfect Square Placement | `prob009_perfect_square` |
| prob010 | Social Golfers Problem | `prob010_social_golfers` |
| prob011 | ACC Basketball Schedule | `prob011_acc_basketball` |
| prob012 | Nonogram | `prob012_nonogram` |
| prob013 | Progressive Party | `prob013_progressive_party` |
| prob014 | Solitaire Battleships | `prob014_battleships` |
| prob015 | Schur's Lemma | `prob015_schur` |
| prob016 | Traffic Lights | `prob016_traffic_lights` |
| prob017 | Ramsey Numbers | `prob017_ramsey` |
| prob018 | Water Bucket Problem | `prob018_water_buckets` |
| prob019 | Magic Squares and Sequences | `prob019_magic_square` |
| prob020 | Darts Tournament | `prob020_darts` |
| prob021 | Crossfigures | `prob021_crossfigures` |
| prob022 | Bus Driver Scheduling | `prob022_bus_driver` |
| prob023 | Magic Hexagon | `prob023_magic_hexagon` |
| prob024 | Langford's number problem | `prob024_langford` |
| prob025 | Lam's Projective-Plane Problem | `prob025_lams_problem` |
| prob026 | Sports Tournament Scheduling | `prob026_sports_tournament` |
| prob027 | Alien Tiles Problem | `prob027_alien_tiles` |
| prob028 | Balanced Incomplete Block Designs | `prob028_bibd` |
| prob029 | Prime queen attacking problem | `prob029_prime_queen` |
| prob030 | Balanced Academic Curriculum | `prob030_bacp` |
| prob031 | Rack Configuration | `prob031_rack_configuration` |
| prob032 | Maximum density still life | `prob032_still_life` |
| prob033 | Word Design for DNA Computing | `prob033_dna_word_design` |
| prob034 | Warehouse Location Problem | `prob034_warehouse_location` |
| prob035 | Molnar's Determinant Problem | `prob035_molnar` |
| prob036 | Fixed Length Error Correcting Codes | `prob036_error_correcting_codes` |
| prob037 | Peg Solitaire | `prob037_peg_solitaire` |
| prob038 | Steel Mill Slab Design | `prob038_steel_mill` |
| prob039 | Rehearsal Scheduling | `prob039_rehearsal` |
| prob040 | Multi-Level Distribution | `prob040_distribution` |
| prob041 | The N-Fractions Puzzle | `prob041_n_fractions` |
| prob042 | Diagnosis of Digital Circuits | `prob042_diagnosis` |
| prob043 | Differential Diagnosis | `prob043_differential_diagnosis` |
| prob044 | Steiner triple systems | `prob044_steiner_triples` |
| prob045 | Covering Arrays | `prob045_covering_array` |
| prob046 | Meeting Scheduling | `prob046_meeting_scheduling` |
| prob047 | Supply Chain Coordination | `prob047_supply_chain` |
| prob048 | Minimum Energy Broadcast | `prob048_minimum_energy_broadcast` |
| prob049 | Number Partitioning | `prob049_number_partitioning` |
| prob050 | Diamond-free Degree Sequences | `prob050_diamond_free` |
| prob051 | Tank Allocation | `prob051_tank_allocation` |
| prob052 | Extremal Graphs with Small Girth | `prob052_extremal_graphs` |
| prob053 | Graceful Graphs | `prob053_graceful_graph` |
| prob054 | N-Queens | `prob054_n_queens` |
| prob055 | Equidistant Frequency Permutation Arrays | `prob055_efpa` |
| prob056 | Synchronous Optical Networking Problem | `prob056_sonet` |
| prob057 | Killer Sudoku | `prob057_killer_sudoku` |
| prob058 | Discrete Lot Sizing | `prob058_discrete_lot_sizing` |
| prob059 | Energy-Cost Aware Scheduling | `prob059_energy_scheduling` |
| prob060 | Ridesharing | `prob060_ridesharing` |
| prob061 | Resource-Constrained Project Scheduling | `prob061_rcpsp` |
| prob062 | Interview Assignment | `prob062_interview_assignment` |
| prob063 | Winner Determination Problem | `prob063_combinatorial_auction` |
| prob064 | Generalized Balanced Academic Curriculum | `prob064_generalized_bacp` |
| prob065 | Optimal Financial Portfolio Design | `prob065_portfolio_design` |
| prob066 | Distance-Based Constrained Clustering | `prob066_constrained_clustering` |
| prob067 | Quasigroup Completion | `prob067_quasigroup_completion` |
| prob068 | Travelling Tournament with Predefined Venues | `prob068_ttppv` |
| prob069 | Balanced Nursing Workload | `prob069_nursing_workload` |
| prob070 | Distributed Channel Assignment | `prob070_channel_assignment` |
| prob071 | Network Design | `prob071_network_design` |
| prob072 | Target Tracking | `prob072_target_tracking` |
| prob073 | Test Scheduling | `prob073_test_scheduling` |
| prob074 | Maximum Clique | `prob074_maximum_clique` |
| prob075 | Product Matrix Travelling Salesman | `prob075_product_matrix_tsp` |
| prob076 | Costas Arrays | `prob076_costas_array` |
| prob077 | Stochastic Assignment and Scheduling | `prob077_stochastic_scheduling` |
| prob078 | Train Traffic Rescheduling | `prob078_train_rescheduling` |
| prob079 | n-Queens Completion and Excluded Diagonals | `prob079_queens_completion` |
| prob080 | Blocked n-Queens Problem | `prob080_blocked_queens` |
| prob081 | Black Hole | `prob081_black_hole` |
| prob082 | Patient Transportation | `prob082_patient_transportation` |
| prob083 | Transshipment problem | `prob083_transshipment` |
| prob084 | Hadamard Legendre Pairs | `prob084_hadamard` |
| prob085 | Bookshelves | `prob085_bookshelves` |
| prob086 | Capacitated Vehicle Routing | `prob086_cvrp` |
| prob087 | Rotating Rostering | `prob087_rotating_roster` |
| prob088 | Plotting | `prob088_plotting` |
| prob089 | Medical Appointment Scheduling | `prob089_medical_appointment` |
| prob090 | WordPress Cloud Deployment | `prob090_wordpress_deployment` |
| prob091 | Medical Appointment Sequence Scheduling | `prob091_medical_sequence` |
| prob110 | Peaceably Co-Existing Armies of Queens | `prob110_peaceable_queens` |
| prob115 | Tail Assignment | `prob115_tail_assignment` |
| prob116 | Vellino's Coloured-Bin Problem | `prob116_vellino` |
| prob131 | Production Line Sequencing | `prob131_production_line` |
| prob132 | Layout Problem | `prob132_layout` |
| prob133 | Knapsack Problem | `prob133_knapsack` |
