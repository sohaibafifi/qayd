import gzip
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest
from qayd.datasets import (
    DatasetParseError,
    load_instance,
    parse_cvrplib,
    parse_jsplib,
    parse_psplib,
    parse_solomon,
    parse_vrp_solution,
)

CVRPLIB = """\
NAME : X-n3-k2
COMMENT : Optimal value: 12
TYPE : CVRP
DIMENSION : 3
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 10
NODE_COORD_SECTION
1 0 0
2 3 4
3 6 8
DEMAND_SECTION
1 0
2 4
3 5
DEPOT_SECTION
1
-1
EOF
"""


EXPLICIT_CVRPLIB = """\
NAME: explicit
TYPE: CVRP
DIMENSION: 3
CAPACITY: 7
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: LOWER_ROW
EDGE_WEIGHT_SECTION
4 7 5
DEMAND_SECTION
1 0
2 2
3 3
DEPOT_SECTION
1 -1
EOF
"""


SOLOMON = """\
C101-mini

VEHICLE
NUMBER     CAPACITY
  2          10

CUSTOMER
CUST NO.  XCOORD. YCOORD. DEMAND READY TIME DUE DATE SERVICE TIME
0 0 0 0 0 100 0
1 3 4 4 5 20 2
2 6 8 5 10 30 3
"""


JSPLIB = """\
# instance toy-jsp
2 2
0 3 1 2
1 4 0 1
"""


PSPLIB_SM = """\
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  4
horizon                       :  20
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  0   N
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          2           2   3
   2        1          1           4
   3        1          1           4
   4        1          0
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2
  1      1     0       0    0
  2      1     3       2    1
  3      1     4       1    2
  4      1     0       0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2
  4    5
************************************************************************
"""


PSPLIB_MM = """\
projects                      :  1
jobs (incl. supersource/sink ):  3
horizon                       :  15
PRECEDENCE RELATIONS:
jobnr. #modes #successors successors
1 1 1 2
2 2 1 3
3 1 0
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration R 1 N 1
1 1 0 0 0
2 1 3 2 1
  2 4 1 2
3 1 0 0 0
************************************************************************
RESOURCEAVAILABILITIES:
R 1 N 1
3 8
************************************************************************
"""


def test_cvrplib_coordinates_are_normalized_and_rounded():
    instance = parse_cvrplib(CVRPLIB, source="mini.vrp")
    assert instance.name == "X-n3-k2"
    assert instance.dimension == 3
    assert instance.node_ids == (1, 2, 3)
    assert instance.depot == 0
    assert instance.customers == (1, 2)
    assert instance.demands == (0, 4, 5)
    assert instance.edge_weights == ((0, 5, 10), (5, 0, 5), (10, 5, 0))
    assert instance.vehicles == 2
    assert instance.best_known == 12


def test_cvrplib_explicit_triangular_matrix():
    instance = parse_cvrplib(EXPLICIT_CVRPLIB)
    assert instance.edge_weights == ((0, 4, 7), (4, 0, 5), (7, 5, 0))
    assert instance.coordinates == ((0.0, 0.0),) * 3


def test_solomon_and_homberger_shared_format_uses_dimacs_trunc1():
    instance = parse_solomon(SOLOMON)
    assert instance.name == "C101-mini"
    assert instance.vehicles == 2
    assert instance.capacity == 10
    assert instance.demands == (0, 4, 5)
    assert instance.time_windows[1] == (5, 20)
    assert instance.distance_matrix() == ((0, 50, 100), (50, 0, 50), (100, 50, 0))


def test_vrp_solution_can_map_original_cvrplib_ids():
    instance = parse_cvrplib(CVRPLIB)
    solution = parse_vrp_solution("Route #1: 2 3\nCost: 12\n", instance=instance)
    assert solution.routes == ((1, 2),)
    assert solution.cost == 12

    solomon = parse_solomon(SOLOMON)
    solomon_solution = parse_vrp_solution(
        "Route #1: 1 2\nCost: 15.5\n", instance=solomon
    )
    assert solomon_solution.routes == ((1, 2),)
    assert solomon_solution.cost == 15.5


def test_jsplib_parses_operations_and_normalizes_one_based_machines():
    instance = parse_jsplib(JSPLIB)
    assert instance.name == "toy-jsp"
    assert instance.num_jobs == 2
    assert instance.num_machines == 2
    assert instance.machines == ((0, 1), (1, 0))
    assert instance.durations == ((3, 2), (4, 1))
    assert instance.horizon == 10

    one_based = parse_jsplib("1 2\n1 3 2 4\n")
    assert one_based.machines == ((0, 1),)


def test_psplib_single_mode_parses_precedence_resources_and_capacities():
    instance = parse_psplib(PSPLIB_SM, source="j301_1.sm")
    assert instance.name == "j301_1"
    assert instance.horizon == 20
    assert instance.resource_names == ("R1", "R2")
    assert instance.resource_kinds == ("renewable", "renewable")
    assert instance.capacities == (4, 5)
    assert not instance.multi_mode
    assert instance.job(1).successors == (2, 3)
    assert instance.job(2).modes[0].demands == (2, 1)


def test_psplib_multimode_continuation_rows_and_nonrenewables():
    instance = parse_psplib(PSPLIB_MM, source="c15_1.mm")
    assert instance.multi_mode
    assert instance.resource_names == ("R1", "N1")
    assert instance.renewable_resources == (0,)
    assert instance.nonrenewable_resources == (1,)
    assert instance.job(2).modes[1].duration == 4
    assert instance.job(2).modes[1].demands == (1, 2)


@pytest.mark.parametrize(
    ("content", "expected_type"),
    [
        (CVRPLIB, "CVRPLibInstance"),
        (SOLOMON, "SolomonInstance"),
        (JSPLIB, "JobShopInstance"),
        (PSPLIB_SM, "PSPLibInstance"),
    ],
)
def test_load_instance_detects_supported_formats(
    tmp_path: Path, content: str, expected_type: str
):
    path = tmp_path / "instance.txt"
    path.write_text(content)
    assert type(load_instance(path)).__name__ == expected_type


def test_load_instance_reads_common_compressed_files(tmp_path: Path):
    path = tmp_path / "instance.vrp.gz"
    with gzip.open(path, "wt") as stream:
        stream.write(CVRPLIB)
    assert load_instance(path).name == "X-n3-k2"


def test_parser_errors_include_source_and_line():
    broken = CVRPLIB.replace("2 4", "2 invalid")
    with pytest.raises(DatasetParseError, match=r"broken\.vrp:\d+:"):
        parse_cvrplib(broken, source="broken.vrp")


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("api_script", "native_script"),
    [
        ("examples/python/routing/api/vrp.py", "examples/python/routing/native/vrp.py"),
        ("examples/python/routing/api/cvrptw.py", "examples/python/routing/native/cvrptw.py"),
        ("examples/python/scheduling/api/jssp.py", "examples/python/scheduling/native/jssp.py"),
        ("examples/python/scheduling/api/rcpsp.py", "examples/python/scheduling/native/rcpsp.py"),
    ],
)
def test_api_and_native_benchmark_launchers_share_the_same_cli(api_script, native_script):
    def help_options(script):
        result = subprocess.run(
            [sys.executable, str(ROOT / script), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        )
        assert "[instance]" in result.stdout
        return set(re.findall(r"--[a-z][a-z-]*", result.stdout))

    assert help_options(api_script) == help_options(native_script)


def test_python_examples_use_arguments_instead_of_qayd_environment_variables():
    for script in (ROOT / "examples" / "python").rglob("*.py"):
        source = script.read_text(encoding="utf-8")
        assert "os.environ" not in source, script
        assert "QAYD_" not in source, script


def _run_example(script: str, *arguments: str):
    result = subprocess.run(
        [sys.executable, str(ROOT / script), *arguments, "--time-limit", "1", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record["verified"] is True
    assert record["objectives"]
    assert record["status"] in {"SATISFIABLE", "OPTIMAL"}
    assert record["dual_bound"] is not None
    assert record["absolute_gap"] is not None
    assert record["relative_gap"] is not None
    assert record["bound_method"]
    return record


@pytest.mark.parametrize(
    ("script", "arguments", "expected_prefix"),
    [
        (
            "examples/python/routing/api/vrp.py",
            ("--customers", "5", "--vehicles", "2", "--capacity", "30"),
            "generated-cvrp",
        ),
        (
            "examples/python/routing/native/vrp.py",
            ("--customers", "5", "--vehicles", "2", "--capacity", "30"),
            "generated-cvrp",
        ),
        (
            "examples/python/routing/api/cvrptw.py",
            ("--customers", "4", "--vehicles", "4"),
            "generated-vrptw",
        ),
        (
            "examples/python/routing/native/cvrptw.py",
            ("--customers", "4", "--vehicles", "4"),
            "generated-vrptw",
        ),
        (
            "examples/python/scheduling/api/jssp.py",
            ("--jobs", "3", "--machines", "2"),
            "generated-jssp",
        ),
        (
            "examples/python/scheduling/native/jssp.py",
            ("--jobs", "3", "--machines", "2"),
            "generated-jssp",
        ),
        (
            "examples/python/scheduling/api/rcpsp.py",
            ("--tasks", "5", "--resources", "1"),
            "generated-rcpsp",
        ),
        (
            "examples/python/scheduling/native/rcpsp.py",
            ("--tasks", "5", "--resources", "1"),
            "generated-rcpsp",
        ),
    ],
)
def test_api_and_native_examples_generate_without_an_instance(script, arguments, expected_prefix):
    record = _run_example(script, *arguments)
    assert record["instance"].startswith(expected_prefix)


@pytest.mark.parametrize(
    ("script", "instance", "expected_name"),
    [
        (
            "examples/python/routing/api/vrp.py",
            "examples/instances/routing/tiny-cvrp.vrp",
            "X-n5-k2",
        ),
        (
            "examples/python/routing/native/vrp.py",
            "examples/instances/routing/tiny-cvrp.vrp",
            "X-n5-k2",
        ),
        (
            "examples/python/routing/api/cvrptw.py",
            "examples/instances/routing/tiny-vrptw.txt",
            "C101-qayd-smoke",
        ),
        (
            "examples/python/routing/native/cvrptw.py",
            "examples/instances/routing/tiny-vrptw.txt",
            "C101-qayd-smoke",
        ),
        (
            "examples/python/scheduling/api/jssp.py",
            "examples/instances/scheduling/tiny-jssp.txt",
            "tiny-qayd-jssp",
        ),
        (
            "examples/python/scheduling/native/jssp.py",
            "examples/instances/scheduling/tiny-jssp.txt",
            "tiny-qayd-jssp",
        ),
        (
            "examples/python/scheduling/api/rcpsp.py",
            "examples/instances/scheduling/tiny-rcpsp.sm",
            "tiny-rcpsp",
        ),
        (
            "examples/python/scheduling/native/rcpsp.py",
            "examples/instances/scheduling/tiny-rcpsp.sm",
            "tiny-rcpsp",
        ),
        (
            "examples/python/scheduling/api/rcpsp.py",
            "examples/instances/scheduling/tiny-mrcpsp.mm",
            "tiny-mrcpsp",
        ),
        (
            "examples/python/scheduling/native/rcpsp.py",
            "examples/instances/scheduling/tiny-mrcpsp.mm",
            "tiny-mrcpsp",
        ),
    ],
)
def test_api_and_native_examples_solve_standard_instance_files(script, instance, expected_name):
    record = _run_example(script, instance)
    assert record["instance"] == expected_name
