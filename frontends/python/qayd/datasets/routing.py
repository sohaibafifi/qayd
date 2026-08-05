"""Parsers for CVRPLIB, Solomon/Homberger, and common VRP solutions."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast

from .common import DatasetParseError, numbered_lines, read_text, require

Coordinate = Tuple[float, float]
IntMatrix = Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class CVRPLibInstance:
    """A CVRPLIB/TSPLIB instance normalized to zero-based node positions."""

    name: str
    problem_type: str
    capacity: int
    node_ids: Tuple[int, ...]
    depot: int
    demands: Tuple[int, ...]
    coordinates: Tuple[Coordinate, ...]
    edge_weights: IntMatrix
    edge_weight_type: str
    edge_weight_format: Optional[str]
    vehicles: Optional[int]
    comment: str
    best_known: Optional[int]
    metadata: Mapping[str, str]

    @property
    def dimension(self) -> int:
        return len(self.node_ids)

    @property
    def customers(self) -> Tuple[int, ...]:
        return tuple(node for node in range(self.dimension) if node != self.depot)

    def normalize_node(self, node_id: int) -> int:
        try:
            return self.node_ids.index(node_id)
        except ValueError as exc:
            raise KeyError(f"unknown CVRPLIB node id {node_id}") from exc


@dataclass(frozen=True)
class SolomonInstance:
    """A Solomon or Gehring-Homberger VRPTW instance."""

    name: str
    vehicles: int
    capacity: int
    node_ids: Tuple[int, ...]
    depot: int
    coordinates: Tuple[Coordinate, ...]
    demands: Tuple[int, ...]
    time_windows: Tuple[Tuple[int, int], ...]
    service_times: Tuple[int, ...]

    @property
    def dimension(self) -> int:
        return len(self.node_ids)

    @property
    def customers(self) -> Tuple[int, ...]:
        return tuple(node for node in range(self.dimension) if node != self.depot)

    def normalize_node(self, node_id: int) -> int:
        try:
            return self.node_ids.index(node_id)
        except ValueError as exc:
            raise KeyError(f"unknown Solomon node id {node_id}") from exc

    def distance_matrix(
        self, *, scale: int = 10, rounding: str = "truncate"
    ) -> IntMatrix:
        """Return reproducible integer distances, with DIMACS trunc1 by default."""

        if scale <= 0:
            raise ValueError("distance scale must be positive")
        if rounding not in {"truncate", "nearest", "ceil"}:
            raise ValueError("rounding must be 'truncate', 'nearest', or 'ceil'")

        def convert(distance: float) -> int:
            scaled = distance * scale
            if rounding == "truncate":
                return math.floor(scaled + 1e-9)
            if rounding == "nearest":
                return math.floor(scaled + 0.5)
            return math.ceil(scaled - 1e-9)

        return tuple(
            tuple(convert(math.hypot(ax - bx, ay - by)) for bx, by in self.coordinates)
            for ax, ay in self.coordinates
        )


@dataclass(frozen=True)
class VRPSolution:
    routes: Tuple[Tuple[int, ...], ...]
    cost: Optional[float]
    metadata: Mapping[str, str]


def _integer(value: str, field: str, *, source: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise DatasetParseError(
            f"{field} must be an integer, got {value!r}", source=source
        ) from exc


def _section_name(line: str) -> Optional[str]:
    candidate = line.strip().rstrip(":").upper()
    return candidate if candidate.endswith("_SECTION") else None


def _parse_vrplib_document(
    text: str, source: str
) -> Tuple[Dict[str, str], Dict[str, List[Tuple[int, str]]]]:
    specifications: Dict[str, str] = {}
    sections: Dict[str, List[Tuple[int, str]]] = {}
    current: Optional[str] = None
    for line_number, raw in numbered_lines(text):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.upper() == "EOF":
            break
        section = _section_name(line)
        if section is not None:
            current = section
            sections.setdefault(section, [])
            continue
        if current is not None:
            # A specification after a section is rare but legal in generated
            # files. Detect it before treating the row as section data.
            if ":" in line and line.split(":", 1)[0].strip().replace("_", "").isalpha():
                current = None
            else:
                sections[current].append((line_number, line))
                continue
        if ":" in line:
            key, value = line.split(":", 1)
        else:
            parts = line.split(None, 1)
            require(
                len(parts) == 2,
                f"expected a KEY: VALUE specification, got {line!r}",
                source=source,
                line=line_number,
            )
            key, value = parts
        specifications[key.strip().upper()] = value.strip()
    return specifications, sections


def _parse_indexed_coordinates(
    rows: Sequence[Tuple[int, str]], source: str
) -> Dict[int, Coordinate]:
    coordinates: Dict[int, Coordinate] = {}
    for line_number, line in rows:
        parts = line.split()
        require(
            len(parts) in {3, 4},
            "coordinate row must contain an id and two or three coordinates",
            source=source,
            line=line_number,
        )
        try:
            node = int(parts[0])
            coordinate = (float(parts[1]), float(parts[2]))
        except ValueError as exc:
            raise DatasetParseError(
                "invalid coordinate row", source=source, line=line_number
            ) from exc
        require(
            node not in coordinates,
            f"duplicate node id {node}",
            source=source,
            line=line_number,
        )
        coordinates[node] = coordinate
    return coordinates


def _parse_demands(rows: Sequence[Tuple[int, str]], source: str) -> Dict[int, int]:
    demands: Dict[int, int] = {}
    for line_number, line in rows:
        parts = line.split()
        require(
            len(parts) == 2,
            "demand row must contain a node id and demand",
            source=source,
            line=line_number,
        )
        try:
            node, demand = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise DatasetParseError(
                "invalid demand row", source=source, line=line_number
            ) from exc
        require(
            demand >= 0, "demands must be non-negative", source=source, line=line_number
        )
        require(
            node not in demands,
            f"duplicate demand for node {node}",
            source=source,
            line=line_number,
        )
        demands[node] = demand
    return demands


def _explicit_matrix(
    values: Sequence[int], dimension: int, matrix_format: str, source: str
) -> IntMatrix:
    matrix = [[0] * dimension for _ in range(dimension)]
    positions: List[Tuple[int, int]] = []
    if matrix_format == "FULL_MATRIX":
        positions = [(row, col) for row in range(dimension) for col in range(dimension)]
    elif matrix_format == "UPPER_ROW":
        positions = [
            (row, col) for row in range(dimension) for col in range(row + 1, dimension)
        ]
    elif matrix_format == "LOWER_ROW":
        positions = [(row, col) for row in range(dimension) for col in range(row)]
    elif matrix_format == "UPPER_DIAG_ROW":
        positions = [
            (row, col) for row in range(dimension) for col in range(row, dimension)
        ]
    elif matrix_format == "LOWER_DIAG_ROW":
        positions = [(row, col) for row in range(dimension) for col in range(row + 1)]
    elif matrix_format == "UPPER_COL":
        positions = [(row, col) for col in range(dimension) for row in range(col)]
    elif matrix_format == "LOWER_COL":
        positions = [
            (row, col) for col in range(dimension) for row in range(col + 1, dimension)
        ]
    elif matrix_format == "UPPER_DIAG_COL":
        positions = [(row, col) for col in range(dimension) for row in range(col + 1)]
    elif matrix_format == "LOWER_DIAG_COL":
        positions = [
            (row, col) for col in range(dimension) for row in range(col, dimension)
        ]
    else:
        raise DatasetParseError(
            f"unsupported EDGE_WEIGHT_FORMAT {matrix_format!r}", source=source
        )

    require(
        len(values) == len(positions),
        f"EDGE_WEIGHT_SECTION has {len(values)} values, expected {len(positions)} for {matrix_format}",
        source=source,
    )
    for value, (row, col) in zip(values, positions):
        matrix[row][col] = value
        if matrix_format != "FULL_MATRIX":
            matrix[col][row] = value
    return tuple(tuple(row) for row in matrix)


def _coordinate_matrix(
    coordinates: Sequence[Coordinate], edge_type: str, source: str
) -> IntMatrix:
    def distance(a: Coordinate, b: Coordinate) -> int:
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        if edge_type == "EUC_2D":
            return math.floor(math.hypot(dx, dy) + 0.5)
        if edge_type == "CEIL_2D":
            return math.ceil(math.hypot(dx, dy))
        if edge_type == "MAN_2D":
            return math.floor(dx + dy + 0.5)
        if edge_type == "MAX_2D":
            return math.floor(max(dx, dy) + 0.5)
        if edge_type == "ATT":
            raw = math.sqrt((dx * dx + dy * dy) / 10.0)
            rounded = math.floor(raw + 0.5)
            return rounded + 1 if rounded < raw else rounded
        raise DatasetParseError(
            f"unsupported EDGE_WEIGHT_TYPE {edge_type!r}", source=source
        )

    return tuple(tuple(distance(a, b) for b in coordinates) for a in coordinates)


def parse_cvrplib(text: str, *, source: str = "<string>") -> CVRPLibInstance:
    """Parse a CVRPLIB/TSPLIB95 CVRP instance."""

    specifications, sections = _parse_vrplib_document(text, source)
    problem_type = specifications.get("TYPE", "CVRP").upper()
    require(
        problem_type in {"CVRP", "CVRPTW", "VRP"},
        f"unsupported TYPE {problem_type!r}",
        source=source,
    )

    coordinates_by_id = _parse_indexed_coordinates(
        sections.get("NODE_COORD_SECTION", []), source
    )
    demands_by_id = _parse_demands(sections.get("DEMAND_SECTION", []), source)
    dimension_text = specifications.get("DIMENSION")
    dimension = (
        _integer(dimension_text, "DIMENSION", source=source)
        if dimension_text is not None
        else max(len(coordinates_by_id), len(demands_by_id))
    )
    require(dimension > 0, "DIMENSION must be positive", source=source)

    node_ids = tuple(sorted(coordinates_by_id or demands_by_id))
    if not node_ids:
        node_ids = tuple(range(1, dimension + 1))
    require(
        len(node_ids) == dimension,
        f"found {len(node_ids)} nodes, expected DIMENSION {dimension}",
        source=source,
    )
    require(
        set(demands_by_id) == set(node_ids),
        "DEMAND_SECTION must define every node exactly once",
        source=source,
    )

    depot_values: List[int] = []
    for line_number, line in sections.get("DEPOT_SECTION", []):
        for token in line.split():
            try:
                node = int(token)
            except ValueError as exc:
                raise DatasetParseError(
                    "invalid depot node id", source=source, line=line_number
                ) from exc
            if node == -1:
                break
            depot_values.append(node)
    require(
        len(depot_values) == 1,
        "DEPOT_SECTION must contain exactly one depot before -1",
        source=source,
    )
    require(
        depot_values[0] in node_ids,
        f"unknown depot node id {depot_values[0]}",
        source=source,
    )
    depot = node_ids.index(depot_values[0])

    edge_type = specifications.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper()
    edge_format = specifications.get("EDGE_WEIGHT_FORMAT")
    if edge_type == "EXPLICIT":
        matrix_format = (edge_format or "FULL_MATRIX").upper()
        tokens = " ".join(
            line for _, line in sections.get("EDGE_WEIGHT_SECTION", [])
        ).split()
        try:
            weights = [int(token) for token in tokens]
        except ValueError as exc:
            raise DatasetParseError(
                "EDGE_WEIGHT_SECTION must contain integers", source=source
            ) from exc
        edge_weights = _explicit_matrix(weights, dimension, matrix_format, source)
    else:
        require(
            set(coordinates_by_id) == set(node_ids),
            "NODE_COORD_SECTION must define every node for coordinate distances",
            source=source,
        )
        edge_weights = _coordinate_matrix(
            [coordinates_by_id[node] for node in node_ids], edge_type, source
        )

    capacity = _integer(specifications.get("CAPACITY", "0"), "CAPACITY", source=source)
    require(capacity > 0, "CAPACITY must be positive", source=source)
    demands = tuple(demands_by_id[node] for node in node_ids)
    require(demands[depot] == 0, "depot demand must be zero", source=source)

    name = specifications.get(
        "NAME", Path(source).stem if source != "<string>" else "unnamed"
    )
    comment = specifications.get("COMMENT", "")
    vehicle_text = specifications.get("VEHICLES")
    vehicle_match = re.search(r"-k(\d+)(?:\b|$)", name, re.IGNORECASE)
    vehicles = (
        _integer(vehicle_text, "VEHICLES", source=source)
        if vehicle_text
        else (int(vehicle_match.group(1)) if vehicle_match else None)
    )
    optimum = re.search(
        r"(?:optimal value|best known(?: solution)?|bks)\s*[:=]\s*(\d+)",
        comment,
        re.IGNORECASE,
    )

    return CVRPLibInstance(
        name=name,
        problem_type=problem_type,
        capacity=capacity,
        node_ids=node_ids,
        depot=depot,
        demands=demands,
        coordinates=tuple(coordinates_by_id.get(node, (0.0, 0.0)) for node in node_ids),
        edge_weights=edge_weights,
        edge_weight_type=edge_type,
        edge_weight_format=edge_format.upper() if edge_format else None,
        vehicles=vehicles,
        comment=comment,
        best_known=int(optimum.group(1)) if optimum else None,
        metadata=MappingProxyType(dict(specifications)),
    )


def read_cvrplib(path: object) -> CVRPLibInstance:
    text, source = read_text(path)
    return parse_cvrplib(text, source=source)


def parse_solomon(text: str, *, source: str = "<string>") -> SolomonInstance:
    """Parse the shared Solomon and Gehring-Homberger VRPTW format."""

    lines = [
        (number, raw.strip()) for number, raw in numbered_lines(text) if raw.strip()
    ]
    require(bool(lines), "empty Solomon instance", source=source)
    name = lines[0][1]

    vehicle_index = next(
        (index for index, (_, line) in enumerate(lines) if line.upper() == "VEHICLE"),
        None,
    )
    customer_index = next(
        (index for index, (_, line) in enumerate(lines) if line.upper() == "CUSTOMER"),
        None,
    )
    require(vehicle_index is not None, "missing VEHICLE section", source=source)
    require(customer_index is not None, "missing CUSTOMER section", source=source)
    vehicle_index = cast(int, vehicle_index)
    customer_index = cast(int, customer_index)

    vehicle_values: Optional[Tuple[int, int]] = None
    for line_number, line in lines[vehicle_index + 1 : customer_index]:
        parts = line.split()
        if len(parts) == 2:
            try:
                vehicle_values = (int(parts[0]), int(parts[1]))
                break
            except ValueError:
                continue
    require(
        vehicle_values is not None,
        "VEHICLE section needs NUMBER and CAPACITY values",
        source=source,
    )
    vehicle_values = cast(Tuple[int, int], vehicle_values)
    vehicles, capacity = vehicle_values
    require(
        vehicles > 0 and capacity > 0,
        "vehicle number and capacity must be positive",
        source=source,
    )

    rows: Dict[int, Tuple[Coordinate, int, Tuple[int, int], int]] = {}
    for line_number, line in lines[customer_index + 1 :]:
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            node = int(parts[0])
            x, y = float(parts[1]), float(parts[2])
            demand, ready, due, service = map(int, parts[3:7])
        except ValueError:
            continue
        require(
            node not in rows,
            f"duplicate customer id {node}",
            source=source,
            line=line_number,
        )
        require(
            demand >= 0, "demand must be non-negative", source=source, line=line_number
        )
        require(
            0 <= ready <= due,
            "time window must satisfy 0 <= ready <= due",
            source=source,
            line=line_number,
        )
        require(
            service >= 0,
            "service time must be non-negative",
            source=source,
            line=line_number,
        )
        rows[node] = ((x, y), demand, (ready, due), service)
    require(bool(rows), "CUSTOMER section contains no data rows", source=source)

    node_ids = tuple(sorted(rows))
    depot_id = 0 if 0 in rows else node_ids[0]
    depot = node_ids.index(depot_id)
    require(rows[depot_id][1] == 0, "depot demand must be zero", source=source)
    return SolomonInstance(
        name=name,
        vehicles=vehicles,
        capacity=capacity,
        node_ids=node_ids,
        depot=depot,
        coordinates=tuple(rows[node][0] for node in node_ids),
        demands=tuple(rows[node][1] for node in node_ids),
        time_windows=tuple(rows[node][2] for node in node_ids),
        service_times=tuple(rows[node][3] for node in node_ids),
    )


def read_solomon(path: object) -> SolomonInstance:
    text, source = read_text(path)
    return parse_solomon(text, source=source)


def parse_vrp_solution(
    text: str,
    *,
    source: str = "<string>",
    instance: Optional[UnionRoutingInstance] = None,
) -> VRPSolution:
    """Parse the common CVRPLIB/Solomon route-and-cost solution format."""

    routes: List[Tuple[int, ...]] = []
    metadata: Dict[str, str] = {}
    cost: Optional[float] = None
    for line_number, raw in numbered_lines(text):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("route"):
            require(
                ":" in line,
                "route line must contain ':'",
                source=source,
                line=line_number,
            )
            _, values = line.split(":", 1)
            try:
                route = tuple(int(token) for token in values.split())
            except ValueError as exc:
                raise DatasetParseError(
                    "route contains a non-integer node id",
                    source=source,
                    line=line_number,
                ) from exc
            if instance is not None:
                try:
                    route = tuple(instance.normalize_node(node) for node in route)
                except KeyError as exc:
                    raise DatasetParseError(
                        str(exc), source=source, line=line_number
                    ) from exc
            routes.append(route)
            continue
        if ":" in line:
            key, value = line.split(":", 1)
        else:
            parts = line.split(None, 1)
            require(
                len(parts) == 2,
                f"unrecognized solution line {line!r}",
                source=source,
                line=line_number,
            )
            key, value = parts
        metadata[key.strip().lower()] = value.strip()
        if key.strip().lower() in {"cost", "objective", "value"}:
            try:
                cost = float(value)
            except ValueError as exc:
                raise DatasetParseError(
                    "solution cost must be numeric", source=source, line=line_number
                ) from exc
    require(bool(routes), "solution contains no routes", source=source)
    return VRPSolution(
        routes=tuple(routes), cost=cost, metadata=MappingProxyType(metadata)
    )


UnionRoutingInstance = Union[CVRPLibInstance, SolomonInstance]


def read_vrp_solution(
    path: object, *, instance: Optional[UnionRoutingInstance] = None
) -> VRPSolution:
    text, source = read_text(path)
    return parse_vrp_solution(text, source=source, instance=instance)
