"""Typed parsers for reproducible routing and scheduling benchmarks.

The module intentionally parses local files only. Dataset download, provenance,
and license acceptance stay explicit in the benchmark harness.
"""

from __future__ import annotations

from typing import Union

from .common import DatasetParseError, read_text
from .routing import (
    CVRPLibInstance,
    SolomonInstance,
    VRPSolution,
    parse_cvrplib,
    parse_solomon,
    parse_vrp_solution,
    read_cvrplib,
    read_solomon,
    read_vrp_solution,
)
from .scheduling import (
    JobShopInstance,
    JobShopOperation,
    PSPLibInstance,
    PSPLibJob,
    PSPLibMode,
    parse_jsplib,
    parse_psplib,
    read_jsplib,
    read_psplib,
)

BenchmarkInstance = Union[
    CVRPLibInstance, SolomonInstance, JobShopInstance, PSPLibInstance
]


_ALIASES = {
    "cvrp": "cvrplib",
    "cvrplib": "cvrplib",
    "tsplib": "cvrplib",
    "vrplib": "cvrplib",
    "solomon": "solomon",
    "homberger": "solomon",
    "vrptw": "solomon",
    "jsp": "jsplib",
    "jsplib": "jsplib",
    "rcpsp": "psplib",
    "mrcpsp": "psplib",
    "psplib": "psplib",
}


def detect_format(text: str, *, source: str = "<string>") -> str:
    """Detect one supported benchmark format from its structural markers."""

    upper = text.upper()
    if "NODE_COORD_SECTION" in upper and "DEMAND_SECTION" in upper:
        return "cvrplib"
    if (
        re_marker(upper, "VEHICLE")
        and re_marker(upper, "CUSTOMER")
        and "READY TIME" in upper
    ):
        return "solomon"
    if "PRECEDENCE RELATIONS" in upper and "REQUESTS/DURATIONS" in upper:
        return "psplib"

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 2:
            try:
                int(parts[0])
                int(parts[1])
                return "jsplib"
            except ValueError:
                break
        break
    raise DatasetParseError(
        "could not detect a supported benchmark format", source=source
    )


def re_marker(text: str, marker: str) -> bool:
    return any(line.strip() == marker for line in text.splitlines())


def load_instance(path: object, *, format: str = "auto") -> BenchmarkInstance:
    """Read a benchmark file, with marker-based format detection by default."""

    text, source = read_text(path)
    normalized = (
        detect_format(text, source=source)
        if format.lower() == "auto"
        else _ALIASES.get(format.lower())
    )
    if normalized is None:
        choices = ", ".join(sorted(_ALIASES))
        raise ValueError(
            f"unknown benchmark format {format!r}; expected one of: {choices}"
        )
    if normalized == "cvrplib":
        return parse_cvrplib(text, source=source)
    if normalized == "solomon":
        return parse_solomon(text, source=source)
    if normalized == "jsplib":
        return parse_jsplib(text, source=source)
    return parse_psplib(text, source=source)


__all__ = [
    "BenchmarkInstance",
    "CVRPLibInstance",
    "DatasetParseError",
    "JobShopInstance",
    "JobShopOperation",
    "PSPLibInstance",
    "PSPLibJob",
    "PSPLibMode",
    "SolomonInstance",
    "VRPSolution",
    "detect_format",
    "load_instance",
    "parse_cvrplib",
    "parse_jsplib",
    "parse_psplib",
    "parse_solomon",
    "parse_vrp_solution",
    "read_cvrplib",
    "read_jsplib",
    "read_psplib",
    "read_solomon",
    "read_vrp_solution",
]
