"""Parsers for JSPLIB job-shop and PSPLIB project scheduling instances."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, cast

from .common import DatasetParseError, numbered_lines, read_text, require


@dataclass(frozen=True)
class JobShopOperation:
    machine: int
    duration: int


@dataclass(frozen=True)
class JobShopInstance:
    name: str
    jobs: Tuple[Tuple[JobShopOperation, ...], ...]
    num_machines: int

    @property
    def num_jobs(self) -> int:
        return len(self.jobs)

    @property
    def horizon(self) -> int:
        return sum(operation.duration for job in self.jobs for operation in job)

    @property
    def machines(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(tuple(operation.machine for operation in job) for job in self.jobs)

    @property
    def durations(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(
            tuple(operation.duration for operation in job) for job in self.jobs
        )


@dataclass(frozen=True)
class PSPLibMode:
    mode: int
    duration: int
    demands: Tuple[int, ...]


@dataclass(frozen=True)
class PSPLibJob:
    job: int
    successors: Tuple[int, ...]
    modes: Tuple[PSPLibMode, ...]


@dataclass(frozen=True)
class PSPLibInstance:
    name: str
    horizon: Optional[int]
    resource_names: Tuple[str, ...]
    resource_kinds: Tuple[str, ...]
    capacities: Tuple[int, ...]
    jobs: Tuple[PSPLibJob, ...]

    @property
    def num_jobs(self) -> int:
        return len(self.jobs)

    @property
    def multi_mode(self) -> bool:
        return any(len(job.modes) > 1 for job in self.jobs)

    @property
    def renewable_resources(self) -> Tuple[int, ...]:
        return tuple(
            index
            for index, kind in enumerate(self.resource_kinds)
            if kind == "renewable"
        )

    @property
    def nonrenewable_resources(self) -> Tuple[int, ...]:
        return tuple(
            index
            for index, kind in enumerate(self.resource_kinds)
            if kind == "nonrenewable"
        )

    def job(self, job_id: int) -> PSPLibJob:
        for job in self.jobs:
            if job.job == job_id:
                return job
        raise KeyError(f"unknown PSPLIB job {job_id}")


def parse_jsplib(text: str, *, source: str = "<string>") -> JobShopInstance:
    """Parse the standard JSPLIB pair format into normalized operations."""

    content: List[Tuple[int, str]] = []
    description: List[str] = []
    for line_number, raw in numbered_lines(text):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            cleaned = line.lstrip("#").strip()
            if cleaned and not set(cleaned) <= {"+", "-", "="}:
                description.append(cleaned)
            continue
        content.append((line_number, line))
    require(bool(content), "empty JSPLIB instance", source=source)

    header_line, header = content[0]
    try:
        header_values = [int(token) for token in header.split()]
    except ValueError as exc:
        raise DatasetParseError(
            "JSPLIB header must contain two integers", source=source, line=header_line
        ) from exc
    require(
        len(header_values) == 2,
        "JSPLIB header must contain exactly job and machine counts",
        source=source,
        line=header_line,
    )
    num_jobs, num_machines = header_values
    require(
        num_jobs > 0 and num_machines > 0,
        "job and machine counts must be positive",
        source=source,
        line=header_line,
    )
    require(
        len(content) - 1 == num_jobs,
        f"found {len(content) - 1} job rows, expected {num_jobs}",
        source=source,
    )

    raw_jobs: List[List[Tuple[int, int]]] = []
    machine_values: List[int] = []
    for line_number, line in content[1:]:
        try:
            values = [int(token) for token in line.split()]
        except ValueError as exc:
            raise DatasetParseError(
                "job row must contain integers", source=source, line=line_number
            ) from exc
        require(
            len(values) == 2 * num_machines,
            f"job row has {len(values)} values, expected {2 * num_machines}",
            source=source,
            line=line_number,
        )
        operations: List[Tuple[int, int]] = []
        for index in range(0, len(values), 2):
            machine, duration = values[index], values[index + 1]
            require(
                duration > 0,
                "operation durations must be positive",
                source=source,
                line=line_number,
            )
            machine_values.append(machine)
            operations.append((machine, duration))
        raw_jobs.append(operations)

    machine_set = set(machine_values)
    zero_based = machine_set == set(range(num_machines))
    one_based = machine_set == set(range(1, num_machines + 1))
    require(
        zero_based or one_based,
        "machine ids must cover either 0..m-1 or 1..m",
        source=source,
    )
    offset = 0 if zero_based else 1
    jobs = tuple(
        tuple(
            JobShopOperation(machine=machine - offset, duration=duration)
            for machine, duration in operations
        )
        for operations in raw_jobs
    )
    name = Path(source).stem if source != "<string>" else "unnamed"
    for line in description:
        match = re.match(r"instance\s+(.+)", line, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            break
    return JobShopInstance(name=name, jobs=jobs, num_machines=num_machines)


def read_jsplib(path: object) -> JobShopInstance:
    text, source = read_text(path)
    return parse_jsplib(text, source=source)


def _section(lines: Sequence[Tuple[int, str]], prefix: str, source: str) -> int:
    found = next(
        (
            index
            for index, (_, line) in enumerate(lines)
            if line.upper().startswith(prefix)
        ),
        None,
    )
    if found is None:
        raise DatasetParseError(f"missing {prefix} section", source=source)
    return found


def _resource_labels(header: str) -> List[Tuple[str, str]]:
    labels: List[Tuple[str, str]] = []
    for kind, number in re.findall(r"\b([RND])\s*(\d+)\b", header.upper()):
        full_kind = {"R": "renewable", "N": "nonrenewable", "D": "doubly_constrained"}[
            kind
        ]
        labels.append((f"{kind}{number}", full_kind))
    return labels


def parse_psplib(text: str, *, source: str = "<string>") -> PSPLibInstance:
    """Parse PSPLIB RCPSP/MRCPSP `.sm` and `.mm` instances."""

    lines = [(number, raw.rstrip()) for number, raw in numbered_lines(text)]
    precedence_start = _section(lines, "PRECEDENCE RELATIONS", source)
    requests_start = _section(lines, "REQUESTS/DURATIONS", source)
    resources_start = _section(lines, "RESOURCEAVAILABILITIES", source)
    require(
        precedence_start < requests_start < resources_start,
        "PSPLIB sections are out of order",
        source=source,
    )

    horizon: Optional[int] = None
    declared_jobs: Optional[int] = None
    for line_number, raw in lines[:precedence_start]:
        line = raw.strip().lower()
        if line.startswith("jobs"):
            match = re.search(r":\s*(\d+)", line)
            if match:
                declared_jobs = int(match.group(1))
        elif line.startswith("horizon"):
            match = re.search(r":\s*(\d+)", line)
            if match:
                horizon = int(match.group(1))

    precedence: Dict[int, Tuple[int, Tuple[int, ...]]] = {}
    for line_number, raw in lines[precedence_start + 1 : requests_start]:
        line = raw.strip()
        if not line or line.startswith(("*", "-")) or not line[0].isdigit():
            continue
        try:
            values = [int(token) for token in line.split()]
        except ValueError as exc:
            raise DatasetParseError(
                "invalid precedence row", source=source, line=line_number
            ) from exc
        require(
            len(values) >= 3,
            "precedence row needs job, mode count, and successor count",
            source=source,
            line=line_number,
        )
        job, mode_count, successor_count = values[:3]
        successors = tuple(values[3:])
        require(
            len(successors) == successor_count,
            f"job {job} declares {successor_count} successors but lists {len(successors)}",
            source=source,
            line=line_number,
        )
        require(
            mode_count > 0,
            f"job {job} must have at least one mode",
            source=source,
            line=line_number,
        )
        require(
            job not in precedence,
            f"duplicate precedence row for job {job}",
            source=source,
            line=line_number,
        )
        precedence[job] = (mode_count, successors)
    require(bool(precedence), "PRECEDENCE RELATIONS contains no jobs", source=source)
    if declared_jobs is not None:
        require(
            len(precedence) == declared_jobs,
            f"found {len(precedence)} jobs, expected {declared_jobs}",
            source=source,
        )

    request_lines = lines[requests_start + 1 : resources_start]
    header_index = next(
        (
            index
            for index, (_, line) in enumerate(request_lines)
            if "duration" in line.lower()
        ),
        None,
    )
    require(
        header_index is not None,
        "missing REQUESTS/DURATIONS column header",
        source=source,
    )
    header_index = cast(int, header_index)
    labels = _resource_labels(request_lines[header_index][1])
    require(
        bool(labels),
        "REQUESTS/DURATIONS header defines no resources",
        source=source,
        line=request_lines[header_index][0],
    )
    resource_names = tuple(label for label, _ in labels)
    resource_kinds = tuple(kind for _, kind in labels)
    resource_count = len(labels)

    modes_by_job: Dict[int, List[PSPLibMode]] = {job: [] for job in precedence}
    current_job: Optional[int] = None
    for line_number, raw in request_lines[header_index + 1 :]:
        line = raw.strip()
        if not line or line.startswith(("*", "-")):
            continue
        try:
            values = [int(token) for token in line.split()]
        except ValueError:
            continue
        if len(values) == resource_count + 3:
            current_job, mode, duration = values[:3]
            demands = values[3:]
        elif len(values) == resource_count + 2 and current_job is not None:
            mode, duration = values[:2]
            demands = values[2:]
        else:
            raise DatasetParseError(
                f"mode row has {len(values)} values, expected {resource_count + 3} or continuation {resource_count + 2}",
                source=source,
                line=line_number,
            )
        require(
            current_job in modes_by_job,
            f"mode row references unknown job {current_job}",
            source=source,
            line=line_number,
        )
        require(
            duration >= 0,
            "mode duration must be non-negative",
            source=source,
            line=line_number,
        )
        require(
            all(demand >= 0 for demand in demands),
            "resource demands must be non-negative",
            source=source,
            line=line_number,
        )
        modes_by_job[current_job].append(
            PSPLibMode(mode=mode, duration=duration, demands=tuple(demands))
        )

    jobs: List[PSPLibJob] = []
    known_jobs = set(precedence)
    for job in sorted(precedence):
        mode_count, successors = precedence[job]
        require(
            all(successor in known_jobs for successor in successors),
            f"job {job} references an unknown successor",
            source=source,
        )
        modes = modes_by_job[job]
        require(
            len(modes) == mode_count,
            f"job {job} declares {mode_count} modes but defines {len(modes)}",
            source=source,
        )
        require(
            len({mode.mode for mode in modes}) == len(modes),
            f"job {job} has duplicate mode ids",
            source=source,
        )
        jobs.append(PSPLibJob(job=job, successors=successors, modes=tuple(modes)))

    capacities: Optional[Tuple[int, ...]] = None
    availability_lines = lines[resources_start + 1 :]
    for line_number, raw in availability_lines:
        line = raw.strip()
        if (
            not line
            or line.startswith(("*", "-"))
            or any(char.isalpha() for char in line)
        ):
            continue
        try:
            values = tuple(int(token) for token in line.split())
        except ValueError:
            continue
        if len(values) == resource_count:
            capacities = values
            break
        raise DatasetParseError(
            f"capacity row has {len(values)} values, expected {resource_count}",
            source=source,
            line=line_number,
        )
    require(
        capacities is not None,
        "RESOURCEAVAILABILITIES contains no capacity row",
        source=source,
    )
    capacities = cast(Tuple[int, ...], capacities)
    require(
        all(capacity >= 0 for capacity in capacities),
        "resource capacities must be non-negative",
        source=source,
    )

    return PSPLibInstance(
        name=Path(source).stem if source != "<string>" else "unnamed",
        horizon=horizon,
        resource_names=resource_names,
        resource_kinds=resource_kinds,
        capacities=capacities,
        jobs=tuple(jobs),
    )


def read_psplib(path: object) -> PSPLibInstance:
    text, source = read_text(path)
    return parse_psplib(text, source=source)
