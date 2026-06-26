"""Scheduling helpers built on the Python qayd model API."""

from __future__ import annotations

from builtins import all as all_builtin
from builtins import any as any_builtin
from dataclasses import dataclass
from functools import reduce
from operator import or_
from typing import Optional, Sequence, Union

from . import Constraint, Expr, IntVar, Model

Duration = Union[int, IntVar, tuple[int, int]]
Demand = Union[int, IntVar]
Capacity = Union[int, IntVar]
TimeExpr = Union[int, IntVar, Expr]


@dataclass(frozen=True)
class IntervalVar:
    start: IntVar
    duration: Duration
    name: Optional[str] = None
    demand: Demand = 1
    presence: Optional[IntVar] = None

    @property
    def end(self) -> TimeExpr:
        return self.start + self.duration

    @property
    def optional(self) -> bool:
        return self.presence is not None

    @property
    def fixed_duration(self) -> bool:
        return isinstance(self.duration, int) and not isinstance(self.duration, bool)


Task = IntervalVar


@dataclass(frozen=True)
class SequenceVar:
    intervals: Sequence[IntervalVar]
    types: Optional[Sequence[int]] = None
    name: Optional[str] = None
    transition_matrix: Optional[Sequence[Sequence[int]]] = None

    def __post_init__(self) -> None:
        if self.types is not None and len(self.types) != len(self.intervals):
            raise ValueError("sequence types must match the number of intervals")


Sequence = SequenceVar


@dataclass(frozen=True)
class IntervalValue:
    start: int
    end: int
    duration: int
    present: bool = True


def interval_var(
    model: Model,
    *,
    start: Union[IntVar, tuple[int, int]],
    size: Duration,
    name: Optional[str] = None,
    demand: Demand = 1,
    optional: bool = False,
    presence: Optional[IntVar] = None,
) -> IntervalVar:
    start_var = start if isinstance(start, IntVar) else model.int_var(start[0], start[1], name=_field_name(name, "start"))
    if isinstance(size, tuple):
        duration = model.int_var(size[0], size[1], name=_field_name(name, "duration"))
    else:
        duration = size
    if optional and presence is None:
        presence = model.bool_var(name=_field_name(name, "present"))
    return IntervalVar(start=start_var, duration=duration, name=name, demand=demand, presence=presence)


def interval_vars(model: Model, durations: Sequence[int], horizon: int, *, name: str = "task") -> list[IntervalVar]:
    return [interval_var(model, start=(0, horizon), size=duration, name=f"{name}[{i}]") for i, duration in enumerate(durations)]


def sequence_var(
    intervals: Sequence[IntervalVar],
    *,
    types: Optional[Sequence[int]] = None,
    name: Optional[str] = None,
    transition_matrix: Optional[Sequence[Sequence[int]]] = None,
) -> SequenceVar:
    return SequenceVar(intervals=intervals, types=types, name=name, transition_matrix=transition_matrix)


def no_overlap(
    model: Model,
    sequence: Union[SequenceVar, Sequence[IntervalVar]],
    transition_matrix: Optional[Sequence[Sequence[int]]] = None,
) -> None:
    seq = _as_sequence(sequence, transition_matrix)
    tasks = list(seq.intervals)
    matrix = transition_matrix if transition_matrix is not None else seq.transition_matrix

    if matrix is not None:
        _post_transition_no_overlap(model, seq, matrix)
        return

    if _can_use_native_no_overlap(tasks):
        _post_native_no_overlap(model, tasks)
        return

    _post_capacity_one_cumulative(model, tasks)


def _post_transition_no_overlap(model: Model, seq: SequenceVar, matrix: Sequence[Sequence[int]]) -> None:
    for i, left in enumerate(seq.intervals):
        for j in range(i + 1, len(seq.intervals)):
            right = seq.intervals[j]
            left_setup = _setup_time(seq, matrix, i, j)
            right_setup = _setup_time(seq, matrix, j, i)
            disjunction = (left.end + left_setup <= right.start) | (right.end + right_setup <= left.start)
            model.add(_guard_optional(disjunction, left, right))


def seq_no_overlap(
    model: Model,
    sequence: Union[SequenceVar, Sequence[IntervalVar]],
    transition_matrix: Optional[Sequence[Sequence[int]]] = None,
) -> None:
    no_overlap(model, sequence, transition_matrix)


def cumulative(model: Model, tasks_or_sequence: Union[SequenceVar, Sequence[IntervalVar]], capacity: Capacity) -> None:
    tasks = list(_as_sequence(tasks_or_sequence).intervals)
    if any_builtin(task.optional for task in tasks):
        raise NotImplementedError("optional intervals are not supported by cumulative yet")

    starts = [task.start for task in tasks]
    durations = [task.duration for task in tasks]
    demands = [task.demand for task in tasks]
    has_variable = isinstance(capacity, IntVar) or any_builtin(isinstance(item, IntVar) for item in durations + demands)
    if has_variable:
        duration_vars = [_as_var(model, item, f"duration[{i}]") for i, item in enumerate(durations)]
        demand_vars = [_as_var(model, item, f"demand[{i}]") for i, item in enumerate(demands)]
        capacity_var = _as_var(model, capacity, "capacity")
        model.cumulative_var(starts, duration_vars, demand_vars, capacity_var)
        return

    model.cumulative(starts, [int(item) for item in durations], [int(item) for item in demands], int(capacity))


def end_before_start(left: IntervalVar, right: IntervalVar, delay: int = 0) -> Constraint:
    return _guard_optional(left.end + delay <= right.start, left, right)


def start_before_start(left: IntervalVar, right: IntervalVar, delay: int = 0) -> Constraint:
    return _guard_optional(left.start + delay <= right.start, left, right)


def end_before_end(left: IntervalVar, right: IntervalVar, delay: int = 0) -> Constraint:
    return _guard_optional(left.end + delay <= right.end, left, right)


def start_before_end(left: IntervalVar, right: IntervalVar, delay: int = 0) -> Constraint:
    return _guard_optional(left.start + delay <= right.end, left, right)


def makespan_var(model: Model, tasks: Sequence[IntervalVar], horizon: int, *, name: str = "makespan") -> IntVar:
    makespan = model.int_var(0, horizon, name=name)
    for task in tasks:
        model.add(_guard_optional(task.end <= makespan, task))
    return makespan


def interval_value(solution, task: IntervalVar) -> IntervalValue:
    present = True if task.presence is None else bool(solution.value(task.presence))
    duration = int(task.duration) if isinstance(task.duration, int) else solution.value(task.duration)
    if not present:
        return IntervalValue(start=0, end=0, duration=duration, present=False)
    start = solution.value(task.start)
    return IntervalValue(start=start, end=start + duration, duration=duration, present=True)


def _field_name(name: Optional[str], field: str) -> Optional[str]:
    return None if name is None else f"{name}.{field}"


def _as_sequence(sequence: Union[SequenceVar, Sequence[IntervalVar]], transition_matrix=None) -> SequenceVar:
    if isinstance(sequence, SequenceVar):
        if transition_matrix is not None and sequence.transition_matrix is not None:
            raise ValueError("transition matrix was supplied twice")
        return sequence
    return SequenceVar(intervals=sequence, transition_matrix=transition_matrix)


def _can_use_native_no_overlap(tasks: Sequence[IntervalVar]) -> bool:
    return all_builtin(not task.optional and task.fixed_duration for task in tasks)


def _post_native_no_overlap(model: Model, tasks: Sequence[IntervalVar]) -> None:
    model.no_overlap([task.start for task in tasks], [int(task.duration) for task in tasks])


def _post_capacity_one_cumulative(model: Model, tasks: Sequence[IntervalVar]) -> None:
    starts = [task.start for task in tasks]
    durations = [_as_var(model, task.duration, f"{task.name or 'task'}.duration") for task in tasks]
    heights = [task.presence if task.presence is not None else model.int_var(1, 1, name=f"{task.name or 'task'}.present") for task in tasks]
    capacity = model.int_var(1, 1, name="no_overlap.capacity")
    model.cumulative_var(starts, durations, heights, capacity)


def _setup_time(seq: SequenceVar, matrix: Optional[Sequence[Sequence[int]]], left: int, right: int) -> int:
    if matrix is None:
        return 0
    if seq.types is None:
        raise ValueError("sequence types are required when a transition matrix is used")
    return int(matrix[seq.types[left]][seq.types[right]])


def _guard_optional(constraint: Constraint, *tasks: IntervalVar) -> Constraint:
    absence = [task.presence == 0 for task in tasks if task.presence is not None]
    if not absence:
        return constraint
    return reduce(or_, absence + [constraint])


def _as_var(model: Model, value: Union[int, IntVar], name: str) -> IntVar:
    if isinstance(value, IntVar):
        return value
    return model.int_var(int(value), int(value), name=name)
