"""Scheduling convenience API built on top of interval primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from . import _core as cp


_ORIGINAL_MODEL_ADD = cp.Model.add
_ORIGINAL_MODEL_MINIMIZE = cp.Model.minimize
_PATCHED = False


def _as_id(item: Union[int, "TaskRef"]) -> int:
    return item.id if isinstance(item, TaskRef) else int(item)


def _unwrap(value: Any) -> Any:
    return value.value if isinstance(value, TaskValue) else value


def _as_i64(value: Any, what: str) -> int:
    value = _unwrap(value)
    if not isinstance(value, int):
        raise TypeError(f"{what} must be an int")
    return value


@dataclass(frozen=True)
class TaskValue:
    """A value attached to one task."""

    task: "TaskRef"
    name: str
    value: Any

    def __int__(self) -> int:
        return _as_i64(self.value, self.name)

    def __index__(self) -> int:
        return int(self)

    def __getitem__(self, key: Any) -> Any:
        key = _unwrap(key)
        return self.value[key]

    def __bool__(self) -> bool:
        return bool(self.value)

    def __eq__(self, other: object) -> bool:
        return self.value == _unwrap(other)

    def __lt__(self, other: Any) -> bool:
        return self.value < _unwrap(other)

    def __le__(self, other: Any) -> bool:
        return self.value <= _unwrap(other)

    def __gt__(self, other: Any) -> bool:
        return self.value > _unwrap(other)

    def __ge__(self, other: Any) -> bool:
        return self.value >= _unwrap(other)

    def __hash__(self) -> int:
        return hash(self.value)


class TaskRef:
    """One task in a scheduling model."""

    __slots__ = ("_set", "id", "_attrs")

    def __init__(self, owner: "TaskSet", item_id: int):
        object.__setattr__(self, "_set", owner)
        object.__setattr__(self, "id", int(item_id))
        object.__setattr__(self, "_attrs", {})

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name == "id":
            object.__setattr__(self, name, value)
            return
        self._attrs[name] = value
        self._set._set_attr_value(name, self.id, value)

    def __getattr__(self, name: str) -> TaskValue:
        try:
            return TaskValue(self, name, self._attrs[name])
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __repr__(self) -> str:
        return f"TaskRef({self.id})"

    def __hash__(self) -> int:
        return hash((id(self._set), self.id))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TaskRef) and self._set is other._set and self.id == other.id


class TaskData:
    """A named value table over tasks."""

    def __init__(self, owner: "TaskSet", name: str):
        self._set = owner
        self.name = name

    def __getitem__(self, task: Union[int, TaskRef]) -> Any:
        return self._set._attrs[self.name][_as_id(task)]

    def __repr__(self) -> str:
        return f"TaskData({self.name})"


class TaskSet:
    """Ordered task ids plus per-task data."""

    def __init__(self, items: Iterable[int]):
        ids = [int(item) for item in items]
        if not ids:
            raise ValueError("tasks cannot be empty")
        if len(set(ids)) != len(ids):
            raise ValueError("task ids must be distinct")
        self.ids = ids
        self._by_id = {item_id: TaskRef(self, item_id) for item_id in ids}
        self._attrs: Dict[str, Dict[int, Any]] = {}

    def __iter__(self) -> Iterator[TaskRef]:
        return (self._by_id[item_id] for item_id in self.ids)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, item: Union[int, TaskRef]) -> TaskRef:
        item_id = _as_id(item)
        try:
            return self._by_id[item_id]
        except KeyError as exc:
            raise KeyError(f"unknown task id {item_id}") from exc

    def attr(self, name: str, values: Union[Sequence[Any], Dict[int, Any]]) -> TaskData:
        if isinstance(values, dict):
            mapped = {int(k): v for k, v in values.items()}
        else:
            mapped = {item_id: values[item_id] for item_id in self.ids}
        missing = [item_id for item_id in self.ids if item_id not in mapped]
        if missing:
            raise ValueError(f"missing {name!r} values for task ids {missing}")
        for item_id, value in mapped.items():
            if item_id in self._by_id:
                self._by_id[item_id]._attrs[name] = value
        self._attrs[name] = {item_id: mapped[item_id] for item_id in self.ids}
        return TaskData(self, name)

    def group_by(self, fn_or_attr: Union[str, Callable[[TaskRef], Any]]) -> Dict[Any, List[TaskRef]]:
        groups: Dict[Any, List[TaskRef]] = {}
        for task in self:
            key = getattr(task, fn_or_attr) if isinstance(fn_or_attr, str) else fn_or_attr(task)
            groups.setdefault(_unwrap(key), []).append(task)
        return groups

    def sum(self, fn: Callable[[TaskRef], Any]) -> Any:
        return cp.sum(_as_i64(fn(task), "task sum item") for task in self)

    def _set_attr_value(self, name: str, item_id: int, value: Any) -> None:
        self._attrs.setdefault(name, {})[item_id] = value

    def __repr__(self) -> str:
        return f"TaskSet({self.ids!r})"


class ScheduledTaskView:
    """Decision view of one task in a schedule."""

    def __init__(self, schedule: "ScheduleSet", task: TaskRef):
        self.schedule = schedule
        self.task = task

    @property
    def interval(self) -> Any:
        return self.schedule._interval_by_id[self.task.id]

    @property
    def start(self) -> "TaskEndpointExpr":
        return TaskEndpointExpr(self, "start")

    @property
    def end(self) -> "TaskEndpointExpr":
        return TaskEndpointExpr(self, "end")

    @property
    def presence(self) -> Any:
        return self.interval.presence

    def before(self, other: "ScheduledTaskView") -> "SchedulePrecedenceConstraint":
        return SchedulePrecedenceConstraint(self, other)

    def __repr__(self) -> str:
        return f"ScheduledTaskView({self.task.id})"


class TaskEndpointExpr:
    def __init__(self, view: ScheduledTaskView, kind: str):
        self.view = view
        self.kind = kind

    def __le__(self, rhs: Any) -> "ScheduleEndpointConstraint":
        return ScheduleEndpointConstraint(self, "le", rhs)

    def __ge__(self, rhs: Any) -> "ScheduleEndpointConstraint":
        return ScheduleEndpointConstraint(self, "ge", rhs)

    def __eq__(self, rhs: object) -> "ScheduleEndpointConstraint":  # type: ignore[override]
        return ScheduleEndpointConstraint(self, "eq", rhs)

    def __hash__(self) -> int:
        return hash((id(self.view.schedule), self.view.task.id, self.kind))

    def _core_expr(self) -> Any:
        interval = self.view.interval
        if self.kind == "start":
            start = interval.start
            if start is None:
                raise ValueError("start constraints require native intervals")
            return start
        if self.kind == "end":
            return interval.end
        raise ValueError(f"unsupported task endpoint {self.kind!r}")

    def __repr__(self) -> str:
        return f"{self.view!r}.{self.kind}"


class ScheduleConstraint:
    def post(self, model: Any) -> None:
        raise NotImplementedError


class ScheduleEndpointConstraint(ScheduleConstraint):
    def __init__(self, left: TaskEndpointExpr, op: str, right: Any):
        self.left = left
        self.op = op
        self.right = right

    def post(self, model: Any) -> None:
        right = self.right
        if isinstance(right, TaskEndpointExpr):
            if self.op == "le" and self.left.kind == "end" and right.kind == "start":
                SchedulePrecedenceConstraint(self.left.view, right.view).post(model)
                return
            self._post_core(model, self.left._core_expr(), right._core_expr())
            return
        self._post_core(model, self.left._core_expr(), _as_i64(right, "task endpoint bound"))

    def _post_core(self, model: Any, left: Any, right: Any) -> None:
        if self.op == "le":
            constraint = left <= right
        elif self.op == "ge":
            constraint = left >= right
        elif self.op == "eq":
            constraint = left == right
        else:
            raise ValueError(f"unsupported endpoint relation {self.op!r}")
        _ORIGINAL_MODEL_ADD(model, constraint)


class SchedulePrecedenceConstraint(ScheduleConstraint):
    def __init__(self, before: ScheduledTaskView, after: ScheduledTaskView):
        self.before = before
        self.after = after

    def post(self, model: Any) -> None:
        self._check_same_schedule()
        model.precedence(self.before.interval, self.after.interval)

    def _check_same_schedule(self) -> None:
        if self.before.schedule is not self.after.schedule:
            raise ValueError("schedule constraints must use one ScheduleSet")


class ScheduleNoOverlapConstraint(ScheduleConstraint):
    def __init__(self, schedule: "ScheduleSet", key: Optional[Callable[[TaskRef], Any]] = None):
        self.schedule = schedule
        self.key = key

    def post(self, model: Any) -> None:
        if self.key is None:
            if self.schedule.moded:
                model.no_overlap_by_machine()
            else:
                model.no_overlap(self.schedule.intervals())
            return
        for group in self.schedule.tasks.group_by(self.key).values():
            if len(group) > 1:
                model.no_overlap(self.schedule.intervals(group))


class ScheduleResourceExpr:
    def __init__(self, schedule: "ScheduleSet", amount: Callable[[TaskRef], Any]):
        self.schedule = schedule
        self.amount = amount

    def __le__(self, capacity: Any) -> "ScheduleResourceConstraint":
        return ScheduleResourceConstraint(self.schedule, self.amount, "le", _as_i64(capacity, "resource capacity"))

    def __ge__(self, capacity: Any) -> "ScheduleResourceConstraint":
        return ScheduleResourceConstraint(self.schedule, self.amount, "ge", _as_i64(capacity, "resource capacity"))

    def __eq__(self, capacity: object) -> "ScheduleResourceConstraint":  # type: ignore[override]
        return ScheduleResourceConstraint(self.schedule, self.amount, "eq", _as_i64(capacity, "resource capacity"))

    def __hash__(self) -> int:
        return hash((id(self.schedule), id(self.amount)))


class ScheduleResourceConstraint(ScheduleConstraint):
    def __init__(self, schedule: "ScheduleSet", amount: Callable[[TaskRef], Any], op: str, capacity: int):
        self.schedule = schedule
        self.amount = amount
        self.op = op
        self.capacity = capacity

    def post(self, model: Any) -> None:
        if self.op != "le":
            raise ValueError("schedule.resource(...) currently supports only <= capacity")
        demands = []
        for task in self.schedule.tasks:
            amount = _as_i64(self.amount(task), "resource demand")
            if amount < 0:
                raise ValueError("resource demand must be non-negative")
            if amount:
                demands.append((self.schedule[task].interval, amount))
        if demands:
            model.resource(demands, self.capacity)


class ScheduleMakespanObjective:
    def __init__(self, schedule: "ScheduleSet"):
        self.schedule = schedule

    def post_minimize(self, model: Any) -> None:
        model.minimize_makespan(self.schedule.intervals())


class ScheduleSet:
    """A set of scheduled tasks over interval variables."""

    def __init__(
        self,
        model: Any,
        tasks: Union[TaskSet, Iterable[int]],
        *,
        horizon: int,
        optional: bool = False,
    ):
        if not isinstance(tasks, TaskSet):
            tasks = TaskSet(tasks)
        self.model = model
        self.tasks = tasks
        self.horizon = int(horizon)
        self.optional = bool(optional)
        self.moded = self._uses_modes()
        if self.moded:
            modes = [self._task_modes(task) for task in tasks]
            self._intervals = model.alternatives(modes, self.horizon)
        else:
            durations = [self._task_duration(task) for task in tasks]
            self._intervals = model.intervals(durations, self.horizon, optional=optional)
        self._interval_by_id = {task.id: interval for task, interval in zip(tasks, self._intervals)}

    def __iter__(self) -> Iterator[ScheduledTaskView]:
        return (self[task] for task in self.tasks)

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, task: Union[int, TaskRef]) -> ScheduledTaskView:
        return ScheduledTaskView(self, self.tasks[task])

    def intervals(self, tasks: Optional[Iterable[Union[int, TaskRef]]] = None) -> List[Any]:
        if tasks is None:
            return list(self._intervals)
        return [self[task].interval for task in tasks]

    def resource(self, amount: Callable[[TaskRef], Any]) -> ScheduleResourceExpr:
        return ScheduleResourceExpr(self, amount)

    def no_overlap(self, key: Optional[Callable[[TaskRef], Any]] = None) -> ScheduleNoOverlapConstraint:
        return ScheduleNoOverlapConstraint(self, key)

    def makespan(self) -> ScheduleMakespanObjective:
        return ScheduleMakespanObjective(self)

    def _uses_modes(self) -> bool:
        has_modes = ["modes" in task._attrs for task in self.tasks]
        if any(has_modes) and not all(has_modes):
            raise ValueError("either all tasks define modes, or none of them do")
        return all(has_modes)

    def _task_duration(self, task: TaskRef) -> int:
        if "duration" not in task._attrs:
            raise ValueError(f"task {task.id} is missing duration")
        return _as_i64(task._attrs["duration"], "task duration")

    def _task_modes(self, task: TaskRef) -> List[Tuple[int, int]]:
        raw_modes = task._attrs["modes"]
        modes = [(int(machine), _as_i64(duration, "mode duration")) for machine, duration in raw_modes]
        if not modes:
            raise ValueError(f"task {task.id} needs at least one mode")
        return modes

    def __repr__(self) -> str:
        return f"ScheduleSet(tasks={len(self)}, horizon={self.horizon})"


def tasks(self: Any, items: Iterable[int]) -> TaskSet:
    return TaskSet(items)


def schedule(
    self: Any,
    tasks: Union[TaskSet, Iterable[int]],
    *,
    horizon: int,
    optional: bool = False,
) -> ScheduleSet:
    return ScheduleSet(self, tasks, horizon=horizon, optional=optional)


def _scheduling_add(self: Any, constraint: Any) -> None:
    if isinstance(constraint, ScheduleConstraint):
        constraint.post(self)
        return
    if isinstance(constraint, Iterable) and not isinstance(constraint, (str, bytes)):
        try:
            items = list(constraint)
        except TypeError:
            pass
        else:
            if any(isinstance(item, ScheduleConstraint) for item in items):
                for item in items:
                    self.add(item)
                return
    _ORIGINAL_MODEL_ADD(self, constraint)


def _scheduling_minimize(self: Any, objective: Any) -> None:
    if isinstance(objective, ScheduleMakespanObjective):
        objective.post_minimize(self)
        return
    _ORIGINAL_MODEL_MINIMIZE(self, objective)


def install_model_api() -> None:
    """Attach scheduling helpers to the extension Model class."""
    global _PATCHED
    if _PATCHED:
        return
    cp.Model.tasks = tasks
    cp.Model.schedule = schedule
    cp.Model.add = _scheduling_add
    cp.Model.minimize = _scheduling_minimize
    _PATCHED = True


__all__ = [
    "ScheduleSet",
    "ScheduledTaskView",
    "TaskData",
    "TaskRef",
    "TaskSet",
    "TaskValue",
    "install_model_api",
]
