"""List-domain resource scans built on top of ``scan_sum``."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from . import _core as cp


_ORIGINAL_MODEL_ADD = cp.Model.add
_PATCHED = False


def _as_i64(value: Any, what: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{what} must be an int")
    return value


def _list_var(route: Any) -> Any:
    return route._list if hasattr(route, "_list") else route


def _bound_violation(value: Any, op: str, rhs: int) -> Any:
    if op == "le":
        return cp.max(0, value - rhs)
    if op == "ge":
        return cp.max(0, rhs - value)
    if op == "eq":
        return cp.abs(value - rhs)
    raise ValueError(f"unsupported resource relation {op!r}")


class ScanResource:
    """A deterministic state threaded along one list variable."""

    def __init__(
        self,
        route: Any,
        *,
        transition: Callable[[Any, Any, Any], Any],
        initial: int = 0,
        boundary: int = 0,
        lower: Optional[int] = None,
        upper: Optional[int] = None,
        end: Optional[int] = None,
    ):
        self.route = _list_var(route)
        self.transition = transition
        self.initial = _as_i64(initial, "resource initial value")
        self.boundary = int(boundary)
        self.lower = None if lower is None else _as_i64(lower, "resource lower bound")
        self.upper = None if upper is None else _as_i64(upper, "resource upper bound")
        # Closing node: when set, the scan folds one more transition for the return
        # edge (last item -> end), so bounds hold over the CLOSED tour including the
        # return arc. None keeps the scan stopping at the last item.
        self.end = None if end is None else int(end)
        self._views: Dict[str, Callable[[Any, Any, Any], Any]] = {
            "after": lambda cur, state, prev: state,
        }

    def view(self, name: str, projection: Callable[[Any, Any, Any], Any]) -> "ScanResource":
        if name in self._views:
            raise ValueError(f"resource view {name!r} already exists")
        self._views[name] = projection
        return self

    def at(self, item: int, *, view: str = "after") -> "ScanResourcePointExpr":
        return ScanResourcePointExpr(self, view, int(item))

    def after(self, item: int) -> "ScanResourcePointExpr":
        return self.at(item, view="after")

    def before(self, item: int) -> "ScanResourcePointExpr":
        return self.at(item, view="before")

    def within_bounds(self, *, view: str = "after", lower: Optional[int] = None, upper: Optional[int] = None) -> "ScanResourceBoundsConstraint":
        if lower is None:
            lower = self.lower
        if upper is None:
            upper = self.upper
        if lower is None and upper is None:
            raise ValueError("within_bounds needs a lower or upper bound")
        return ScanResourceBoundsConstraint(self, view, lower, upper)

    def _projection(self, view: str) -> Callable[[Any, Any, Any], Any]:
        try:
            return self._views[view]
        except KeyError as exc:
            raise ValueError(f"unknown resource view {view!r}") from exc

    def _scan_violation_for_item(self, view: str, item: int, op: str, rhs: int) -> Any:
        projection = self._projection(view)

        def emit(cur: Any, state: Any, prev: Any) -> Any:
            value = projection(cur, state, prev)
            return cp.if_(cp.eq(cur, item), _bound_violation(value, op, rhs), 0)

        return cp.scan_sum(self.route, step=self.transition, emit=emit, init=self.initial, boundary=self.boundary, end=self.end)

    def _scan_bounds_violation(self, view: str, lower: Optional[int], upper: Optional[int]) -> Any:
        projection = self._projection(view)

        def emit(cur: Any, state: Any, prev: Any) -> Any:
            value = projection(cur, state, prev)
            violation = 0
            if lower is not None:
                violation = violation + cp.max(0, lower - value)
            if upper is not None:
                violation = violation + cp.max(0, value - upper)
            return violation

        return cp.scan_sum(self.route, step=self.transition, emit=emit, init=self.initial, boundary=self.boundary, end=self.end)


class ScanResourcePointExpr:
    def __init__(self, resource: ScanResource, view: str, item: int):
        self.resource = resource
        self.view = view
        self.item = int(item)

    def __le__(self, rhs: Any) -> "ScanResourcePointConstraint":
        return ScanResourcePointConstraint(self, "le", _as_i64(rhs, "resource bound"))

    def __ge__(self, rhs: Any) -> "ScanResourcePointConstraint":
        return ScanResourcePointConstraint(self, "ge", _as_i64(rhs, "resource bound"))

    def __eq__(self, rhs: object) -> "ScanResourcePointConstraint":  # type: ignore[override]
        return ScanResourcePointConstraint(self, "eq", _as_i64(rhs, "resource bound"))

    def __hash__(self) -> int:
        return hash((id(self.resource), self.view, self.item))


class ScanResourceConstraint:
    def post(self, model: Any) -> None:
        raise NotImplementedError


class ScanResourcePointConstraint(ScanResourceConstraint):
    def __init__(self, expr: ScanResourcePointExpr, op: str, rhs: int):
        self.expr = expr
        self.op = op
        self.rhs = rhs

    def post(self, model: Any) -> None:
        term = self.expr.resource._scan_violation_for_item(self.expr.view, self.expr.item, self.op, self.rhs)
        _ORIGINAL_MODEL_ADD(model, term <= 0)


class ScanResourceBoundsConstraint(ScanResourceConstraint):
    def __init__(self, resource: ScanResource, view: str, lower: Optional[int], upper: Optional[int]):
        self.resource = resource
        self.view = view
        self.lower = lower
        self.upper = upper

    def post(self, model: Any) -> None:
        term = self.resource._scan_bounds_violation(self.view, self.lower, self.upper)
        _ORIGINAL_MODEL_ADD(model, term <= 0)


def scan_resource(
    route: Any,
    *,
    initial: int = 0,
    lower: Optional[int] = None,
    upper: Optional[int] = None,
    bounds: Optional[Tuple[int, int]] = None,
    transition: Optional[Callable[[Any, Any, Any], Any]] = None,
    delta: Optional[Callable[[Any, Any], Any]] = None,
    boundary: int = 0,
    end: Optional[int] = None,
) -> ScanResource:
    """Thread one deterministic resource state along a list variable.

    ``end`` closes the tour: with it set, the scan folds one more transition for
    the return edge to ``end``, so ``lower``/``upper``/``within_bounds`` also hold
    at the closing node (the state after the return arc). ``None`` stops at the
    last item.
    """

    if bounds is not None:
        if lower is not None or upper is not None:
            raise ValueError("use either bounds=(lower, upper) or lower=/upper=, not both")
        lower, upper = bounds
    if transition is None:
        if delta is None:
            raise ValueError("scan_resource needs transition= or delta=")

        def transition(cur: Any, state: Any, prev: Any) -> Any:
            return state + delta(prev, cur)

    return ScanResource(route, transition=transition, initial=initial, boundary=boundary, lower=lower, upper=upper, end=end)


def _resource_add(self: Any, constraint: Any) -> None:
    if isinstance(constraint, ScanResourceConstraint):
        constraint.post(self)
        return
    if isinstance(constraint, Iterable) and not isinstance(constraint, (str, bytes)):
        try:
            items = list(constraint)
        except TypeError:
            pass
        else:
            if any(isinstance(item, ScanResourceConstraint) for item in items):
                for item in items:
                    self.add(item)
                return
    _ORIGINAL_MODEL_ADD(self, constraint)


def install_model_api() -> None:
    """Attach list-resource constraints to ``Model.add``."""
    global _PATCHED
    if _PATCHED:
        return
    cp.Model.add = _resource_add
    _PATCHED = True


__all__ = [
    "ScanResource",
    "ScanResourceConstraint",
    "ScanResourcePointExpr",
    "scan_resource",
    "install_model_api",
]
