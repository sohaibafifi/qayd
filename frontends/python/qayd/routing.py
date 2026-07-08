"""Routing convenience API built on top of the list-domain primitives.

This module is intentionally a thin Python layer. It gives routing models names
that match the domain, while lowering to the existing ``list_vars`` and lambda
reductions in the Rust extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from . import _core as cp


_ORIGINAL_MODEL_ADD = cp.Model.add
_PATCHED = False


def _as_id(item: Union[int, "CustomerRef"]) -> int:
    return item.id if isinstance(item, CustomerRef) else int(item)


def _as_i64(value: Any, what: str) -> int:
    if isinstance(value, CustomerValue):
        value = value.value
    if not isinstance(value, int):
        raise TypeError(f"{what} must be an int")
    return value


def _dense_array(values: Dict[int, int], *, default: int = 0, what: str = "values") -> List[int]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if lo < 0:
        raise ValueError(f"{what} requires non-negative customer ids")
    data = [default] * (hi + 1)
    for key, value in values.items():
        data[key] = value
    return data


def _core_array(data: Any) -> Any:
    if data.__class__.__name__ == "Array":
        return data
    return cp.array(list(data))


def _core_matrix(data: Any) -> Any:
    if data.__class__.__name__ == "Matrix":
        return data
    return cp.matrix(data)


@dataclass(frozen=True)
class CustomerValue:
    """A scalar value attached to one customer."""

    customer: "CustomerRef"
    name: str
    value: int

    def __int__(self) -> int:
        return self.value


class CustomerRef:
    """One item in a routing model."""

    __slots__ = ("_set", "id", "_attrs")

    def __init__(self, owner: "CustomerSet", item_id: int):
        object.__setattr__(self, "_set", owner)
        object.__setattr__(self, "id", int(item_id))
        object.__setattr__(self, "_attrs", {})

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name == "id":
            object.__setattr__(self, name, value)
            return
        if not isinstance(value, int):
            raise TypeError(f"customer attribute {name!r} must be an int")
        self._attrs[name] = value
        self._set._set_attr_value(name, self.id, value)

    def __getattr__(self, name: str) -> CustomerValue:
        try:
            return CustomerValue(self, name, self._attrs[name])
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __repr__(self) -> str:
        return f"CustomerRef({self.id})"

    def __hash__(self) -> int:
        return hash((id(self._set), self.id))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CustomerRef) and self._set is other._set and self.id == other.id


class CustomerData:
    """A named dense value table over customers."""

    def __init__(self, owner: "CustomerSet", name: str):
        self._set = owner
        self.name = name

    def __getitem__(self, customer: Union[int, CustomerRef]) -> int:
        return self._set._attrs[self.name][_as_id(customer)]

    def array(self, *, default: int = 0) -> Any:
        return cp.array(self._set._dense_attr(self.name, default=default))

    def __repr__(self) -> str:
        return f"CustomerData({self.name})"


class CustomerSet:
    """Ordered customer ids plus per-customer data."""

    def __init__(self, items: Iterable[int]):
        ids = [int(item) for item in items]
        if not ids:
            raise ValueError("customers cannot be empty")
        if len(set(ids)) != len(ids):
            raise ValueError("customer ids must be distinct")
        self.ids = ids
        self._by_id = {item_id: CustomerRef(self, item_id) for item_id in ids}
        self._attrs: Dict[str, Dict[int, int]] = {}

    def __iter__(self) -> Iterator[CustomerRef]:
        return (self._by_id[item_id] for item_id in self.ids)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, item: Union[int, CustomerRef]) -> CustomerRef:
        item_id = _as_id(item)
        try:
            return self._by_id[item_id]
        except KeyError as exc:
            raise KeyError(f"unknown customer id {item_id}") from exc

    def attr(self, name: str, values: Union[Sequence[int], Dict[int, int]]) -> CustomerData:
        """Attach one integer value to each customer.

        A sequence is indexed by customer id, matching the existing list lambda
        style. A dict may be used for sparse ids.
        """
        if isinstance(values, dict):
            mapped = {int(k): int(v) for k, v in values.items()}
        else:
            mapped = {item_id: int(values[item_id]) for item_id in self.ids}
        missing = [item_id for item_id in self.ids if item_id not in mapped]
        if missing:
            raise ValueError(f"missing {name!r} values for customer ids {missing}")
        for item_id, value in mapped.items():
            if item_id in self._by_id:
                self._by_id[item_id]._attrs[name] = value
        self._attrs[name] = {item_id: mapped[item_id] for item_id in self.ids}
        return CustomerData(self, name)

    def sum(self, fn: Callable[[CustomerRef], Any]) -> Any:
        return cp.sum(fn(customer) for customer in self)

    def _set_attr_value(self, name: str, item_id: int, value: int) -> None:
        self._attrs.setdefault(name, {})[item_id] = value

    def _dense_attr(self, name: str, *, default: int = 0) -> List[int]:
        values = self._attrs.get(name, {})
        dense = {item_id: values.get(item_id, default) for item_id in self.ids}
        return _dense_array(dense, default=default, what=name)

    def __repr__(self) -> str:
        return f"CustomerSet({self.ids!r})"


class RouteItemRef:
    """Symbolic item used inside ``route.sum(lambda c: ...)``."""

    __slots__ = ("_routes", "_node")

    def __init__(self, routes: "RouteSet", node: Any):
        self._routes = routes
        self._node = node

    @property
    def id(self) -> Any:
        return self._node

    def __getattr__(self, name: str) -> Any:
        values = self._routes._value_array(None, default_attr=name, default=None)
        return values[self._node]

    def __repr__(self) -> str:
        return "RouteItemRef(*)"


class RouteVar:
    """One visible route."""

    def __init__(self, owner: "RouteSet", index: int, list_var: Any):
        self._routes = owner
        self.index = index
        self._list = list_var

    def used(self) -> Any:
        return cp.used(self._list)

    def count(self) -> Any:
        return cp.count(self._list)

    def sum(self, fn: Callable[[RouteItemRef], Any]) -> Any:
        return cp.sum(self._list, lambda item: fn(RouteItemRef(self._routes, item)))

    def distance(self, travel: Optional[Any] = None) -> Any:
        matrix = self._routes._travel_matrix(travel)
        depot = self._routes.depot
        return cp.sum_edges(self._list, lambda i, j: matrix[i][j], start=depot, end=depot)

    def load(self, demand: Optional[Any] = None) -> Any:
        if demand is None:
            return self.sum(lambda customer: customer.demand)
        values = self._routes._value_array(demand, default_attr="demand", default=None)
        return cp.sum(self._list, lambda i: values[i])

    def profit(self, profit: Optional[Any] = None) -> Any:
        values = self._routes._value_array(profit, default_attr="profit", default=0)
        return cp.sum(self._list, lambda i: values[i])

    def __repr__(self) -> str:
        return f"RouteVar({self.index})"


class RouteSet:
    """A set of visible routes over a customer set."""

    def __init__(
        self,
        model: Any,
        customers: Union[CustomerSet, Iterable[int]],
        *,
        vehicles: int,
        depot: int = 0,
        travel: Optional[Any] = None,
        optional: bool = False,
    ):
        if not isinstance(customers, CustomerSet):
            customers = CustomerSet(customers)
        self.model = model
        self.customers = customers
        self.depot = int(depot)
        self.travel = _core_matrix(travel) if travel is not None else None
        self.optional = bool(optional)
        self._lists = model.list_vars(customers.ids, count=int(vehicles), optional=optional)
        self._routes = [RouteVar(self, index, route) for index, route in enumerate(self._lists)]

    def __iter__(self) -> Iterator[RouteVar]:
        return iter(self._routes)

    def __len__(self) -> int:
        return len(self._routes)

    def __getitem__(self, customer: Union[int, CustomerRef]) -> "VisitView":
        return VisitView(self, self.customers[customer])

    def sum(self, fn: Callable[[RouteVar], Any]) -> Any:
        return cp.sum(fn(route) for route in self._routes)

    def used_count(self) -> Any:
        return self.sum(lambda route: route.used())

    def total_distance(self, travel: Optional[Any] = None) -> Any:
        return self.sum(lambda route: route.distance(travel))

    def total_profit(self, profit: Optional[Any] = None) -> Any:
        return self.sum(lambda route: route.profit(profit))

    def _travel_matrix(self, travel: Optional[Any] = None) -> Any:
        matrix = self.travel if travel is None else _core_matrix(travel)
        if matrix is None:
            raise ValueError("routes need a travel matrix for distance/start views")
        return matrix

    def _value_array(self, data: Optional[Any], *, default_attr: str, default: Optional[int]) -> Any:
        if isinstance(data, CustomerData):
            return data.array(default=0 if default is None else default)
        if data is not None:
            return _core_array(data)
        if default_attr in self.customers._attrs:
            return cp.array(self.customers._dense_attr(default_attr, default=0 if default is None else default))
        if default is None:
            raise ValueError(f"missing customer attribute {default_attr!r}")
        values = {item_id: default for item_id in self.customers.ids}
        return cp.array(_dense_array(values, default=default, what=default_attr))

    def _start_arrays(self) -> Tuple[Any, Any, Any]:
        travel = self._travel_matrix()
        release = cp.array(self.customers._dense_attr("earliest", default=0))
        service = cp.array(self.customers._dense_attr("service", default=0))
        return travel, release, service

    def __repr__(self) -> str:
        return f"RouteSet(vehicles={len(self)}, depot={self.depot})"


class VisitView:
    """Per-customer view induced by a route set."""

    def __init__(self, routes: RouteSet, customer: CustomerRef):
        self.routes = routes
        self.customer = customer

    @property
    def start(self) -> "VisitStateExpr":
        return VisitStateExpr(self, "start")

    @property
    def load_after(self) -> "VisitStateExpr":
        return VisitStateExpr(self, "load_after")

    @property
    def route(self) -> "VisitRouteExpr":
        return VisitRouteExpr(self)

    @property
    def position(self) -> "VisitPositionExpr":
        return VisitPositionExpr(self)

    def same_route(self, other: "VisitView") -> "SameRouteConstraint":
        return SameRouteConstraint(self, other)

    def before(self, other: "VisitView") -> "VisitOrderConstraint":
        return VisitOrderConstraint(self, other)

    def __repr__(self) -> str:
        return f"VisitView({self.customer.id})"


class VisitStateExpr:
    def __init__(self, visit: VisitView, kind: str):
        self.visit = visit
        self.kind = kind

    def __le__(self, rhs: Any) -> "VisitStateConstraint":
        return VisitStateConstraint(self, "le", _as_i64(rhs, "visit bound"))

    def __ge__(self, rhs: Any) -> "VisitStateConstraint":
        return VisitStateConstraint(self, "ge", _as_i64(rhs, "visit bound"))

    def __eq__(self, rhs: Any) -> "VisitStateConstraint":  # type: ignore[override]
        return VisitStateConstraint(self, "eq", _as_i64(rhs, "visit bound"))

    def __hash__(self) -> int:
        return hash((id(self.visit.routes), self.visit.customer.id, self.kind))

    def __repr__(self) -> str:
        return f"{self.visit!r}.{self.kind}"


class VisitRouteExpr:
    def __init__(self, visit: VisitView):
        self.visit = visit

    def __eq__(self, other: object) -> "SameRouteConstraint":  # type: ignore[override]
        if not isinstance(other, VisitRouteExpr):
            return NotImplemented
        return SameRouteConstraint(self.visit, other.visit)

    def __hash__(self) -> int:
        return hash((id(self.visit.routes), self.visit.customer.id, "route"))


class VisitPositionExpr:
    def __init__(self, visit: VisitView):
        self.visit = visit

    def __lt__(self, other: object) -> "VisitOrderConstraint":
        if not isinstance(other, VisitPositionExpr):
            return NotImplemented
        return VisitOrderConstraint(self.visit, other.visit)

    def __hash__(self) -> int:
        return hash((id(self.visit.routes), self.visit.customer.id, "position"))


class VisitConstraint:
    def post(self, model: Any) -> None:
        raise NotImplementedError


class VisitStateConstraint(VisitConstraint):
    def __init__(self, expr: VisitStateExpr, op: str, rhs: int):
        self.expr = expr
        self.op = op
        self.rhs = rhs

    def post(self, model: Any) -> None:
        visit = self.expr.visit
        routes = visit.routes
        customer_id = visit.customer.id
        for route in routes:
            if self.expr.kind == "start":
                term = self._start_violation(routes, route, customer_id)
            elif self.expr.kind == "load_after":
                term = self._load_violation(routes, route, customer_id)
            else:
                raise ValueError(f"unsupported visit state {self.expr.kind!r}")
            _ORIGINAL_MODEL_ADD(model, term <= 0)

    def _bound_violation(self, value: Any) -> Any:
        if self.op == "le":
            return cp.max(0, value - self.rhs)
        if self.op == "ge":
            return cp.max(0, self.rhs - value)
        if self.op == "eq":
            return cp.abs(value - self.rhs)
        raise ValueError(f"unsupported visit relation {self.op!r}")

    def _start_violation(self, routes: RouteSet, route: RouteVar, customer_id: int) -> Any:
        travel, release, service = routes._start_arrays()

        def step(cur: Any, acc: Any, prev: Any) -> Any:
            return cp.max(release[cur], acc + travel[prev][cur]) + service[cur]

        def emit(cur: Any, departure: Any, prev: Any) -> Any:
            del prev
            start = departure - service[cur]
            return cp.if_(cp.eq(cur, customer_id), self._bound_violation(start), 0)

        return cp.scan_sum(route._list, step=step, emit=emit, init=0, boundary=routes.depot)

    def _load_violation(self, routes: RouteSet, route: RouteVar, customer_id: int) -> Any:
        demand = routes._value_array(None, default_attr="demand", default=None)

        def step(cur: Any, acc: Any, prev: Any) -> Any:
            del prev
            return acc + demand[cur]

        def emit(cur: Any, load_after: Any, prev: Any) -> Any:
            del prev
            return cp.if_(cp.eq(cur, customer_id), self._bound_violation(load_after), 0)

        return cp.scan_sum(route._list, step=step, emit=emit, init=0, boundary=routes.depot)


class SameRouteConstraint(VisitConstraint):
    def __init__(self, left: VisitView, right: VisitView):
        self.left = left
        self.right = right

    def post(self, model: Any) -> None:
        self._check_same_routes()
        model.same_list(self.left.customer.id, self.right.customer.id)

    def _check_same_routes(self) -> None:
        if self.left.routes is not self.right.routes:
            raise ValueError("visit constraints must use one RouteSet")


class VisitOrderConstraint(VisitConstraint):
    def __init__(self, before: VisitView, after: VisitView):
        self.before = before
        self.after = after

    def post(self, model: Any) -> None:
        if self.before.routes is not self.after.routes:
            raise ValueError("visit constraints must use one RouteSet")
        routes = self.before.routes
        before = self.before.customer.id
        after = self.after.customer.id
        size = max([before, after, *routes.customers.ids, routes.depot]) + 1
        if min(before, after, *routes.customers.ids, routes.depot) < 0:
            raise ValueError("position constraints require non-negative customer ids")
        marker = [[0 for _ in range(size)] for _ in range(size)]
        marker[after][before] = 1
        matrix = cp.matrix(marker)
        for route in routes:
            term = cp.pos_pairs(route._list, lambda a, b, i, j: (i < j) * matrix[a][b])
            _ORIGINAL_MODEL_ADD(model, term <= 0)


def customers(self: Any, items: Iterable[int]) -> CustomerSet:
    return CustomerSet(items)


def routes(
    self: Any,
    customers: Union[CustomerSet, Iterable[int]],
    *,
    vehicles: int,
    depot: int = 0,
    travel: Optional[Any] = None,
    optional: bool = False,
) -> RouteSet:
    return RouteSet(self, customers, vehicles=vehicles, depot=depot, travel=travel, optional=optional)


def _routing_add(self: Any, constraint: Any) -> None:
    if isinstance(constraint, VisitConstraint):
        constraint.post(self)
        return
    if isinstance(constraint, Iterable) and not isinstance(constraint, (str, bytes)):
        try:
            items = list(constraint)
        except TypeError:
            pass
        else:
            if any(isinstance(item, VisitConstraint) for item in items):
                for item in items:
                    self.add(item)
                return
    _ORIGINAL_MODEL_ADD(self, constraint)


def install_model_api() -> None:
    """Attach ``customers`` and ``routes`` helpers to the extension Model class."""
    global _PATCHED
    if _PATCHED:
        return
    cp.Model.customers = customers
    cp.Model.routes = routes
    cp.Model.add = _routing_add
    _PATCHED = True


__all__ = [
    "CustomerData",
    "CustomerRef",
    "CustomerSet",
    "CustomerValue",
    "RouteSet",
    "RouteVar",
    "VisitView",
    "install_model_api",
]
