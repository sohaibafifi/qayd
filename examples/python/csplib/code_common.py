"""Shared finite-alphabet distance encodings for CSPLib code problems."""

from __future__ import annotations

import qayd as cp


def symbol_distance(
    model: cp.Model,
    left: cp.IntVar,
    right: cp.IntVar,
    *,
    alphabet: int,
    metric: str,
    name: str,
) -> cp.IntVar:
    if metric == "hamming":
        distance = model.bool_var(name=name)
        allowed = [
            (first, second, int(first != second))
            for first in range(alphabet)
            for second in range(alphabet)
        ]
    elif metric == "lee":
        distance = model.int_var(0, alphabet // 2, name=name)
        allowed = [
            (first, second, min(abs(first - second), alphabet - abs(first - second)))
            for first in range(alphabet)
            for second in range(alphabet)
        ]
    else:
        raise ValueError("metric must be 'hamming' or 'lee'")
    model.table([left, right, distance], allowed)
    return distance
