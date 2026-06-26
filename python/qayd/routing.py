"""Routing helpers built on the Python qayd model API."""

from __future__ import annotations

from typing import Sequence

from . import IntVar, Model


def circuit(model: Model, successors: Sequence[IntVar]) -> None:
    model.circuit(successors)
