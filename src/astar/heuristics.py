"""Heuristic functions.

All heuristics share a common signature::

    heuristic(current: tuple[float, ...], goal: tuple[float, ...]) -> float

where ``current`` is the current node's coordinates and ``goal`` the goal node's coordinates. For a
2D matrix the coordinates are ``(row, col)``; for a vertex/edge graph they are the vertex's ``(x, y)``
position. The same heuristic can therefore be used with both data structures.

Built-in:

=============  ==========================================  =====================================
Name           Formula                                     Use case
=============  ==========================================  =====================================
``zero``       0                                           Degenerates to Dijkstra
``manhattan``  |dx| + |dy|                                 4-connected grids
``euclidean``  sqrt(dx² + dy²)                             Movement in any direction / planar graphs
``chebyshev``  max(|dx|, |dy|)                             8-connected with diagonal cost 1
``octile``     max + (√2 - 1)·min                          8-connected with diagonal cost √2
=============  ==========================================  =====================================

There are three ways to provide a custom heuristic:

1. Pass a callable directly to ``astar(..., heuristic=my_func)``;
2. Register a name with the :func:`register_heuristic` decorator, then refer to it by string;
3. Use the ``module.path:function`` form on the command line (see :func:`get_heuristic`).
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Sequence

Point = Sequence[float]
Heuristic = Callable[[Point, Point], float]

_REGISTRY: dict[str, Heuristic] = {}
_DESCRIPTIONS: dict[str, str] = {}


def register_heuristic(
    name: str, description: str = ""
) -> Callable[[Heuristic], Heuristic]:
    """Decorator: register a function as a named heuristic, usable afterwards by its string name.

    >>> @register_heuristic("half_manhattan", "half the Manhattan distance")
    ... def half(current, goal):
    ...     return (abs(current[0] - goal[0]) + abs(current[1] - goal[1])) / 2
    """

    def decorator(heuristic: Heuristic) -> Heuristic:
        _REGISTRY[name] = heuristic
        _DESCRIPTIONS[name] = (
            description or (heuristic.__doc__ or "").strip().split("\n")[0]
        )
        return heuristic

    return decorator


@register_heuristic("zero", "h = 0; A* degenerates to Dijkstra")
def zero(current: Point, goal: Point) -> float:
    return 0.0


@register_heuristic("manhattan", "|dx| + |dy|; exact lower bound on 4-connected grids")
def manhattan(current: Point, goal: Point) -> float:
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1])


@register_heuristic(
    "euclidean", "straight-line distance; admissible for movement in any direction"
)
def euclidean(current: Point, goal: Point) -> float:
    return math.hypot(current[0] - goal[0], current[1] - goal[1])


@register_heuristic("chebyshev", "max(|dx|, |dy|); lower bound when diagonal cost is 1")
def chebyshev(current: Point, goal: Point) -> float:
    return max(abs(current[0] - goal[0]), abs(current[1] - goal[1]))


@register_heuristic(
    "octile", "max + (√2-1)·min; exact lower bound when diagonal cost is √2"
)
def octile(current: Point, goal: Point) -> float:
    delta_0, delta_1 = abs(current[0] - goal[0]), abs(current[1] - goal[1])
    return max(delta_0, delta_1) + (math.sqrt(2) - 1) * min(delta_0, delta_1)


def weighted(base_heuristic: Heuristic, weight: float) -> Heuristic:
    """Return ``weight * base_heuristic``. weight > 1 is Weighted A*: fewer expansions but no optimality guarantee (cost ≤ weight·optimal)."""

    def weighted_heuristic(current: Point, goal: Point) -> float:
        return weight * base_heuristic(current, goal)

    weighted_heuristic.__name__ = (
        f"{getattr(base_heuristic, '__name__', 'heuristic')}x{weight:g}"
    )
    return weighted_heuristic


def available_heuristics() -> dict[str, str]:
    """Return {name: description}."""
    return dict(_DESCRIPTIONS)


def get_heuristic(spec: str | Heuristic) -> Heuristic:
    """Resolve a name / ``module:function`` / callable into a heuristic.

    - ``"manhattan"``: a built-in or registered name;
    - ``"manhattan*2"``: append ``*weight`` to a name for a weighted version;
    - ``"my_pkg.my_mod:my_func"``: import dynamically from a module (custom heuristic).
    """
    if callable(spec):
        return spec
    weight = 1.0
    if "*" in spec:
        spec, weight_text = spec.rsplit("*", 1)
        weight = float(weight_text)
    if spec in _REGISTRY:
        heuristic = _REGISTRY[spec]
    elif ":" in spec:
        module_name, function_name = spec.split(":", 1)
        heuristic = getattr(importlib.import_module(module_name), function_name)
        if not callable(heuristic):
            raise TypeError(f"{spec} is not callable")
    else:
        raise KeyError(
            f"unknown heuristic {spec!r}; choose from: {', '.join(_REGISTRY)}, or use module:function"
        )
    return weighted(heuristic, weight) if weight != 1.0 else heuristic


def heuristic_name(heuristic: Heuristic | str) -> str:
    if isinstance(heuristic, str):
        return heuristic
    for name, registered in _REGISTRY.items():
        if registered is heuristic:
            return name
    return getattr(heuristic, "__name__", "custom")
