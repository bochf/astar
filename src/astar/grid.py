"""Graph and A* implementation based on a **2D matrix**.

The map is a 2D ``numpy`` array ``cell_costs[row, col]``:

- a positive value: the cost of entering that cell (1 for ordinary ground; swamps/mountains can be higher);
- ``inf`` (the constant :data:`WALL`): an obstacle, impassable.

Nodes are coordinates ``(row, col)``; neighbors are given implicitly by matrix indices, so edges
need not be stored explicitly.
"""

from __future__ import annotations

import heapq
import itertools
import math
import time
from collections.abc import Iterator

import numpy as np

from .heuristics import Heuristic, get_heuristic, heuristic_name
from .result import SearchResult

Cell = tuple[int, int]
WALL = math.inf

_ORTHOGONAL_STEPS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIAGONAL_STEPS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_ALL_STEPS = _ORTHOGONAL_STEPS + _DIAGONAL_STEPS
SQRT2 = math.sqrt(2)


class GridMap:
    """2D matrix map.

    Parameters
    ----------
    cell_costs:
        2D array whose elements are the cost of entering each cell; ``inf`` means wall.
    diagonal:
        Whether 8-directional movement is allowed. A diagonal step costs √2 × the target cell's cost,
        and "corner cutting" is forbidden (no diagonal move if either orthogonally adjacent cell is a wall).
    """

    def __init__(
        self,
        cell_costs: np.ndarray,
        diagonal: bool = False,
        start: Cell | None = None,
        goal: Cell | None = None,
        name: str = "grid",
    ):
        cell_costs = np.asarray(cell_costs, dtype=float)
        if cell_costs.ndim != 2:
            raise ValueError("cell_costs must be a 2D matrix")
        if np.any(cell_costs <= 0):
            raise ValueError("cell costs must be positive (use inf for walls)")
        self.cell_costs = cell_costs
        self.diagonal = diagonal
        self.start = start
        self.goal = goal
        self.name = name

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_ascii(
        cls, text: str, diagonal: bool = False, name: str = "ascii"
    ) -> "GridMap":
        """Build a map from ASCII art.

        ``.`` or space = cost 1, ``#`` = wall, ``1``-``9`` = that cost,
        ``S`` = start, ``G`` = goal (both cost 1).
        """
        lines = text.strip("\n").splitlines()
        width = max(len(line) for line in lines)
        cell_costs = np.ones((len(lines), width))
        start = goal = None
        for row, line in enumerate(lines):
            for col, char in enumerate(line.ljust(width, ".")):
                if char == "#":
                    cell_costs[row, col] = WALL
                elif char.isdigit() and char != "0":
                    cell_costs[row, col] = int(char)
                elif char == "S":
                    start = (row, col)
                elif char == "G":
                    goal = (row, col)
        return cls(cell_costs, diagonal=diagonal, start=start, goal=goal, name=name)

    # ------------------------------------------------------------------ queries
    @property
    def shape(self) -> tuple[int, int]:
        return self.cell_costs.shape  # type: ignore[return-value]

    def in_bounds(self, cell: Cell) -> bool:
        row, col = cell
        return (
            0 <= row < self.cell_costs.shape[0] and 0 <= col < self.cell_costs.shape[1]
        )

    def passable(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and math.isfinite(self.cell_costs[cell])

    def neighbors(self, cell: Cell) -> Iterator[tuple[Cell, float]]:
        """Yield (neighbor cell, move cost)."""
        row, col = cell
        for row_step, col_step in _ALL_STEPS if self.diagonal else _ORTHOGONAL_STEPS:
            neighbor = (row + row_step, col + col_step)
            if not self.passable(neighbor):
                continue
            is_diagonal = row_step != 0 and col_step != 0
            if is_diagonal:
                # No corner cutting: both orthogonal cells passed by a diagonal move must be passable
                if not (
                    self.passable((row + row_step, col))
                    and self.passable((row, col + col_step))
                ):
                    continue
                yield neighbor, SQRT2 * float(self.cell_costs[neighbor])
            else:
                yield neighbor, float(self.cell_costs[neighbor])

    def path_cost(self, path: list[Cell]) -> float:
        total_cost = 0.0
        for current, following in itertools.pairwise(path):
            step_cost = dict(self.neighbors(current)).get(following)
            if step_cost is None:
                raise ValueError(f"{current} -> {following} is not a legal move")
            total_cost += step_cost
        return total_cost

    def default_heuristic(self) -> str:
        return "octile" if self.diagonal else "manhattan"


def astar_grid(
    grid: GridMap,
    start: Cell | None = None,
    goal: Cell | None = None,
    heuristic: str | Heuristic | None = None,
    label: str | None = None,
) -> SearchResult[Cell]:
    """Run A* on a 2D matrix.

    With ``heuristic="zero"`` this is Dijkstra. To keep heuristics admissible, the built-in heuristics
    assume a minimum cell cost of 1; if the map's minimum cost is below 1, they are automatically
    scaled by that minimum.

    Ties in f are broken in favor of nodes with smaller h (closer to the goal).
    If the heuristic is inconsistent (e.g. custom or weighted), nodes may be reopened and the result
    is still correct.
    """
    start = start if start is not None else grid.start
    goal = goal if goal is not None else grid.goal
    if start is None or goal is None:
        raise ValueError("start and goal must be specified")
    for endpoint in (start, goal):
        if not grid.passable(endpoint):
            raise ValueError(f"{endpoint} is impassable or out of bounds")

    heuristic_spec = heuristic if heuristic is not None else grid.default_heuristic()
    heuristic_fn = get_heuristic(heuristic_spec)
    heuristic_label = heuristic_name(heuristic_spec)

    # If any cell costs < 1, scale the heuristic by the minimum cost so it never overestimates
    passable_costs = grid.cell_costs[np.isfinite(grid.cell_costs)]
    heuristic_scale = min(1.0, float(passable_costs.min()))

    def estimate_to_goal(cell: Cell) -> float:
        return heuristic_scale * heuristic_fn(cell, goal)

    start_time = time.perf_counter()
    insertion_order = (
        itertools.count()
    )  # FIFO on heap ties, and avoids comparing nodes themselves
    cost_from_start: dict[Cell, float] = {
        start: 0.0
    }  # g values: best known cost from start to each cell
    came_from: dict[Cell, Cell] = {}  # parent pointers for reconstructing the path
    start_estimate = estimate_to_goal(start)
    # Heap entry: (f = g + h, h, insertion order, g, cell)
    open_heap: list[tuple[float, float, int, float, Cell]] = [
        (start_estimate, start_estimate, next(insertion_order), 0.0, start)
    ]
    expanded: list[Cell] = []
    discovered: list[Cell] = [start]

    while open_heap:
        _, _, _, current_cost, current = heapq.heappop(open_heap)
        if current_cost > cost_from_start[current]:  # stale entry (lazy deletion)
            continue
        expanded.append(current)
        if current == goal:
            break
        for neighbor, step_cost in grid.neighbors(current):
            new_cost = current_cost + step_cost
            if new_cost < cost_from_start.get(neighbor, math.inf):
                if neighbor not in cost_from_start:
                    discovered.append(neighbor)
                cost_from_start[neighbor] = new_cost
                came_from[neighbor] = current
                neighbor_estimate = estimate_to_goal(neighbor)
                heapq.heappush(
                    open_heap,
                    (
                        new_cost + neighbor_estimate,
                        neighbor_estimate,
                        next(insertion_order),
                        new_cost,
                        neighbor,
                    ),
                )
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    path = None
    if goal in cost_from_start:
        path = [goal]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path.reverse()
    algorithm_name = label or (
        "Dijkstra" if heuristic_label == "zero" else f"A* ({heuristic_label})"
    )
    return SearchResult(
        path=path,
        cost=cost_from_start.get(goal, math.inf),
        expanded=expanded,
        discovered=discovered,
        algorithm=algorithm_name,
        heuristic=heuristic_label,
        elapsed_ms=elapsed_ms,
    )


def dijkstra_grid(
    grid: GridMap, start: Cell | None = None, goal: Cell | None = None
) -> SearchResult[Cell]:
    """Dijkstra = A* with a heuristic that is always 0."""
    return astar_grid(grid, start, goal, heuristic="zero", label="Dijkstra")
