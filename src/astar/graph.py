"""Graph and A* implementation based on **vertices / edges (adjacency list)**.

Each vertex has a unique id and a planar position ``(x, y)`` (the heuristic uses positions to
estimate the remaining distance); edges are stored explicitly in an adjacency list with
non-negative weights. Both directed and undirected graphs are supported.
"""

from __future__ import annotations

import heapq
import itertools
import math
import time
from collections.abc import Hashable, Iterable, Iterator

from .heuristics import Heuristic, get_heuristic, heuristic_name
from .result import SearchResult

Vertex = Hashable


class Graph:
    """Adjacency-list graph.

    >>> graph = Graph()
    >>> graph.add_vertex("A", (0, 0))
    ... graph.add_vertex("B", (3, 4))
    >>> graph.add_edge("A", "B")  # weight defaults to the Euclidean distance = 5
    >>> astar_graph(graph, "A", "B").cost
    5.0
    """

    def __init__(self, directed: bool = False, name: str = "graph"):
        self.directed = directed
        self.name = name
        self.positions: dict[Vertex, tuple[float, float]] = {}
        self.adjacency: dict[
            Vertex, dict[Vertex, float]
        ] = {}  # {vertex: {neighbor: edge weight}}
        self.start: Vertex | None = None
        self.goal: Vertex | None = None

    # ------------------------------------------------------------------ construction
    def add_vertex(self, vertex: Vertex, position: tuple[float, float]) -> None:
        self.positions[vertex] = (float(position[0]), float(position[1]))
        self.adjacency.setdefault(vertex, {})

    def add_edge(
        self, source: Vertex, target: Vertex, weight: float | None = None
    ) -> None:
        """Add an edge; ``weight`` defaults to the Euclidean distance between its endpoints."""
        if source not in self.positions or target not in self.positions:
            raise KeyError("call add_vertex before add_edge")
        if weight is None:
            weight = self.distance(source, target)
        if weight < 0:
            raise ValueError("A*/Dijkstra require non-negative edge weights")
        self.adjacency[source][target] = float(weight)
        if not self.directed:
            self.adjacency[target][source] = float(weight)

    @classmethod
    def from_edges(
        cls,
        positions: dict[Vertex, tuple[float, float]],
        edges: Iterable[tuple[Vertex, Vertex] | tuple[Vertex, Vertex, float]],
        directed: bool = False,
        name: str = "graph",
    ) -> "Graph":
        graph = cls(directed=directed, name=name)
        for vertex, position in positions.items():
            graph.add_vertex(vertex, position)
        for edge in edges:
            graph.add_edge(*edge)
        return graph

    # ------------------------------------------------------------------ queries
    def distance(self, first: Vertex, second: Vertex) -> float:
        (first_x, first_y), (second_x, second_y) = (
            self.positions[first],
            self.positions[second],
        )
        return math.hypot(first_x - second_x, first_y - second_y)

    def neighbors(self, vertex: Vertex) -> Iterator[tuple[Vertex, float]]:
        yield from self.adjacency[vertex].items()

    def edges(self) -> Iterator[tuple[Vertex, Vertex, float]]:
        """Iterate over all edges (source, target, weight); in undirected graphs each edge appears once."""
        seen_edges = set()
        for source, neighbors in self.adjacency.items():
            for target, weight in neighbors.items():
                edge_key = (
                    (source, target) if self.directed else frozenset((source, target))
                )
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    yield source, target, weight

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def edge_count(self) -> int:
        return sum(1 for _ in self.edges())

    def path_cost(self, path: list[Vertex]) -> float:
        return sum(
            self.adjacency[current][following]
            for current, following in itertools.pairwise(path)
        )


def astar_graph(
    graph: Graph,
    start: Vertex | None = None,
    goal: Vertex | None = None,
    heuristic: str | Heuristic = "euclidean",
    label: str | None = None,
) -> SearchResult:
    """Run A* on a vertex/edge graph; the heuristic operates on vertex positions. ``heuristic="zero"`` is Dijkstra.

    Note: Euclidean distance is only admissible if *every edge weight ≥ the Euclidean distance between
    its endpoints*. If edge weights are on a different scale than positions, scale it with
    ``"euclidean*k"`` (k = min(weight / distance)).
    """
    start = start if start is not None else graph.start
    goal = goal if goal is not None else graph.goal
    if start not in graph.positions or goal not in graph.positions:
        raise KeyError("start or goal is not in the graph")

    heuristic_fn = get_heuristic(heuristic)
    heuristic_label = heuristic_name(heuristic)
    positions = graph.positions
    goal_position = positions[goal]

    start_time = time.perf_counter()
    insertion_order = (
        itertools.count()
    )  # FIFO on heap ties, and avoids comparing vertices themselves
    cost_from_start: dict[Vertex, float] = {
        start: 0.0
    }  # g values: best known cost from start to each vertex
    came_from: dict[Vertex, Vertex] = {}  # parent pointers for reconstructing the path
    start_estimate = heuristic_fn(positions[start], goal_position)
    # Heap entry: (f = g + h, h, insertion order, g, vertex)
    open_heap = [(start_estimate, start_estimate, next(insertion_order), 0.0, start)]
    expanded: list[Vertex] = []
    discovered: list[Vertex] = [start]

    while open_heap:
        _, _, _, current_cost, current = heapq.heappop(open_heap)
        if current_cost > cost_from_start[current]:  # stale entry (lazy deletion)
            continue
        expanded.append(current)
        if current == goal:
            break
        for neighbor, edge_weight in graph.adjacency[current].items():
            new_cost = current_cost + edge_weight
            if new_cost < cost_from_start.get(neighbor, math.inf):
                if neighbor not in cost_from_start:
                    discovered.append(neighbor)
                cost_from_start[neighbor] = new_cost
                came_from[neighbor] = current
                neighbor_estimate = heuristic_fn(positions[neighbor], goal_position)
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


def dijkstra_graph(
    graph: Graph, start: Vertex | None = None, goal: Vertex | None = None
) -> SearchResult:
    """Dijkstra = A* with a heuristic that is always 0."""
    return astar_graph(graph, start, goal, heuristic="zero", label="Dijkstra")
