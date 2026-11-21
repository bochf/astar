"""基于 **顶点 / 边（邻接表）** 的图与 A* 实现。

每个顶点有一个唯一 id 和一个平面坐标 ``(x, y)``（启发函数用坐标估计剩余距离）；
边显式存储在邻接表中，带非负权重。支持有向图与无向图。
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
    """邻接表图。

    >>> graph = Graph()
    >>> graph.add_vertex("A", (0, 0)); graph.add_vertex("B", (3, 4))
    >>> graph.add_edge("A", "B")          # 权重缺省为两点欧氏距离 = 5
    >>> astar_graph(graph, "A", "B").cost
    5.0
    """

    def __init__(self, directed: bool = False, name: str = "graph"):
        self.directed = directed
        self.name = name
        self.positions: dict[Vertex, tuple[float, float]] = {}
        self.adjacency: dict[Vertex, dict[Vertex, float]] = {}   # {顶点: {邻居: 边权}}
        self.start: Vertex | None = None
        self.goal: Vertex | None = None

    # ------------------------------------------------------------------ 构造
    def add_vertex(self, vertex: Vertex, position: tuple[float, float]) -> None:
        self.positions[vertex] = (float(position[0]), float(position[1]))
        self.adjacency.setdefault(vertex, {})

    def add_edge(self, source: Vertex, target: Vertex, weight: float | None = None) -> None:
        """添加边；``weight`` 缺省时取两端点的欧氏距离。"""
        if source not in self.positions or target not in self.positions:
            raise KeyError("请先 add_vertex 再 add_edge")
        if weight is None:
            weight = self.distance(source, target)
        if weight < 0:
            raise ValueError("A*/Dijkstra 要求边权非负")
        self.adjacency[source][target] = float(weight)
        if not self.directed:
            self.adjacency[target][source] = float(weight)

    @classmethod
    def from_edges(cls, positions: dict[Vertex, tuple[float, float]],
                   edges: Iterable[tuple[Vertex, Vertex] | tuple[Vertex, Vertex, float]],
                   directed: bool = False, name: str = "graph") -> "Graph":
        graph = cls(directed=directed, name=name)
        for vertex, position in positions.items():
            graph.add_vertex(vertex, position)
        for edge in edges:
            graph.add_edge(*edge)
        return graph

    # ------------------------------------------------------------------ 查询
    def distance(self, first: Vertex, second: Vertex) -> float:
        (first_x, first_y), (second_x, second_y) = self.positions[first], self.positions[second]
        return math.hypot(first_x - second_x, first_y - second_y)

    def neighbors(self, vertex: Vertex) -> Iterator[tuple[Vertex, float]]:
        yield from self.adjacency[vertex].items()

    def edges(self) -> Iterator[tuple[Vertex, Vertex, float]]:
        """遍历所有边 (source, target, weight)；无向图每条边只出现一次。"""
        seen_edges = set()
        for source, neighbors in self.adjacency.items():
            for target, weight in neighbors.items():
                edge_key = (source, target) if self.directed else frozenset((source, target))
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    yield source, target, weight

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def edge_count(self) -> int:
        return sum(1 for _ in self.edges())

    def path_cost(self, path: list[Vertex]) -> float:
        return sum(self.adjacency[current][following] for current, following in itertools.pairwise(path))


def astar_graph(graph: Graph, start: Vertex | None = None, goal: Vertex | None = None,
                heuristic: str | Heuristic = "euclidean", label: str | None = None) -> SearchResult:
    """在顶点/边图上运行 A*；启发函数作用于顶点坐标。``heuristic="zero"`` 即 Dijkstra。

    注意：欧氏距离只有在 *每条边权 ≥ 其端点欧氏距离* 时才可采纳。若边权与坐标
    比例不同，可用 ``"euclidean*k"`` 缩放（k = min(权重/距离)）。
    """
    start = start if start is not None else graph.start
    goal = goal if goal is not None else graph.goal
    if start not in graph.positions or goal not in graph.positions:
        raise KeyError("起点或终点不在图中")

    heuristic_fn = get_heuristic(heuristic)
    heuristic_label = heuristic_name(heuristic)
    positions = graph.positions
    goal_position = positions[goal]

    start_time = time.perf_counter()
    insertion_order = itertools.count()               # 堆中平局时先进先出，且避免比较顶点本身
    cost_from_start: dict[Vertex, float] = {start: 0.0}  # g 值：目前已知的起点到该顶点的最小代价
    came_from: dict[Vertex, Vertex] = {}              # 父指针：用于回溯路径
    start_estimate = heuristic_fn(positions[start], goal_position)
    # 堆条目：(f = g + h, h, 插入序号, g, 顶点)
    open_heap = [(start_estimate, start_estimate, next(insertion_order), 0.0, start)]
    expanded: list[Vertex] = []
    discovered: list[Vertex] = [start]

    while open_heap:
        _, _, _, current_cost, current = heapq.heappop(open_heap)
        if current_cost > cost_from_start[current]:  # 过期条目（惰性删除）
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
                heapq.heappush(open_heap, (new_cost + neighbor_estimate, neighbor_estimate,
                                           next(insertion_order), new_cost, neighbor))
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    path = None
    if goal in cost_from_start:
        path = [goal]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path.reverse()
    algorithm_name = label or ("Dijkstra" if heuristic_label == "zero" else f"A* ({heuristic_label})")
    return SearchResult(path=path, cost=cost_from_start.get(goal, math.inf), expanded=expanded,
                        discovered=discovered, algorithm=algorithm_name, heuristic=heuristic_label,
                        elapsed_ms=elapsed_ms)


def dijkstra_graph(graph: Graph, start: Vertex | None = None, goal: Vertex | None = None) -> SearchResult:
    """Dijkstra = 启发函数恒为 0 的 A*。"""
    return astar_graph(graph, start, goal, heuristic="zero", label="Dijkstra")
