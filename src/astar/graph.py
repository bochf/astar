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

    >>> g = Graph()
    >>> g.add_vertex("A", (0, 0)); g.add_vertex("B", (3, 4))
    >>> g.add_edge("A", "B")          # 权重缺省为两点欧氏距离 = 5
    >>> astar_graph(g, "A", "B").cost
    5.0
    """

    def __init__(self, directed: bool = False, name: str = "graph"):
        self.directed = directed
        self.name = name
        self.pos: dict[Vertex, tuple[float, float]] = {}
        self.adj: dict[Vertex, dict[Vertex, float]] = {}
        self.start: Vertex | None = None
        self.goal: Vertex | None = None

    # ------------------------------------------------------------------ 构造
    def add_vertex(self, v: Vertex, pos: tuple[float, float]) -> None:
        self.pos[v] = (float(pos[0]), float(pos[1]))
        self.adj.setdefault(v, {})

    def add_edge(self, u: Vertex, v: Vertex, weight: float | None = None) -> None:
        """添加边；``weight`` 缺省时取两端点的欧氏距离。"""
        if u not in self.pos or v not in self.pos:
            raise KeyError("请先 add_vertex 再 add_edge")
        if weight is None:
            weight = self.distance(u, v)
        if weight < 0:
            raise ValueError("A*/Dijkstra 要求边权非负")
        self.adj[u][v] = float(weight)
        if not self.directed:
            self.adj[v][u] = float(weight)

    @classmethod
    def from_edges(cls, positions: dict[Vertex, tuple[float, float]],
                   edges: Iterable[tuple[Vertex, Vertex] | tuple[Vertex, Vertex, float]],
                   directed: bool = False, name: str = "graph") -> "Graph":
        g = cls(directed=directed, name=name)
        for v, p in positions.items():
            g.add_vertex(v, p)
        for e in edges:
            g.add_edge(*e)
        return g

    # ------------------------------------------------------------------ 查询
    def distance(self, u: Vertex, v: Vertex) -> float:
        (x1, y1), (x2, y2) = self.pos[u], self.pos[v]
        return math.hypot(x1 - x2, y1 - y2)

    def neighbors(self, v: Vertex) -> Iterator[tuple[Vertex, float]]:
        yield from self.adj[v].items()

    def edges(self) -> Iterator[tuple[Vertex, Vertex, float]]:
        seen = set()
        for u, nbs in self.adj.items():
            for v, w in nbs.items():
                key = (u, v) if self.directed else frozenset((u, v))
                if key not in seen:
                    seen.add(key)
                    yield u, v, w

    def __len__(self) -> int:
        return len(self.pos)

    @property
    def n_edges(self) -> int:
        return sum(1 for _ in self.edges())

    def path_cost(self, path: list[Vertex]) -> float:
        return sum(self.adj[a][b] for a, b in itertools.pairwise(path))


def astar_graph(graph: Graph, start: Vertex | None = None, goal: Vertex | None = None,
                heuristic: str | Heuristic = "euclidean", label: str | None = None) -> SearchResult:
    """在顶点/边图上运行 A*；启发函数作用于顶点坐标。``heuristic="zero"`` 即 Dijkstra。

    注意：欧氏距离只有在 *每条边权 ≥ 其端点欧氏距离* 时才可采纳。若边权与坐标
    比例不同，可用 ``"euclidean*k"`` 缩放（k = min(权重/距离)）。
    """
    start = start if start is not None else graph.start
    goal = goal if goal is not None else graph.goal
    if start not in graph.pos or goal not in graph.pos:
        raise KeyError("起点或终点不在图中")

    h_func = get_heuristic(heuristic)
    hname = heuristic_name(heuristic)
    goal_pos = graph.pos[goal]
    pos = graph.pos

    t0 = time.perf_counter()
    counter = itertools.count()
    g: dict[Vertex, float] = {start: 0.0}
    parent: dict[Vertex, Vertex] = {}
    h0 = h_func(pos[start], goal_pos)
    open_heap = [(h0, h0, next(counter), 0.0, start)]
    expanded: list[Vertex] = []
    discovered: list[Vertex] = [start]

    while open_heap:
        _, _, _, gc, cur = heapq.heappop(open_heap)
        if gc > g[cur]:  # 过期条目
            continue
        expanded.append(cur)
        if cur == goal:
            break
        for nb, w in graph.adj[cur].items():
            ng = gc + w
            if ng < g.get(nb, math.inf):
                if nb not in g:
                    discovered.append(nb)
                g[nb] = ng
                parent[nb] = cur
                hn = h_func(pos[nb], goal_pos)
                heapq.heappush(open_heap, (ng + hn, hn, next(counter), ng, nb))
    elapsed = (time.perf_counter() - t0) * 1000

    path = None
    if goal in g:
        path = [goal]
        while path[-1] != start:
            path.append(parent[path[-1]])
        path.reverse()
    algo = label or ("Dijkstra" if hname == "zero" else f"A* ({hname})")
    return SearchResult(path=path, cost=g.get(goal, math.inf), expanded=expanded,
                        discovered=discovered, algorithm=algo, heuristic=hname, elapsed_ms=elapsed)


def dijkstra_graph(graph: Graph, start: Vertex | None = None, goal: Vertex | None = None) -> SearchResult:
    return astar_graph(graph, start, goal, heuristic="zero", label="Dijkstra")
