"""基于 **二维矩阵** 的图与 A* 实现。

地图是一个 ``numpy`` 二维数组 ``cost[row, col]``：

- 值为正数：进入该格子的代价（普通地面为 1，沼泽/山地可设更高）；
- 值为 ``inf``（常量 :data:`WALL`）：障碍物，不可通行。

节点即坐标 ``(row, col)``，邻居由矩阵下标隐式给出，不需要显式存储边。
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

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = _DIRS4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
SQRT2 = math.sqrt(2)


class GridMap:
    """二维矩阵地图。

    Parameters
    ----------
    cost:
        二维数组，元素为进入该格的代价，``inf`` 表示墙。
    diagonal:
        是否允许 8 方向移动。对角步代价 = √2 × 目标格代价，且禁止“切角”
        （两个正交相邻格中任何一个是墙时不能斜穿）。
    """

    def __init__(self, cost: np.ndarray, diagonal: bool = False,
                 start: Cell | None = None, goal: Cell | None = None, name: str = "grid"):
        cost = np.asarray(cost, dtype=float)
        if cost.ndim != 2:
            raise ValueError("cost 必须是二维矩阵")
        if np.any(cost <= 0):
            raise ValueError("格子代价必须为正数（墙使用 inf）")
        self.cost = cost
        self.diagonal = diagonal
        self.start = start
        self.goal = goal
        self.name = name

    # ------------------------------------------------------------------ 构造
    @classmethod
    def from_ascii(cls, text: str, diagonal: bool = False, name: str = "ascii") -> "GridMap":
        """从字符画构造地图。

        ``.`` 或空格 = 代价 1，``#`` = 墙，``1``-``9`` = 对应代价，
        ``S`` = 起点，``G`` = 终点（二者代价均为 1）。
        """
        lines = [ln for ln in text.strip("\n").splitlines()]
        width = max(len(ln) for ln in lines)
        cost = np.ones((len(lines), width))
        start = goal = None
        for r, line in enumerate(lines):
            for c, ch in enumerate(line.ljust(width, ".")):
                if ch == "#":
                    cost[r, c] = WALL
                elif ch.isdigit() and ch != "0":
                    cost[r, c] = int(ch)
                elif ch == "S":
                    start = (r, c)
                elif ch == "G":
                    goal = (r, c)
        return cls(cost, diagonal=diagonal, start=start, goal=goal, name=name)

    # ------------------------------------------------------------------ 查询
    @property
    def shape(self) -> tuple[int, int]:
        return self.cost.shape  # type: ignore[return-value]

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.cost.shape[0] and 0 <= c < self.cost.shape[1]

    def passable(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and math.isfinite(self.cost[cell])

    def neighbors(self, cell: Cell) -> Iterator[tuple[Cell, float]]:
        """生成 (邻居, 边代价)。"""
        r, c = cell
        for dr, dc in (_DIRS8 if self.diagonal else _DIRS4):
            nb = (r + dr, c + dc)
            if not self.passable(nb):
                continue
            if dr and dc:  # 对角：禁止切角
                if not (self.passable((r + dr, c)) and self.passable((r, c + dc))):
                    continue
                yield nb, SQRT2 * self.cost[nb]
            else:
                yield nb, float(self.cost[nb])

    def path_cost(self, path: list[Cell]) -> float:
        total = 0.0
        for a, b in itertools.pairwise(path):
            step = dict(self.neighbors(a)).get(b)
            if step is None:
                raise ValueError(f"{a} -> {b} 不是合法移动")
            total += step
        return total

    def default_heuristic(self) -> str:
        return "octile" if self.diagonal else "manhattan"


def astar_grid(grid: GridMap, start: Cell | None = None, goal: Cell | None = None,
               heuristic: str | Heuristic | None = None, label: str | None = None) -> SearchResult[Cell]:
    """在二维矩阵上运行 A*。

    ``heuristic`` 为 ``"zero"`` 时即 Dijkstra。为保证启发函数可采纳，内置启发函数
    假定最小格子代价为 1；若地图最小代价小于 1，会自动乘以该最小值进行缩放。

    同 f 值时优先 h 更小（更靠近目标）的节点，可显著减少开阔地带的扩展数。
    若启发函数不一致（例如自定义或加权），算法允许节点被重新打开，结果仍正确。
    """
    start = start if start is not None else grid.start
    goal = goal if goal is not None else grid.goal
    if start is None or goal is None:
        raise ValueError("需要指定起点和终点")
    for p in (start, goal):
        if not grid.passable(p):
            raise ValueError(f"{p} 不可通行或越界")

    h_func = get_heuristic(heuristic if heuristic is not None else grid.default_heuristic())
    hname = heuristic_name(heuristic if heuristic is not None else grid.default_heuristic())
    finite = grid.cost[np.isfinite(grid.cost)]
    scale = min(1.0, float(finite.min()))
    h = (lambda a: scale * h_func(a, goal)) if scale < 1 else (lambda a: h_func(a, goal))

    t0 = time.perf_counter()
    counter = itertools.count()
    g: dict[Cell, float] = {start: 0.0}
    parent: dict[Cell, Cell] = {}
    h0 = h(start)
    open_heap: list[tuple[float, float, int, float, Cell]] = [(h0, h0, next(counter), 0.0, start)]
    expanded: list[Cell] = []
    discovered: list[Cell] = [start]

    while open_heap:
        _, _, _, gc, cur = heapq.heappop(open_heap)
        if gc > g[cur]:  # 过期条目（惰性删除）
            continue
        expanded.append(cur)
        if cur == goal:
            break
        for nb, step in grid.neighbors(cur):
            ng = gc + step
            if ng < g.get(nb, math.inf):
                if nb not in g:
                    discovered.append(nb)
                g[nb] = ng
                parent[nb] = cur
                hn = h(nb)
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


def dijkstra_grid(grid: GridMap, start: Cell | None = None, goal: Cell | None = None) -> SearchResult[Cell]:
    """Dijkstra = 启发函数恒为 0 的 A*。"""
    return astar_grid(grid, start, goal, heuristic="zero", label="Dijkstra")
