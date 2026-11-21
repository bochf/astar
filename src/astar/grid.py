"""基于 **二维矩阵** 的图与 A* 实现。

地图是一个 ``numpy`` 二维数组 ``cell_costs[row, col]``：

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

_ORTHOGONAL_STEPS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIAGONAL_STEPS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_ALL_STEPS = _ORTHOGONAL_STEPS + _DIAGONAL_STEPS
SQRT2 = math.sqrt(2)


class GridMap:
    """二维矩阵地图。

    Parameters
    ----------
    cell_costs:
        二维数组，元素为进入该格的代价，``inf`` 表示墙。
    diagonal:
        是否允许 8 方向移动。对角步代价 = √2 × 目标格代价，且禁止“切角”
        （两个正交相邻格中任何一个是墙时不能斜穿）。
    """

    def __init__(self, cell_costs: np.ndarray, diagonal: bool = False,
                 start: Cell | None = None, goal: Cell | None = None, name: str = "grid"):
        cell_costs = np.asarray(cell_costs, dtype=float)
        if cell_costs.ndim != 2:
            raise ValueError("cell_costs 必须是二维矩阵")
        if np.any(cell_costs <= 0):
            raise ValueError("格子代价必须为正数（墙使用 inf）")
        self.cell_costs = cell_costs
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

    # ------------------------------------------------------------------ 查询
    @property
    def shape(self) -> tuple[int, int]:
        return self.cell_costs.shape  # type: ignore[return-value]

    def in_bounds(self, cell: Cell) -> bool:
        row, col = cell
        return 0 <= row < self.cell_costs.shape[0] and 0 <= col < self.cell_costs.shape[1]

    def passable(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and math.isfinite(self.cell_costs[cell])

    def neighbors(self, cell: Cell) -> Iterator[tuple[Cell, float]]:
        """生成 (邻居格子, 移动代价)。"""
        row, col = cell
        for row_step, col_step in (_ALL_STEPS if self.diagonal else _ORTHOGONAL_STEPS):
            neighbor = (row + row_step, col + col_step)
            if not self.passable(neighbor):
                continue
            is_diagonal = row_step != 0 and col_step != 0
            if is_diagonal:
                # 禁止切角：斜向移动经过的两个正交格子都必须可通行
                if not (self.passable((row + row_step, col)) and self.passable((row, col + col_step))):
                    continue
                yield neighbor, SQRT2 * float(self.cell_costs[neighbor])
            else:
                yield neighbor, float(self.cell_costs[neighbor])

    def path_cost(self, path: list[Cell]) -> float:
        total_cost = 0.0
        for current, following in itertools.pairwise(path):
            step_cost = dict(self.neighbors(current)).get(following)
            if step_cost is None:
                raise ValueError(f"{current} -> {following} 不是合法移动")
            total_cost += step_cost
        return total_cost

    def default_heuristic(self) -> str:
        return "octile" if self.diagonal else "manhattan"


def astar_grid(grid: GridMap, start: Cell | None = None, goal: Cell | None = None,
               heuristic: str | Heuristic | None = None, label: str | None = None) -> SearchResult[Cell]:
    """在二维矩阵上运行 A*。

    ``heuristic`` 为 ``"zero"`` 时即 Dijkstra。为保证启发函数可采纳，内置启发函数
    假定最小格子代价为 1；若地图最小代价小于 1，会自动乘以该最小值进行缩放。

    同 f 值时优先 h 更小（更靠近目标）的节点。
    若启发函数不一致（例如自定义或加权），算法允许节点被重新打开，结果仍正确。
    """
    start = start if start is not None else grid.start
    goal = goal if goal is not None else grid.goal
    if start is None or goal is None:
        raise ValueError("需要指定起点和终点")
    for endpoint in (start, goal):
        if not grid.passable(endpoint):
            raise ValueError(f"{endpoint} 不可通行或越界")

    heuristic_spec = heuristic if heuristic is not None else grid.default_heuristic()
    heuristic_fn = get_heuristic(heuristic_spec)
    heuristic_label = heuristic_name(heuristic_spec)

    # 若存在代价 < 1 的格子，按最小代价缩放启发值，保证不高估
    passable_costs = grid.cell_costs[np.isfinite(grid.cell_costs)]
    heuristic_scale = min(1.0, float(passable_costs.min()))

    def estimate_to_goal(cell: Cell) -> float:
        return heuristic_scale * heuristic_fn(cell, goal)

    start_time = time.perf_counter()
    insertion_order = itertools.count()          # 堆中平局时先进先出，且避免比较节点本身
    cost_from_start: dict[Cell, float] = {start: 0.0}   # g 值：目前已知的起点到该格的最小代价
    came_from: dict[Cell, Cell] = {}             # 父指针：用于回溯路径
    start_estimate = estimate_to_goal(start)
    # 堆条目：(f = g + h, h, 插入序号, g, 格子)
    open_heap: list[tuple[float, float, int, float, Cell]] = [
        (start_estimate, start_estimate, next(insertion_order), 0.0, start)
    ]
    expanded: list[Cell] = []
    discovered: list[Cell] = [start]

    while open_heap:
        _, _, _, current_cost, current = heapq.heappop(open_heap)
        if current_cost > cost_from_start[current]:  # 过期条目（惰性删除）
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


def dijkstra_grid(grid: GridMap, start: Cell | None = None, goal: Cell | None = None) -> SearchResult[Cell]:
    """Dijkstra = 启发函数恒为 0 的 A*。"""
    return astar_grid(grid, start, goal, heuristic="zero", label="Dijkstra")
