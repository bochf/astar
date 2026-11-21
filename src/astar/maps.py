"""演示用地图生成器：简单 / 复杂的矩阵地图与顶点/边图。所有随机地图都可通过 seed 复现。"""

from __future__ import annotations

import math

import numpy as np

from .grid import WALL, GridMap
from .graph import Graph

# ============================================================ 矩阵地图

SIMPLE_ASCII = """
.........................
.........................
.........................
.........................
.........#######.........
...............#.........
...............#.........
...S...........#.....G...
...............#.........
...............#.........
.........#######.........
.........................
.........................
.........................
.........................
"""


def simple_grid(diagonal: bool = False) -> GridMap:
    """15×25 的简单地图：起点与终点之间隔着一道“匚”形墙。"""
    return GridMap.from_ascii(SIMPLE_ASCII, diagonal=diagonal, name="simple")


def open_grid(h: int = 30, w: int = 50, diagonal: bool = False) -> GridMap:
    """无障碍空地：最能体现启发函数对扩展范围的约束效果。"""
    return GridMap(np.ones((h, w)), diagonal=diagonal, start=(h // 2, 3), goal=(h // 2, w - 4), name="open")


def maze_grid(h: int = 41, w: int = 61, loops: float = 0.08, seed: int = 0, diagonal: bool = False) -> GridMap:
    """随机深度优先生成的迷宫（尺寸取奇数），再随机打通 ``loops`` 比例的墙以产生多条路径。"""
    rng = np.random.default_rng(seed)
    h, w = h | 1, w | 1
    cost = np.full((h, w), WALL)
    stack = [(1, 1)]
    cost[1, 1] = 1
    while stack:
        r, c = stack[-1]
        nbs = [(r + dr, c + dc, dr, dc) for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2))
               if 0 < r + dr < h - 1 and 0 < c + dc < w - 1 and cost[r + dr, c + dc] == WALL]
        if not nbs:
            stack.pop()
            continue
        nr, nc, dr, dc = nbs[rng.integers(len(nbs))]
        cost[r + dr // 2, c + dc // 2] = 1
        cost[nr, nc] = 1
        stack.append((nr, nc))
    # 打通部分内墙，制造环路
    inner = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)
             if cost[r, c] == WALL and ((r % 2) != (c % 2))]
    for i in rng.choice(len(inner), int(len(inner) * loops), replace=False):
        cost[inner[i]] = 1
    return GridMap(cost, diagonal=diagonal, start=(1, 1), goal=(h - 2, w - 2), name="maze")


def random_obstacle_grid(h: int = 60, w: int = 90, density: float = 0.3, seed: int = 0,
                         diagonal: bool = True) -> GridMap:
    """随机障碍地图（障碍为大小不一的矩形块），保证起终点连通。"""
    from .grid import astar_grid  # 避免循环导入

    rng = np.random.default_rng(seed)
    for _ in range(100):
        cost = np.ones((h, w))
        while np.isinf(cost).mean() < density:
            bh, bw = rng.integers(1, 8, size=2)
            r, c = rng.integers(0, h - bh), rng.integers(0, w - bw)
            cost[r:r + bh, c:c + bw] = WALL
        start, goal = (h - 3, 2), (2, w - 3)
        cost[start[0] - 2:start[0] + 3, :5] = 1
        cost[:5, goal[1] - 2:goal[1] + 3] = 1
        grid = GridMap(cost, diagonal=diagonal, start=start, goal=goal, name="random")
        if astar_grid(grid).found:
            return grid
        rng = np.random.default_rng(rng.integers(1 << 31))
    raise RuntimeError("无法生成连通的随机地图，请降低 density")


def _smooth_noise(h: int, w: int, rng: np.random.Generator, scale: int = 8) -> np.ndarray:
    """简单的值噪声：低分辨率随机网格 + 双线性插值，结果归一化到 [0, 1]。"""
    gh, gw = h // scale + 2, w // scale + 2
    coarse = rng.random((gh, gw))
    ys = np.linspace(0, gh - 1.001, h)
    xs = np.linspace(0, gw - 1.001, w)
    y0, x0 = ys.astype(int), xs.astype(int)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    a = coarse[np.ix_(y0, x0)]
    b = coarse[np.ix_(y0, x0 + 1)]
    c = coarse[np.ix_(y0 + 1, x0)]
    d = coarse[np.ix_(y0 + 1, x0 + 1)]
    n = (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy
    return (n - n.min()) / (np.ptp(n) or 1)


def weighted_terrain_grid(h: int = 60, w: int = 90, seed: int = 0, diagonal: bool = True) -> GridMap:
    """加权地形：代价 1（平地）~ 9（山地），并在噪声最高处放置不可通行的山峰，
    另有一条只留两个渡口的“河流”。最优路径会绕开高代价区域，而不是走直线。"""
    rng = np.random.default_rng(seed)
    n = _smooth_noise(h, w, rng, scale=10) * 0.7 + _smooth_noise(h, w, rng, scale=4) * 0.3
    cost = 1 + np.floor(n ** 1.5 * 9).clip(0, 8)
    cost[n > 0.9] = WALL
    # 河流：沿一条正弦曲线的竖向墙，留两个渡口
    for r in range(h):
        c = int(w * 0.55 + 6 * math.sin(r / 6))
        cost[r, c:c + 2] = WALL
    for bridge in (h // 5, h * 4 // 5):
        c = int(w * 0.55 + 6 * math.sin(bridge / 6))
        cost[bridge - 1:bridge + 2, c - 1:c + 3] = 1
    start, goal = (h // 2, 2), (h // 2, w - 3)
    cost[start], cost[goal] = 1, 1
    return GridMap(cost, diagonal=diagonal, start=start, goal=goal, name="terrain")


GRID_MAPS = {
    "simple": simple_grid,
    "open": open_grid,
    "maze": maze_grid,
    "random": random_obstacle_grid,
    "terrain": weighted_terrain_grid,
}

# ============================================================ 顶点 / 边图


def simple_graph() -> Graph:
    """10 个顶点的小型“城市路网”，便于手工核对结果。边权 = 欧氏距离（部分道路拥堵 ×1.5）。"""
    pos = {
        "A": (0, 0), "B": (2, 1), "C": (1, 3), "D": (4, 0), "E": (4, 3),
        "F": (6, 1), "G": (3, 5), "H": (6, 4), "I": (8, 2), "J": (8, 5),
    }
    g = Graph(name="simple_graph")
    for v, p in pos.items():
        g.add_vertex(v, p)
    for u, v in ["AB", "AC", "BD", "BE", "CG", "CE", "DF", "EF", "EH", "GH", "FI", "HJ", "IJ", "HI"]:
        g.add_edge(u, v)
    for u, v in ["BE", "FI"]:  # 拥堵路段
        g.add_edge(u, v, g.distance(u, v) * 1.5)
    g.start, g.goal = "A", "J"
    return g


def random_road_graph(n: int = 800, k: int = 4, seed: int = 0, detour: float = 0.3,
                      holes: int = 4) -> Graph:
    """随机“路网”：在 100×60 的平面撒 n 个点（挖掉若干圆形“湖泊”），每个点连向最近的
    k 个邻居；边权 = 欧氏距离 × U(1, 1+detour)，因此欧氏启发函数可采纳。保证连通。"""
    rng = np.random.default_rng(seed)
    W, H = 100.0, 60.0
    lakes = [(rng.uniform(25, 75), rng.uniform(10, 50), rng.uniform(6, 12)) for _ in range(holes)]
    pts: list[tuple[float, float]] = []
    while len(pts) < n:
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        if all(math.hypot(x - cx, y - cy) > r for cx, cy, r in lakes):
            pts.append((x, y))
    P = np.array(pts)
    g = Graph(name="road_graph")
    for i, p in enumerate(pts):
        g.add_vertex(i, p)
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nearest = np.argsort(d2, axis=1)[:, :k]
    for i in range(n):
        for j in nearest[i]:
            j = int(j)
            if j not in g.adj[i]:
                g.add_edge(i, j, g.distance(i, j) * rng.uniform(1, 1 + detour))

    # 连通性修复：把每个非主连通分量连到主分量最近的点
    def components() -> list[set[int]]:
        seen, comps = set(), []
        for s in range(n):
            if s in seen:
                continue
            comp, stack = set(), [s]
            while stack:
                u = stack.pop()
                if u in comp:
                    continue
                comp.add(u)
                stack.extend(v for v in g.adj[u] if v not in comp)
            seen |= comp
            comps.append(comp)
        return sorted(comps, key=len, reverse=True)

    comps = components()
    main = comps[0]
    for comp in comps[1:]:
        a_idx = np.array(sorted(comp)); b_idx = np.array(sorted(main))
        sub = d2[np.ix_(a_idx, b_idx)]
        ai, bi = np.unravel_index(np.argmin(sub), sub.shape)
        g.add_edge(int(a_idx[ai]), int(b_idx[bi]))
        main |= comp

    g.start = int(np.argmin(P[:, 0] + P[:, 1]))               # 左下
    g.goal = int(np.argmax(P[:, 0] + P[:, 1]))                # 右上
    return g


GRAPH_MAPS = {
    "simple": simple_graph,
    "road": random_road_graph,
}
