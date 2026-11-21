"""正确性验证：A* 结果与独立实现的参考最短路算法（Bellman-Ford / Floyd-Warshall）对比。"""

import math
import random

import numpy as np
import pytest

from astar import (WALL, Graph, GridMap, astar_graph, astar_grid, dijkstra_graph, dijkstra_grid,
                   get_heuristic, register_heuristic, weighted)
from astar import maps


def reference_grid(grid: GridMap, start, goal) -> float:
    """独立参考：在矩阵上做 Bellman-Ford 松弛直至收敛。"""
    dist = {start: 0.0}
    changed = True
    while changed:
        changed = False
        for cell, d in list(dist.items()):
            for nb, w in grid.neighbors(cell):
                if d + w < dist.get(nb, math.inf) - 1e-12:
                    dist[nb] = d + w
                    changed = True
    return dist.get(goal, math.inf)


def floyd(graph: Graph):
    vs = list(graph.pos)
    idx = {v: i for i, v in enumerate(vs)}
    n = len(vs)
    d = np.full((n, n), np.inf)
    np.fill_diagonal(d, 0)
    for u, nbs in graph.adj.items():
        for v, w in nbs.items():
            d[idx[u], idx[v]] = min(d[idx[u], idx[v]], w)
    for k in range(n):
        d = np.minimum(d, d[:, [k]] + d[[k], :])
    return d, idx


def random_grid(seed, h=14, w=18, density=0.25, weighted_terrain=False, diagonal=False):
    rng = np.random.default_rng(seed)
    cost = rng.integers(1, 6, (h, w)).astype(float) if weighted_terrain else np.ones((h, w))
    cost[rng.random((h, w)) < density] = WALL
    cost[0, 0] = cost[-1, -1] = 1
    return GridMap(cost, diagonal=diagonal, start=(0, 0), goal=(h - 1, w - 1))


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("terrain", [False, True])
def test_grid_matches_reference(seed, diagonal, terrain):
    grid = random_grid(seed, weighted_terrain=terrain, diagonal=diagonal)
    ref = reference_grid(grid, grid.start, grid.goal)
    hs = ["zero", "octile", "euclidean", "chebyshev"] if diagonal else ["zero", "manhattan", "euclidean", "chebyshev", "octile"]
    for h in hs:
        r = astar_grid(grid, heuristic=h)
        assert r.cost == pytest.approx(ref), h
        if r.found:
            assert r.path[0] == grid.start and r.path[-1] == grid.goal
            assert grid.path_cost(r.path) == pytest.approx(r.cost)
        else:
            assert math.isinf(ref)


@pytest.mark.parametrize("seed", range(20))
def test_graph_matches_floyd(seed):
    rng = random.Random(seed)
    g = Graph(directed=seed % 2 == 1)
    for i in range(25):
        g.add_vertex(i, (rng.uniform(0, 10), rng.uniform(0, 10)))
    for _ in range(50):
        u, v = rng.sample(range(25), 2)
        g.add_edge(u, v, g.distance(u, v) * rng.uniform(1, 2))  # 保证欧氏距离可采纳
    d, idx = floyd(g)
    for s, t in [(0, 24), (3, 17), (5, 5)]:
        for h in ["zero", "euclidean", "chebyshev"]:
            r = astar_graph(g, s, t, heuristic=h)
            assert r.cost == pytest.approx(d[idx[s], idx[t]]), (s, t, h)
            if r.found:
                assert g.path_cost(r.path) == pytest.approx(r.cost)


def test_astar_expands_no_more_than_dijkstra_on_demo_maps():
    for name, f in maps.GRID_MAPS.items():
        grid = f()
        a, d = astar_grid(grid), dijkstra_grid(grid)
        assert a.cost == pytest.approx(d.cost), name
        assert a.n_expanded <= d.n_expanded, name
    for name, f in maps.GRAPH_MAPS.items():
        graph = f()
        a, d = astar_graph(graph), dijkstra_graph(graph)
        assert a.cost == pytest.approx(d.cost), name
        assert a.n_expanded <= d.n_expanded, name


def test_unreachable():
    grid = GridMap.from_ascii("S#.\n.#.\n.#G")
    r = astar_grid(grid)
    assert r.path is None and math.isinf(r.cost)
    g = Graph()
    g.add_vertex("a", (0, 0)); g.add_vertex("b", (1, 1))
    assert not astar_graph(g, "a", "b").found


def test_ascii_parsing_and_weights():
    grid = GridMap.from_ascii("S9G\n...")
    assert grid.start == (0, 0) and grid.goal == (0, 2)
    r = astar_grid(grid)
    assert r.cost == 4  # 绕过代价 9 的格子：下-右-右-上
    assert r.path == [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)]


def test_no_corner_cutting():
    grid = GridMap.from_ascii("S#\n#G", diagonal=True)
    assert not astar_grid(grid).found


def test_custom_heuristics():
    grid = maps.simple_grid()
    base = astar_grid(grid, heuristic="manhattan")

    # 1) 直接传入可调用对象
    r1 = astar_grid(grid, heuristic=lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]))
    assert r1.cost == base.cost and r1.n_expanded == base.n_expanded

    # 2) 注册名称
    @register_heuristic("test_half", "half manhattan")
    def half(a, b):
        return (abs(a[0] - b[0]) + abs(a[1] - b[1])) / 2

    r2 = astar_grid(grid, heuristic="test_half")
    assert r2.cost == base.cost and r2.heuristic == "test_half"

    # 3) module:function
    assert get_heuristic("astar.heuristics:manhattan")((0, 0), (3, 4)) == 7

    # 4) 加权
    assert get_heuristic("manhattan*2")((0, 0), (3, 4)) == 14
    assert weighted(get_heuristic("manhattan"), 3)((0, 0), (1, 1)) == 6


def test_inconsistent_heuristic_still_valid():
    """不一致（甚至随机）的启发函数：仍应得到合法路径（允许节点重新打开）。"""
    rng = random.Random(0)
    grid = maps.maze_grid(seed=1)
    r = astar_grid(grid, heuristic=lambda a, b: rng.uniform(0, 50))
    assert r.found and grid.path_cost(r.path) == pytest.approx(r.cost)


def test_cheap_cells_scale_heuristic():
    """格子代价 < 1 时，内置启发函数自动缩放以保持可采纳。"""
    cost = np.full((10, 10), 0.25)
    grid = GridMap(cost, start=(0, 0), goal=(9, 9))
    assert astar_grid(grid).cost == pytest.approx(dijkstra_grid(grid).cost) == pytest.approx(18 * 0.25)
