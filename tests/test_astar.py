"""正确性验证：A* 结果与独立实现的参考最短路算法（Bellman-Ford / Floyd-Warshall）对比。"""

import math
import random

import numpy as np
import pytest

from astar import (WALL, Graph, GridMap, astar_graph, astar_grid, dijkstra_graph, dijkstra_grid,
                   get_heuristic, register_heuristic, weighted)
from astar import maps


def reference_grid_cost(grid: GridMap, start, goal) -> float:
    """独立参考：在矩阵上做 Bellman-Ford 松弛直至收敛。"""
    shortest = {start: 0.0}
    changed = True
    while changed:
        changed = False
        for cell, cell_cost in list(shortest.items()):
            for neighbor, step_cost in grid.neighbors(cell):
                if cell_cost + step_cost < shortest.get(neighbor, math.inf) - 1e-12:
                    shortest[neighbor] = cell_cost + step_cost
                    changed = True
    return shortest.get(goal, math.inf)


def floyd_warshall(graph: Graph):
    vertices = list(graph.positions)
    index_of = {vertex: index for index, vertex in enumerate(vertices)}
    vertex_count = len(vertices)
    distances = np.full((vertex_count, vertex_count), np.inf)
    np.fill_diagonal(distances, 0)
    for source, neighbors in graph.adjacency.items():
        for target, weight in neighbors.items():
            current = distances[index_of[source], index_of[target]]
            distances[index_of[source], index_of[target]] = min(current, weight)
    for middle in range(vertex_count):
        distances = np.minimum(distances, distances[:, [middle]] + distances[[middle], :])
    return distances, index_of


def random_grid(seed, height=14, width=18, density=0.25, weighted_terrain=False, diagonal=False):
    rng = np.random.default_rng(seed)
    if weighted_terrain:
        cell_costs = rng.integers(1, 6, (height, width)).astype(float)
    else:
        cell_costs = np.ones((height, width))
    cell_costs[rng.random((height, width)) < density] = WALL
    cell_costs[0, 0] = cell_costs[-1, -1] = 1
    return GridMap(cell_costs, diagonal=diagonal, start=(0, 0), goal=(height - 1, width - 1))


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("weighted_terrain", [False, True])
def test_grid_matches_reference(seed, diagonal, weighted_terrain):
    grid = random_grid(seed, weighted_terrain=weighted_terrain, diagonal=diagonal)
    expected_cost = reference_grid_cost(grid, grid.start, grid.goal)
    if diagonal:
        heuristic_specs = ["zero", "octile", "euclidean", "chebyshev"]
    else:
        heuristic_specs = ["zero", "manhattan", "euclidean", "chebyshev", "octile"]
    for spec in heuristic_specs:
        result = astar_grid(grid, heuristic=spec)
        assert result.cost == pytest.approx(expected_cost), spec
        if result.found:
            assert result.path[0] == grid.start and result.path[-1] == grid.goal
            assert grid.path_cost(result.path) == pytest.approx(result.cost)
        else:
            assert math.isinf(expected_cost)


@pytest.mark.parametrize("seed", range(20))
def test_graph_matches_floyd_warshall(seed):
    rng = random.Random(seed)
    graph = Graph(directed=seed % 2 == 1)
    for vertex in range(25):
        graph.add_vertex(vertex, (rng.uniform(0, 10), rng.uniform(0, 10)))
    for _ in range(50):
        source, target = rng.sample(range(25), 2)
        graph.add_edge(source, target, graph.distance(source, target) * rng.uniform(1, 2))  # 保证欧氏距离可采纳
    distances, index_of = floyd_warshall(graph)
    for start, goal in [(0, 24), (3, 17), (5, 5)]:
        for spec in ["zero", "euclidean", "chebyshev"]:
            result = astar_graph(graph, start, goal, heuristic=spec)
            assert result.cost == pytest.approx(distances[index_of[start], index_of[goal]]), (start, goal, spec)
            if result.found:
                assert graph.path_cost(result.path) == pytest.approx(result.cost)


def test_astar_expands_no_more_than_dijkstra_on_demo_maps():
    for map_name, make_grid in maps.GRID_MAPS.items():
        grid = make_grid()
        astar_result, dijkstra_result = astar_grid(grid), dijkstra_grid(grid)
        assert astar_result.cost == pytest.approx(dijkstra_result.cost), map_name
        assert astar_result.expanded_count <= dijkstra_result.expanded_count, map_name
    for map_name, make_graph in maps.GRAPH_MAPS.items():
        graph = make_graph()
        astar_result, dijkstra_result = astar_graph(graph), dijkstra_graph(graph)
        assert astar_result.cost == pytest.approx(dijkstra_result.cost), map_name
        assert astar_result.expanded_count <= dijkstra_result.expanded_count, map_name


def test_unreachable():
    grid = GridMap.from_ascii("S#.\n.#.\n.#G")
    result = astar_grid(grid)
    assert result.path is None and math.isinf(result.cost)
    graph = Graph()
    graph.add_vertex("a", (0, 0)); graph.add_vertex("b", (1, 1))
    assert not astar_graph(graph, "a", "b").found


def test_ascii_parsing_and_weights():
    grid = GridMap.from_ascii("S9G\n...")
    assert grid.start == (0, 0) and grid.goal == (0, 2)
    result = astar_grid(grid)
    assert result.cost == 4  # 绕过代价 9 的格子：下-右-右-上
    assert result.path == [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)]


def test_no_corner_cutting():
    grid = GridMap.from_ascii("S#\n#G", diagonal=True)
    assert not astar_grid(grid).found


def test_custom_heuristics():
    grid = maps.simple_grid()
    baseline = astar_grid(grid, heuristic="manhattan")

    # 1) 直接传入可调用对象
    lambda_result = astar_grid(grid, heuristic=lambda current, goal: abs(current[0] - goal[0]) + abs(current[1] - goal[1]))
    assert lambda_result.cost == baseline.cost and lambda_result.expanded_count == baseline.expanded_count

    # 2) 注册名称
    @register_heuristic("test_half", "half manhattan")
    def half_manhattan(current, goal):
        return (abs(current[0] - goal[0]) + abs(current[1] - goal[1])) / 2

    registered_result = astar_grid(grid, heuristic="test_half")
    assert registered_result.cost == baseline.cost and registered_result.heuristic == "test_half"

    # 3) module:function
    assert get_heuristic("astar.heuristics:manhattan")((0, 0), (3, 4)) == 7

    # 4) 加权
    assert get_heuristic("manhattan*2")((0, 0), (3, 4)) == 14
    assert weighted(get_heuristic("manhattan"), 3)((0, 0), (1, 1)) == 6


def test_inconsistent_heuristic_still_valid():
    """不一致（甚至随机）的启发函数：仍应得到合法路径（允许节点重新打开）。"""
    rng = random.Random(0)
    grid = maps.maze_grid(seed=1)
    result = astar_grid(grid, heuristic=lambda current, goal: rng.uniform(0, 50))
    assert result.found and grid.path_cost(result.path) == pytest.approx(result.cost)


def test_cheap_cells_scale_heuristic():
    """格子代价 < 1 时，内置启发函数自动缩放以保持可采纳。"""
    grid = GridMap(np.full((10, 10), 0.25), start=(0, 0), goal=(9, 9))
    assert astar_grid(grid).cost == pytest.approx(dijkstra_grid(grid).cost) == pytest.approx(18 * 0.25)
