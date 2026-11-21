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


def open_grid(height: int = 30, width: int = 50, diagonal: bool = False) -> GridMap:
    """无障碍空地：最能体现启发函数对扩展范围的约束效果。"""
    return GridMap(np.ones((height, width)), diagonal=diagonal,
                   start=(height // 2, 3), goal=(height // 2, width - 4), name="open")


def maze_grid(height: int = 41, width: int = 61, loop_ratio: float = 0.08, seed: int = 0,
              diagonal: bool = False) -> GridMap:
    """随机深度优先生成的迷宫（尺寸取奇数），再随机打通 ``loop_ratio`` 比例的内墙以产生多条路径。"""
    rng = np.random.default_rng(seed)
    height, width = height | 1, width | 1
    cell_costs = np.full((height, width), WALL)
    carve_stack = [(1, 1)]
    cell_costs[1, 1] = 1
    while carve_stack:
        row, col = carve_stack[-1]
        unvisited = [(row + row_step, col + col_step, row_step, col_step)
                     for row_step, col_step in ((-2, 0), (2, 0), (0, -2), (0, 2))
                     if 0 < row + row_step < height - 1 and 0 < col + col_step < width - 1
                     and cell_costs[row + row_step, col + col_step] == WALL]
        if not unvisited:
            carve_stack.pop()
            continue
        next_row, next_col, row_step, col_step = unvisited[rng.integers(len(unvisited))]
        cell_costs[row + row_step // 2, col + col_step // 2] = 1   # 打通两格之间的墙
        cell_costs[next_row, next_col] = 1
        carve_stack.append((next_row, next_col))
    # 打通部分内墙，制造环路
    inner_walls = [(row, col) for row in range(1, height - 1) for col in range(1, width - 1)
                   if cell_costs[row, col] == WALL and (row % 2) != (col % 2)]
    for wall_index in rng.choice(len(inner_walls), int(len(inner_walls) * loop_ratio), replace=False):
        cell_costs[inner_walls[wall_index]] = 1
    return GridMap(cell_costs, diagonal=diagonal, start=(1, 1), goal=(height - 2, width - 2), name="maze")


def random_obstacle_grid(height: int = 60, width: int = 90, density: float = 0.3, seed: int = 0,
                         diagonal: bool = True) -> GridMap:
    """随机障碍地图（障碍为大小不一的矩形块），保证起终点连通。"""
    from .grid import astar_grid  # 避免循环导入

    rng = np.random.default_rng(seed)
    for _attempt in range(100):
        cell_costs = np.ones((height, width))
        while np.isinf(cell_costs).mean() < density:
            block_height, block_width = rng.integers(1, 8, size=2)
            top, left = rng.integers(0, height - block_height), rng.integers(0, width - block_width)
            cell_costs[top:top + block_height, left:left + block_width] = WALL
        start, goal = (height - 3, 2), (2, width - 3)
        cell_costs[start[0] - 2:start[0] + 3, :5] = 1      # 清空起点周围
        cell_costs[:5, goal[1] - 2:goal[1] + 3] = 1        # 清空终点周围
        grid = GridMap(cell_costs, diagonal=diagonal, start=start, goal=goal, name="random")
        if astar_grid(grid).found:
            return grid
        rng = np.random.default_rng(rng.integers(1 << 31))
    raise RuntimeError("无法生成连通的随机地图，请降低 density")


def _smooth_noise(height: int, width: int, rng: np.random.Generator, scale: int = 8) -> np.ndarray:
    """简单的值噪声：低分辨率随机网格 + 双线性插值，结果归一化到 [0, 1]。"""
    coarse_height, coarse_width = height // scale + 2, width // scale + 2
    coarse = rng.random((coarse_height, coarse_width))
    sample_rows = np.linspace(0, coarse_height - 1.001, height)
    sample_cols = np.linspace(0, coarse_width - 1.001, width)
    row_index, col_index = sample_rows.astype(int), sample_cols.astype(int)
    row_fraction = (sample_rows - row_index)[:, None]
    col_fraction = (sample_cols - col_index)[None, :]
    top_left = coarse[np.ix_(row_index, col_index)]
    top_right = coarse[np.ix_(row_index, col_index + 1)]
    bottom_left = coarse[np.ix_(row_index + 1, col_index)]
    bottom_right = coarse[np.ix_(row_index + 1, col_index + 1)]
    top = top_left * (1 - col_fraction) + top_right * col_fraction
    bottom = bottom_left * (1 - col_fraction) + bottom_right * col_fraction
    noise = top * (1 - row_fraction) + bottom * row_fraction
    return (noise - noise.min()) / (np.ptp(noise) or 1)


def weighted_terrain_grid(height: int = 60, width: int = 90, seed: int = 0, diagonal: bool = True) -> GridMap:
    """加权地形：代价 1（平地）~ 9（山地），并在噪声最高处放置不可通行的山峰，
    另有一条只留两个渡口的“河流”。最优路径会绕开高代价区域，而不是走直线。"""
    rng = np.random.default_rng(seed)
    elevation = _smooth_noise(height, width, rng, scale=10) * 0.7 + _smooth_noise(height, width, rng, scale=4) * 0.3
    cell_costs = 1 + np.floor(elevation ** 1.5 * 9).clip(0, 8)
    cell_costs[elevation > 0.9] = WALL      # 山峰
    # 河流：沿一条正弦曲线的竖向墙，留两个渡口

    def river_col(row: int) -> int:
        return int(width * 0.55 + 6 * math.sin(row / 6))

    for row in range(height):
        cell_costs[row, river_col(row):river_col(row) + 2] = WALL
    for bridge_row in (height // 5, height * 4 // 5):
        bridge_col = river_col(bridge_row)
        cell_costs[bridge_row - 1:bridge_row + 2, bridge_col - 1:bridge_col + 3] = 1
    start, goal = (height // 2, 2), (height // 2, width - 3)
    cell_costs[start], cell_costs[goal] = 1, 1
    return GridMap(cell_costs, diagonal=diagonal, start=start, goal=goal, name="terrain")


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
    positions = {
        "A": (0, 0), "B": (2, 1), "C": (1, 3), "D": (4, 0), "E": (4, 3),
        "F": (6, 1), "G": (3, 5), "H": (6, 4), "I": (8, 2), "J": (8, 5),
    }
    graph = Graph(name="simple_graph")
    for vertex, position in positions.items():
        graph.add_vertex(vertex, position)
    for source, target in ["AB", "AC", "BD", "BE", "CG", "CE", "DF", "EF", "EH", "GH", "FI", "HJ", "IJ", "HI"]:
        graph.add_edge(source, target)
    for source, target in ["BE", "FI"]:  # 拥堵路段
        graph.add_edge(source, target, graph.distance(source, target) * 1.5)
    graph.start, graph.goal = "A", "J"
    return graph


def random_road_graph(vertex_count: int = 800, neighbors_per_vertex: int = 4, seed: int = 0,
                      max_detour: float = 0.3, lake_count: int = 4) -> Graph:
    """随机“路网”：在 100×60 的平面撒点（挖掉若干圆形“湖泊”），每个点连向最近的
    ``neighbors_per_vertex`` 个邻居；边权 = 欧氏距离 × U(1, 1+max_detour)，
    因此欧氏启发函数可采纳。保证连通。"""
    rng = np.random.default_rng(seed)
    area_width, area_height = 100.0, 60.0
    lakes = [(rng.uniform(25, 75), rng.uniform(10, 50), rng.uniform(6, 12)) for _ in range(lake_count)]
    points: list[tuple[float, float]] = []
    while len(points) < vertex_count:
        x, y = rng.uniform(0, area_width), rng.uniform(0, area_height)
        if all(math.hypot(x - lake_x, y - lake_y) > lake_radius for lake_x, lake_y, lake_radius in lakes):
            points.append((x, y))
    point_array = np.array(points)
    graph = Graph(name="road_graph")
    for vertex, position in enumerate(points):
        graph.add_vertex(vertex, position)
    squared_distances = ((point_array[:, None, :] - point_array[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(squared_distances, np.inf)
    nearest_neighbors = np.argsort(squared_distances, axis=1)[:, :neighbors_per_vertex]
    for source in range(vertex_count):
        for target in nearest_neighbors[source]:
            target = int(target)
            if target not in graph.adjacency[source]:
                detour_factor = rng.uniform(1, 1 + max_detour)
                graph.add_edge(source, target, graph.distance(source, target) * detour_factor)

    # 连通性修复：把每个非主连通分量连到主分量最近的点
    def connected_components() -> list[set[int]]:
        visited, components = set(), []
        for seed_vertex in range(vertex_count):
            if seed_vertex in visited:
                continue
            component, stack = set(), [seed_vertex]
            while stack:
                vertex = stack.pop()
                if vertex in component:
                    continue
                component.add(vertex)
                stack.extend(neighbor for neighbor in graph.adjacency[vertex] if neighbor not in component)
            visited |= component
            components.append(component)
        return sorted(components, key=len, reverse=True)

    components = connected_components()
    main_component = components[0]
    for component in components[1:]:
        component_vertices = np.array(sorted(component))
        main_vertices = np.array(sorted(main_component))
        between = squared_distances[np.ix_(component_vertices, main_vertices)]
        component_index, main_index = np.unravel_index(np.argmin(between), between.shape)
        graph.add_edge(int(component_vertices[component_index]), int(main_vertices[main_index]))
        main_component |= component

    graph.start = int(np.argmin(point_array[:, 0] + point_array[:, 1]))   # 左下
    graph.goal = int(np.argmax(point_array[:, 0] + point_array[:, 1]))    # 右上
    return graph


GRAPH_MAPS = {
    "simple": simple_graph,
    "road": random_road_graph,
}
