"""A* 寻路：二维矩阵与顶点/边图两种实现，附地图生成与可视化。"""

from .graph import Graph, astar_graph, dijkstra_graph
from .grid import WALL, GridMap, astar_grid, dijkstra_grid
from .heuristics import (available_heuristics, chebyshev, euclidean, get_heuristic, manhattan, octile,
                         register_heuristic, weighted, zero)
from .result import SearchResult

__all__ = [
    "Graph", "GridMap", "SearchResult", "WALL",
    "astar_grid", "dijkstra_grid", "astar_graph", "dijkstra_graph",
    "register_heuristic", "get_heuristic", "available_heuristics", "weighted",
    "zero", "manhattan", "euclidean", "chebyshev", "octile",
    "main",
]


def main() -> None:
    from .cli import main as _main

    _main()
