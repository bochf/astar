"""可视化：静态对比图、搜索过程动画（GIF）、启发函数基准柱状图。

颜色约定（所有图一致）：
- 已扩展节点：按扩展顺序由浅到深的蓝色（展示搜索“波前”的推进）；
- 开放列表中尚未扩展的节点：浅蓝；
- 最终路径：橙色粗线；起点/终点：带 S / G 标签的黑色标记。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from .graph import Graph
from .grid import GridMap
from .result import SearchResult

INK = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"
PATH_COLOR = "#eb6834"
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FRONTIER_COLOR = "#cfe0f5"
EXPANSION_ORDER_CMAP = LinearSegmentedColormap.from_list("order", ["#a9c9ef", "#2a78d6", "#0d3b75"])
TERRAIN_CMAP = LinearSegmentedColormap.from_list("terrain", ["#ffffff", "#bdb8ad"])
WALL_COLOR = "#3a3936"
EDGE_COLOR = "#c9c6bd"
VERTEX_COLOR = "#bdb8ad"
END_HOLD_FRAMES = 12   # 动画结尾停留的帧数

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#d6d4ce", "axes.titlesize": 10, "axes.titlecolor": INK,
    "font.size": 9, "text.color": INK, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
})


def use_backend(show: bool) -> None:
    if not show:
        matplotlib.use("Agg")


def _animation_steps(results: Sequence[SearchResult], frames: int) -> list[int]:
    """每帧显示的扩展数量：所有算法同速推进，最后停留若干帧。"""
    most_expanded = max(result.expanded_count for result in results)
    expansions_per_frame = max(1, math.ceil(most_expanded / frames))
    return (list(range(0, most_expanded + expansions_per_frame, expansions_per_frame))
            + [most_expanded] * END_HOLD_FRAMES)


# ============================================================ 矩阵地图

def _grid_base_rgb(grid: GridMap) -> np.ndarray:
    """底图：格子代价越高颜色越深，墙为深灰。"""
    cell_costs = grid.cell_costs
    passable_mask = np.isfinite(cell_costs)
    lowest_cost, highest_cost = cell_costs[passable_mask].min(), cell_costs[passable_mask].max()
    if highest_cost == lowest_cost:
        normalized_cost = np.zeros_like(cell_costs)
    else:
        normalized_cost = (np.where(passable_mask, cell_costs, lowest_cost) - lowest_cost) / (highest_cost - lowest_cost)
    image = TERRAIN_CMAP(normalized_cost)[..., :3]
    image[~passable_mask] = matplotlib.colors.to_rgb(WALL_COLOR)
    return image


def _grid_overlay(grid: GridMap, result: SearchResult, shown_count: int | None = None) -> np.ndarray:
    """RGBA 叠加层：前 shown_count 个扩展节点按顺序着色，其余已发现节点为浅色。"""
    height, width = grid.shape
    overlay = np.zeros((height, width, 4))
    expanded = result.expanded if shown_count is None else result.expanded[:shown_count]
    expanded_set = set(expanded)
    if shown_count is None:
        frontier = [cell for cell in result.discovered if cell not in expanded_set]
    else:
        frontier = _frontier_of(grid, expanded, expanded_set)
    frontier_rgba = matplotlib.colors.to_rgba(FRONTIER_COLOR, 0.9)
    for cell in frontier:
        overlay[cell] = frontier_rgba
    last_index = max(len(result.expanded) - 1, 1)
    for order, cell in enumerate(expanded):
        overlay[cell] = EXPANSION_ORDER_CMAP(order / last_index)[:3] + (0.85,)
    return overlay


def _frontier_of(grid: GridMap, expanded: list, expanded_set: set) -> list:
    """已扩展节点的、尚未扩展的邻居（即动画某一帧的开放列表）。"""
    frontier = set()
    for cell in expanded:
        for neighbor, _ in grid.neighbors(cell):
            if neighbor not in expanded_set:
                frontier.add(neighbor)
    return list(frontier)


def _mark_endpoints(ax, start_xy, goal_xy, with_letters: bool = True) -> None:
    for (x, y), letter in ((start_xy, "S"), (goal_xy, "G")):
        if with_letters:
            ax.plot(x, y, marker="o", markersize=11, color=INK, markeredgecolor=SURFACE,
                    markeredgewidth=2, zorder=5)
            ax.text(x, y, letter, color=SURFACE, ha="center", va="center", fontsize=7,
                    fontweight="bold", zorder=6)
        else:  # 已有顶点名的小图：用黑色圆环，避免与顶点名冲突
            ax.plot(x, y, marker="o", markersize=15, markerfacecolor="none", markeredgecolor=INK,
                    markeredgewidth=2.2, zorder=5)


def draw_grid(ax, grid: GridMap, result: SearchResult | None = None, shown_count: int | None = None,
              show_path: bool = True, title: str | None = None) -> None:
    ax.clear()
    ax.imshow(_grid_base_rgb(grid), interpolation="nearest")
    if result is not None:
        ax.imshow(_grid_overlay(grid, result, shown_count), interpolation="nearest")
        search_finished = shown_count is None or shown_count >= result.expanded_count
        if show_path and result.path and search_finished:
            rows, cols = zip(*result.path)
            ax.plot(cols, rows, color=PATH_COLOR, linewidth=2.5, solid_capstyle="round", zorder=4)
    if grid.start and grid.goal:
        _mark_endpoints(ax, grid.start[::-1], grid.goal[::-1])   # (row, col) -> (x, y)
    ax.set_xticks([]); ax.set_yticks([])
    if title is None and result is not None:
        title = _result_title(result, shown_count)
    if title:
        ax.set_title(title, loc="left")


def _result_title(result: SearchResult, shown_count: int | None = None, optimal_cost: float | None = None) -> str:
    expanded_shown = result.expanded_count if shown_count is None else min(shown_count, result.expanded_count)
    cost_text = f"{result.cost:.2f}" if result.found else "unreachable"
    if shown_count is not None and shown_count < result.expanded_count:
        cost_text = "…"
    suboptimal_note = ""
    if optimal_cost is not None and result.found and result.cost > optimal_cost + 1e-9:
        suboptimal_note = f"  (+{(result.cost / optimal_cost - 1) * 100:.1f}% suboptimal)"
    return f"{result.algorithm}\nexpanded {expanded_shown}   cost {cost_text}{suboptimal_note}"


def compare_grid(grid: GridMap, results: Sequence[SearchResult], suptitle: str | None = None):
    """多个结果并排的静态对比图。"""
    panel_count = len(results)
    column_count = min(panel_count, 3)
    row_count = math.ceil(panel_count / column_count)
    height, width = grid.shape
    figure_size = (column_count * max(4.2, width / 14), row_count * max(3.4, height / 14) + 0.6)
    fig, axes = plt.subplots(row_count, column_count, figsize=figure_size, squeeze=False, layout="constrained")
    optimal_cost = min((result.cost for result in results if result.found), default=None)
    for ax, result in zip(axes.flat, results):
        draw_grid(ax, grid, result, title=_result_title(result, optimal_cost=optimal_cost))
    for unused_ax in axes.flat[panel_count:]:
        unused_ax.axis("off")
    connectivity = "8" if grid.diagonal else "4"
    fig.suptitle(suptitle or f"{grid.name}  {height}×{width}  ({connectivity}-connected)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    return fig


def animate_grid(grid: GridMap, results: Sequence[SearchResult], frames: int = 90, interval: int = 60):
    """并排动画：所有算法以相同速度（每帧扩展相同数量的节点）推进，直观对比搜索范围。"""
    panel_count = len(results)
    height, width = grid.shape
    figure_size = (panel_count * max(4.2, width / 14), max(3.4, height / 14) + 0.6)
    fig, axes = plt.subplots(1, panel_count, figsize=figure_size, squeeze=False, layout="constrained")
    axes = axes[0]
    steps = _animation_steps(results, frames)

    # 底图只画一次，每帧只更新叠加层、路径和标题
    overlay_images, path_lines = [], []
    for ax in axes:
        draw_grid(ax, grid, None, title=" ")
        overlay_images.append(ax.imshow(np.zeros((height, width, 4)), interpolation="nearest"))
        path_line, = ax.plot([], [], color=PATH_COLOR, linewidth=2.5, solid_capstyle="round", zorder=4)
        path_lines.append(path_line)

    def update(shown_count):
        for ax, result, overlay_image, path_line in zip(axes, results, overlay_images, path_lines):
            overlay_image.set_data(_grid_overlay(grid, result, shown_count=shown_count))
            if result.path and shown_count >= result.expanded_count:
                rows, cols = zip(*result.path)
                path_line.set_data(cols, rows)
            else:
                path_line.set_data([], [])
            ax.set_title(_result_title(result, shown_count=shown_count), loc="left")
        return []

    fig.suptitle(f"{grid.name}: same number of expansions per frame", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    return animation.FuncAnimation(fig, update, frames=steps, interval=interval, blit=False), fig


# ============================================================ 顶点 / 边图

def draw_graph(ax, graph: Graph, result: SearchResult | None = None, shown_count: int | None = None,
               title: str | None = None, labels: bool | None = None) -> None:
    ax.clear()
    positions = graph.positions
    is_small = len(graph) <= 30 if labels is None else labels
    edge_segments = [(positions[source], positions[target]) for source, target, _ in graph.edges()]
    ax.add_collection(LineCollection(edge_segments, colors=EDGE_COLOR,
                                     linewidths=1.2 if is_small else 0.6, zorder=1))
    all_points = np.array(list(positions.values()))
    ax.scatter(all_points[:, 0], all_points[:, 1], s=60 if is_small else 6, color=VERTEX_COLOR,
               zorder=2, linewidths=0)

    if result is not None:
        expanded = result.expanded if shown_count is None else result.expanded[:shown_count]
        expanded_set = set(expanded)
        if shown_count is None:
            frontier = [vertex for vertex in result.discovered if vertex not in expanded_set]
        else:
            frontier = {neighbor for vertex in expanded for neighbor in graph.adjacency[vertex]
                        if neighbor not in expanded_set}
        if frontier:
            frontier_points = np.array([positions[vertex] for vertex in frontier])
            ax.scatter(frontier_points[:, 0], frontier_points[:, 1], s=90 if is_small else 14,
                       color=FRONTIER_COLOR, zorder=3, edgecolors=SERIES_COLORS[0], linewidths=0.6)
        if expanded:
            expanded_points = np.array([positions[vertex] for vertex in expanded])
            last_index = max(result.expanded_count - 1, 1)
            ax.scatter(expanded_points[:, 0], expanded_points[:, 1], s=110 if is_small else 16,
                       c=np.arange(len(expanded)) / last_index, cmap=EXPANSION_ORDER_CMAP,
                       vmin=0, vmax=1, zorder=3, linewidths=0)
        search_finished = shown_count is None or shown_count >= result.expanded_count
        if result.path and search_finished:
            path_points = np.array([positions[vertex] for vertex in result.path])
            ax.plot(path_points[:, 0], path_points[:, 1], color=PATH_COLOR, linewidth=3, zorder=4,
                    solid_capstyle="round")
        if title is None:
            title = _result_title(result, shown_count)

    if is_small:
        endpoint_suffix = {graph.start: " (start)", graph.goal: " (goal)"}
        for vertex, (x, y) in positions.items():
            ax.text(x, y + 0.28, str(vertex) + endpoint_suffix.get(vertex, ""), ha="center", va="bottom",
                    fontsize=8, color=INK, zorder=7)
        for source, target, weight in graph.edges():
            (source_x, source_y), (target_x, target_y) = positions[source], positions[target]
            ax.text((source_x + target_x) / 2, (source_y + target_y) / 2, f"{weight:.1f}", fontsize=6.5,
                    color=MUTED, ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor=SURFACE, edgecolor="none"))
    if graph.start is not None and graph.goal is not None:
        _mark_endpoints(ax, positions[graph.start], positions[graph.goal], with_letters=not is_small)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.margins(0.06)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, loc="left")


def compare_graph(graph: Graph, results: Sequence[SearchResult], suptitle: str | None = None):
    panel_count = len(results)
    column_count = min(panel_count, 3)
    row_count = math.ceil(panel_count / column_count)
    fig, axes = plt.subplots(row_count, column_count, figsize=(column_count * 5.2, row_count * 3.8 + 0.6),
                             squeeze=False, layout="constrained")
    optimal_cost = min((result.cost for result in results if result.found), default=None)
    for ax, result in zip(axes.flat, results):
        draw_graph(ax, graph, result, title=_result_title(result, optimal_cost=optimal_cost))
    for unused_ax in axes.flat[panel_count:]:
        unused_ax.axis("off")
    fig.suptitle(suptitle or f"{graph.name}  |V|={len(graph)}  |E|={graph.edge_count}",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    return fig


def animate_graph(graph: Graph, results: Sequence[SearchResult], frames: int = 90, interval: int = 60):
    panel_count = len(results)
    fig, axes = plt.subplots(1, panel_count, figsize=(panel_count * 5.2, 4.4), squeeze=False,
                             layout="constrained")
    axes = axes[0]
    steps = _animation_steps(results, frames)

    def update(shown_count):
        for ax, result in zip(axes, results):
            draw_graph(ax, graph, result, shown_count=shown_count)
        return []

    fig.suptitle(f"{graph.name}: same number of expansions per frame", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    return animation.FuncAnimation(fig, update, frames=steps, interval=interval, blit=False), fig


# ============================================================ 基准柱状图

def benchmark_chart(benchmarks: Sequence[tuple[str, Sequence[SearchResult]]]):
    """每张地图一个子图：横向柱 = 扩展节点数；柱尾标注路径代价，非最优的加注偏差。"""
    panel_count = len(benchmarks)
    most_bars = max(len(results) for _, results in benchmarks)
    fig, axes = plt.subplots(1, panel_count, figsize=(3.6 * panel_count, 0.45 * most_bars + 1.4),
                             squeeze=False, layout="constrained")
    for ax, (map_name, results) in zip(axes[0], benchmarks):
        optimal_cost = min(result.cost for result in results if result.found)
        bar_labels = [result.algorithm.replace("A* ", "").strip("()") for result in results]
        expanded_counts = [result.expanded_count for result in results]
        bar_positions = np.arange(len(results))[::-1]
        ax.barh(bar_positions, expanded_counts, height=0.62, color=SERIES_COLORS[0],
                edgecolor=SURFACE, linewidth=2)
        largest_count = max(expanded_counts)
        for bar_position, result in zip(bar_positions, results):
            is_suboptimal = result.cost > optimal_cost + 1e-9
            annotation = f"{result.expanded_count}  ·  cost {result.cost:.1f}"
            if is_suboptimal:
                annotation += f"  (+{(result.cost / optimal_cost - 1) * 100:.1f}%)"
            ax.text(result.expanded_count + largest_count * 0.02, bar_position, annotation, va="center",
                    fontsize=7.5, color=INK if is_suboptimal else MUTED,
                    fontweight="bold" if is_suboptimal else "normal")
        ax.set_yticks(bar_positions, bar_labels)
        ax.set_xlim(0, largest_count * 1.9)
        ax.set_title(map_name, loc="left", fontweight="bold")
        ax.set_xlabel("expanded nodes")
        ax.grid(axis="x", color="#e7e5df", linewidth=0.6)
        ax.set_axisbelow(True)
        for spine_name in ("top", "right"):
            ax.spines[spine_name].set_visible(False)
    fig.suptitle("Expanded nodes by heuristic (fewer is better; bold = suboptimal path)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    return fig


def save(figure_or_animation, path: Path, fig=None, fps: int = 15) -> Path:
    """保存静态图（PNG）或动画（GIF，需同时传入 fig 以便关闭）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(figure_or_animation, animation.Animation):
        figure_or_animation.save(path, writer=animation.PillowWriter(fps=fps), dpi=80)
        plt.close(fig)
    else:
        figure_or_animation.savefig(path, dpi=130)
        plt.close(figure_or_animation)
    return path
