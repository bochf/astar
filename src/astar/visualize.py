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
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from .graph import Graph
from .grid import GridMap
from .result import SearchResult

INK = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"
PATH_COLOR = "#eb6834"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FRONTIER = "#cfe0f5"
ORDER_CMAP = LinearSegmentedColormap.from_list("order", ["#a9c9ef", "#2a78d6", "#0d3b75"])
TERRAIN_CMAP = LinearSegmentedColormap.from_list("terrain", ["#ffffff", "#bdb8ad"])
WALL_COLOR = "#3a3936"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#d6d4ce", "axes.titlesize": 10, "axes.titlecolor": INK,
    "font.size": 9, "text.color": INK, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
})


def use_backend(show: bool) -> None:
    if not show:
        matplotlib.use("Agg")


# ============================================================ 矩阵地图

def _grid_base_rgb(grid: GridMap) -> np.ndarray:
    cost = grid.cost
    finite = np.isfinite(cost)
    lo, hi = cost[finite].min(), cost[finite].max()
    norm = np.zeros_like(cost) if hi == lo else (np.where(finite, cost, lo) - lo) / (hi - lo)
    rgb = TERRAIN_CMAP(norm)[..., :3]
    rgb[~finite] = matplotlib.colors.to_rgb(WALL_COLOR)
    return rgb


def _grid_overlay(grid: GridMap, result: SearchResult, upto: int | None = None) -> np.ndarray:
    """RGBA 叠加层：前 upto 个扩展节点按顺序着色，其余已发现节点为浅色。"""
    h, w = grid.shape
    over = np.zeros((h, w, 4))
    expanded = result.expanded if upto is None else result.expanded[:upto]
    exp_set = set(expanded)
    if upto is None:
        frontier = [c for c in result.discovered if c not in exp_set]
    else:
        frontier = _discovered_until(grid, expanded, exp_set)
    fr = matplotlib.colors.to_rgba(FRONTIER, 0.9)
    for c in frontier:
        over[c] = fr
    n = max(len(result.expanded) - 1, 1)
    for i, c in enumerate(expanded):
        over[c] = ORDER_CMAP(i / n)[:3] + (0.85,)
    return over


def _discovered_until(grid: GridMap, expanded: list, exp_set: set) -> list:
    out = set()
    for c in expanded:
        for nb, _ in grid.neighbors(c):
            if nb not in exp_set:
                out.add(nb)
    return list(out)


def _mark_endpoints(ax, s, g, letters: bool = True) -> None:
    for (x, y), lbl in ((s, "S"), (g, "G")):
        if letters:
            ax.plot(x, y, marker="o", ms=11, color=INK, mec=SURFACE, mew=2, zorder=5)
            ax.text(x, y, lbl, color=SURFACE, ha="center", va="center", fontsize=7, fontweight="bold", zorder=6)
        else:  # 已有顶点名的小图：用黑色圆环，避免与顶点名冲突
            ax.plot(x, y, marker="o", ms=15, mfc="none", mec=INK, mew=2.2, zorder=5)


def draw_grid(ax, grid: GridMap, result: SearchResult | None = None, upto: int | None = None,
              show_path: bool = True, title: str | None = None) -> None:
    ax.clear()
    ax.imshow(_grid_base_rgb(grid), interpolation="nearest")
    if result is not None:
        ax.imshow(_grid_overlay(grid, result, upto), interpolation="nearest")
        if show_path and result.path and (upto is None or upto >= result.n_expanded):
            rows, cols = zip(*result.path)
            ax.plot(cols, rows, color=PATH_COLOR, lw=2.5, solid_capstyle="round", zorder=4)
    if grid.start and grid.goal:
        _mark_endpoints(ax, grid.start[::-1], grid.goal[::-1])
    ax.set_xticks([]); ax.set_yticks([])
    if title is None and result is not None:
        title = _result_title(result, upto)
    if title:
        ax.set_title(title, loc="left")


def _result_title(result: SearchResult, upto: int | None = None, optimal: float | None = None) -> str:
    n = result.n_expanded if upto is None else min(upto, result.n_expanded)
    cost = f"{result.cost:.2f}" if result.found else "unreachable"
    if upto is not None and upto < result.n_expanded:
        cost = "…"
    tag = ""
    if optimal is not None and result.found and result.cost > optimal + 1e-9:
        tag = f"  (+{(result.cost / optimal - 1) * 100:.1f}% suboptimal)"
    return f"{result.algorithm}\nexpanded {n}   cost {cost}{tag}"


def compare_grid(grid: GridMap, results: Sequence[SearchResult], suptitle: str | None = None):
    """多个结果并排的静态对比图。"""
    n = len(results)
    cols = min(n, 3)
    rows = math.ceil(n / cols)
    h, w = grid.shape
    fig, axes = plt.subplots(rows, cols, figsize=(cols * max(4.2, w / 14), rows * max(3.4, h / 14) + 0.6),
                             squeeze=False, layout="constrained")
    optimal = min((r.cost for r in results if r.found), default=None)
    for ax, r in zip(axes.flat, results):
        draw_grid(ax, grid, r, title=_result_title(r, optimal=optimal))
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(suptitle or f"{grid.name}  {h}×{w}  ({'8' if grid.diagonal else '4'}-connected)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    return fig


def animate_grid(grid: GridMap, results: Sequence[SearchResult], frames: int = 90, interval: int = 60):
    """并排动画：所有算法以相同速度（每帧扩展相同数量的节点）推进，直观对比搜索范围。"""
    n = len(results)
    h, w = grid.shape
    fig, axes = plt.subplots(1, n, figsize=(n * max(4.2, w / 14), max(3.4, h / 14) + 0.6),
                             squeeze=False, layout="constrained")
    axes = axes[0]
    total = max(r.n_expanded for r in results)
    per = max(1, math.ceil(total / frames))
    steps = list(range(0, total + per, per)) + [total] * 12  # 结尾停留

    overlays, lines = [], []
    for ax, r in zip(axes, results):
        draw_grid(ax, grid, None, title=" ")
        overlays.append(ax.imshow(np.zeros((h, w, 4)), interpolation="nearest"))
        lines.append(ax.plot([], [], color=PATH_COLOR, lw=2.5, solid_capstyle="round", zorder=4)[0])

    def update(k):
        for ax, r, im, ln in zip(axes, results, overlays, lines):
            im.set_data(_grid_overlay(grid, r, upto=k))
            if r.path and k >= r.n_expanded:
                rows, cols = zip(*r.path)
                ln.set_data(cols, rows)
            else:
                ln.set_data([], [])
            ax.set_title(_result_title(r, upto=k), loc="left")
        return []

    fig.suptitle(f"{grid.name}: same number of expansions per frame", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    return animation.FuncAnimation(fig, update, frames=steps, interval=interval, blit=False), fig


# ============================================================ 顶点 / 边图

def draw_graph(ax, graph: Graph, result: SearchResult | None = None, upto: int | None = None,
               title: str | None = None, labels: bool | None = None) -> None:
    ax.clear()
    pos = graph.pos
    small = len(graph) <= 30 if labels is None else labels
    segs = [(pos[u], pos[v]) for u, v, _ in graph.edges()]
    ax.add_collection(LineCollection(segs, colors="#c9c6bd", linewidths=1.2 if small else 0.6, zorder=1))
    xy = np.array(list(pos.values()))
    ax.scatter(xy[:, 0], xy[:, 1], s=60 if small else 6, color="#bdb8ad", zorder=2, linewidths=0)

    if result is not None:
        expanded = result.expanded if upto is None else result.expanded[:upto]
        exp_set = set(expanded)
        if upto is None:
            frontier = [v for v in result.discovered if v not in exp_set]
        else:
            frontier = {nb for v in expanded for nb in graph.adj[v] if nb not in exp_set}
        if frontier:
            f = np.array([pos[v] for v in frontier])
            ax.scatter(f[:, 0], f[:, 1], s=90 if small else 14, color=FRONTIER, zorder=3,
                       edgecolors="#2a78d6", linewidths=0.6)
        if expanded:
            e = np.array([pos[v] for v in expanded])
            nmax = max(result.n_expanded - 1, 1)
            ax.scatter(e[:, 0], e[:, 1], s=110 if small else 16, c=np.arange(len(expanded)) / nmax,
                       cmap=ORDER_CMAP, vmin=0, vmax=1, zorder=3, linewidths=0)
        if result.path and (upto is None or upto >= result.n_expanded):
            p = np.array([pos[v] for v in result.path])
            ax.plot(p[:, 0], p[:, 1], color=PATH_COLOR, lw=3, zorder=4, solid_capstyle="round")
        if title is None:
            title = _result_title(result, upto)

    if small:
        for v, (x, y) in pos.items():
            name = str(v) + {graph.start: " (start)", graph.goal: " (goal)"}.get(v, "")
            ax.text(x, y + 0.28, name, ha="center", va="bottom", fontsize=8, color=INK, zorder=7)
        for u, v, wgt in graph.edges():
            (x1, y1), (x2, y2) = pos[u], pos[v]
            ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{wgt:.1f}", fontsize=6.5, color=MUTED,
                    ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none"))
    if graph.start is not None and graph.goal is not None:
        _mark_endpoints(ax, pos[graph.start], pos[graph.goal], letters=not small)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.margins(0.06)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, loc="left")


def compare_graph(graph: Graph, results: Sequence[SearchResult], suptitle: str | None = None):
    n = len(results)
    cols = min(n, 3)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 3.8 + 0.6), squeeze=False,
                             layout="constrained")
    optimal = min((r.cost for r in results if r.found), default=None)
    for ax, r in zip(axes.flat, results):
        draw_graph(ax, graph, r, title=_result_title(r, optimal=optimal))
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(suptitle or f"{graph.name}  |V|={len(graph)}  |E|={graph.n_edges}",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    return fig


def animate_graph(graph: Graph, results: Sequence[SearchResult], frames: int = 90, interval: int = 60):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(n * 5.2, 4.4), squeeze=False, layout="constrained")
    axes = axes[0]
    total = max(r.n_expanded for r in results)
    per = max(1, math.ceil(total / frames))
    steps = list(range(0, total + per, per)) + [total] * 12

    def update(k):
        for ax, r in zip(axes, results):
            draw_graph(ax, graph, r, upto=k)
        return []

    fig.suptitle(f"{graph.name}: same number of expansions per frame", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    return animation.FuncAnimation(fig, update, frames=steps, interval=interval, blit=False), fig


# ============================================================ 基准柱状图

def benchmark_chart(rows: Sequence[tuple[str, Sequence[SearchResult]]]):
    """每张地图一个子图：横向柱 = 扩展节点数；柱尾标注路径代价，非最优的加注偏差。"""
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 0.45 * max(len(r) for _, r in rows) + 1.4),
                             squeeze=False, layout="constrained")
    for ax, (name, results) in zip(axes[0], rows):
        optimal = min(r.cost for r in results if r.found)
        labels = [r.algorithm.replace("A* ", "").strip("()") for r in results]
        vals = [r.n_expanded for r in results]
        y = np.arange(len(results))[::-1]
        ax.barh(y, vals, height=0.62, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
        vmax = max(vals)
        for yi, r in zip(y, results):
            subopt = r.cost > optimal + 1e-9
            txt = f"{r.n_expanded}  ·  cost {r.cost:.1f}" + (f"  (+{(r.cost / optimal - 1) * 100:.1f}%)" if subopt else "")
            ax.text(r.n_expanded + vmax * 0.02, yi, txt, va="center", fontsize=7.5,
                    color=INK if subopt else MUTED, fontweight="bold" if subopt else "normal")
        ax.set_yticks(y, labels)
        ax.set_xlim(0, vmax * 1.9)
        ax.set_title(name, loc="left", fontweight="bold")
        ax.set_xlabel("expanded nodes")
        ax.grid(axis="x", color="#e7e5df", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Expanded nodes by heuristic (fewer is better; bold = suboptimal path)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    return fig


def save(fig_or_anim, path: Path, fig=None, fps: int = 15) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(fig_or_anim, animation.Animation):
        fig_or_anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=80)
        plt.close(fig)
    else:
        fig_or_anim.savefig(path, dpi=130)
        plt.close(fig_or_anim)
    return path
