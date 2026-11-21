"""命令行入口：``astar demo | grid | graph | heuristics``。"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from pathlib import Path

from . import maps
from .graph import Graph, astar_graph
from .grid import GridMap, astar_grid
from .heuristics import available_heuristics
from .result import SearchResult


def _run_grid(grid: GridMap, heuristics: list[str]) -> list[SearchResult]:
    return [astar_grid(grid, heuristic=h) for h in heuristics]


def _run_graph(graph: Graph, heuristics: list[str]) -> list[SearchResult]:
    return [astar_graph(graph, heuristic=h) for h in heuristics]


def verify(struct: GridMap | Graph, results: list[SearchResult], admissible: set[str]) -> bool:
    """校验：① 路径合法且代价与报告一致；② 可采纳启发函数得到的代价 = Dijkstra 最优代价。"""
    ok = True
    dijkstra = next((r for r in results if r.heuristic == "zero"), None)
    for r in results:
        if not r.found:
            continue
        real = struct.path_cost(r.path)
        if not math.isclose(real, r.cost, rel_tol=1e-9):
            print(f"  ✗ {r.algorithm}: 路径实际代价 {real} ≠ 报告值 {r.cost}")
            ok = False
        if dijkstra and r.heuristic in admissible and not math.isclose(r.cost, dijkstra.cost, rel_tol=1e-9):
            print(f"  ✗ {r.algorithm}: 代价 {r.cost} ≠ Dijkstra {dijkstra.cost}")
            ok = False
    print("  ✓ 校验通过：路径合法，可采纳启发函数的代价与 Dijkstra 一致" if ok else "  ✗ 校验失败")
    return ok


def _report(title: str, results: list[SearchResult]) -> None:
    print(f"\n== {title}")
    for r in results:
        print("  " + r.summary())


def _output(fig_or_anim, out: Path | None, name: str, show: bool, fig=None) -> None:
    from . import visualize as viz

    if out is not None:
        p = viz.save(fig_or_anim, out / name, fig=fig)
        print(f"  -> {p}")
    elif show:
        import matplotlib.pyplot as plt

        plt.show()


def _load_plugins(mods: list[str]) -> None:
    """导入用户模块，以便其中的 @register_heuristic 生效。"""
    for m in mods:
        if m.endswith(".py"):
            spec = importlib.util.spec_from_file_location(Path(m).stem, m)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        else:
            importlib.import_module(m)


# ------------------------------------------------------------------ commands

def cmd_demo(args) -> int:
    from . import visualize as viz

    out = Path(args.out)
    all_ok = True
    bench = []

    # 1. 矩阵地图：Dijkstra vs A*
    grid_cases = [
        ("simple", maps.simple_grid(), ["zero", "manhattan", "euclidean"]),
        ("open", maps.open_grid(), ["zero", "manhattan"]),
        ("maze", maps.maze_grid(seed=args.seed), ["zero", "manhattan", "manhattan*3"]),
        ("random", maps.random_obstacle_grid(seed=args.seed), ["zero", "octile", "euclidean", "chebyshev", "octile*2"]),
        ("terrain", maps.weighted_terrain_grid(seed=args.seed), ["zero", "octile", "octile*3"]),
    ]
    admissible_grid = {"zero", "manhattan", "euclidean", "chebyshev", "octile"}
    for name, grid, hs in grid_cases:
        res = _run_grid(grid, hs)
        _report(f"grid/{name} {grid.shape} diagonal={grid.diagonal}", res)
        all_ok &= verify(grid, res, admissible_grid if grid.diagonal else admissible_grid)
        viz.save(viz.compare_grid(grid, res), out / f"grid_{name}.png")
        if not args.no_anim:
            anim, fig = viz.animate_grid(grid, res[:2])
            viz.save(anim, out / f"grid_{name}.gif", fig=fig)

    # 2. 启发函数基准（8 邻接随机障碍图 + 4 邻接迷宫）
    hs_bench = ["zero", "chebyshev", "euclidean", "octile", "manhattan", "octile*2"]
    bench.append(("random 60×90 (8-conn)", _run_grid(maps.random_obstacle_grid(seed=args.seed), hs_bench)))
    bench.append(("terrain 60×90 (8-conn, weighted)", _run_grid(maps.weighted_terrain_grid(seed=args.seed), hs_bench)))
    hs4 = ["zero", "chebyshev", "euclidean", "manhattan", "manhattan*2"]
    bench.append(("maze 41×61 (4-conn)", _run_grid(maps.maze_grid(seed=args.seed), hs4)))

    # 3. 顶点/边图
    admissible_graph = {"zero", "euclidean", "chebyshev"}
    sg = maps.simple_graph()
    res = _run_graph(sg, ["zero", "euclidean", "manhattan"])
    _report(f"graph/simple |V|={len(sg)} |E|={sg.n_edges}", res)
    all_ok &= verify(sg, res, admissible_graph)
    viz.save(viz.compare_graph(sg, res), out / "graph_simple.png")
    if not args.no_anim:
        # 顶点很少：每帧扩展 1 个顶点，低帧率便于逐步观察
        anim, fig = viz.animate_graph(sg, res, frames=len(sg))
        viz.save(anim, out / "graph_simple.gif", fig=fig, fps=2)

    rg = maps.random_road_graph(seed=args.seed)
    res = _run_graph(rg, ["zero", "euclidean", "euclidean*2"])
    _report(f"graph/road |V|={len(rg)} |E|={rg.n_edges}", res)
    all_ok &= verify(rg, res, admissible_graph)
    viz.save(viz.compare_graph(rg, res), out / "graph_road.png")
    if not args.no_anim:
        anim, fig = viz.animate_graph(rg, res[:2])
        viz.save(anim, out / "graph_road.gif", fig=fig)
    bench.append(("road graph 800 V", _run_graph(rg, ["zero", "chebyshev", "euclidean", "manhattan", "euclidean*2"])))

    viz.save(viz.benchmark_chart(bench), out / "benchmark.png")
    print(f"\n所有图像已写入 {out.resolve()}")
    return 0 if all_ok else 1


def _grid_from_args(args) -> GridMap:
    if args.map == "file":
        grid = GridMap.from_ascii(Path(args.file).read_text(), name=Path(args.file).stem)
    else:
        kw = {"seed": args.seed} if args.map in {"maze", "random", "terrain"} else {}
        grid = maps.GRID_MAPS[args.map](**kw)
    if args.diagonal is not None:
        grid.diagonal = args.diagonal
    return grid


def cmd_grid(args) -> int:
    _load_plugins(args.plugin)
    from . import visualize as viz

    viz.use_backend(args.show and not args.out)
    grid = _grid_from_args(args)
    hs = args.heuristic or ["zero", grid.default_heuristic()]
    res = _run_grid(grid, hs)
    _report(f"grid/{grid.name} {grid.shape} diagonal={grid.diagonal}", res)
    out = Path(args.out) if args.out else None
    if args.animate:
        anim, fig = viz.animate_grid(grid, res)
        _output(anim, out, f"grid_{grid.name}.gif", args.show, fig=fig)
    else:
        _output(viz.compare_grid(grid, res), out, f"grid_{grid.name}.png", args.show)
    return 0


def cmd_graph(args) -> int:
    _load_plugins(args.plugin)
    from . import visualize as viz

    viz.use_backend(args.show and not args.out)
    graph = maps.GRAPH_MAPS[args.map](**({"seed": args.seed, "n": args.nodes} if args.map == "road" else {}))
    hs = args.heuristic or ["zero", "euclidean"]
    res = _run_graph(graph, hs)
    _report(f"graph/{graph.name} |V|={len(graph)} |E|={graph.n_edges}", res)
    out = Path(args.out) if args.out else None
    if args.animate:
        anim, fig = viz.animate_graph(graph, res)
        _output(anim, out, f"graph_{graph.name}.gif", args.show, fig=fig)
    else:
        _output(viz.compare_graph(graph, res), out, f"graph_{graph.name}.png", args.show)
    return 0


def cmd_heuristics(args) -> int:
    _load_plugins(args.plugin)
    for name, desc in available_heuristics().items():
        print(f"  {name:<12} {desc}")
    print("\n  任意名称后加 *w 表示加权（如 manhattan*2）；也可用 module:function 引用自定义函数。")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="astar", description="A* 寻路演示（二维矩阵 & 顶点/边图）")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="运行全部演示并输出图片/动画到目录")
    d.add_argument("--out", default="output")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--no-anim", action="store_true", help="跳过 GIF 生成（更快）")
    d.set_defaults(func=cmd_demo)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-H", "--heuristic", action="append",
                        help="启发函数，可多次指定以对比；zero=Dijkstra；支持 name*w 与 module:func")
    common.add_argument("--plugin", action="append", default=[],
                        help="先导入的模块或 .py 文件（其中可用 @register_heuristic 注册启发函数）")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--animate", action="store_true", help="生成搜索过程动画")
    common.add_argument("--show", action="store_true", help="弹出窗口显示")
    common.add_argument("--out", help="输出目录（PNG/GIF）")

    g = sub.add_parser("grid", parents=[common], help="在二维矩阵地图上运行")
    g.add_argument("--map", choices=[*maps.GRID_MAPS, "file"], default="simple")
    g.add_argument("--file", help="--map file 时的 ASCII 地图路径（. 空地 # 墙 1-9 代价 S 起点 G 终点）")
    diag = g.add_mutually_exclusive_group()
    diag.add_argument("--diagonal", dest="diagonal", action="store_true", default=None, help="8 邻接")
    diag.add_argument("--no-diagonal", dest="diagonal", action="store_false", help="4 邻接")
    g.set_defaults(func=cmd_grid)

    gr = sub.add_parser("graph", parents=[common], help="在顶点/边图上运行")
    gr.add_argument("--map", choices=list(maps.GRAPH_MAPS), default="simple")
    gr.add_argument("--nodes", type=int, default=800, help="road 图的顶点数")
    gr.set_defaults(func=cmd_graph)

    h = sub.add_parser("heuristics", help="列出可用启发函数")
    h.add_argument("--plugin", action="append", default=[])
    h.set_defaults(func=cmd_heuristics)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd in ("demo", "heuristics"):
        import matplotlib

        matplotlib.use("Agg")
    sys.exit(args.func(args))
