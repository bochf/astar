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


def _run_grid(grid: GridMap, heuristic_specs: list[str]) -> list[SearchResult]:
    return [astar_grid(grid, heuristic=spec) for spec in heuristic_specs]


def _run_graph(graph: Graph, heuristic_specs: list[str]) -> list[SearchResult]:
    return [astar_graph(graph, heuristic=spec) for spec in heuristic_specs]


def verify(map_structure: GridMap | Graph, results: list[SearchResult], admissible: set[str]) -> bool:
    """校验：① 路径合法且代价与报告一致；② 可采纳启发函数得到的代价 = Dijkstra 最优代价。"""
    all_passed = True
    dijkstra_result = next((result for result in results if result.heuristic == "zero"), None)
    for result in results:
        if not result.found:
            continue
        actual_cost = map_structure.path_cost(result.path)
        if not math.isclose(actual_cost, result.cost, rel_tol=1e-9):
            print(f"  ✗ {result.algorithm}: 路径实际代价 {actual_cost} ≠ 报告值 {result.cost}")
            all_passed = False
        if (dijkstra_result and result.heuristic in admissible
                and not math.isclose(result.cost, dijkstra_result.cost, rel_tol=1e-9)):
            print(f"  ✗ {result.algorithm}: 代价 {result.cost} ≠ Dijkstra {dijkstra_result.cost}")
            all_passed = False
    print("  ✓ 校验通过：路径合法，可采纳启发函数的代价与 Dijkstra 一致" if all_passed else "  ✗ 校验失败")
    return all_passed


def _report(title: str, results: list[SearchResult]) -> None:
    print(f"\n== {title}")
    for result in results:
        print("  " + result.summary())


def _output(figure_or_animation, output_dir: Path | None, file_name: str, show: bool, fig=None) -> None:
    from . import visualize

    if output_dir is not None:
        saved_path = visualize.save(figure_or_animation, output_dir / file_name, fig=fig)
        print(f"  -> {saved_path}")
    elif show:
        import matplotlib.pyplot as plt

        plt.show()


def _load_plugins(plugins: list[str]) -> None:
    """导入用户模块，以便其中的 @register_heuristic 生效。"""
    for plugin in plugins:
        if plugin.endswith(".py"):
            module_spec = importlib.util.spec_from_file_location(Path(plugin).stem, plugin)
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
        else:
            importlib.import_module(plugin)


# ------------------------------------------------------------------ commands

def cmd_demo(args) -> int:
    from . import visualize

    output_dir = Path(args.out)
    all_passed = True
    benchmarks = []

    # 1. 矩阵地图：Dijkstra vs A*
    grid_cases = [
        ("simple", maps.simple_grid(), ["zero", "manhattan", "euclidean"]),
        ("open", maps.open_grid(), ["zero", "manhattan"]),
        ("maze", maps.maze_grid(seed=args.seed), ["zero", "manhattan", "manhattan*3"]),
        ("random", maps.random_obstacle_grid(seed=args.seed), ["zero", "octile", "euclidean", "chebyshev", "octile*2"]),
        ("terrain", maps.weighted_terrain_grid(seed=args.seed), ["zero", "octile", "octile*3"]),
    ]
    admissible_on_grid = {"zero", "manhattan", "euclidean", "chebyshev", "octile"}
    for map_name, grid, heuristic_specs in grid_cases:
        results = _run_grid(grid, heuristic_specs)
        _report(f"grid/{map_name} {grid.shape} diagonal={grid.diagonal}", results)
        all_passed &= verify(grid, results, admissible_on_grid)
        visualize.save(visualize.compare_grid(grid, results), output_dir / f"grid_{map_name}.png")
        if not args.no_anim:
            grid_animation, fig = visualize.animate_grid(grid, results[:2])
            visualize.save(grid_animation, output_dir / f"grid_{map_name}.gif", fig=fig)

    # 2. 启发函数基准（8 邻接随机障碍图、加权地形 + 4 邻接迷宫）
    eight_connected_specs = ["zero", "chebyshev", "euclidean", "octile", "manhattan", "octile*2"]
    benchmarks.append(("random 60×90 (8-conn)",
                       _run_grid(maps.random_obstacle_grid(seed=args.seed), eight_connected_specs)))
    benchmarks.append(("terrain 60×90 (8-conn, weighted)",
                       _run_grid(maps.weighted_terrain_grid(seed=args.seed), eight_connected_specs)))
    four_connected_specs = ["zero", "chebyshev", "euclidean", "manhattan", "manhattan*2"]
    benchmarks.append(("maze 41×61 (4-conn)", _run_grid(maps.maze_grid(seed=args.seed), four_connected_specs)))

    # 3. 顶点/边图
    admissible_on_graph = {"zero", "euclidean", "chebyshev"}
    small_graph = maps.simple_graph()
    results = _run_graph(small_graph, ["zero", "euclidean", "manhattan"])
    _report(f"graph/simple |V|={len(small_graph)} |E|={small_graph.edge_count}", results)
    all_passed &= verify(small_graph, results, admissible_on_graph)
    visualize.save(visualize.compare_graph(small_graph, results), output_dir / "graph_simple.png")
    if not args.no_anim:
        # 顶点很少：每帧扩展 1 个顶点，低帧率便于逐步观察
        graph_animation, fig = visualize.animate_graph(small_graph, results, frames=len(small_graph))
        visualize.save(graph_animation, output_dir / "graph_simple.gif", fig=fig, fps=2)

    road_graph = maps.random_road_graph(seed=args.seed)
    results = _run_graph(road_graph, ["zero", "euclidean", "euclidean*2"])
    _report(f"graph/road |V|={len(road_graph)} |E|={road_graph.edge_count}", results)
    all_passed &= verify(road_graph, results, admissible_on_graph)
    visualize.save(visualize.compare_graph(road_graph, results), output_dir / "graph_road.png")
    if not args.no_anim:
        graph_animation, fig = visualize.animate_graph(road_graph, results[:2])
        visualize.save(graph_animation, output_dir / "graph_road.gif", fig=fig)
    benchmarks.append(("road graph 800 V",
                       _run_graph(road_graph, ["zero", "chebyshev", "euclidean", "manhattan", "euclidean*2"])))

    visualize.save(visualize.benchmark_chart(benchmarks), output_dir / "benchmark.png")
    print(f"\n所有图像已写入 {output_dir.resolve()}")
    return 0 if all_passed else 1


def _grid_from_args(args) -> GridMap:
    if args.map == "file":
        grid = GridMap.from_ascii(Path(args.file).read_text(), name=Path(args.file).stem)
    else:
        generator_kwargs = {"seed": args.seed} if args.map in {"maze", "random", "terrain"} else {}
        grid = maps.GRID_MAPS[args.map](**generator_kwargs)
    if args.diagonal is not None:
        grid.diagonal = args.diagonal
    return grid


def cmd_grid(args) -> int:
    _load_plugins(args.plugin)
    from . import visualize

    visualize.use_backend(args.show and not args.out)
    grid = _grid_from_args(args)
    heuristic_specs = args.heuristic or ["zero", grid.default_heuristic()]
    results = _run_grid(grid, heuristic_specs)
    _report(f"grid/{grid.name} {grid.shape} diagonal={grid.diagonal}", results)
    output_dir = Path(args.out) if args.out else None
    if args.animate:
        grid_animation, fig = visualize.animate_grid(grid, results)
        _output(grid_animation, output_dir, f"grid_{grid.name}.gif", args.show, fig=fig)
    else:
        _output(visualize.compare_grid(grid, results), output_dir, f"grid_{grid.name}.png", args.show)
    return 0


def cmd_graph(args) -> int:
    _load_plugins(args.plugin)
    from . import visualize

    visualize.use_backend(args.show and not args.out)
    generator_kwargs = {"seed": args.seed, "vertex_count": args.nodes} if args.map == "road" else {}
    graph = maps.GRAPH_MAPS[args.map](**generator_kwargs)
    heuristic_specs = args.heuristic or ["zero", "euclidean"]
    results = _run_graph(graph, heuristic_specs)
    _report(f"graph/{graph.name} |V|={len(graph)} |E|={graph.edge_count}", results)
    output_dir = Path(args.out) if args.out else None
    if args.animate:
        graph_animation, fig = visualize.animate_graph(graph, results)
        _output(graph_animation, output_dir, f"graph_{graph.name}.gif", args.show, fig=fig)
    else:
        _output(visualize.compare_graph(graph, results), output_dir, f"graph_{graph.name}.png", args.show)
    return 0


def cmd_heuristics(args) -> int:
    _load_plugins(args.plugin)
    for name, description in available_heuristics().items():
        print(f"  {name:<12} {description}")
    print("\n  任意名称后加 *w 表示加权（如 manhattan*2）；也可用 module:function 引用自定义函数。")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="astar", description="A* 寻路演示（二维矩阵 & 顶点/边图）")
    subcommands = parser.add_subparsers(dest="cmd", required=True)

    demo_parser = subcommands.add_parser("demo", help="运行全部演示并输出图片/动画到目录")
    demo_parser.add_argument("--out", default="output")
    demo_parser.add_argument("--seed", type=int, default=0)
    demo_parser.add_argument("--no-anim", action="store_true", help="跳过 GIF 生成（更快）")
    demo_parser.set_defaults(func=cmd_demo)

    common_options = argparse.ArgumentParser(add_help=False)
    common_options.add_argument("-H", "--heuristic", action="append",
                                help="启发函数，可多次指定以对比；zero=Dijkstra；支持 name*w 与 module:func")
    common_options.add_argument("--plugin", action="append", default=[],
                                help="先导入的模块或 .py 文件（其中可用 @register_heuristic 注册启发函数）")
    common_options.add_argument("--seed", type=int, default=0)
    common_options.add_argument("--animate", action="store_true", help="生成搜索过程动画")
    common_options.add_argument("--show", action="store_true", help="弹出窗口显示")
    common_options.add_argument("--out", help="输出目录（PNG/GIF）")

    grid_parser = subcommands.add_parser("grid", parents=[common_options], help="在二维矩阵地图上运行")
    grid_parser.add_argument("--map", choices=[*maps.GRID_MAPS, "file"], default="simple")
    grid_parser.add_argument("--file", help="--map file 时的 ASCII 地图路径（. 空地 # 墙 1-9 代价 S 起点 G 终点）")
    connectivity_group = grid_parser.add_mutually_exclusive_group()
    connectivity_group.add_argument("--diagonal", dest="diagonal", action="store_true", default=None, help="8 邻接")
    connectivity_group.add_argument("--no-diagonal", dest="diagonal", action="store_false", help="4 邻接")
    grid_parser.set_defaults(func=cmd_grid)

    graph_parser = subcommands.add_parser("graph", parents=[common_options], help="在顶点/边图上运行")
    graph_parser.add_argument("--map", choices=list(maps.GRAPH_MAPS), default="simple")
    graph_parser.add_argument("--nodes", type=int, default=800, help="road 图的顶点数")
    graph_parser.set_defaults(func=cmd_graph)

    heuristics_parser = subcommands.add_parser("heuristics", help="列出可用启发函数")
    heuristics_parser.add_argument("--plugin", action="append", default=[])
    heuristics_parser.set_defaults(func=cmd_heuristics)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd in ("demo", "heuristics"):
        import matplotlib

        matplotlib.use("Agg")
    sys.exit(args.func(args))
