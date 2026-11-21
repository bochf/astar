# A* 寻路算法演示

用两种图数据结构分别实现 A*，配有简单/复杂地图生成器、与 Dijkstra 的可视化对比、
多种经典启发函数，以及自定义启发函数接口。

| 数据结构 | 模块 | 节点 | 边 |
|---|---|---|---|
| 二维矩阵 | [grid.py](src/astar/grid.py) `GridMap` + `astar_grid` | `(row, col)` | 由下标隐式给出（4/8 邻接），代价 = 目标格代价（对角 ×√2） |
| 顶点/边 | [graph.py](src/astar/graph.py) `Graph` + `astar_graph` | 任意可哈希 id + `(x, y)` 坐标 | 邻接表显式存储，有向/无向，非负权重 |

Dijkstra 就是启发函数恒为 0 的 A*（`heuristic="zero"`，或用 `dijkstra_grid` / `dijkstra_graph`）。

## 快速开始

```bash
uv sync
uv run astar demo                 # 运行全部演示，PNG/GIF 写入 output/（约 1 分钟；--no-anim 跳过 GIF 只要几秒）
uv run astar heuristics           # 列出启发函数
```

`demo` 会在终端打印每张地图上各算法的路径代价和扩展节点数，并自动校验：
① 路径是合法移动且代价与报告值一致；② 可采纳启发函数得到的代价与 Dijkstra 相同。

生成的文件：

| 文件 | 内容 |
|---|---|
| `grid_simple.png/gif` | 简单地图：15×25，起终点间有一道“匚”形墙 |
| `grid_open.png/gif` | 空地：Dijkstra 扩展 1185 个格子，A* 只扩展路径上的 44 个 |
| `grid_maze.png/gif` | 41×61 迷宫（带环路） |
| `grid_random.png/gif` | 60×90 随机矩形障碍，8 邻接 |
| `grid_terrain.png/gif` | 60×90 加权地形（代价 1–9），有山峰和只有两个渡口的河流 |
| `graph_simple.png/gif` | 10 个顶点的小路网，标注了边权，方便手工核对；动画每帧扩展一个顶点 |
| `graph_road.png/gif` | 800 个顶点的随机路网（中间有“湖泊”空洞） |
| `benchmark.png` | 各启发函数在几张地图上的扩展节点数和路径代价对比 |

图例：蓝色深浅表示扩展顺序（浅色先扩展），浅蓝表示仍在开放列表中的节点，橙线是最终路径，S/G 是起点和终点。
动画里所有算法每帧扩展相同数量的节点，可以直接看出 A* 搜索范围更集中。

### 单独运行

```bash
# 矩阵地图：--map simple|open|maze|random|terrain|file，-H 可重复以并排对比
uv run astar grid --map maze -H zero -H manhattan -H manhattan*2 --show
uv run astar grid --map terrain --seed 3 -H zero -H octile --animate --out output
uv run astar grid --map file --file my_map.txt --diagonal --show   # ASCII：. 空地  # 墙  1-9 代价  S 起点  G 终点

# 顶点/边图：--map simple|road
uv run astar graph --map road --nodes 1500 -H zero -H euclidean --animate --out output
```

不加 `--show` 也不加 `--out` 时只打印统计结果。

## 启发函数

所有启发函数签名都是 `h(a, b) -> float`，`a`、`b` 是坐标元组，矩阵和图共用。

| 名称 | 公式 | 何时可采纳（保证最优） |
|---|---|---|
| `zero` | 0 | 总是（即 Dijkstra） |
| `manhattan` | \|dx\|+\|dy\| | 4 邻接网格（是精确下界）；8 邻接或平面图上会高估 |
| `euclidean` | √(dx²+dy²) | 总是，只要边权 ≥ 欧氏距离 |
| `chebyshev` | max(\|dx\|,\|dy\|) | 总是（偏保守） |
| `octile` | max+(√2−1)·min | 8 邻接、对角代价 √2 时的精确下界 |
| `name*w` | w·h | w>1 时是 Weighted A*：扩展更少，代价 ≤ w × 最优 |

内置启发函数按最小格子代价为 1 设计；地图中有代价小于 1 的格子时，`astar_grid` 会自动按比例缩小启发值。

### 自定义启发函数

```python
from astar import astar_grid, register_heuristic, maps

# 方式 1：直接传可调用对象
astar_grid(grid, heuristic=lambda a, b: max(abs(a[0]-b[0]), abs(a[1]-b[1])))

# 方式 2：注册成名称，之后可以用字符串（包括命令行 -H）引用
@register_heuristic("my_h", "说明文字")
def my_h(a, b): ...
```

命令行中用 `--plugin path/to/file.py`（文件里用 `@register_heuristic` 注册）或 `-H module:function` 引用自定义函数。
完整示例见 [examples/custom_heuristic.py](examples/custom_heuristic.py)：

```bash
uv run astar grid --map random --plugin examples/custom_heuristic.py -H zero -H tie_breaking_octile --show
PYTHONPATH=examples uv run astar grid --map maze -H zero -H custom_heuristic:manhattan_sq --show
```

启发函数不一致（例如加权或随机的启发函数）时，实现允许节点被重新打开，返回的路径仍然合法。

## 实现要点

- 开放列表用二叉堆，条目为 `(f, h, 序号, g, 节点)`。f 相同时优先扩展 h 更小的节点，这在开阔区域能大幅减少扩展数。
- 采用惰性删除：节点的 g 值被改进后直接压入新条目，弹出时如果条目里的 g 大于当前最优 g 就跳过。
- 8 邻接时禁止切角：两个正交相邻格中有一个是墙，就不能斜穿过去。
- `SearchResult` 记录 `path`、`cost`、扩展顺序 `expanded`、发现顺序 `discovered` 和耗时，可视化用的就是这些数据。

## 测试

```bash
uv run pytest
```

测试把 A* 和独立实现的参考算法对比：矩阵地图用 Bellman-Ford，顶点/边图用 Floyd-Warshall。覆盖 4/8 邻接、加权地形、
有向图和无向图、不可达、禁止切角、自定义/加权/不一致启发函数，并检查演示地图上 A* 的扩展数不超过 Dijkstra。

> 如果系统 `PYTHONPATH` 里带有第三方 pytest 插件（例如 ROS），请用 `env -u PYTHONPATH uv run pytest`。
