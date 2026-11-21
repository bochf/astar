"""自定义启发函数示例。

运行方式（任选其一）：

    # 1. 作为脚本直接调用 API
    uv run python examples/custom_heuristic.py

    # 2. 通过命令行插件注册后按名称使用
    uv run astar grid --map random --plugin examples/custom_heuristic.py \
        -H zero -H octile -H tie_breaking_octile --show

    # 3. 通过 module:function 引用（需要模块在 sys.path 中）
    PYTHONPATH=examples uv run astar grid --map maze -H custom_heuristic:manhattan_sq --out output
"""

import math

from astar import astar_grid, maps, octile, register_heuristic


@register_heuristic("tie_breaking_octile", "octile × (1 + 1e-3)：轻微放大以打破平局")
def tie_breaking_octile(a, b):
    # 放大极小倍数：在大片等 f 值的区域里更偏向靠近目标的节点，扩展更少；
    # 代价至多比最优多 0.1%。
    return octile(a, b) * (1 + 1e-3)


def manhattan_sq(a, b):
    """反例：平方曼哈顿距离严重高估，搜索很快但路径可能明显非最优。"""
    return (abs(a[0] - b[0]) + abs(a[1] - b[1])) ** 2


if __name__ == "__main__":
    grid = maps.random_obstacle_grid(seed=0)
    for h in ["zero", "octile", tie_breaking_octile, manhattan_sq,
              lambda a, b: math.dist(a, b) * 0.5]:  # lambda：任何 (a, b) -> float 都可以
        print(astar_grid(grid, heuristic=h).summary())
