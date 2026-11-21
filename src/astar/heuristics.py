"""启发函数（heuristic functions）。

所有启发函数都使用统一签名::

    heuristic(current: tuple[float, ...], goal: tuple[float, ...]) -> float

其中 ``current`` 为当前节点坐标，``goal`` 为目标节点坐标。对二维矩阵来说坐标是
``(row, col)``；对顶点/边图来说坐标是顶点的 ``(x, y)`` 位置。因此同一个启发
函数可以同时用于两种数据结构。

内置：

=============  ==========================================  =====================
名称            公式                                         适用场景
=============  ==========================================  =====================
``zero``       0                                           退化为 Dijkstra
``manhattan``  |dx| + |dy|                                 4 邻接网格
``euclidean``  sqrt(dx² + dy²)                             任意方向移动 / 平面图
``chebyshev``  max(|dx|, |dy|)                             8 邻接且对角代价为 1
``octile``     max + (√2 - 1)·min                          8 邻接且对角代价为 √2
=============  ==========================================  =====================

自定义启发函数有三种方式：

1. 直接把可调用对象传给 ``astar(..., heuristic=my_func)``；
2. 用 :func:`register_heuristic` 装饰器注册名称，之后用字符串引用；
3. 命令行中使用 ``module.path:function`` 形式（见 :func:`get_heuristic`）。
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Sequence

Point = Sequence[float]
Heuristic = Callable[[Point, Point], float]

_REGISTRY: dict[str, Heuristic] = {}
_DESCRIPTIONS: dict[str, str] = {}


def register_heuristic(name: str, description: str = "") -> Callable[[Heuristic], Heuristic]:
    """装饰器：把函数注册为具名启发函数，之后可通过字符串名称使用。

    >>> @register_heuristic("half_manhattan", "曼哈顿距离的一半")
    ... def half(current, goal):
    ...     return (abs(current[0] - goal[0]) + abs(current[1] - goal[1])) / 2
    """

    def decorator(heuristic: Heuristic) -> Heuristic:
        _REGISTRY[name] = heuristic
        _DESCRIPTIONS[name] = description or (heuristic.__doc__ or "").strip().split("\n")[0]
        return heuristic

    return decorator


@register_heuristic("zero", "h = 0，A* 退化为 Dijkstra")
def zero(current: Point, goal: Point) -> float:
    return 0.0


@register_heuristic("manhattan", "|dx| + |dy|，4 邻接网格的精确下界")
def manhattan(current: Point, goal: Point) -> float:
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1])


@register_heuristic("euclidean", "直线距离，任意方向下都可采纳")
def euclidean(current: Point, goal: Point) -> float:
    return math.hypot(current[0] - goal[0], current[1] - goal[1])


@register_heuristic("chebyshev", "max(|dx|, |dy|)，对角代价为 1 时的下界")
def chebyshev(current: Point, goal: Point) -> float:
    return max(abs(current[0] - goal[0]), abs(current[1] - goal[1]))


@register_heuristic("octile", "max + (√2-1)·min，对角代价为 √2 时的精确下界")
def octile(current: Point, goal: Point) -> float:
    delta_0, delta_1 = abs(current[0] - goal[0]), abs(current[1] - goal[1])
    return max(delta_0, delta_1) + (math.sqrt(2) - 1) * min(delta_0, delta_1)


def weighted(base_heuristic: Heuristic, weight: float) -> Heuristic:
    """返回 ``weight * base_heuristic``。weight > 1 即 Weighted A*：扩展更少但不保证最优（代价 ≤ weight·最优）。"""

    def weighted_heuristic(current: Point, goal: Point) -> float:
        return weight * base_heuristic(current, goal)

    weighted_heuristic.__name__ = f"{getattr(base_heuristic, '__name__', 'heuristic')}x{weight:g}"
    return weighted_heuristic


def available_heuristics() -> dict[str, str]:
    """返回 {名称: 说明}。"""
    return dict(_DESCRIPTIONS)


def get_heuristic(spec: str | Heuristic) -> Heuristic:
    """把名称 / ``module:function`` / 可调用对象 解析为启发函数。

    - ``"manhattan"``：内置或已注册名称；
    - ``"manhattan*2"``：名称后加 ``*权重`` 表示加权；
    - ``"my_pkg.my_mod:my_func"``：从模块动态导入（自定义启发函数）。
    """
    if callable(spec):
        return spec
    weight = 1.0
    if "*" in spec:
        spec, weight_text = spec.rsplit("*", 1)
        weight = float(weight_text)
    if spec in _REGISTRY:
        heuristic = _REGISTRY[spec]
    elif ":" in spec:
        module_name, function_name = spec.split(":", 1)
        heuristic = getattr(importlib.import_module(module_name), function_name)
        if not callable(heuristic):
            raise TypeError(f"{spec} 不是可调用对象")
    else:
        raise KeyError(f"未知启发函数 {spec!r}，可选：{', '.join(_REGISTRY)}，或使用 module:function")
    return weighted(heuristic, weight) if weight != 1.0 else heuristic


def heuristic_name(heuristic: Heuristic | str) -> str:
    if isinstance(heuristic, str):
        return heuristic
    for name, registered in _REGISTRY.items():
        if registered is heuristic:
            return name
    return getattr(heuristic, "__name__", "custom")
