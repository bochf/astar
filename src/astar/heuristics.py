"""启发函数（heuristic functions）。

所有启发函数都使用统一签名::

    h(a: tuple[float, ...], b: tuple[float, ...]) -> float

其中 ``a`` 为当前节点坐标，``b`` 为目标节点坐标。对二维矩阵来说坐标是
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
    ... def half(a, b):
    ...     return (abs(a[0] - b[0]) + abs(a[1] - b[1])) / 2
    """

    def deco(func: Heuristic) -> Heuristic:
        _REGISTRY[name] = func
        _DESCRIPTIONS[name] = description or (func.__doc__ or "").strip().split("\n")[0]
        return func

    return deco


@register_heuristic("zero", "h = 0，A* 退化为 Dijkstra")
def zero(a: Point, b: Point) -> float:
    return 0.0


@register_heuristic("manhattan", "|dx| + |dy|，4 邻接网格的精确下界")
def manhattan(a: Point, b: Point) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@register_heuristic("euclidean", "直线距离，任意方向下都可采纳")
def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@register_heuristic("chebyshev", "max(|dx|, |dy|)，对角代价为 1 时的下界")
def chebyshev(a: Point, b: Point) -> float:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


@register_heuristic("octile", "max + (√2-1)·min，对角代价为 √2 时的精确下界")
def octile(a: Point, b: Point) -> float:
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)


def weighted(h: Heuristic, w: float) -> Heuristic:
    """返回 ``w * h``。w > 1 即 Weighted A*：扩展更少但不保证最优（代价 ≤ w·最优）。"""

    def _h(a: Point, b: Point) -> float:
        return w * h(a, b)

    _h.__name__ = f"{getattr(h, '__name__', 'h')}x{w:g}"
    return _h


def available_heuristics() -> dict[str, str]:
    """返回 {名称: 说明}。"""
    return dict(_DESCRIPTIONS)


def get_heuristic(spec: str | Heuristic) -> Heuristic:
    """把名称 / ``module:function`` / 可调用对象 解析为启发函数。

    - ``"manhattan"``：内置或已注册名称；
    - ``"manhattan*2"``：名称后加 ``*w`` 表示加权；
    - ``"my_pkg.my_mod:my_func"``：从模块动态导入（自定义启发函数）。
    """
    if callable(spec):
        return spec
    weight = 1.0
    if "*" in spec:
        spec, w = spec.rsplit("*", 1)
        weight = float(w)
    if spec in _REGISTRY:
        h = _REGISTRY[spec]
    elif ":" in spec:
        mod_name, func_name = spec.split(":", 1)
        h = getattr(importlib.import_module(mod_name), func_name)
        if not callable(h):
            raise TypeError(f"{spec} 不是可调用对象")
    else:
        raise KeyError(f"未知启发函数 {spec!r}，可选：{', '.join(_REGISTRY)}，或使用 module:function")
    return weighted(h, weight) if weight != 1.0 else h


def heuristic_name(h: Heuristic | str) -> str:
    if isinstance(h, str):
        return h
    for name, func in _REGISTRY.items():
        if func is h:
            return name
    return getattr(h, "__name__", "custom")
