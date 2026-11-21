"""搜索结果数据结构，两种图实现共用。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Hashable, TypeVar

Node = TypeVar("Node", bound=Hashable)


@dataclass
class SearchResult(Generic[Node]):
    path: list[Node] | None
    """从起点到终点的节点序列；不可达时为 None。"""
    cost: float
    """路径总代价；不可达时为 inf。"""
    expanded: list[Node] = field(default_factory=list)
    """按顺序被弹出并扩展的节点（用于可视化动画）。"""
    discovered: list[Node] = field(default_factory=list)
    """按顺序首次加入开放列表的节点。"""
    algorithm: str = "A*"
    heuristic: str = ""
    elapsed_ms: float = 0.0

    @property
    def found(self) -> bool:
        return self.path is not None

    @property
    def expanded_count(self) -> int:
        return len(self.expanded)

    def summary(self) -> str:
        cost = f"{self.cost:.2f}" if self.found else "unreachable"
        return (
            f"{self.algorithm:<28} cost={cost:<10} expanded={self.expanded_count:<6} "
            f"path_len={len(self.path) if self.path else 0:<5} time={self.elapsed_ms:.1f}ms"
        )
