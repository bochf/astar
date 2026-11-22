"""Search result data structure shared by both graph implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Hashable, TypeVar

Node = TypeVar("Node", bound=Hashable)


@dataclass
class SearchResult(Generic[Node]):
    path: list[Node] | None
    """Sequence of nodes from start to goal; None if unreachable."""
    cost: float
    """Total path cost; inf if unreachable."""
    expanded: list[Node] = field(default_factory=list)
    """Nodes popped and expanded, in order (used for visualization animations)."""
    discovered: list[Node] = field(default_factory=list)
    """Nodes in the order they were first added to the open list."""
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
