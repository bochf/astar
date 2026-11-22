"""Custom heuristic examples.

How to run (pick one):

    # 1. Call the API directly as a script
    uv run python examples/custom_heuristic.py

    # 2. Register via a command-line plugin, then use by name
    uv run astar grid --map random --plugin examples/custom_heuristic.py \
        -H zero -H octile -H tie_breaking_octile --show

    # 3. Refer to it as module:function (the module must be on sys.path)
    PYTHONPATH=examples uv run astar grid --map maze -H custom_heuristic:manhattan_sq --out output
"""

import math

from astar import astar_grid, maps, octile, register_heuristic


@register_heuristic(
    "tie_breaking_octile", "octile × (1 + 1e-3): slightly inflated to break ties"
)
def tie_breaking_octile(current, goal):
    # Inflate by a tiny factor: in large regions of equal f, nodes closer to the goal are preferred,
    # so fewer nodes are expanded; the cost is at most 0.1% above optimal.
    return octile(current, goal) * (1 + 1e-3)


def manhattan_sq(current, goal):
    """Counterexample: squared Manhattan distance overestimates heavily; fast search, but the path may be clearly suboptimal."""
    return (abs(current[0] - goal[0]) + abs(current[1] - goal[1])) ** 2


if __name__ == "__main__":
    grid = maps.random_obstacle_grid(seed=0)
    for heuristic in [
        "zero",
        "octile",
        tie_breaking_octile,
        manhattan_sq,
        lambda current, goal: math.dist(current, goal) * 0.5,
    ]:  # any (current, goal) -> float works
        print(astar_grid(grid, heuristic=heuristic).summary())
