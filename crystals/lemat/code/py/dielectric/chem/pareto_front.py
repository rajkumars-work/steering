"""
Pareto front ranking utilities.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple


class ParetoFront:
    """
    Maintain a set of (id, bandgap, eps_0) points and compute Pareto ranks.

    Rank definitions:
    - Front (rank 0): non-dominated points.
    - Rank 1: points dominated only by the front.
    - Rank 2: points dominated only by rank 1, and so on.
    """

    def __init__(self, maximize_bandgap: bool = True, maximize_eps_0: bool = True) -> None:
        self.maximize_bandgap = maximize_bandgap
        self.maximize_eps_0 = maximize_eps_0
        self._items: Dict[str, Tuple[float, float]] = {}

    def add(self, item_id: str, bandgap: float, eps_0: float) -> None:
        """Add or replace a point."""
        self._items[item_id] = (float(bandgap), float(eps_0))

    def add_many(self, items: Iterable[Tuple[str, float, float]]) -> None:
        """Add multiple points."""
        for item_id, bandgap, eps_0 in items:
            self.add(item_id, bandgap, eps_0)

    def pareto(self, rank: Optional[int] = None) -> Dict[str, Tuple[float, float]]:
        """
        Return the Pareto front (rank 0) or a specific rank.

        Parameters
        ----------
        rank : int | None
            None returns the front. rank=1 returns points dominated only by the front,
            rank=2 returns points dominated only by rank-1, etc.

        Returns
        -------
        dict
            {id: (bandgap, eps_0)} for the requested rank. Empty if not found.
        """
        if not self._items:
            return {}

        if rank is not None and rank < 0:
            raise ValueError("rank must be >= 0 or None")

        remaining: Dict[str, Tuple[float, float]] = dict(self._items)

        if rank is None:
            return self._front(remaining)

        current_rank = 0
        while remaining:
            front = self._front(remaining)
            if current_rank == rank:
                return front
            for key in front:
                remaining.pop(key, None)
            current_rank += 1

        return {}

    def _front(self, items: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        """Compute the non-dominated front for the given items."""
        front: Dict[str, Tuple[float, float]] = {}
        for item1, (bg1, eps1) in items.items():
            is_dominated = False
            for item2, (bg2, eps2) in items.items():
                if item1 == item2:
                    continue
                if self._dominates(bg2, eps2, bg1, eps1):
                    is_dominated = True
                    break
            if not is_dominated:
                front[item1] = (bg1, eps1)
        return front

    def _dominates(self, bg_a: float, eps_a: float, bg_b: float, eps_b: float) -> bool:
        """Return True if point A dominates point B."""
        if self.maximize_bandgap:
            bg_better = bg_a > bg_b
            bg_equal = bg_a == bg_b
        else:
            bg_better = bg_a < bg_b
            bg_equal = bg_a == bg_b

        if self.maximize_eps_0:
            eps_better = eps_a > eps_b
            eps_equal = eps_a == eps_b
        else:
            eps_better = eps_a < eps_b
            eps_equal = eps_a == eps_b

        return (
            (bg_better or bg_equal)
            and (eps_better or eps_equal)
            and (bg_better or eps_better)
        )
