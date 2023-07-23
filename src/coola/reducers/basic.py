from __future__ import annotations

__all__ = ["BasicReducer"]

from collections.abc import Sequence
from statistics import mean, median, stdev

from coola.reducers.base import BaseBasicReducer
from coola.utils.stats import quantile


class BasicReducer(BaseBasicReducer):
    r"""Implements a basic reducer.

    Note that this reducer does not implement the method ``quantiles``.
    """

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def _max(self, values: Sequence[int | float]) -> int | float:
        return max(values)

    def _mean(self, values: Sequence[int | float]) -> float:
        return float(mean(values))

    def _median(self, values: Sequence[int | float]) -> int | float:
        return median(values)

    def _min(self, values: Sequence[int | float]) -> int | float:
        return min(values)

    def _quantile(self, values: Sequence[int | float], quantiles: Sequence[float]) -> list[float]:
        return quantile(values, quantiles)

    def _std(self, values: Sequence[int | float]) -> float:
        if len(values) == 1:
            return float("nan")
        return stdev(values)

    def sort(self, values: Sequence[int | float], descending: bool = False) -> list[int | float]:
        return sorted(values, reverse=descending)
