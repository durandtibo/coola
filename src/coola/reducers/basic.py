r"""Implement a simple reducer that uses only standard libray
functions."""

from __future__ import annotations

__all__ = ["BasicReducer"]

from statistics import mean, median, stdev
from typing import TYPE_CHECKING

from coola.reducers.base import BaseBasicReducer
from coola.utils.stats import quantile

if TYPE_CHECKING:
    from collections.abc import Sequence


class BasicReducer(BaseBasicReducer):
    r"""Implement a basic reducer.

    Example usage:

    ```pycon
    >>> from coola.reducers import BasicReducer
    >>> reducer = BasicReducer()
    >>> reducer.max([-2, -1, 0, 1, 2])
    2
    >>> reducer.median([-2, -1, 0, 1, 2])
    0
    >>> reducer.sort([2, 1, -2, 3, 0])
    [-2, 0, 1, 2, 3]

    ```
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
