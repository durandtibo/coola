r"""Implement a simple reducer that uses only standard libray
functions."""

from __future__ import annotations

__all__ = ["NativeReducer"]

from collections.abc import Sequence
from statistics import mean, median, stdev
from typing import TypeVar, Union

from coola.reducers.base import BaseBasicReducer
from coola.utils.stats import quantile

T = TypeVar("T", bound=Sequence[Union[int, float]])


class NativeReducer(BaseBasicReducer[T]):
    r"""Implement a native reducer.

    Example usage:

    ```pycon

    >>> from coola.reducers import NativeReducer
    >>> reducer = NativeReducer()
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

    def _is_empty(self, values: T) -> bool:
        return not values

    def _max(self, values: T) -> int | float:
        return max(values)

    def _mean(self, values: T) -> float:
        return float(mean(values))

    def _median(self, values: T) -> int | float:
        return median(values)

    def _min(self, values: T) -> int | float:
        return min(values)

    def _quantile(self, values: T, quantiles: Sequence[float]) -> list[float]:
        return quantile(values, quantiles)

    def _std(self, values: T) -> float:
        if len(values) == 1:
            return float("nan")
        return stdev(values)

    def sort(self, values: T, descending: bool = False) -> list[int | float]:
        return sorted(values, reverse=descending)
