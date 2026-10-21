r"""Implement a simple reducer that uses only standard library
functions."""

from __future__ import annotations

__all__ = ["NativeReducer"]

import math
from collections.abc import Sequence
from statistics import mean, median, stdev
from typing import TypeVar

from coola.display import InlineDisplayMixin, NoArgsDisplayMixin
from coola.reducer.base import BaseBasicReducer
from coola.utils.stats import quantile

T = TypeVar("T", bound=Sequence[float])


def _has_nan(values: Sequence[float]) -> bool:
    return any(value != value for value in values)  # noqa: PLR0124


def _nan_last_key(value: float) -> tuple[bool, float]:
    return value != value, value  # noqa: PLR0124


class NativeReducer(InlineDisplayMixin, NoArgsDisplayMixin, BaseBasicReducer[T]):
    r"""Implement a native reducer.

    Example:
        ```pycon
        >>> from coola.reducer import NativeReducer
        >>> reducer = NativeReducer()
        >>> reducer.max([-2, -1, 0, 1, 2])
        2
        >>> reducer.median([-2, -1, 0, 1, 2])
        0
        >>> reducer.sort([2, 1, -2, 3, 0])
        [-2, 0, 1, 2, 3]
        >>> reducer.max([1.0, float("nan"), 2.0])
        nan

        ```
    """

    def _is_empty(self, values: T) -> bool:
        return not values

    def _max(self, values: T) -> int | float:
        # ``max`` depends on the order of the values if one is NaN.
        if _has_nan(values):
            return float("nan")
        return max(values)

    def _mean(self, values: T) -> float:
        return float(mean(values))

    def _median(self, values: T) -> int | float:
        if _has_nan(values):
            return float("nan")
        return median(values)

    def _min(self, values: T) -> int | float:
        if _has_nan(values):
            return float("nan")
        return min(values)

    def _quantile(self, values: T, quantiles: Sequence[float]) -> list[float]:
        if _has_nan(values):
            return [float("nan")] * len(quantiles)
        return quantile(values, quantiles)

    def _std(self, values: T) -> float:
        # ``statistics.stdev`` raises an error for the NaN and infinite values.
        if len(values) == 1 or not all(math.isfinite(value) for value in values):
            return float("nan")
        return stdev(values)

    def sort(self, values: T, descending: bool = False) -> list[int | float]:
        # The NaN values are last (first if descending), like with NumPy and PyTorch.
        return sorted(values, key=_nan_last_key, reverse=descending)
