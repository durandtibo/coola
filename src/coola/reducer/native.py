r"""Implement a simple reducer that uses only standard library
functions."""

from __future__ import annotations

__all__ = ["NativeReducer"]

import math
from collections.abc import Sequence
from statistics import mean, median, stdev

from coola.display import InlineDisplayMixin
from coola.reducer.base import BaseReducer
from coola.utils.stats import quantile


def _has_nan(values: Sequence[float]) -> bool:
    return any(math.isnan(value) for value in values)


def _nan_last_key(value: float) -> tuple[bool, float]:
    return math.isnan(value), value


class NativeReducer(InlineDisplayMixin, BaseReducer[Sequence[float]]):
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

    def _max(self, values: Sequence[float]) -> int | float:
        # ``max`` depends on the order of the values if one is NaN.
        if _has_nan(values):
            return float("nan")
        return max(values)

    def _mean(self, values: Sequence[float]) -> float:
        return float(mean(values))

    def _median(self, values: Sequence[float]) -> int | float:
        if _has_nan(values):
            return float("nan")
        result = median(values)
        if math.isinf(result) and all(math.isfinite(value) for value in values):
            # The mean of the two middle values overflowed: halve the values first.
            return median([value / 2 for value in values]) * 2
        return result

    def _min(self, values: Sequence[float]) -> int | float:
        if _has_nan(values):
            return float("nan")
        return min(values)

    def _quantile(self, values: Sequence[float], quantiles: Sequence[float]) -> list[float]:
        if _has_nan(values):
            return [float("nan")] * len(quantiles)
        return quantile(values, quantiles)

    def _std(self, values: Sequence[float]) -> float:
        # ``statistics.stdev`` raises an error for the NaN and infinite values.
        if len(values) == 1 or not all(math.isfinite(value) for value in values):
            return float("nan")
        return stdev(values)

    def _sort(self, values: Sequence[float], *, descending: bool) -> list[int | float]:
        # The NaN values are last (first if descending), like with NumPy and PyTorch.
        return sorted(values, key=_nan_last_key, reverse=descending)
