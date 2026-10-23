r"""Implement a reducer using NumPy functions.

The reducer is registered only if ``numpy`` is available.
"""

from __future__ import annotations

__all__ = ["NumpyReducer"]

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

from coola.display import InlineDisplayMixin, NoArgsDisplayMixin
from coola.reducer.base import BaseBasicReducer
from coola.utils.imports import check_numpy, is_numpy_available

if TYPE_CHECKING or is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    from coola.utils.fallback.numpy import numpy as np


T = TypeVar("T", Sequence[float], np.ndarray)


class NumpyReducer(InlineDisplayMixin, NoArgsDisplayMixin, BaseBasicReducer[T]):
    r"""Implement a reducer based on NumPy functions.

    Raises:
        MissingPackageError: if ``numpy`` is not installed.

    Example:
        ```pycon
        >>> from coola.reducer import NumpyReducer
        >>> reducer = NumpyReducer()
        >>> reducer.max([-2, -1, 0, 1, 2])
        2
        >>> reducer.median([-2, -1, 0, 1, 2])
        0
        >>> reducer.sort([2, 1, -2, 3, 0])
        [-2, 0, 1, 2, 3]

        ```
    """

    def __init__(self) -> None:
        check_numpy()

    def _is_empty(self, values: T) -> bool:
        if isinstance(values, np.ndarray):
            return values.size == 0
        return not values

    def _max(self, values: T) -> int | float:
        return np.max(np.asarray(values)).item()

    def _mean(self, values: T) -> float:
        return np.mean(np.asarray(values)).item()

    def _median(self, values: T) -> int | float:
        array = np.asarray(values)
        if array.dtype.kind in "iu" and array.size % 2 == 1:
            # Like the other reducers, the median of an odd number of integers is an integer.
            return np.sort(array, axis=None)[array.size // 2].item()
        with np.errstate(over="ignore"):
            result = np.median(array).item()
        if math.isinf(result) and np.isfinite(array).all():
            # The mean of the two middle values overflowed: halve the values first.
            return np.median(array / 2).item() * 2
        return result

    def _min(self, values: T) -> int | float:
        return np.min(np.asarray(values)).item()

    def _quantile(self, values: T, quantiles: Sequence[float]) -> list[float]:
        return np.quantile(np.asarray(values), q=quantiles).tolist()

    def sort(self, values: T, descending: bool = False) -> list[int | float]:
        array = np.sort(np.asarray(values))
        if descending:
            return array[::-1].tolist()
        return array.tolist()

    def _std(self, values: T) -> float:
        if len(values) <= 1:
            return float("nan")
        return np.std(np.asarray(values), ddof=1).item()
