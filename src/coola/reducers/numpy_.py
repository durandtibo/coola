r"""Implement a reducer using NumPy functions.

The reducer is registered only if ``numpy`` is available.
"""

from __future__ import annotations

__all__ = ["NumpyReducer"]

from collections.abc import Sequence
from typing import TypeVar, Union
from unittest.mock import Mock

from coola.reducers.base import BaseBasicReducer
from coola.reducers.registry import ReducerRegistry
from coola.utils import check_numpy, is_numpy_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()  # pragma: no cover


T = TypeVar("T", Sequence[Union[int, float]], np.ndarray)


class NumpyReducer(BaseBasicReducer[T]):
    r"""Implement a reducer based on NumPy functions.

    Raises:
        RuntimeError: if ``numpy`` is not installed.

    Example usage:

    ```pycon

    >>> from coola.reducers import NumpyReducer
    >>> reducer = NumpyReducer()
    >>> reducer.max([-2, -1, 0, 1, 2])
    2
    >>> reducer.median([-2, -1, 0, 1, 2])
    0.0
    >>> reducer.sort([2, 1, -2, 3, 0])
    [-2, 0, 1, 2, 3]

    ```
    """

    def __init__(self) -> None:
        check_numpy()

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def _is_empty(self, values: T) -> bool:
        if isinstance(values, np.ndarray):
            return values.size == 0
        return not values

    def _max(self, values: T) -> int | float:
        return np.max(np.asarray(values)).item()

    def _mean(self, values: T) -> float:
        return np.mean(np.asarray(values)).item()

    def _median(self, values: T) -> int | float:
        return np.median(np.asarray(values)).item()

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


if is_numpy_available() and not ReducerRegistry.has_reducer("numpy"):  # pragma: no cover
    ReducerRegistry.add_reducer("numpy", NumpyReducer())
