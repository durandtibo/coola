from __future__ import annotations

__all__ = ["NumpyReducer"]

from collections.abc import Sequence
from unittest.mock import Mock

from coola.reducers.base import BaseBasicReducer
from coola.reducers.registry import ReducerRegistry
from coola.utils import check_numpy, is_numpy_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()  # pragma: no cover


class NumpyReducer(BaseBasicReducer):
    r"""Implement a reducer based on NumPy functions.

    Raises:
        RuntimeError if ``numpy`` is not installed.
    """

    def __init__(self) -> None:
        check_numpy()

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def _max(self, values: Sequence[int | float]) -> int | float:
        return np.max(np.asarray(values)).item()

    def _mean(self, values: Sequence[int | float]) -> float:
        return np.mean(np.asarray(values)).item()

    def _median(self, values: Sequence[int | float]) -> int | float:
        return np.median(np.asarray(values)).item()

    def _min(self, values: Sequence[int | float]) -> int | float:
        return np.min(np.asarray(values)).item()

    def _quantile(self, values: Sequence[int | float], quantiles: Sequence[float]) -> list[float]:
        return np.quantile(np.asarray(values), q=quantiles).tolist()

    def sort(self, values: Sequence[int | float], descending: bool = False) -> list[int | float]:
        array = np.sort(np.asarray(values))
        if descending:
            return array[::-1].tolist()
        return array.tolist()

    def _std(self, values: Sequence[int | float]) -> float:
        return np.std(np.asarray(values), ddof=1).item()


if is_numpy_available():  # pragma: no cover
    if not ReducerRegistry.has_reducer("numpy"):
        ReducerRegistry.add_reducer("numpy", NumpyReducer())
