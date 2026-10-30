r"""Implement a reducer using NumPy functions.

The reducer is registered only if ``numpy`` is available.
"""

from __future__ import annotations

__all__ = ["NumpyReducer"]

import math
from typing import TYPE_CHECKING

from coola.display import InlineDisplayMixin
from coola.reducer.base import BaseReducer
from coola.utils.imports import check_numpy, is_numpy_available

if TYPE_CHECKING:
    from collections.abc import Sequence

if TYPE_CHECKING or is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    from coola.utils.fallback.numpy import numpy as np


class NumpyReducer(InlineDisplayMixin, BaseReducer["Sequence[float] | np.ndarray"]):
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

    def _prepare(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        r"""Convert the values to a one-dimensional NumPy array.

        Args:
            values: The values: a sequence or a NumPy array.

        Returns:
            The values as a one-dimensional array of booleans, integers
                or floats.

        Raises:
            TypeError: If the values are nested sequences of different
                lengths or have another dtype (an integer beyond 64
                bits, a complex number, ...).
            ValueError: If the values are not one-dimensional.
        """
        try:
            array = np.asarray(values)
        except ValueError:  # ragged nested sequences
            self._raise_unsupported("nested sequences of different lengths")
        if array.dtype.kind not in "biuf":
            self._raise_unsupported(
                f"values of dtype {array.dtype} (an integer beyond 64 bits, a complex number or "
                "another type)",
                "use NativeReducer for arbitrary precision integers",
            )
        if array.ndim != 1:
            self._raise_ndim(array.ndim)
        return array

    def _max(self, values: np.ndarray) -> int | float:
        r"""Compute the maximum value with NumPy.

        Args:
            values: The prepared values, which are not empty.

        Returns:
            The result as a Python number.
        """
        return values.max().item()

    def _mean(self, values: np.ndarray) -> float:
        r"""Compute the mean value with NumPy.

        Args:
            values: The prepared values, which are not empty.

        Returns:
            The result as a Python float.
        """
        return values.mean().item()

    def _median(self, values: np.ndarray) -> int | float:
        r"""Compute the median value with NumPy.

        Args:
            values: The prepared values, which are not empty.

        Returns:
            The result as a Python number.
        """
        if values.dtype.kind in "iu" and values.size % 2 == 1:
            # Like the other reducers, the median of an odd number of integers is an integer.
            return np.sort(values)[values.size // 2].item()
        with np.errstate(over="ignore"):
            result = np.median(values).item()
        if math.isinf(result) and np.isfinite(values).all():
            # The mean of the two middle values overflowed: halve the values first.
            return np.median(values / 2).item() * 2
        return result

    def _min(self, values: np.ndarray) -> int | float:
        r"""Compute the minimum value with NumPy.

        Args:
            values: The prepared values, which are not empty.

        Returns:
            The result as a Python number.
        """
        return values.min().item()

    def _quantile(self, values: np.ndarray, quantiles: Sequence[float]) -> list[float]:
        r"""Compute the quantiles with NumPy.

        Args:
            values: The prepared values, which are not empty.
            quantiles: The quantile values, which are in the range ``[0,
                1]``.

        Returns:
            The quantiles as Python floats.
        """
        if values.dtype.kind == "b":
            values = values.astype(np.float64)  # ``np.quantile`` does not support booleans
        return np.quantile(values, q=quantiles).tolist()

    def _sort(self, values: np.ndarray, *, descending: bool) -> list[int | float]:
        r"""Sort the values with NumPy.

        Args:
            values: The prepared values, which can be empty.
            descending: If ``True``, sorts in descending order.

        Returns:
            A new list with the sorted values. The ``NaN`` values are
                last (first if descending).
        """
        array = np.sort(values)
        if descending:
            return array[::-1].tolist()
        return array.tolist()

    def _std(self, values: np.ndarray) -> float:
        r"""Compute the sample standard deviation with NumPy.

        Args:
            values: The prepared values, which are not empty.

        Returns:
            The result as a Python float, ``NaN`` for a single value.
        """
        if values.size <= 1:
            return float("nan")
        return np.std(values, ddof=1).item()
