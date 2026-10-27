r"""Define the reducer base class."""

from __future__ import annotations

__all__ = ["BaseReducer", "EmptySequenceError"]

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sized
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

T = TypeVar("T", bound=Sized)


class EmptySequenceError(ValueError):
    r"""Raise when the sequence is empty because it is not possible to
    reduce an empty sequence."""


class BaseReducer(ABC, Generic[T]):
    r"""Define the base class to implement a reducer.

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

    def max(self, values: T) -> int | float:
        r"""Compute the maximum value.

        Args:
            values: The values.

        Returns:
            The maximum value.

        Raises:
            EmptySequenceError: if the input sequence is empty.
            TypeError: if the input is an iterator (e.g. a generator).

        Example:
            ```pycon
            >>> from coola.reducer import TorchReducer
            >>> reducer = TorchReducer()
            >>> reducer.max([-2, -1, 0, 1, 2])
            2

            ```
        """
        return self._max(self._prepare_non_empty(values, "the maximum"))

    def mean(self, values: T) -> float:
        r"""Compute the mean value.

        Args:
            values: The values.

        Returns:
            The mean value.

        Raises:
            EmptySequenceError: if the input sequence is empty.

        Example:
            ```pycon
            >>> from coola.reducer import TorchReducer
            >>> reducer = TorchReducer()
            >>> reducer.mean([-2, -1, 0, 1, 2])
            0.0

            ```
        """
        return self._mean(self._prepare_non_empty(values, "the mean"))

    def median(self, values: T) -> int | float:
        r"""Compute the median value.

        Args:
            values: The values.

        Returns:
            The median value.

        Raises:
            EmptySequenceError: if the input sequence is empty.

        Example:
            ```pycon
            >>> from coola.reducer import TorchReducer
            >>> reducer = TorchReducer()
            >>> reducer.median([-2, -1, 0, 1, 2])
            0

            ```
        """
        return self._median(self._prepare_non_empty(values, "the median"))

    def min(self, values: T) -> int | float:
        r"""Compute the minimum value.

        Args:
            values: The values.

        Returns:
            The minimum value.

        Raises:
            EmptySequenceError: if the input sequence is empty.

        Example:
            ```pycon
            >>> from coola.reducer import TorchReducer
            >>> reducer = TorchReducer()
            >>> reducer.min([-2, -1, 0, 1, 2])
            -2

            ```
        """
        return self._min(self._prepare_non_empty(values, "the minimum"))

    def quantile(self, values: T, quantiles: Sequence[float]) -> list[float]:
        r"""Compute the quantiles.

        Args:
            values: The values.
            quantiles: The quantile values in the range ``[0, 1]``.

        Returns:
            The quantiles.

        Raises:
            ValueError: if a quantile is not in the range ``[0, 1]``
                (or is ``NaN``). This is checked before the values.
            EmptySequenceError: if the input sequence is empty.

        Example:
            ```pycon
            >>> from coola.reducer import TorchReducer
            >>> reducer = TorchReducer()
            >>> reducer.quantile([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], (0.2, 0.5, 0.9))
            [2.0, 5.0, 9.0]

            ```
        """
        _check_quantiles(quantiles)
        return self._quantile(self._prepare_non_empty(values, "the quantiles"), quantiles)

    def sort(self, values: T, *, descending: bool = False) -> list[int | float]:
        r"""Sort the values.

        Args:
            values: The values.
            descending: If ``True``, sorts in descending order.
                Defaults to ``False`` (ascending order).

        Returns:
            The sorted values. An empty input returns an empty list,
            unlike the other reduction methods.

        Example:
            ```pycon
            >>> from coola.reducer import TorchReducer
            >>> reducer = TorchReducer()
            >>> reducer.sort([2, 1, -2, 3, 0])
            [-2, 0, 1, 2, 3]
            >>> reducer.sort([2, 1, -2, 3, 0], descending=True)
            [3, 2, 1, 0, -2]
            >>> reducer.sort([])
            []

            ```
        """
        return self._sort(self._prepare(values), descending=descending)

    def std(self, values: T) -> float:
        r"""Compute the standard deviation.

        Args:
            values: The values.

        Returns:
            The standard deviation.

        Raises:
            EmptySequenceError: if the input sequence is empty.

        Example:
            ```pycon
            >>> from coola.reducer import TorchReducer
            >>> reducer = TorchReducer()
            >>> reducer.std([-2, -1, 0, 1, 2])
            1.581138...

            ```
        """
        return self._std(self._prepare_non_empty(values, "the standard deviation"))

    def _prepare(self, values: T) -> T:
        r"""Return the values as the container used by the computations.

        The default implementation returns the values unchanged. A
        backend overrides it to convert and validate the values.

        Args:
            values: The values.

        Returns:
            The values to reduce.
        """
        return values

    def _prepare_non_empty(self, values: T, name: str) -> T:
        r"""Prepare the values and check that they are not empty.

        Args:
            values: The values.
            name: The name of the reduction, used in the error message.

        Returns:
            The prepared values.

        Raises:
            TypeError: if ``values`` is an iterator (e.g. a generator),
                because the reducers need to go over the values
                several times. Convert it to a list first.
            EmptySequenceError: if the input sequence is empty.
        """
        if isinstance(values, Iterator):
            msg = f"Cannot compute {name} of an iterator, convert it to a list first"
            raise TypeError(msg)
        values = self._prepare(values)
        if len(values) == 0:
            msg = f"Cannot compute {name} because the sequence is empty"
            raise EmptySequenceError(msg)
        return values

    @abstractmethod
    def _max(self, values: T) -> int | float:
        r"""Compute the maximum value of non-empty values.

        Args:
            values: The prepared values, which are not empty.

        Returns:
            The result as a Python number.
        """

    @abstractmethod
    def _mean(self, values: T) -> float:
        r"""Compute the mean value of non-empty values.

        Args:
            values: The prepared values, which are not empty.

        Returns:
            The result as a Python number.
        """

    @abstractmethod
    def _median(self, values: T) -> int | float:
        r"""Compute the median value of non-empty values.

        Args:
            values: The prepared values, which are not empty.

        Returns:
            The result as a Python number.
        """

    @abstractmethod
    def _min(self, values: T) -> int | float:
        r"""Compute the minimum value of non-empty values.

        Args:
            values: The prepared values, which are not empty.

        Returns:
            The result as a Python number.
        """

    @abstractmethod
    def _std(self, values: T) -> float:
        r"""Compute the standard deviation of non-empty values.

        Args:
            values: The prepared values, which are not empty.

        Returns:
            The result as a Python number.
        """

    @abstractmethod
    def _quantile(self, values: T, quantiles: Sequence[float]) -> list[float]:
        r"""Compute the quantiles of non-empty values.

        Args:
            values: The prepared values, which are not empty.
            quantiles: The quantile values, which are validated: they
                are in the range ``[0, 1]``.

        Returns:
            The quantiles as Python floats.
        """

    @abstractmethod
    def _sort(self, values: T, *, descending: bool) -> list[int | float]:
        r"""Sort the values.

        Args:
            values: The prepared values, which can be empty.
            descending: If ``True``, sorts in descending order.

        Returns:
            A new list with the sorted values.
        """


def _check_quantiles(quantiles: Sequence[float]) -> None:
    r"""Check that the quantiles are in the range ``[0, 1]``.

    Args:
        quantiles: The quantile values.

    Raises:
        ValueError: if a quantile is not in the range ``[0, 1]`` or
            is ``NaN``.
    """
    for q in quantiles:
        if math.isnan(q) or not 0 <= q <= 1:
            msg = f"quantiles must be in [0, 1], got {q}"
            raise ValueError(msg)
