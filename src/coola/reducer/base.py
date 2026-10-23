r"""Define the reducer base class."""

from __future__ import annotations

__all__ = ["BaseBasicReducer", "BaseReducer", "EmptySequenceError"]

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

T = TypeVar("T")


class EmptySequenceError(Exception):
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def quantile(self, values: T, quantiles: Sequence[float]) -> list[float]:
        r"""Compute the quantiles.

        Args:
            values: The values.
            quantiles: The quantile values in the range ``[0, 1]``.

        Returns:
            The quantiles.

        Raises:
            EmptySequenceError: if the input sequence is empty.

        Example:
            ```pycon
            >>> from coola.reducer import TorchReducer
            >>> reducer = TorchReducer()
            >>> reducer.quantile([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], (0.2, 0.5, 0.9))
            [2.0, 5.0, 9.0]

            ```
        """

    @abstractmethod
    def sort(self, values: T, descending: bool = False) -> list[int | float]:
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

    @abstractmethod
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


class BaseBasicReducer(BaseReducer[T]):
    r"""Extension of ``BaseReducer`` that validates the input sequence is
    not empty before calling the reduction methods.

    This class provides concrete implementations of all ``BaseReducer``
    abstract methods by first checking for empty sequences (raising
    ``EmptySequenceError``) and then delegating to private ``_max``,
    ``_mean``, etc. methods that subclasses must implement.
    """

    @abstractmethod
    def _is_empty(self, values: T) -> bool:
        r"""Return ``True`` if the values are empty.

        Returns:
            ``True`` if the values are empty, ``False`` otherwise.
        """

    def _check_values(self, values: T, name: str) -> None:
        r"""Check that the values can be reduced.

        Args:
            values: The values.
            name: The name of the reduction, used in the error message.

        Raises:
            TypeError: if ``values`` is an iterator (e.g. a generator),
                because the reducers need to go over the values
                several times. Convert it to a list first.
            EmptySequenceError: if the input sequence is empty.
        """
        if isinstance(values, Iterator):
            msg = f"Cannot compute {name} of an iterator, convert it to a list first"
            raise TypeError(msg)
        if self._is_empty(values):
            msg = f"Cannot compute {name} because the sequence is empty"
            raise EmptySequenceError(msg)

    def max(self, values: T) -> int | float:
        self._check_values(values, "the maximum")
        return self._max(values)

    @abstractmethod
    def _max(self, values: T) -> int | float:
        r"""Compute the maximum value.

        Args:
            values: The values.

        Returns:
            The maximum value.
        """

    def mean(self, values: T) -> int | float:
        self._check_values(values, "the mean")
        return self._mean(values)

    @abstractmethod
    def _mean(self, values: T) -> float:
        r"""Compute the mean value.

        Args:
            values: The values.

        Returns:
            The mean value.
        """

    def median(self, values: T) -> int | float:
        self._check_values(values, "the median")
        return self._median(values)

    @abstractmethod
    def _median(self, values: T) -> int | float:
        r"""Compute the median value.

        Args:
            values: The values.

        Returns:
            The median value.
        """

    def min(self, values: T) -> int | float:
        self._check_values(values, "the minimum")
        return self._min(values)

    @abstractmethod
    def _min(self, values: T) -> int | float:
        r"""Compute the minimum value.

        Args:
            values: The values.

        Returns:
            The minimum value.
        """

    def quantile(self, values: T, quantiles: Sequence[float]) -> list[float]:
        self._check_values(values, "the quantiles")
        return self._quantile(values, quantiles)

    @abstractmethod
    def _quantile(self, values: T, quantiles: Sequence[float]) -> list[float]:
        r"""Compute the quantiles.

        Args:
            values: The values.
            quantiles: The quantile values in the
                range ``[0, 1]``.

        Returns:
            The quantiles.
        """

    # Unlike the other reductions, sorting is well-defined for an empty
    # sequence (``sorted([]) == []``), so ``sort`` intentionally has no
    # concrete override here that would route it through the
    # ``_is_empty`` check guarding max/mean/median/min/quantile/std;
    # subclasses implement ``sort`` (inherited from ``BaseReducer``)
    # directly.

    def std(self, values: T) -> float:
        self._check_values(values, "the standard deviation")
        return self._std(values)

    @abstractmethod
    def _std(self, values: T) -> float:
        r"""Compute the standard deviation.

        Args:
            values: The values.

        Returns:
            The standard deviation.
        """
