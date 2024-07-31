r"""Define the reducer base class."""

from __future__ import annotations

__all__ = ["BaseBasicReducer", "BaseReducer", "EmptySequenceError"]

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar, Union

T = TypeVar("T", bound=Sequence[Union[int, float]])


class EmptySequenceError(Exception):
    r"""Raise when the sequence is empty because it is not possible to
    reduce an empty sequence."""


class BaseReducer(ABC, Generic[T]):
    r"""Define the base class to implement a reducer.

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

    @abstractmethod
    def max(self, values: T) -> int | float:
        r"""Compute the maximum value.

        Args:
            values: The values.

        Returns:
            The maximum value.

        Raises:
            EmptySequenceError: if the input sequence is empty.

        Example usage:

        .. code-block:: pycon

            >>> from coola.reducers import TorchReducer
            >>> reducer = TorchReducer()
            >>> reducer.max([-2, -1, 0, 1, 2])
            2
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

        Example usage:

        .. code-block:: pycon

            >>> from coola.reducers import TorchReducer
            >>> reducer = TorchReducer()
            >>> reducer.mean([-2, -1, 0, 1, 2])
            0.0
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

        Example usage:

        .. code-block:: pycon

            >>> from coola.reducers import TorchReducer
            >>> reducer = TorchReducer()
            >>> reducer.median([-2, -1, 0, 1, 2])
            0
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

        Example usage:

        .. code-block:: pycon

            >>> from coola.reducers import TorchReducer
            >>> reducer = TorchReducer()
            >>> reducer.min([-2, -1, 0, 1, 2])
            -2
        """

    @abstractmethod
    def quantile(self, values: T, quantiles: Sequence[float]) -> list[float]:
        r"""Compute the quantiles.

        Args:
            values: The values.
            quantiles (sequence of float): The quantile
                values in the range ``[0, 1]``.

        Returns:
            The quantiles.

        Raises:
            EmptySequenceError: if the input sequence is empty.

        Example usage:

        .. code-block:: pycon

            >>> from coola.reducers import TorchReducer
            >>> reducer = TorchReducer()
            >>> reducer.quantile([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], (0.2, 0.5, 0.9))
            [2.0, 5.0, 9.0]
        """

    @abstractmethod
    def sort(self, values: T, descending: bool = False) -> list[int | float]:
        r"""Sorts the values.

        Args:
            values: The values.
            descending: The sorting order.

        Returns:
            The sorted values.

        Example usage:

        .. code-block:: pycon

            >>> from coola.reducers import TorchReducer
            >>> reducer = TorchReducer()
            >>> reducer.sort([2, 1, -2, 3, 0])
            [-2, 0, 1, 2, 3]
            >>> reducer.sort([2, 1, -2, 3, 0], descending=True)
            [3, 2, 1, 0, -2]
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

        Example usage:

        .. code-block:: pycon

            >>> from coola.reducers import TorchReducer
            >>> reducer = TorchReducer()
            >>> reducer.std([-2, -1, 0, 1, 2])
            1.581138...
        """


class BaseBasicReducer(BaseReducer[T]):
    r"""Extension of ``BaseReducer`` to check if the input sequence is
    empty before to call the reduction methods."""

    @abstractmethod
    def _is_empty(self, values: T) -> bool:
        r"""Return ``True`` if the values are empty.

        Returns:
            ``True`` if the values are empty, ``False`` otherwise.
        """

    def max(self, values: T) -> int | float:
        if self._is_empty(values):
            msg = "Cannot compute the maximum because the summary is empty"
            raise EmptySequenceError(msg)
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
        if self._is_empty(values):
            msg = "Cannot compute the mean because the summary is empty"
            raise EmptySequenceError(msg)
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
        if self._is_empty(values):
            msg = "Cannot compute the median because the summary is empty"
            raise EmptySequenceError(msg)
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
        if self._is_empty(values):
            msg = "Cannot compute the minimum because the summary is empty"
            raise EmptySequenceError(msg)
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
        if self._is_empty(values):
            msg = "Cannot compute the quantiles because the summary is empty"
            raise EmptySequenceError(msg)
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

    def std(self, values: T) -> float:
        if self._is_empty(values):
            msg = "Cannot compute the standard deviation because the summary is empty"
            raise EmptySequenceError(msg)
        return self._std(values)

    @abstractmethod
    def _std(self, values: T) -> float:
        r"""Compute the standard deviation.

        Args:
            values: The values.

        Returns:
            The standard deviation.
        """
