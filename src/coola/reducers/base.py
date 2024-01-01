r"""Define the reducer base class."""

from __future__ import annotations

__all__ = ["BaseReducer", "BaseBasicReducer", "EmptySequenceError"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class EmptySequenceError(Exception):
    r"""Raise when the sequence is empty because it is not possible to
    reduce an empty sequence."""


class BaseReducer(ABC):
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
    def max(self, values: Sequence[int | float]) -> int | float:
        r"""Compute the maximum value.

        Args:
            values: Specifies the values.

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
    def mean(self, values: Sequence[int | float]) -> float:
        r"""Compute the mean value.

        Args:
            values: Specifies the values.

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
    def median(self, values: Sequence[int | float]) -> int | float:
        r"""Compute the median value.

        Args:
            values: Specifies the values.

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
    def min(self, values: Sequence[int | float]) -> int | float:
        r"""Compute the minimum value.

        Args:
            values: Specifies the values.

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
    def quantile(self, values: Sequence[int | float], quantiles: Sequence[float]) -> list[float]:
        r"""Compute the quantiles.

        Args:
            values: Specifies the values.
            quantiles (sequence of float): Specifies the quantile
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
    def sort(self, values: Sequence[int | float], descending: bool = False) -> list[int | float]:
        r"""Sorts the values.

        Args:
            values: Specifies the values.
            descending: Specifies the sorting order.

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
    def std(self, values: Sequence[int | float]) -> float:
        r"""Compute the standard deviation.

        Args:
            values: Specifies the values.

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


class BaseBasicReducer(BaseReducer):
    r"""Extension of ``BaseReducer`` to check if the input sequence is
    empty before to call the reduction methods."""

    def max(self, values: Sequence[int | float]) -> int | float:
        if not values:
            msg = "Cannot compute the maximum because the summary is empty"
            raise EmptySequenceError(msg)
        return self._max(values)

    @abstractmethod
    def _max(self, values: Sequence[int | float]) -> int | float:
        r"""Compute the maximum value.

        Args:
            values: Specifies the values.

        Returns:
            The maximum value.
        """

    def mean(self, values: Sequence[int | float]) -> int | float:
        if not values:
            msg = "Cannot compute the mean because the summary is empty"
            raise EmptySequenceError(msg)
        return self._mean(values)

    @abstractmethod
    def _mean(self, values: Sequence[int | float]) -> float:
        r"""Compute the mean value.

        Args:
            values: Specifies the values.

        Returns:
            The mean value.
        """

    def median(self, values: Sequence[int | float]) -> int | float:
        if not values:
            msg = "Cannot compute the median because the summary is empty"
            raise EmptySequenceError(msg)
        return self._median(values)

    @abstractmethod
    def _median(self, values: Sequence[int | float]) -> int | float:
        r"""Compute the median value.

        Args:
            values: Specifies the values.

        Returns:
            The median value.
        """

    def min(self, values: Sequence[int | float]) -> int | float:
        if not values:
            msg = "Cannot compute the minimum because the summary is empty"
            raise EmptySequenceError(msg)
        return self._min(values)

    @abstractmethod
    def _min(self, values: Sequence[int | float]) -> int | float:
        r"""Compute the minimum value.

        Args:
            values: Specifies the values.

        Returns:
            The minimum value.
        """

    def quantile(self, values: Sequence[int | float], quantiles: Sequence[float]) -> list[float]:
        if not values:
            msg = "Cannot compute the quantiles because the summary is empty"
            raise EmptySequenceError(msg)
        return self._quantile(values, quantiles)

    @abstractmethod
    def _quantile(self, values: Sequence[int | float], quantiles: Sequence[float]) -> list[float]:
        r"""Compute the quantiles.

        Args:
            values: Specifies the values.
            quantiles: Specifies the quantile values in the
                range ``[0, 1]``.

        Returns:
            The quantiles.
        """

    def std(self, values: Sequence[int | float]) -> float:
        if not values:
            msg = "Cannot compute the standard deviation because the summary is empty"
            raise EmptySequenceError(msg)
        return self._std(values)

    @abstractmethod
    def _std(self, values: Sequence[int | float]) -> float:
        r"""Compute the standard deviation.

        Args:
            values: Specifies the values.

        Returns:
            The standard deviation.
        """
