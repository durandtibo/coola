r"""Define the equality comparator base class."""

from __future__ import annotations

__all__ = ["BaseEqualityComparator"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseEqualityComparator(ABC, Generic[T]):
    r"""Define the base class to implement an equality operator.

    Example usage:

    ```pycon

    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.comparators import DefaultEqualityComparator
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> comparator = DefaultEqualityComparator()
    >>> comparator.equal(42, 42, config)
    True
    >>> comparator.equal("meow", "meov", config)
    False

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    @abstractmethod
    def clone(self) -> BaseEqualityComparator:
        r"""Return a copy of the equality operator.

        Returns:
            A copy of the equality operator.

        Example usage:

        ```pycon
        >>> from coola.equality.comparators import DefaultEqualityComparator
        >>> op = DefaultEqualityComparator()
        >>> op_cloned = op.clone()
        >>> op_cloned
        DefaultEqualityComparator()
        >>> op is op_cloned
        False

        ```
        """

    @abstractmethod
    def equal(self, actual: T, expected: Any, config: EqualityConfig) -> bool:
        r"""Indicate if two objects are equal or not.

        Args:
            actual: The actual input.
            expected: The expected input.
            config: The equality configuration.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``.

        Example usage:

        ```pycon
        >>> from coola.equality import EqualityConfig
        >>> from coola.equality.comparators import DefaultEqualityComparator
        >>> from coola.equality.testers import EqualityTester
        >>> config = EqualityConfig(tester=EqualityTester())
        >>> comparator = DefaultEqualityComparator()
        >>> comparator.equal(42, 42, config)
        True
        >>> comparator.equal("meow", "meov", config)
        False

        ```
        """
