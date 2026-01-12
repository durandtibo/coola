r"""Define the equality comparator base class."""

from __future__ import annotations

__all__ = ["BaseEqualityTester"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig2
    from coola.equality.tester.registry import EqualityTesterRegistry

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseEqualityTester(ABC, Generic[T]):
    r"""Define the base class to implement an equality operator.

    Example:
        ```pycon
        >>> from coola.equality import EqualityConfig
        >>> from coola.equality.tester import DefaultEqualityTester
        >>> config = EqualityConfig2()
        >>> comparator = DefaultEqualityTester()
        >>> comparator.equal(42, 42, config)
        True
        >>> comparator.equal("meow", "meov", config)
        False

        ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    @abstractmethod
    def objects_are_equal(
        self,
        actual: T,
        expected: object,
        registry: EqualityTesterRegistry,
        config: EqualityConfig2,
    ) -> bool:
        r"""Indicate if two objects are equal or not.

        Args:
            actual: The actual object.
            expected: The expected object.
            registry: The registry with the equality tester to use.
            config: The equality configuration.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``.

        Example:
            ```pycon
            >>> from coola.equality import EqualityConfig2
            >>> from coola.equality.tester import DefaultEqualityTester
            >>> config = EqualityConfig2()
            >>> comparator = DefaultEqualityTester()
            >>> comparator.equal(42, 42, config)
            True
            >>> comparator.equal("meow", "meov", config)
            False

            ```
        """
