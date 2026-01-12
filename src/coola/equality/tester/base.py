r"""Define the equality tester base class."""

from __future__ import annotations

__all__ = ["BaseEqualityTester"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig2

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseEqualityTester(ABC, Generic[T]):
    r"""Define the base class to implement an equality operator.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig2
        >>> from coola.equality.tester import DefaultEqualityTester
        >>> config = EqualityConfig2()
        >>> tester = DefaultEqualityTester()
        >>> tester.objects_are_equal(42, 42, config)
        True
        >>> tester.objects_are_equal("meow", "meov", config)
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
        config: EqualityConfig2,
    ) -> bool:
        r"""Indicate if two objects are equal or not.

        Args:
            actual: The actual object.
            expected: The expected object.
            config: The equality configuration.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``.

        Example:
            ```pycon
            >>> from coola.equality.config import EqualityConfig2
            >>> from coola.equality.tester import DefaultEqualityTester
            >>> config = EqualityConfig2()
            >>> tester = DefaultEqualityTester()
            >>> tester.objects_are_equal(42, 42, config)
            True
            >>> tester.objects_are_equal("meow", "meov", config)
            False

            ```
        """
