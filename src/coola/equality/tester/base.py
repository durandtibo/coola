r"""Define the equality tester base class."""

from __future__ import annotations

__all__ = ["BaseEqualityTester"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseEqualityTester(ABC, Generic[T]):
    r"""Define the base class to implement an equality operator.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import DefaultEqualityTester
        >>> config = EqualityConfig()
        >>> tester = DefaultEqualityTester()
        >>> tester.objects_are_equal(42, 42, config)
        True
        >>> tester.objects_are_equal("meow", "meov", config)
        False

        ```
    """

    @abstractmethod
    def objects_are_equal(
        self,
        actual: T,
        expected: object,
        config: EqualityConfig,
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
            >>> from coola.equality.config import EqualityConfig
            >>> from coola.equality.tester import DefaultEqualityTester
            >>> config = EqualityConfig()
            >>> tester = DefaultEqualityTester()
            >>> tester.objects_are_equal(42, 42, config)
            True
            >>> tester.objects_are_equal("meow", "meov", config)
            False

            ```
        """
