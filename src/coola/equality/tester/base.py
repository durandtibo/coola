r"""Define the equality tester base class.

This module provides the base class for implementing custom equality testers
that use a chain of responsibility pattern with handlers to check if two objects
are equal. Equality testers are the core abstraction in coola's equality checking
system, providing type-specific comparison logic.
"""

from __future__ import annotations

__all__ = ["BaseEqualityTester"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


T = TypeVar("T")


class BaseEqualityTester(ABC, Generic[T]):
    r"""Define the base class to implement an equality operator.

    This abstract base class defines the interface for all equality testers in coola.
    Equality testers are responsible for comparing objects of a specific type using
    a chain of handlers that implement the chain of responsibility pattern.

    The generic type parameter T indicates the primary type this tester is designed
    to handle, though the actual implementation may handle related types as well.

    Subclasses must implement:
        - `equal()`: Check if another tester is of the same type
        - `objects_are_equal()`: Check if two objects are equal using handler chain

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
    def equal(self, other: object) -> bool:
        r"""Indicate if two equality testers are equal (same type).

        This method checks if another object is an equality tester of the same
        type as this one. It's used for comparing equality tester instances
        themselves, not the objects they test.

        Args:
            other: The other object to compare against.

        Returns:
            ``True`` if the other object is an equality tester of the same type,
            otherwise ``False``.

        Example:
            ```pycon
            >>> from coola.equality.tester import DefaultEqualityTester, MappingEqualityTester
            >>> tester1 = DefaultEqualityTester()
            >>> tester2 = DefaultEqualityTester()
            >>> tester3 = MappingEqualityTester()
            >>> tester1.equal(tester2)
            True
            >>> tester1.equal(tester3)
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

        This method delegates equality checking to a chain of handlers that
        implement various checks (e.g., same object, same type, same values).
        The handler chain is typically set up in the tester's __init__ method.

        Args:
            actual: The actual object to compare.
            expected: The expected object to compare against.
            config: The equality configuration controlling comparison behavior
                (e.g., tolerance for floating point, whether to treat NaN as equal).

        Returns:
            ``True`` if the two objects are equal according to this tester's
            logic and the provided configuration, otherwise ``False``.

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
