r"""Define the equality tester base class.

This module provides the base class for implementing custom equality
testers. Equality testers are the core abstraction in coola's equality
checking system, providing type-specific comparison logic. A tester
never recurses: it returns a verdict, or the child pairs that must all
be equal, and the engine does the rest.
"""

from __future__ import annotations

__all__ = ["BaseEqualityTester"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from coola.display import InlineDisplayMixin

if TYPE_CHECKING:
    from collections.abc import Iterator

    from coola.equality.engine import Engine

T = TypeVar("T")


class BaseEqualityTester(InlineDisplayMixin, ABC, Generic[T]):
    r"""Define the base class to implement an equality operator.

    The engine owns the invariants of a comparison (identity, strict
    type, depth, cycles, location of the first difference). A tester
    never recurses: ``objects_are_equal`` receives the two objects,
    which have the same type, and the engine, and returns

    - a verdict: ``True``, or ``ctx.fail(reason, *path)`` (which
      returns ``False``) for a leaf;
    - or the child pairs ``(actual, expected, path part)`` that must
      all be equal, produced lazily and in the order the first
      difference is wanted, for a container.

    Two testers are equal if they have the same type and the same
    attributes. A child class with parameters must override
    ``_display_fields``.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import BaseEqualityTester, get_default_registry
        >>> class PointTester(BaseEqualityTester[tuple]):
        ...     def objects_are_equal(self, actual, expected, ctx):
        ...         return ((actual[0], expected[0], 0), (actual[1], expected[1], 1))
        ...
        >>> registry = get_default_registry().copy()
        >>> registry.register(tuple, PointTester(), exist_ok=True)
        >>> Engine(EqualityConfig(registry=registry)).run((1, 2), (1, 2))
        True
        >>> Engine(EqualityConfig(registry=registry)).run((1, 2), (1, 3))
        False
        >>> PointTester() == PointTester()
        True

        ```
    """

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and vars(self) == vars(other)

    __hash__ = None

    @abstractmethod
    def objects_are_equal(
        self, actual: T, expected: T, ctx: Engine, /
    ) -> bool | Iterator[tuple[object, object, object]]:
        r"""Test two objects that have the same type.

        Args:
            actual: The actual object.
            expected: The expected object, with the same type.
            ctx: The ``Engine`` of the comparison.

        Returns:
            A verdict, or the child pairs that must all be equal.

        Example:
            ```pycon
            >>> import numpy as np
            >>> from coola.equality.config import EqualityConfig
            >>> from coola.equality.engine import Engine
            >>> from coola.equality.tester import NumpyArrayEqualityTester, get_default_registry
            >>> engine = Engine(EqualityConfig())
            >>> tester = NumpyArrayEqualityTester()
            >>> tester.objects_are_equal(np.ones(2), np.ones(2), engine)
            True
            >>> tester.objects_are_equal(np.ones(2), np.zeros(2), engine)
            False

            ```
        """
