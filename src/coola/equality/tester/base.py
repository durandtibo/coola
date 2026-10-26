r"""Define the equality tester base class.

This module provides the base class for implementing custom equality
testers that use a chain of responsibility pattern with handlers to
check if two objects are equal. Equality testers are the core
abstraction in coola's equality checking system, providing type-specific
comparison logic.
"""

from __future__ import annotations

__all__ = ["BaseEngineEqualityTester", "BaseEqualityTester"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from coola.display import InlineDisplayMixin
from coola.utils.introspection import is_same_type

if TYPE_CHECKING:
    from collections.abc import Iterator


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
        - `objects_are_equal()`: Check if two objects are equal: a chain tester runs a handler
          chain, a tester that uses the engine returns a verdict or the child pairs

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import DefaultEqualityTester, get_default_registry
        >>> engine = Engine(get_default_registry(), EqualityConfig())
        >>> tester = DefaultEqualityTester()
        >>> tester.objects_are_equal(42, 42, engine)
        True
        >>> tester.objects_are_equal("meow", "meov", engine)
        False

        ```
    """

    # ``True`` for a tester that uses the engine: its ``objects_are_equal`` receives the
    # ``Engine`` and returns a verdict or the child pairs (see ``coola.equality.engine``).
    # The other testers receive the ``EqualityConfig``. The flag is removed once all the
    # testers use the engine.
    engine_protocol: ClassVar[bool] = False

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
        expected: Any,
        config: Any,
        /,
    ) -> bool | Iterator[tuple[object, object, object]]:
        r"""Indicate if two objects are equal or not.

        A chain tester (``BaseChainEqualityTester``) delegates the
        comparison to a chain of handlers and receives the
        ``EqualityConfig``. A tester that uses the engine
        (``BaseEngineEqualityTester``) receives the ``Engine``, which has
        already checked the identity and the type of the two objects.

        Args:
            actual: The actual object to compare.
            expected: The expected object to compare against.
            config: The ``EqualityConfig`` of a chain tester, or the
                ``Engine`` of a tester that uses the engine.

        Returns:
            ``True`` if the two objects are equal according to this tester's
            logic, otherwise ``False``. A tester that uses the engine can
            also return the child pairs that must all be equal.

        Example:
            ```pycon
            >>> from coola.equality.config import EqualityConfig
            >>> from coola.equality.tester import NumpyArrayEqualityTester
            >>> import numpy as np
            >>> config = EqualityConfig()
            >>> tester = NumpyArrayEqualityTester()
            >>> tester.objects_are_equal(np.ones(2), np.ones(2), config)
            True
            >>> tester.objects_are_equal(np.ones(2), np.zeros(2), config)
            False

            ```
        """


class BaseEngineEqualityTester(InlineDisplayMixin, BaseEqualityTester[T]):
    r"""Define the base class of the testers that use the engine.

    The engine owns the invariants of a comparison (identity, strict
    type, depth, cycles, location of the first difference). A tester
    never recurses: ``objects_are_equal`` receives the two objects,
    which have the same type, and the engine, and returns

    - a verdict: ``True``, or ``ctx.fail(reason, *path)`` (which
      returns ``False``) for a leaf;
    - or the child pairs ``(actual, expected, path part)`` that must
      all be equal, produced lazily and in the order the first
      difference is wanted, for a container.

    Two testers of the same class are equal. A child class with
    parameters must override ``_display_fields`` and ``equal``.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import BaseEngineEqualityTester, get_default_registry
        >>> class PointTester(BaseEngineEqualityTester[tuple]):
        ...     def objects_are_equal(self, actual, expected, ctx):
        ...         return ((actual[0], expected[0], 0), (actual[1], expected[1], 1))
        ...
        >>> registry = get_default_registry().copy()
        >>> registry.register(tuple, PointTester(), exist_ok=True)
        >>> Engine(registry, EqualityConfig(registry=registry)).run((1, 2), (1, 2))
        True
        >>> Engine(registry, EqualityConfig(registry=registry)).run((1, 2), (1, 3))
        False

        ```
    """

    engine_protocol: ClassVar[bool] = True

    def equal(self, other: object) -> bool:
        return is_same_type(other, self)

    @abstractmethod
    def objects_are_equal(
        self, actual: T, expected: T, config: Any, /
    ) -> bool | Iterator[tuple[object, object, object]]:
        r"""Test two objects that have the same type.

        Args:
            actual: The actual object.
            expected: The expected object, with the same type.
            config: The ``Engine`` of the comparison. The testers name it
                ``ctx``.

        Returns:
            A verdict, or the child pairs that must all be equal.
        """
