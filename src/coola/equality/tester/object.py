r"""Implement an equality tester for objects compared by their
attributes."""

from __future__ import annotations

__all__ = ["ObjectAttributesEqualityTester"]

from typing import TYPE_CHECKING, Any

from coola.equality.handler import (
    ObjectSameAttributesHandler,
    SameObjectHandler,
    SameTypeHandler,
    TrueHandler,
    create_chain,
)
from coola.equality.tester.chain import BaseChainEqualityTester

if TYPE_CHECKING:
    from coola.equality.handler import BaseEqualityHandler


class ObjectAttributesEqualityTester(BaseChainEqualityTester[Any]):
    r"""Implement an equality tester for objects without value
    semantics.

    A plain class, an exception or a ``functools.partial`` is equal to
    another object only if it is the same object, unless the class
    defines ``__eq__``. This tester compares the attributes instead:
    the entries of ``vars``, the slots, the ``args`` of an exception
    and the content of a ``functools.partial`` (see
    ``get_object_attributes``). The attributes are compared through the
    registry, so the tolerance and ``equal_nan`` apply to them.

    It is not registered by default, because it changes the result for
    the classes that rely on the identity: register it for the classes
    that must be compared by their attributes. The handler chain:
    1. SameObjectHandler: Check for object identity
    2. SameTypeHandler: Verify both have the same type
    3. ObjectSameAttributesHandler: Compare the attributes
    4. TrueHandler: Return True if all checks pass

    Example:
        ```pycon
        >>> from coola.equality import objects_are_equal
        >>> from coola.equality.tester import (
        ...     DefaultEqualityTester,
        ...     EqualityTesterRegistry,
        ...     ObjectAttributesEqualityTester,
        ... )
        >>> class Point:
        ...     def __init__(self, x):
        ...         self.x = x
        ...
        >>> registry = EqualityTesterRegistry({object: DefaultEqualityTester()})
        >>> objects_are_equal(Point(1), Point(1), registry=registry)
        False
        >>> registry.register(Point, ObjectAttributesEqualityTester())
        >>> objects_are_equal(Point(1), Point(1), registry=registry)
        True
        >>> objects_are_equal(Point(1), Point(2), registry=registry)
        False

        ```
    """

    def _create_handler(self) -> BaseEqualityHandler:
        return create_chain(
            SameObjectHandler(),
            SameTypeHandler(),
            ObjectSameAttributesHandler(),
            TrueHandler(),
        )
