r"""Implement an equality tester for objects compared by their
attributes."""

from __future__ import annotations

__all__ = ["ObjectAttributesEqualityTester"]

from typing import TYPE_CHECKING, Any

from coola.equality._helpers.object import get_object_attributes
from coola.equality._helpers.utils import AttributeName
from coola.equality.tester.base import BaseEqualityTester

if TYPE_CHECKING:
    from collections.abc import Iterator

    from coola.equality.engine import Engine


class ObjectAttributesEqualityTester(BaseEqualityTester[Any]):
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
    that must be compared by their attributes. The two objects have the
    same type, the same attribute names and equal values. The location
    of a difference is the name of the attribute.

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

    def objects_are_equal(
        self, actual: Any, expected: Any, ctx: Engine
    ) -> bool | Iterator[tuple[object, object, object]]:
        attributes_a = get_object_attributes(actual)
        attributes_e = get_object_attributes(expected)
        if attributes_a.keys() != attributes_e.keys():
            return ctx.fail(
                lambda: (
                    f"objects have different attributes: {sorted(attributes_a)} "
                    f"vs {sorted(attributes_e)}"
                )
            )
        return (
            (value, attributes_e[name], AttributeName(name)) for name, value in attributes_a.items()
        )
