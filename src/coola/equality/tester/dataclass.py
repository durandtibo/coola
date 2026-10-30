r"""Implement an equality tester for dataclass instances."""

from __future__ import annotations

__all__ = ["DataclassEqualityTester"]

import dataclasses
from typing import TYPE_CHECKING, Any

from coola.equality._helpers.utils import AttributeName
from coola.equality.tester.base import BaseEqualityTester

if TYPE_CHECKING:
    from collections.abc import Iterator

    from coola.equality.engine import Engine


class DataclassEqualityTester(BaseEqualityTester[Any]):
    r"""Implement an equality tester for dataclass instances.

    The two instances have the same type, and the tester compares their
    fields, in the order of declaration. The fields with
    ``compare=False`` are ignored. The fields are compared with the
    engine, so they support the tolerance and ``equal_nan``, and the
    location of a difference is the name of the field.

    Example:
        ```pycon
        >>> from dataclasses import dataclass
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import get_default_registry
        >>> @dataclass
        ... class Point:
        ...     x: float
        ...
        >>> engine = Engine(EqualityConfig())
        >>> engine.run(Point(1.0), Point(1.0))
        True
        >>> engine.run(Point(1.0), Point(2.0))
        False

        ```
    """

    def objects_are_equal(
        self, actual: Any, expected: Any, _ctx: Engine
    ) -> bool | Iterator[tuple[object, object, object]]:
        return (
            (getattr(actual, field.name), getattr(expected, field.name), AttributeName(field.name))
            for field in dataclasses.fields(actual)
            if field.compare
        )
