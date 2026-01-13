r"""Implement an equality tester."""

from __future__ import annotations

__all__ = ["EqualEqualityTester"]

from typing import TYPE_CHECKING

from coola.equality.handler import (
    EqualHandler,
    SameObjectHandler,
    SameTypeHandler,
)
from coola.equality.tester.base import BaseEqualityTester

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


class EqualEqualityTester(BaseEqualityTester[object]):
    r"""Implement an equality tester for objects with equal method.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import EqualEqualityTester
        >>> class MyFloat:
        ...     def __init__(self, value: float) -> None:
        ...         self._value = float(value)
        ...     def equal(self, other: object) -> bool:
        ...         if type(other) is not type(self):
        ...             return False
        ...         return self._value == other._value
        ...
        >>> config = EqualityConfig()
        >>> tester = EqualEqualityTester()
        >>> tester.objects_are_equal(MyFloat(42), MyFloat(42), config=config)
        True
        >>> tester.objects_are_equal(MyFloat(42), MyFloat(1), config=config)
        False

        ```
    """

    def __init__(self) -> None:
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(EqualHandler())

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self,
        actual: object,
        expected: object,
        config: EqualityConfig,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)
