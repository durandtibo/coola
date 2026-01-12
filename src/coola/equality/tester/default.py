r"""Implement the default equality tester."""

from __future__ import annotations

__all__ = ["DefaultEqualityTester"]

from typing import TYPE_CHECKING

from coola.equality.handler import (
    ObjectEqualHandler,
    SameObjectHandler,
    SameTypeHandler,
)
from coola.equality.tester.base import BaseEqualityTester

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig2


class DefaultEqualityTester(BaseEqualityTester[object]):
    r"""Implement a default equality tester.

    The ``==`` operator is used to test the equality between the
    objects.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig2
        >>> from coola.equality.tester import DefaultEqualityTester
        >>> config = EqualityConfig2()
        >>> tester = DefaultEqualityTester()
        >>> tester.objects_are_equal(42, 42, config=config)
        True
        >>> tester.objects_are_equal("meow", "meov", config)
        False

        ```
    """

    def __init__(self) -> None:
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(ObjectEqualHandler())

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self,
        actual: object,
        expected: object,
        config: EqualityConfig2,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)
