r"""Implement scalar equality comparators."""

from __future__ import annotations

__all__ = ["ScalarEqualityTester"]

from typing import TYPE_CHECKING

from coola.equality.handler import (
    NanEqualHandler,
    SameObjectHandler,
    SameTypeHandler,
    ScalarEqualHandler,
)
from coola.equality.tester.base import BaseEqualityTester

if TYPE_CHECKING:
    from coola.equality import EqualityConfig2


class ScalarEqualityTester(BaseEqualityTester[float]):
    r"""Implement a default equality comparator.

    Example:
        ```pycon
        >>> from coola.equality import EqualityConfig2
        >>> from coola.equality.tester import ScalarEqualityTester
        >>> config = EqualityConfig2()
        >>> comparator = ScalarEqualityTester()
        >>> comparator.objects_are_equal(42.0, 42.0, config)
        True
        >>> comparator.objects_are_equal(42.0, 1.0, config)
        False

        ```
    """

    def __init__(self) -> None:
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(NanEqualHandler()).chain(ScalarEqualHandler())

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self,
        actual: float,
        expected: object,
        config: EqualityConfig2,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)
