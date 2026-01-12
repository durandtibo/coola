r"""Implement an equality tester that uses an equality handler."""

from __future__ import annotations

__all__ = ["HandlerEqualityTester"]

from typing import TYPE_CHECKING, TypeVar

from coola.equality.tester.base import BaseEqualityTester

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig2
    from coola.equality.handler import BaseEqualityHandler

T = TypeVar("T")


class HandlerEqualityTester(BaseEqualityTester[T]):
    r"""Implement an equality tester that uses an equality handler.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig2
        >>> from coola.equality.handler import (
        ...     ObjectEqualHandler,
        ...     SameObjectHandler,
        ...     SameTypeHandler,
        ... )
        >>> from coola.equality.tester import DefaultEqualityTester
        >>> config = EqualityConfig2()
        >>> handler = SameObjectHandler()
        >>> handler.chain(SameTypeHandler()).chain(ObjectEqualHandler())
        >>> tester = DefaultEqualityTester()
        >>> tester.objects_are_equal(42, 42, config=config)
        True
        >>> tester.objects_are_equal("meow", "meov", config)
        False

        ```
    """

    def __init__(self, handler: BaseEqualityHandler) -> None:
        self._handler = handler

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def equal(self, other: object) -> bool:
        return type(other) is type(self)  # TODO(tibo): not accurate. Need to be fixed

    def objects_are_equal(
        self,
        actual: object,
        expected: object,
        config: EqualityConfig2,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)
