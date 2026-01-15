r"""Implement an equality tester that uses an equality handler."""

from __future__ import annotations

__all__ = ["HandlerEqualityTester"]

from typing import TYPE_CHECKING, TypeVar

from coola.equality.tester.base import BaseEqualityTester

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig
    from coola.equality.handler import BaseEqualityHandler

T = TypeVar("T")


class HandlerEqualityTester(BaseEqualityTester[T]):
    r"""Implement an equality tester that uses an equality handler.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import (
        ...     ObjectEqualHandler,
        ...     SameObjectHandler,
        ...     SameTypeHandler,
        ... )
        >>> from coola.equality.tester import HandlerEqualityTester
        >>> config = EqualityConfig()
        >>> handler = SameObjectHandler()
        >>> handler.chain(SameTypeHandler()).chain(ObjectEqualHandler())
        >>> tester = HandlerEqualityTester(handler)
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
        return self._handler.equal(other._handler)

    def objects_are_equal(self, actual: object, expected: object, config: EqualityConfig) -> bool:
        return self._handler.handle(actual, expected, config=config)
