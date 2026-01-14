r"""Implement equality testers for objects with equal methods.

This module provides specialized testers for objects that implement an ``equal``
method, supporting both standard equality and NaN-aware equality comparisons.
"""

from __future__ import annotations

__all__ = ["EqualEqualityTester", "EqualNanEqualityTester"]

from typing import TYPE_CHECKING

from coola.equality.handler import (
    EqualHandler,
    EqualNanHandler,
    SameObjectHandler,
    SameTypeHandler,
)
from coola.equality.tester.base import BaseEqualityTester

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


class EqualEqualityTester(BaseEqualityTester[object]):
    r"""Implement an equality tester for objects with equal method.

    This tester is designed for objects that implement an ``equal(other)`` method
    for equality comparison. It uses a handler chain that checks:
    1. SameObjectHandler: Check for object identity
    2. SameTypeHandler: Verify same type
    3. EqualHandler: Call the object's equal() method

    This tester is used for comparing BaseEqualityTester instances themselves
    in the default registry.

    Example:
        Custom class with equal method:

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


class EqualNanEqualityTester(BaseEqualityTester[object]):
    r"""Implement an equality tester for objects with NaN-aware equal
    method.

    This tester is designed for objects that implement an ``equal(other, equal_nan)``
    method that supports NaN equality. When config.equal_nan is True, NaN values
    are considered equal to each other. The handler chain:
    1. SameObjectHandler: Check for object identity
    2. SameTypeHandler: Verify same type
    3. EqualNanHandler: Call object's equal() with equal_nan parameter

    Example:
        Custom class with NaN-aware equal method:

        ```pycon
        >>> import math
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import EqualNanEqualityTester
        >>> class MyFloat:
        ...     def __init__(self, value: float) -> None:
        ...         self._value = float(value)
        ...     def equal(self, other: object, equal_nan: bool = False) -> bool:
        ...         if type(other) is not type(self):
        ...             return False
        ...         if equal_nan and math.isnan(self._value) and math.isnan(other._value):
        ...             return True
        ...         return self._value == other._value
        ...
        >>> config = EqualityConfig()
        >>> tester = EqualNanEqualityTester()
        >>> tester.objects_are_equal(MyFloat(42), MyFloat(42), config=config)
        True
        >>> tester.objects_are_equal(MyFloat(float("nan")), MyFloat(float("nan")), config=config)
        False
        >>> config.equal_nan = True
        >>> tester.objects_are_equal(MyFloat(float("nan")), MyFloat(float("nan")), config=config)
        True

        ```
    """

    def __init__(self) -> None:
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(EqualNanHandler())

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
