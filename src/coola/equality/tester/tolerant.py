r"""Implement equality testers for objects with tolerant equal methods.

This module provides a tester for objects that implement both an
``allclose`` method (for tolerance-based comparison) and an ``equal``
method (for exact comparison), with optional NaN-aware equality.
"""

from __future__ import annotations

__all__ = ["TolerantEqualEqualityTester"]

from typing import TYPE_CHECKING

from coola.equality.handler import SameObjectHandler, SameTypeHandler, create_chain
from coola.equality.handler.tolerant import TolerantEqualHandler
from coola.equality.tester.base import BaseEqualityTester

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


class TolerantEqualEqualityTester(BaseEqualityTester[object]):
    r"""Implement an equality tester for objects that support both
    ``allclose`` and ``equal`` methods.

        This tester is designed for objects that implement both an
        ``allclose(other, rtol, atol, equal_nan)`` method for
        tolerance-based comparison and an ``equal(other, equal_nan)``
        method for exact comparison. It uses a handler chain that checks:

        1. ``SameObjectHandler``: Check for object identity.
        2. ``SameTypeHandler``: Verify same type.
        3. ``TolerantEqualHandler``: Dispatch to ``allclose`` when
           tolerances are non-zero, otherwise to ``equal``.

    Example:
        ```pycon
        >>> import math
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import TolerantEqualEqualityTester
        >>> class MyFloat:
        ...     def __init__(self, value: float) -> None:
        ...         self._value = float(value)
        ...     def allclose(
        ...         self,
        ...         other: object,
        ...         rtol: float = 1e-5,
        ...         atol: float = 1e-8,
        ...         equal_nan: bool = False,
        ...     ) -> bool:
        ...         if type(other) is not type(self):
        ...             return False
        ...         if equal_nan and math.isnan(self._value) and math.isnan(other._value):
        ...             return True
        ...         return math.isclose(self._value, other._value, rel_tol=rtol, abs_tol=atol)
        ...     def equal(self, other: object, equal_nan: bool = False) -> bool:
        ...         if type(other) is not type(self):
        ...             return False
        ...         if equal_nan and math.isnan(self._value) and math.isnan(other._value):
        ...             return True
        ...         return self._value == other._value
        ...
        >>> config = EqualityConfig()
        >>> tester = TolerantEqualEqualityTester()
        >>> tester.objects_are_equal(MyFloat(42), MyFloat(42), config)
        True
        >>> tester.objects_are_equal(MyFloat(42), MyFloat(1), config)
        False
        >>> tester.objects_are_equal(MyFloat(float("nan")), MyFloat(float("nan")), config)
        False
        >>> config.equal_nan = True
        >>> tester.objects_are_equal(MyFloat(float("nan")), MyFloat(float("nan")), config)
        True
        >>> config.atol = 0.5
        >>> tester.objects_are_equal(MyFloat(1.0), MyFloat(1.4), config)
        True

        ```
    """

    def __init__(self) -> None:
        self._handler = create_chain(SameObjectHandler(), SameTypeHandler(), TolerantEqualHandler())

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
        return self._handler.handle(actual, expected, config)
