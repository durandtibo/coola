r"""Implement equality testers for objects with equal methods.

This module provides specialized testers for objects that implement an
``equal`` method, supporting both standard equality and NaN-aware
equality comparisons.
"""

from __future__ import annotations

__all__ = ["EqualEqualityTester", "EqualNanEqualityTester"]

from typing import TYPE_CHECKING

from coola.equality._helpers.format import format_value_difference
from coola.equality.tester.base import BaseEqualityTester

if TYPE_CHECKING:
    from coola.equality.engine import Engine


class EqualEqualityTester(BaseEqualityTester[object]):
    r"""Implement an equality tester for objects with equal method.

    This tester is designed for objects that implement an
    ``equal(other)`` method for equality comparison. The two objects
    have the same type (the engine checks it) and the tester calls the
    ``equal`` method of the first one. An object without a callable
    ``equal`` is different.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import EqualEqualityTester, get_default_registry
        >>> class MyFloat:
        ...     def __init__(self, value: float) -> None:
        ...         self._value = float(value)
        ...     def equal(self, other: object) -> bool:
        ...         if type(other) is not type(self):
        ...             return False
        ...         return self._value == other._value
        ...
        >>> engine = Engine(EqualityConfig())
        >>> tester = EqualEqualityTester()
        >>> tester.objects_are_equal(MyFloat(42), MyFloat(42), engine)
        True
        >>> tester.objects_are_equal(MyFloat(42), MyFloat(1), engine)
        False

        ```
    """

    def objects_are_equal(self, actual: object, expected: object, ctx: Engine) -> bool:
        r"""Test two objects with their ``equal`` method.

        Args:
            actual: The actual object.
            expected: The expected object, with the same type.
            ctx: The ``Engine`` of the comparison.

        Returns:
            ``True`` if the objects are equal, otherwise ``False`` (the
                reason is recorded in the engine).
        """
        equal = getattr(actual, "equal", None)
        if not callable(equal):
            return False
        return bool(equal(expected)) or ctx.fail(
            lambda: format_value_difference(actual=actual, expected=expected)
        )


class EqualNanEqualityTester(BaseEqualityTester[object]):
    r"""Implement an equality tester for objects with equal method that
    supports NaN equality.

    This tester is designed for objects that implement an
    ``equal(other, equal_nan)`` method. The tester calls the ``equal``
    method of the first object with ``equal_nan`` of the comparison. An
    object without a callable ``equal`` is different.

    Example:
        ```pycon
        >>> import math
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import EqualNanEqualityTester, get_default_registry
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
        >>> registry = get_default_registry()
        >>> tester = EqualNanEqualityTester()
        >>> nan1, nan2 = MyFloat(float("nan")), MyFloat(float("nan"))
        >>> tester.objects_are_equal(nan1, nan2, Engine(EqualityConfig()))
        False
        >>> tester.objects_are_equal(nan1, nan2, Engine(EqualityConfig(equal_nan=True)))
        True

        ```
    """

    def objects_are_equal(self, actual: object, expected: object, ctx: Engine) -> bool:
        r"""Test two objects with their ``equal`` method and
        ``equal_nan``.

        Args:
            actual: The actual object.
            expected: The expected object, with the same type.
            ctx: The ``Engine`` of the comparison.

        Returns:
            ``True`` if the objects are equal, otherwise ``False`` (the
                reason is recorded in the engine).
        """
        equal = getattr(actual, "equal", None)
        if not callable(equal):
            return False
        return bool(equal(expected, equal_nan=ctx.equal_nan)) or ctx.fail(
            lambda: format_value_difference(actual=actual, expected=expected)
        )
