r"""Implement scalar equality testers.

This module provides equality testers for scalar numeric types (int,
float) with support for NaN equality and tolerance-based comparisons.
"""

from __future__ import annotations

__all__ = ["ScalarEqualityTester"]

import cmath
from typing import TYPE_CHECKING

from coola.equality._helpers.format import format_value_difference
from coola.equality._helpers.scalar import complex_are_equal, number_equal
from coola.equality.tester.base import BaseEqualityTester

if TYPE_CHECKING:
    from coola.equality.engine import Engine


class ScalarEqualityTester(BaseEqualityTester[float]):
    r"""Implement a scalar equality tester.

    This tester handles numeric scalar types (int, float, complex) with
    support for NaN equality and tolerance-based comparisons. Two NaNs
    are equal only if ``equal_nan`` is ``True``, a ``bool`` and an
    enum member are labels and are compared exactly, and the other
    numbers are close if
    ``|a - b| <= atol + rtol * max(|a|, |b|)``.

    This tester is registered for ``int``, ``float`` and ``complex`` in
    the default registry.

    Example:
        ```pycon
        >>> from coola.equality import Tolerance
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import ScalarEqualityTester, get_default_registry
        >>> registry = get_default_registry()
        >>> tester = ScalarEqualityTester()
        >>> tester.objects_are_equal(42.0, 42.0, Engine(EqualityConfig()))
        True
        >>> tester.objects_are_equal(42.0, 1.0, Engine(EqualityConfig()))
        False
        >>> tester.objects_are_equal(42.0, 42.0001, Engine(EqualityConfig(tolerance=Tolerance(atol=1e-3))))
        True

        ```
    """

    def objects_are_equal(self, actual: float, expected: float, ctx: Engine) -> bool:
        if actual == expected:
            # The two numbers have the same type: equal numbers are always close.
            return True
        if ctx.equal_nan and cmath.isnan(actual) and cmath.isnan(expected):
            # ``complex(nan, 1)`` and ``complex(nan, 2)`` differ: compare the parts.
            equal = complex_are_equal(actual, expected, ctx.config)
        else:
            equal = number_equal(actual, expected, ctx.config)
        return equal or ctx.fail(
            lambda: format_value_difference(actual=actual, expected=expected, name="numbers")
        )
