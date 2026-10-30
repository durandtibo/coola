r"""Implement an equality tester for the values that are compared with
their difference.

This module provides an equality tester for ``decimal.Decimal``,
``fractions.Fraction`` and ``datetime.timedelta``.
"""

from __future__ import annotations

__all__ = ["DifferenceCloseEqualityTester"]

from typing import TYPE_CHECKING, Any

from coola.equality._helpers.difference import difference_close
from coola.equality._helpers.format import format_value_difference
from coola.equality.tester.base import BaseEqualityTester

if TYPE_CHECKING:
    from coola.equality.engine import Engine


class DifferenceCloseEqualityTester(BaseEqualityTester[Any]):
    r"""Implement an equality tester for ``decimal.Decimal``,
    ``fractions.Fraction`` and ``datetime.timedelta``.

    These types have no ``allclose``, so the tester compares their
    difference with the tolerance (see ``difference_close``). A
    ``Decimal`` NaN is equal to a NaN only if ``equal_nan`` is
    ``True``, and ``atol`` of a ``timedelta`` is a number of seconds.

    This tester is registered for ``Decimal``, ``Fraction`` and
    ``timedelta`` in the default registry.

    Example:
        ```pycon
        >>> from coola.equality import Tolerance
        >>> from decimal import Decimal
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import DifferenceCloseEqualityTester, get_default_registry
        >>> registry = get_default_registry()
        >>> tester = DifferenceCloseEqualityTester()
        >>> tester.objects_are_equal(
        ...     Decimal("1.0"), Decimal("1.00001"), Engine(EqualityConfig())
        ... )
        False
        >>> tester.objects_are_equal(
        ...     Decimal("1.0"), Decimal("1.00001"), Engine(EqualityConfig(tolerance=Tolerance(atol=1e-3)))
        ... )
        True

        ```
    """

    def objects_are_equal(self, actual: Any, expected: Any, ctx: Engine) -> bool:
        r"""Test two objects that support a difference.

        The objects are equal if the absolute difference is within the
        tolerance.

        Args:
            actual: The actual object.
            expected: The expected object, with the same type.
            ctx: The ``Engine`` of the comparison.

        Returns:
            ``True`` if the objects are equal, otherwise ``False`` (the
                reason is recorded in the engine).
        """
        return difference_close(actual, expected, ctx.config) or ctx.fail(
            lambda: format_value_difference(actual=actual, expected=expected, name="values")
        )
