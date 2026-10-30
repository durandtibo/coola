r"""Implement equality testers for objects with tolerant equal methods.

This module provides a tester for objects that implement both an
``allclose`` method (for tolerance-based comparison) and an ``equal``
method (for exact comparison), with optional NaN-aware equality.
"""

from __future__ import annotations

__all__ = ["TolerantEqualEqualityTester"]

from typing import TYPE_CHECKING, Any

from coola.equality._helpers.format import format_value_difference
from coola.equality.tester.base import BaseEqualityTester
from coola.utils.introspection import supports_methods

if TYPE_CHECKING:
    from coola.equality.engine import Engine


class TolerantEqualEqualityTester(BaseEqualityTester[object]):
    r"""Implement an equality tester for objects that support both
    ``allclose`` and ``equal`` methods.

    This tester is designed for objects that implement both an
    ``allclose(other, rtol, atol, equal_nan)`` method for
    tolerance-based comparison and an ``equal(other, equal_nan)``
    method for exact comparison. The tester calls ``allclose`` when a
    tolerance is not zero, otherwise ``equal``. An object that does not
    have both methods is different.

    Example:
        ```pycon
        >>> from coola.equality import Tolerance
        >>> import math
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import TolerantEqualEqualityTester, get_default_registry
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
        >>> registry = get_default_registry()
        >>> tester = TolerantEqualEqualityTester()
        >>> tester.objects_are_equal(MyFloat(42), MyFloat(42), Engine(EqualityConfig()))
        True
        >>> tester.objects_are_equal(MyFloat(1.0), MyFloat(1.4), Engine(EqualityConfig()))
        False
        >>> tester.objects_are_equal(
        ...     MyFloat(1.0), MyFloat(1.4), Engine(EqualityConfig(tolerance=Tolerance(atol=0.5)))
        ... )
        True

        ```
    """

    def objects_are_equal(self, actual: Any, expected: object, ctx: Engine) -> bool:
        r"""Test two objects with ``allclose`` or ``equal``.

        ``allclose`` is used if a tolerance is set, otherwise ``equal``.

        Args:
            actual: The actual object.
            expected: The expected object, with the same type.
            ctx: The ``Engine`` of the comparison.

        Returns:
            ``True`` if the objects are equal, otherwise ``False`` (the
                reason is recorded in the engine).
        """
        if not supports_methods(actual, "allclose", "equal"):
            return False
        if ctx.atol != 0 or ctx.rtol != 0:
            result = actual.allclose(
                expected, rtol=ctx.rtol, atol=ctx.atol, equal_nan=ctx.equal_nan
            )
        else:
            result = actual.equal(expected, equal_nan=ctx.equal_nan)
        return bool(result) or ctx.fail(
            lambda: format_value_difference(actual=actual, expected=expected)
        )
