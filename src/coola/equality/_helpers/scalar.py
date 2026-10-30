r"""Implement handlers to check if the objects are equal."""

from __future__ import annotations

__all__ = ["complex_are_equal", "int_isclose", "number_equal"]

import cmath
import enum
import math
import numbers
from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


def complex_are_equal(actual: complex, expected: complex, config: EqualityConfig) -> bool:
    r"""Compare the real and imaginary parts, NaN being equal to NaN.

    The parts are compared one by one with the tolerances of ``config``,
    so ``complex(nan, 1)`` and ``complex(nan, 2)`` are different.

    Args:
        actual: The actual complex number.
        expected: The expected complex number.
        config: The equality configuration, which gives the tolerances
            ``atol`` and ``rtol``.

    Returns:
        ``True`` if the real parts and the imaginary parts are equal
            (a NaN part is equal to a NaN part), otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality._helpers.scalar import complex_are_equal
        >>> config = EqualityConfig()
        >>> complex_are_equal(complex(float("nan"), 1), complex(float("nan"), 1), config)
        True
        >>> complex_are_equal(complex(float("nan"), 1), complex(float("nan"), 2), config)
        False

        ```
    """
    for a, b in ((actual.real, expected.real), (actual.imag, expected.imag)):
        if math.isnan(a) and math.isnan(b):
            continue
        if not number_equal(a, b, config):
            return False
    return True


def number_equal(number1: float, number2: float, config: EqualityConfig) -> bool:
    r"""Indicate if the two numbers are equal within a tolerance.

    Args:
        number1: The first number to compare.
        number2: The second number to compare.
        config: The equality configuration.

    Returns:
        ``True`` if the two numbers are equal within a tolerance,
            otherwise ``False``. Values that are not numbers (or that
            cannot be converted to a ``float``) are compared with ``==``.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality._helpers.scalar import number_equal
        >>> config = EqualityConfig()
        >>> number_equal(42.0, 42.0, config)
        True
        >>> number_equal(42.0, 42.1, config)
        False

        ```
    """
    # A ``bool`` and an enum member are labels, not quantities: ``True`` is close to ``True``
    # only, and ``IntEnum`` members are compared exactly.
    if (config.atol > 0.0 or config.rtol > 0.0) and (
        isinstance(number1, numbers.Number)
        and isinstance(number2, numbers.Number)
        and not isinstance(number1, (bool, enum.Enum))
        and not isinstance(number2, (bool, enum.Enum))
    ):
        # ``numbers.Real`` also covers the NumPy scalars (NumPy registers them), so
        # only the complex numbers are converted to ``complex`` (``float`` would raise
        # for ``complex`` and drop the imaginary part for ``numpy.complex64``).
        if not isinstance(number1, numbers.Real) or not isinstance(number2, numbers.Real):
            return _isclose(complex(number1), complex(number2), rtol=config.rtol, atol=config.atol)
        if isinstance(number1, int) and isinstance(number2, int):
            # Exact arithmetic: converting large integers to ``float`` rounds them.
            # NumPy integers are not ``int`` and keep the ``float`` comparison.
            return int_isclose(number1, number2, rtol=config.rtol, atol=config.atol)
        try:
            return _isclose(float(number1), float(number2), rtol=config.rtol, atol=config.atol)
        except OverflowError:
            # An ``int`` too large for a float: compare with exact arithmetic.
            return int_isclose(number1, number2, rtol=config.rtol, atol=config.atol)
        except TypeError:
            # A ``numbers.Real`` that has no ``float`` (``numpy.timedelta64`` is registered as
            # an integer): ``number_equal`` never raises, it compares with ``==``.
            return bool(number1 == number2)
    # ``bool(...)``: NumPy scalars return ``numpy.bool_``. A value that is not a number has no
    # tolerance either.
    return bool(number1 == number2)


def _isclose(number1: complex, number2: complex, rtol: float, atol: float) -> bool:
    r"""Indicate if two numbers are close with the NumPy rule.

    The numbers are close if ``abs(number1 - number2) <= atol + rtol *
    max(abs(number1), abs(number2))``, which is the rule of the NumPy,
    PyTorch and JAX handlers. Two different numbers are never close if
    one of them is infinite or NaN.

    Args:
        number1: The first number.
        number2: The second number.
        rtol: The relative tolerance.
        atol: The absolute tolerance.

    Returns:
        ``True`` if the numbers are close, otherwise ``False``.
    """
    if number1 == number2:
        return True
    if not (cmath.isfinite(number1) and cmath.isfinite(number2)):
        return False
    return abs(number1 - number2) <= atol + rtol * max(abs(number1), abs(number2))


def int_isclose(number1: float, number2: float, rtol: float = 0.0, atol: float = 0.0) -> bool:
    r"""Indicate if two numbers are close, using exact rational
    arithmetic.

    This is a fallback for integers that are too large to be converted
    to a float. The numbers are close if ``abs(number1 - number2) <=
    atol + rtol * max(abs(number1), abs(number2))``.

    Args:
        number1: The first number.
        number2: The second number.
        rtol: The relative tolerance.
        atol: The absolute tolerance.

    Returns:
        ``True`` if the two numbers are close, otherwise ``False``.
            Infinite or NaN inputs, which cannot be converted to exact
            fractions, are compared with ``==``.

    Example:
        ```pycon
        >>> from coola.equality._helpers.scalar import int_isclose
        >>> int_isclose(10**400, 10**400 + 1, rtol=1e-5)
        True
        >>> int_isclose(10**400, 2 * 10**400, rtol=1e-5)
        False

        ```
    """
    try:
        value1, value2 = Fraction(number1), Fraction(number2)
    except (OverflowError, ValueError):  # infinity or NaN
        return number1 == number2
    tolerance = Fraction(atol) + Fraction(rtol) * max(abs(value1), abs(value2))
    return abs(value1 - value2) <= tolerance
