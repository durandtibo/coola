r"""Implement a handler to check if two values are equal within a
tolerance by using their difference."""

from __future__ import annotations

__all__ = ["difference_close"]

from datetime import timedelta
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


def difference_close(actual: Any, expected: Any, config: EqualityConfig) -> bool:
    r"""Indicate if two values are equal within a tolerance.

    The values are close if
    ``|actual - expected| <= atol + rtol * max(|actual|, |expected|)``
    (the rule of the NumPy, PyTorch and JAX handlers). The tolerances are converted to the
    type of the values: ``atol`` of a ``timedelta`` is a number of
    seconds. Two values that are equal are always close, and a
    ``Decimal`` NaN is equal to a NaN only if ``config.equal_nan`` is
    ``True``.

    Args:
        actual: The actual value.
        expected: The expected value, with the same type.
        config: The equality configuration, which gives ``atol``,
            ``rtol`` and ``equal_nan``.

    Returns:
        ``True`` if the values are equal within the tolerance,
            otherwise ``False``.

    Example:
        ```pycon
        >>> from fractions import Fraction
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler.difference import difference_close
        >>> config = EqualityConfig(atol=1e-3)
        >>> difference_close(Fraction(1, 3), Fraction(333333, 1000000), config)
        True
        >>> difference_close(Fraction(1, 3), Fraction(1, 2), config)
        False

        ```
    """
    if isinstance(actual, Decimal) and (actual.is_nan() or expected.is_nan()):
        return config.equal_nan and actual.is_nan() and expected.is_nan()
    if actual == expected:
        return True
    if config.atol <= 0.0 and config.rtol <= 0.0:
        return False
    if isinstance(actual, Decimal):
        if not (actual.is_finite() and expected.is_finite()):
            return False  # Two different values, and one is infinite
        atol, rtol = Decimal(config.atol), Decimal(config.rtol)
    elif isinstance(actual, timedelta):
        atol, rtol = timedelta(seconds=config.atol), config.rtol
    else:
        atol, rtol = Fraction(config.atol), Fraction(config.rtol)
    return abs(actual - expected) <= atol + rtol * max(abs(actual), abs(expected))
