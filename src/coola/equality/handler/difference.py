r"""Implement a handler to check if two values are equal within a
tolerance by using their difference."""

from __future__ import annotations

__all__ = ["DifferenceCloseHandler", "difference_close"]

import logging
from datetime import timedelta
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.format import format_value_difference
from coola.equality.handler.mixin import HandlerEqualityMixin
from coola.equality.handler.utils import report_difference

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig

logger: logging.Logger = logging.getLogger(__name__)


class DifferenceCloseHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if two values are equal within a tolerance.

    The values are compared with their difference, so this handler
    supports ``decimal.Decimal``, ``fractions.Fraction`` and
    ``datetime.timedelta``. See ``difference_close`` for the rule.
    This handler is designed to be used at the end of the chain of
    responsibility. It does not call the next handler.

    Example:
        ```pycon
        >>> from decimal import Decimal
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import DifferenceCloseHandler
        >>> config = EqualityConfig(atol=1e-3)
        >>> handler = DifferenceCloseHandler()
        >>> handler.handle(Decimal("1.0"), Decimal("1.00001"), config)
        True
        >>> handler.handle(Decimal("1.0"), Decimal("1.1"), config)
        False

        ```
    """

    def handle(self, actual: Any, expected: Any, config: EqualityConfig) -> bool:
        object_equal = difference_close(actual, expected, config)
        if not object_equal and config.show_difference:
            report_difference(
                logger,
                config,
                format_value_difference(actual=actual, expected=expected, name="values"),
            )
        return object_equal


def difference_close(actual: Any, expected: Any, config: EqualityConfig) -> bool:
    r"""Indicate if two values are equal within a tolerance.

    The values are close if
    ``|actual - expected| <= max(rtol * max(|actual|, |expected|), atol)``
    (the rule of ``math.isclose``). The tolerances are converted to the
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
    return abs(actual - expected) <= max(rtol * max(abs(actual), abs(expected)), atol)
