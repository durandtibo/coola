r"""Helpers to convert integers to strings."""

from __future__ import annotations

__all__ = ["int_repr"]

from decimal import Decimal

# The default limit of the int-to-str conversion of CPython. It is a constant
# here (not ``sys.get_int_max_str_digits()``) so that the representation of an
# integer does not depend on the setting of the interpreter.
_MAX_DECIMAL_DIGITS = 4300

# An integer with more bits than this has more than ``_MAX_DECIMAL_DIGITS``
# digits (2**15000 has 4515 digits).
_MAX_DECIMAL_BITS = 15000


def int_repr(value: int) -> str:
    r"""Return the ``repr`` of an integer, in hexadecimal if it has more
    than 4300 decimal digits.

    The limit is fixed, so the result does not depend on
    ``sys.set_int_max_str_digits`` or ``PYTHONINTMAXSTRDIGITS``.

    Args:
        value: The integer to convert.

    Returns:
        The decimal ``repr`` of ``value``, or ``0x...`` (``-0x...`` for
            a negative integer) when it has more than 4300 digits.

    Example:
        ```pycon
        >>> from coola.utils.integer import int_repr
        >>> int_repr(42)
        '42'
        >>> int_repr(10**5000)[:6]
        '0x31e2'

        ```
    """
    if value.bit_length() <= _MAX_DECIMAL_BITS:
        # ``Decimal`` is not subject to the limit of the interpreter
        digits = str(Decimal(abs(value)))
        if len(digits) <= _MAX_DECIMAL_DIGITS:
            return f"-{digits}" if value < 0 else digits
    sign = "-" if value < 0 else ""
    return f"{sign}0x{abs(value):x}"
