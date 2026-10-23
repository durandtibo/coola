r"""Helpers to convert integers to strings."""

from __future__ import annotations

__all__ = ["int_repr"]


def int_repr(value: int) -> str:
    r"""Return the ``repr`` of an integer, in hexadecimal if it exceeds
    the int-to-str conversion limit of the interpreter.

    Args:
        value: The integer to convert.

    Returns:
        The decimal ``repr`` of ``value``, or ``0x...`` when it has
            more digits than ``sys.get_int_max_str_digits()``.

    Example:
        ```pycon
        >>> from coola.utils.integer import int_repr
        >>> int_repr(42)
        '42'
        >>> int_repr(10**5000)[:6]
        '0x31e2'

        ```
    """
    try:
        return int.__repr__(value)
    except ValueError:  # more digits than ``sys.get_int_max_str_digits()``
        return f"0x{value:x}"
