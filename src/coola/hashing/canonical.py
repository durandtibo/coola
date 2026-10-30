r"""Define the hashers of the values that have equal representations.

``Decimal('1.0')``, ``Decimal('1.00')`` and ``Decimal(1)`` are equal,
and so are ``range(0)`` and ``range(5, 5)``, but their ``repr`` is
different. The hashers of this module hash a canonical form, so equal
values have the same hash.
"""

from __future__ import annotations

__all__ = ["DecimalHasher", "FractionHasher", "RangeHasher"]

from decimal import Decimal
from fractions import Fraction

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher, Encoding, tagged
from coola.utils.integer import int_repr


def _canonical_decimal(data: Decimal) -> str:
    r"""Return a canonical string of a decimal.

    The string is computed from ``as_tuple()`` instead of
    ``Decimal.normalize()``, because ``normalize()`` rounds to the
    precision of the current context, so it can merge different decimals
    and depends on the thread-local context.

    Args:
        data: The decimal.

    Returns:
        The canonical string: equal decimals give the same string.
    """
    sign, digits, exponent = data.as_tuple()
    if not isinstance(exponent, int):  # 'F', 'n' or 'N': infinity or NaN
        return str(data)
    coefficient = "".join(map(str, digits)).lstrip("0")  # no int(): the digits are not limited
    if not coefficient:  # ``Decimal('-0') == Decimal('0')``
        return "0"
    stripped = coefficient.rstrip("0")
    exponent += len(coefficient) - len(stripped)
    return f"{'-' if sign else ''}{stripped}e{exponent}"


class DecimalHasher(InlineDisplayMixin, BaseHasher[Decimal]):
    r"""Hasher for ``decimal.Decimal``.

    The decimal is normalized first, so the equal decimals
    (``Decimal('1.0')`` and ``Decimal('1.00')``) have the same hash.

    Example:
        ```pycon
        >>> from decimal import Decimal
        >>> from coola.hashing.canonical import DecimalHasher
        >>> hasher = DecimalHasher()
        >>> hasher.encode(Decimal("1.0")) == hasher.encode(Decimal("1.00"))
        True

        ```
    """

    def encode(self, data: Decimal) -> Encoding:
        r"""Describe the canonical form of the value.

        Args:
            data: The value to describe.

        Returns:
            The leaf of the canonical form.
        """
        return tagged("Decimal", _canonical_decimal(data))


class FractionHasher(InlineDisplayMixin, BaseHasher[Fraction]):
    r"""Hasher for ``fractions.Fraction``.

    A ``Fraction`` is always stored in lowest terms, so the numerator
    and the denominator are a canonical form.

    Example:
        ```pycon
        >>> from fractions import Fraction
        >>> from coola.hashing.canonical import FractionHasher
        >>> hasher = FractionHasher()
        >>> hasher.encode(Fraction(1, 2)) == hasher.encode(Fraction(2, 4))
        True

        ```
    """

    def encode(self, data: Fraction) -> Encoding:
        r"""Describe the canonical form of the value.

        Args:
            data: The value to describe.

        Returns:
            The leaf of the canonical form.
        """
        return tagged("Fraction", f"{int_repr(data.numerator)}/{int_repr(data.denominator)}")


class RangeHasher(InlineDisplayMixin, BaseHasher[range]):
    r"""Hasher for ``range``.

    Two ranges are equal if they contain the same items, so the hash is
    computed from the length, and from the first item and the step when
    they matter. A huge range is hashed in constant time.

    Example:
        ```pycon
        >>> from coola.hashing.canonical import RangeHasher
        >>> hasher = RangeHasher()
        >>> hasher.encode(range(0)) == hasher.encode(range(5, 5))
        True

        ```
    """

    def encode(self, data: range) -> Encoding:
        r"""Describe the canonical form of the value.

        Args:
            data: The value to describe.

        Returns:
            The leaf of the canonical form.
        """
        size = len(data)
        if size == 0:
            payload = "0"
        elif size == 1:
            payload = f"1:{int_repr(data[0])}"
        else:
            payload = f"{size}:{int_repr(data[0])}:{int_repr(data.step)}"
        return tagged("range", payload)
