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
from typing import TYPE_CHECKING

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_tagged
from coola.utils.integer import int_repr

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry


def _canonical_decimal(data: Decimal) -> str:
    """Return a canonical string of a decimal.

    The string is computed from ``as_tuple()`` instead of
    ``Decimal.normalize()``, because ``normalize()`` rounds to the
    precision of the current context, so it can merge different decimals
    and depends on the thread-local context.
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
        >>> from coola.hashing import HasherRegistry
        >>> from coola.hashing.canonical import DecimalHasher
        >>> hasher = DecimalHasher()
        >>> registry = HasherRegistry()
        >>> hasher.hash(Decimal("1.0"), registry) == hasher.hash(Decimal("1.00"), registry)
        True

        ```
    """

    def hash(
        self,
        data: Decimal,
        registry: HasherRegistry,  # noqa: ARG002
        *,
        length: int = 64,
        ignore_unhashable: bool = False,  # noqa: ARG002
    ) -> str:
        return hash_tagged("Decimal", _canonical_decimal(data), length=length)


class FractionHasher(InlineDisplayMixin, BaseHasher[Fraction]):
    r"""Hasher for ``fractions.Fraction``.

    A ``Fraction`` is always stored in lowest terms, so the numerator
    and the denominator are a canonical form.

    Example:
        ```pycon
        >>> from fractions import Fraction
        >>> from coola.hashing import HasherRegistry
        >>> from coola.hashing.canonical import FractionHasher
        >>> hasher = FractionHasher()
        >>> registry = HasherRegistry()
        >>> hasher.hash(Fraction(1, 2), registry) == hasher.hash(Fraction(2, 4), registry)
        True

        ```
    """

    def hash(
        self,
        data: Fraction,
        registry: HasherRegistry,  # noqa: ARG002
        *,
        length: int = 64,
        ignore_unhashable: bool = False,  # noqa: ARG002
    ) -> str:
        return hash_tagged(
            "Fraction", f"{int_repr(data.numerator)}/{int_repr(data.denominator)}", length=length
        )


class RangeHasher(InlineDisplayMixin, BaseHasher[range]):
    r"""Hasher for ``range``.

    Two ranges are equal if they contain the same items, so the hash is
    computed from the length, and from the first item and the step when
    they matter. A huge range is hashed in constant time.

    Example:
        ```pycon
        >>> from coola.hashing import HasherRegistry
        >>> from coola.hashing.canonical import RangeHasher
        >>> hasher = RangeHasher()
        >>> registry = HasherRegistry()
        >>> hasher.hash(range(0), registry) == hasher.hash(range(5, 5), registry)
        True

        ```
    """

    def hash(
        self,
        data: range,
        registry: HasherRegistry,  # noqa: ARG002
        *,
        length: int = 64,
        ignore_unhashable: bool = False,  # noqa: ARG002
    ) -> str:
        size = len(data)
        if size == 0:
            payload = "0"
        elif size == 1:
            payload = f"1:{int_repr(data[0])}"
        else:
            payload = f"{size}:{int_repr(data[0])}:{int_repr(data.step)}"
        return hash_tagged("range", payload, length=length)
