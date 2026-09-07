r"""Provide a reversible, obfuscated identifier for a sequential integer.

Unlike every other generator in this package, this one does not mint a
fresh value: it takes an existing non-negative integer (typically a
database autoincrement primary key or a ``SnowflakeIdGenerator`` value)
and reshapes it into a short opaque string that hides the original
integer's magnitude and ordering from anyone who does not know the
``salt``, while remaining exactly reversible via
``decode_obfuscated_id``. This solves a different problem than the rest
of the package: obfuscating and de-obfuscating an already-unique value,
rather than generating uniqueness. Use it when internal code already has
sequential integer IDs (so consecutive rows would otherwise get
consecutive public-facing IDs, leaking e.g. total row counts) but a
public-facing identifier should not reveal that.

The technique (sometimes called "Optimus" or "Hashids"-style ID
obfuscation) applies an affine map, ``number * multiplier + increment``
modulo 2**64, using an odd, salt-derived ``multiplier`` and a second,
independent salt-derived ``increment``: multiplication by an odd number
is a bijection on 64-bit integers, and adding a constant is trivially
invertible too, so the whole map is exactly invertible (subtract
``increment``, then multiply by the modular inverse of ``multiplier``)
without needing to store anything. The ``increment`` term matters
specifically for the sequential-integer case this module targets: with
multiplication alone, ``obfuscated(n + 1) - obfuscated(n) == multiplier``
for every ``n``, so a single known ``(plaintext, obfuscated)`` pair next
to an unknown one immediately recovers ``multiplier`` and breaks every
other ID under the same ``salt``. Adding ``increment`` does not remove
that difference (it still equals ``multiplier``, since ``increment``
cancels out), so this is still obfuscation, not encryption: an attacker
who can observe two ``(plaintext, obfuscated)`` pairs for a known
``salt`` (adjacent or not) can solve the resulting linear system for
both constants. Do not rely on it to hide data from anyone who can query
your API repeatedly under a fixed ``salt``.
"""

from __future__ import annotations

__all__ = ["decode_obfuscated_id", "generate_obfuscated_id"]

import hashlib

from coola.identifier.validation import validate_bit_range, validate_non_negative

_BITS = 64
_MASK = (1 << _BITS) - 1
_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_BASE = len(_ALPHABET)


def _multiplier_and_increment(salt: str) -> tuple[int, int]:
    r"""Derive the affine map's odd 64-bit multiplier and 64-bit
    increment from ``salt``.

    The multiplier must be odd for multiplication modulo 2**64 to be a
    bijection (any even multiplier collapses distinct inputs onto the
    same output for at least half of the value space). The increment
    has no such constraint: adding any constant modulo 2**64 is already
    a bijection.
    """
    digest = hashlib.sha256(salt.encode("utf-8")).digest()
    multiplier = (int.from_bytes(digest[:8], byteorder="big") & _MASK) | 1
    increment = int.from_bytes(digest[8:16], byteorder="big") & _MASK
    return multiplier, increment


def _encode_base62(number: int, min_length: int) -> str:
    if number == 0:
        chars = ["0"]
    else:
        chars = []
        while number > 0:
            number, rem = divmod(number, _BASE)
            chars.append(_ALPHABET[rem])
        chars.reverse()
    text = "".join(chars)
    if len(text) < min_length:
        text = _ALPHABET[0] * (min_length - len(text)) + text
    return text


def _decode_base62(text: str) -> int:
    number = 0
    for char in text:
        number = number * _BASE + _ALPHABET.index(char)
    return number


def generate_obfuscated_id(number: int, salt: str = "", min_length: int = 0) -> str:
    r"""Obfuscate a non-negative integer into a short, reversible
    identifier.

    Args:
        number: The integer to obfuscate. Must fit in 64 bits (``0``
            to ``2**64 - 1``), e.g. a database autoincrement ID or a
            ``SnowflakeIdGenerator`` value.
        salt: A key controlling the obfuscation. Two different values
            of ``salt`` produce unrelated encodings for the same
            ``number``, and ``decode_obfuscated_id`` must be called
            with the same ``salt`` used here to recover ``number``.
        min_length: The minimum length of the returned string; shorter
            results are left-padded with ``'0'``. Defaults to ``0``
            (no padding). Must be non-negative.

    Returns:
        A base62 string that ``decode_obfuscated_id`` can turn back
        into ``number`` given the same ``salt``.

    Raises:
        ValueError: If ``number`` is negative or does not fit in 64
            bits, or if ``min_length`` is negative.

    Example:
        ```pycon
        >>> from coola.identifier import generate_obfuscated_id, decode_obfuscated_id
        >>> encoded = generate_obfuscated_id(42, salt="orders")
        >>> decode_obfuscated_id(encoded, salt="orders")
        42
        >>> generate_obfuscated_id(42, salt="orders") == generate_obfuscated_id(43, salt="orders")
        False

        ```
    """
    validate_bit_range(number, _BITS, name="number")
    validate_non_negative(min_length, name="min_length")
    multiplier, increment = _multiplier_and_increment(salt)
    obfuscated = (number * multiplier + increment) & _MASK
    return _encode_base62(obfuscated, min_length=min_length)


def decode_obfuscated_id(encoded: str, salt: str = "") -> int:
    r"""Reverse ``generate_obfuscated_id`` and recover the original
    integer.

    Args:
        encoded: The string previously returned by
            ``generate_obfuscated_id``.
        salt: The same ``salt`` passed to ``generate_obfuscated_id``
            when ``encoded`` was produced.

    Returns:
        The original non-negative integer.

    Raises:
        ValueError: If ``encoded`` is empty or contains a character
            outside the base62 alphabet used by
            ``generate_obfuscated_id``.

    Example:
        ```pycon
        >>> from coola.identifier import generate_obfuscated_id, decode_obfuscated_id
        >>> decode_obfuscated_id(generate_obfuscated_id(1234, salt="k"), salt="k")
        1234

        ```
    """
    if not encoded:
        msg = "encoded must not be empty"
        raise ValueError(msg)
    try:
        obfuscated = _decode_base62(encoded)
    except ValueError as error:
        msg = f"encoded contains a character outside the base62 alphabet, got {encoded!r}"
        raise ValueError(msg) from error
    multiplier, increment = _multiplier_and_increment(salt)
    inverse = pow(multiplier, -1, 1 << _BITS)
    return ((obfuscated - increment) * inverse) & _MASK
