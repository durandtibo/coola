r"""Provide validation helpers shared by the identifier generators.

Not part of the public API: several generators in this package validate
the same shape of input (e.g. a value that must fit in a fixed number of
bits, or a length/size that must be positive) and previously duplicated
the same checks and error messages.
"""

from __future__ import annotations

__all__ = [
    "CROCKFORD_BASE32_ALPHABET",
    "decode_crockford_base32",
    "resolve_timestamp_ms",
    "validate_bit_range",
    "validate_non_negative",
    "validate_positive",
    "validate_timestamp_ms",
]

import time

_TIMESTAMP_BITS = 48

# Crockford's Base32 alphabet (excludes I, L, O, U to avoid transcription
# ambiguity). Shared by ``coola.identifier.ulid`` and
# ``coola.identifier.checksummed``, which both encode/decode it.
CROCKFORD_BASE32_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def validate_bit_range(value: int, bits: int, *, name: str) -> None:
    r"""Validate that ``value`` fits in ``bits`` unsigned bits.

    Args:
        value: The value to validate.
        bits: The number of bits ``value`` must fit in.
        name: The name of the value, used in the error message.

    Raises:
        ValueError: If ``value`` is negative or exceeds ``2**bits - 1``.
    """
    maximum = (1 << bits) - 1
    if not 0 <= value <= maximum:
        msg = f"{name} must fit in {bits} bits (0 to {maximum}), got {value}"
        raise ValueError(msg)


def validate_timestamp_ms(timestamp_ms: int) -> None:
    r"""Validate that ``timestamp_ms`` fits in 48 bits.

    Shared by the generators (``generate_ulid``, ``generate_uuid7``)
    that encode a 48-bit millisecond timestamp.

    Args:
        timestamp_ms: The Unix timestamp in milliseconds to validate.

    Raises:
        ValueError: If ``timestamp_ms`` does not fit in 48 bits (i.e.
            is negative or exceeds ``2**48 - 1``).
    """
    validate_bit_range(timestamp_ms, _TIMESTAMP_BITS, name="timestamp_ms")


def validate_non_negative(value: int, *, name: str) -> None:
    r"""Validate that ``value`` is not negative.

    Args:
        value: The value to validate.
        name: The name of the value, used in the error message.

    Raises:
        ValueError: If ``value`` is negative.
    """
    if value < 0:
        msg = f"{name} must be non-negative, got {value}"
        raise ValueError(msg)


def validate_positive(value: int, *, name: str) -> None:
    r"""Validate that ``value`` is strictly positive.

    Args:
        value: The value to validate.
        name: The name of the value, used in the error message.

    Raises:
        ValueError: If ``value`` is not positive.
    """
    if value <= 0:
        msg = f"{name} must be positive, got {value}"
        raise ValueError(msg)


def resolve_timestamp_ms(timestamp_ms: int | None) -> int:
    r"""Return ``timestamp_ms``, defaulting to the current time and
    validating that the result fits in 48 bits.

    Shared by the generators (``generate_ulid``, ``generate_uuid7``)
    that accept an optional 48-bit millisecond timestamp, defaulting to
    ``time.time_ns() // 1_000_000`` when not given.

    Args:
        timestamp_ms: The Unix timestamp in milliseconds, or ``None``
            to use the current time.

    Returns:
        ``timestamp_ms``, or the current time in milliseconds if it
        was ``None``.

    Raises:
        ValueError: If the resolved timestamp does not fit in 48 bits
            (i.e. is negative or exceeds ``2**48 - 1``).
    """
    if timestamp_ms is None:
        timestamp_ms = time.time_ns() // 1_000_000
    validate_timestamp_ms(timestamp_ms)
    return timestamp_ms


def decode_crockford_base32(text: str, *, name: str) -> int:
    r"""Decode a Crockford Base32 string into its integer value.

    Shared by ``coola.identifier.ulid`` and
    ``coola.identifier.checksummed``, which both decode strings drawn
    from ``CROCKFORD_BASE32_ALPHABET``.

    Args:
        text: The Crockford Base32 string to decode.
        name: The name of the value, used in the error message.

    Returns:
        The decoded integer value.

    Raises:
        ValueError: If ``text`` contains a character outside
            ``CROCKFORD_BASE32_ALPHABET``.
    """
    value = 0
    for char in text:
        try:
            digit = CROCKFORD_BASE32_ALPHABET.index(char)
        except ValueError as error:
            msg = f"{name} contains a character outside the Crockford Base32 alphabet, got {char!r}"
            raise ValueError(msg) from error
        value = value * 32 + digit
    return value
