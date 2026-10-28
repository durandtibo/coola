r"""Provide the Crockford Base32 encoding shared by the identifiers.

The ULID and the checksummed identifiers use the same alphabet. This
module is the only place that encodes and decodes it, so both follow the
same rule: decoding is case-insensitive and reads ``O`` as ``0`` and
``I``/``L`` as ``1``, as the specification says.
"""

from __future__ import annotations

__all__ = ["ALPHABET", "decode", "encode"]

# Crockford's Base32 alphabet (excludes I, L, O, U to avoid transcription ambiguity).
ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

# The upper-case and lower-case characters of the alphabet, and the confusable letters.
# A dictionary rejects the non-ASCII characters that ``str.upper`` maps to ASCII ones.
_DIGITS = {
    **{char: digit for digit, char in enumerate(ALPHABET)},
    **{char.lower(): digit for digit, char in enumerate(ALPHABET)},
    "O": 0,
    "o": 0,
    "I": 1,
    "i": 1,
    "L": 1,
    "l": 1,
}


def encode(value: int, *, length: int) -> str:
    r"""Encode a non-negative integer as a fixed-length Crockford Base32
    string.

    Args:
        value: The integer to encode.
        length: The number of characters of the result. The string is
            left-padded with ``'0'``.

    Returns:
        The upper-case Crockford Base32 encoding of ``value``.

    Raises:
        ValueError: If ``value`` is negative or does not fit in
            ``length`` characters.

    Example:
        ```pycon
        >>> from coola.identifier.crockford import encode
        >>> encode(31, length=3)
        '00Z'

        ```
    """
    if not 0 <= value < 32**length:
        msg = f"value must fit in {length} Crockford Base32 characters, got {value}"
        raise ValueError(msg)
    chars = [""] * length
    for i in range(length - 1, -1, -1):
        chars[i] = ALPHABET[value & 0x1F]
        value >>= 5
    return "".join(chars)


def decode(text: str, *, name: str = "value") -> int:
    r"""Decode a Crockford Base32 string into its integer value.

    Decoding is case-insensitive, and ``'O'`` is read as ``'0'`` and
    ``'I'``/``'L'`` as ``'1'``.

    Args:
        text: The Crockford Base32 string to decode.
        name: The name of the value, used in the error message.

    Returns:
        The decoded integer value.

    Raises:
        ValueError: If ``text`` is empty or contains a character
            outside the alphabet.

    Example:
        ```pycon
        >>> from coola.identifier.crockford import decode
        >>> decode("00z")
        31

        ```
    """
    if not text:
        msg = f"{name} must not be empty"
        raise ValueError(msg)
    value = 0
    for char in text:
        try:
            digit = _DIGITS[char]
        except KeyError as error:
            msg = f"{name} contains a character outside the Crockford Base32 alphabet, got {char!r}"
            raise ValueError(msg) from error
        value = value * 32 + digit
    return value
