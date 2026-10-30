r"""Provide a human-transcribable identifier with a check symbol.

None of the other generators in this package guard against transcription
errors: a single mistyped character in a UUID or a ULID is silently
accepted as a different, valid-looking identifier. This module adds a
check symbol (following the optional check-symbol scheme from the
[Crockford Base32](
https://www.crockford.com/base32.html)
specification also used by ``generate_ulid``) so a single mistyped or
transposed character is caught locally, before the identifier is looked
up anywhere. Use it for identifiers a human is expected to read back or
retype, e.g. a support code, license key, or invoice number; use
``generate_ulid`` instead for identifiers only handled by machines,
since the check symbol trades away one character of entropy for the
error-detection guarantee.
"""

from __future__ import annotations

__all__ = ["generate_checksummed_id", "verify_checksummed_id"]

import os

from coola.identifier import crockford
from coola.validation import validate_gt, validate_int, validate_isinstance

# The 5 extra check symbols from the Crockford spec, extending the
# alphabet from 32 to the 37 values needed for a mod-37 checksum.
# ``verify_checksummed_id`` accepts them, ``generate_checksummed_id``
# never emits them (they are awkward to type in a shell, a URL or a
# spreadsheet).
_CHECK_SYMBOLS = "*~$=U"

_DEFAULT_LENGTH = 12
_MAX_LENGTH = 1024
_DEFAULT_GROUP_SIZE = 4

# Crockford Base32 decoding is case-insensitive and normalizes the
# characters most often confused when hand-transcribed: 'O' with '0',
# and 'I'/'L' with '1'. Applied to the check symbol before verification,
# since none of the mapped letters collide with the '*~$=U' check symbols.
_NORMALIZE_TABLE = str.maketrans("oOiIlL", "001111")

# The characters (upper case) that a separator cannot use: the decoder reads them as digits.
_DIGIT_CHARACTERS = frozenset(crockford.ALPHABET + "OIL")


def _normalize(identifier: str) -> str:
    r"""Apply the Crockford Base32 case/confusable normalization to
    ``identifier``.

    Args:
        identifier: The identifier.

    Returns:
        The identifier in upper case, with the confusable characters
            replaced.
    """
    return identifier.upper().translate(_NORMALIZE_TABLE)


def _checksum_symbol(payload: str) -> str:
    r"""Compute the mod-37 Crockford check symbol for ``payload``.

    Args:
        payload: The payload, in the Crockford Base32 alphabet.

    Returns:
        The check symbol.

    Raises:
        ValueError: If ``payload`` contains a character outside the
            Crockford Base32 alphabet.
    """
    value = crockford.decode(payload, name="payload")
    remainder = value % 37
    if remainder < 32:
        return crockford.ALPHABET[remainder]
    return _CHECK_SYMBOLS[remainder - 32]


def _validate_sep(sep: str) -> None:
    r"""Validate that ``sep`` is at most one character and is not read
    as a digit, which would make the grouped output ambiguous to parse
    back: the separator is stripped before the check.

    Args:
        sep: The separator of the groups.

    Raises:
        TypeError: If ``sep`` is not a ``str``.
        ValueError: If ``sep`` is more than one character long, or
            is a character of the Crockford Base32 alphabet (the case
            is ignored, ``O``, ``I`` and ``L`` are read as digits).
    """
    validate_isinstance(sep, str, name="sep")
    if len(sep) > 1:
        msg = f"sep must be empty or a single character, got {sep!r}"
        raise ValueError(msg)
    if any(char.upper() in _DIGIT_CHARACTERS for char in sep):
        msg = (
            "sep must not contain a character from the Crockford Base32 "
            f"alphabet ({crockford.ALPHABET!r}, case-insensitive, 'O', 'I' and 'L' included), "
            f"got {sep!r}"
        )
        raise ValueError(msg)


def generate_checksummed_id(
    *, length: int = _DEFAULT_LENGTH, group_size: int = _DEFAULT_GROUP_SIZE, sep: str = "-"
) -> str:
    r"""Generate a random identifier with a trailing check symbol.

    Draws ``length`` random Crockford Base32 characters, appends one
    check symbol computed from them (a mod-37 checksum, per the
    Crockford Base32 spec), and groups the result into chunks of
    ``group_size`` characters separated by ``sep`` for readability.

    The check symbol is always a character of the 32-character
    alphabet (``0-9A-Z`` without ``I``, ``L``, ``O`` and ``U``): when the
    checksum of a payload is one of the 5 extra symbols of the
    specification (``*~$=U``), the payload is drawn again (5 payloads in
    37, so 1.16 draws on average). The identifier is therefore safe in
    a shell, a URL and a spreadsheet, at the cost of 0.21 bit of entropy.

    Args:
        length: The number of random (non-check) characters to
            generate. Must be positive.
        group_size: The number of characters per group in the
            formatted output. Must be positive. Pass a number greater
            than or equal to ``length + 1`` (or ``sep=""``) to disable
            grouping.
        sep: The separator inserted between groups. Must be empty or
            a single character that is not in the Crockford Base32
            alphabet (the case is ignored), or ``verify_checksummed_id``
            would not be able to tell a separator character apart from
            a payload/check one.

    Returns:
        A string of ``length`` random characters plus one check
        symbol, grouped by ``group_size`` and joined by ``sep``.

    Raises:
        TypeError: If ``length`` or ``group_size`` is not an ``int`` (a
            ``bool`` is rejected), or ``sep`` is not a ``str``.
        ValueError: If ``length`` or ``group_size`` is not positive,
            if ``length`` exceeds 1024, or if ``sep`` is more than one
            character long or a character of the Crockford Base32
            alphabet.

    Example:
        ```pycon
        >>> from coola.identifier import generate_checksummed_id, verify_checksummed_id
        >>> checksummed_id = generate_checksummed_id()
        >>> verify_checksummed_id(checksummed_id)
        True
        >>> len(generate_checksummed_id(length=8, group_size=3, sep=" "))
        11

        ```
    """
    validate_int(length, name="length")
    validate_gt(length, 0, name="length")
    if length > _MAX_LENGTH:
        msg = f"length must be at most {_MAX_LENGTH}, got {length}"
        raise ValueError(msg)
    validate_int(group_size, name="group_size")
    validate_gt(group_size, 0, name="group_size")
    _validate_sep(sep)
    while True:
        payload = "".join(crockford.ALPHABET[b % 32] for b in os.urandom(length))
        symbol = _checksum_symbol(payload)
        if symbol not in _CHECK_SYMBOLS:
            break
    full = payload + symbol
    return sep.join(full[i : i + group_size] for i in range(0, len(full), group_size))


def verify_checksummed_id(identifier: str, *, sep: str = "-") -> bool:
    r"""Verify the check symbol of an identifier from
    ``generate_checksummed_id``.

    Per the Crockford Base32 spec, decoding is case-insensitive and
    normalizes the characters most often confused when an identifier is
    hand-transcribed: ``'O'`` with ``'0'``, and ``'I'``/``'L'`` with
    ``'1'``. ``identifier`` is normalized this way before its check
    symbol is verified, so e.g. a lowercase retype or an ``'O'`` typed
    for a ``'0'`` still verifies correctly. The 5 extra check symbols of
    the specification (``*~$=U``) are accepted, so an identifier from
    another implementation verifies.

    Args:
        identifier: The identifier to verify, in the grouped form
            returned by ``generate_checksummed_id``.
        sep: The group separator used when ``identifier`` was
            generated.

    Returns:
        ``True`` if the trailing character is a valid check symbol for
        the characters preceding it, ``False`` otherwise (including
        when ``identifier`` is not a ``str``, contains a non-ASCII
        character or a character outside the extended Crockford Base32
        alphabet, or is too short to contain a payload and a check
        symbol).

    Raises:
        TypeError: If ``sep`` is not a ``str``.
        ValueError: If ``sep`` is more than one character long or a
            character of the Crockford Base32 alphabet.

    Example:
        ```pycon
        >>> from coola.identifier import generate_checksummed_id, verify_checksummed_id
        >>> verify_checksummed_id(generate_checksummed_id())
        True
        >>> verify_checksummed_id("not-a-valid-id")
        False
        >>> verify_checksummed_id(generate_checksummed_id().lower())
        True

        ```
    """
    _validate_sep(sep)
    if not isinstance(identifier, str) or not identifier.isascii():
        # Non-ASCII look-alikes (the dotless i upper-cases to ``'I'``) must not verify.
        return False
    full = identifier.replace(sep, "") if sep else identifier
    if len(full) < 2:
        return False
    try:
        expected = _checksum_symbol(full[:-1])
    except ValueError:
        return False
    return _normalize(full[-1]) == expected
