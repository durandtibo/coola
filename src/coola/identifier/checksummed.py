r"""Provide a human-transcribable identifier with a check symbol.

None of the other generators in this package guard against transcription
errors: a single mistyped character in a UUID, ULID, or Nano ID is
silently accepted as a different, valid-looking identifier. This module
adds a check symbol (following the optional check-symbol scheme from the
[Crockford Base32](
https://www.crockford.com/base32.html)
specification also used by ``generate_ulid``) so a single mistyped or
transposed character is caught locally, before the identifier is looked
up anywhere. Use it for identifiers a human is expected to read back or
retype, e.g. a support code, license key, or invoice number; use
``generate_ulid`` or ``generate_nano_id`` instead for identifiers only
handled by machines, since the check symbol trades away one character
of entropy for the error-detection guarantee.
"""

from __future__ import annotations

__all__ = ["generate_checksummed_id", "verify_checksummed_id"]

import os

from coola.identifier.validation import validate_positive

# Crockford's Base32 alphabet, matching coola.identifier.ulid.
_ENCODING = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
# The 5 extra check symbols from the Crockford spec, extending the
# alphabet from 32 to the 37 values needed for a mod-37 checksum.
_CHECK_SYMBOLS = "*~$=U"

_DEFAULT_LENGTH = 12
_DEFAULT_GROUP_SIZE = 4

# Crockford Base32 decoding is case-insensitive and normalizes the
# characters most often confused when hand-transcribed: 'O' with '0',
# and 'I'/'L' with '1'. Applied to the whole identifier (check symbol
# included) before verification, since none of the mapped letters
# collide with the '*~$=U' check symbols.
_NORMALIZE_TABLE = str.maketrans("oOiIlL", "001111")


def _normalize(identifier: str) -> str:
    r"""Apply the Crockford Base32 case/confusable normalization to
    ``identifier``."""
    return identifier.upper().translate(_NORMALIZE_TABLE)


def _checksum_symbol(payload: str) -> str:
    r"""Compute the mod-37 Crockford check symbol for ``payload``.

    Raises:
        ValueError: If ``payload`` contains a character outside the
            Crockford Base32 alphabet.
    """
    value = 0
    for char in payload:
        try:
            digit = _ENCODING.index(char)
        except ValueError as error:
            msg = (
                f"payload contains a character outside the Crockford Base32 alphabet, got {char!r}"
            )
            raise ValueError(msg) from error
        value = value * 32 + digit
    remainder = value % 37
    if remainder < 32:
        return _ENCODING[remainder]
    return _CHECK_SYMBOLS[remainder - 32]


def _validate_sep(sep: str) -> None:
    r"""Validate that ``sep`` does not overlap the extended Crockford
    Base32 alphabet, which would make grouped output ambiguous to parse
    back.

    Raises:
        ValueError: If ``sep`` contains a character from the extended
            Crockford Base32 alphabet.
    """
    alphabet = _ENCODING + _CHECK_SYMBOLS
    if any(char in alphabet for char in sep):
        msg = (
            "sep must not contain a character from the extended Crockford Base32 "
            f"alphabet ({alphabet!r}), got {sep!r}"
        )
        raise ValueError(msg)


def generate_checksummed_id(
    length: int = _DEFAULT_LENGTH, group_size: int = _DEFAULT_GROUP_SIZE, sep: str = "-"
) -> str:
    r"""Generate a random identifier with a trailing check symbol.

    Draws ``length`` random Crockford Base32 characters, appends one
    check symbol computed from them (a mod-37 checksum, per the
    Crockford Base32 spec), and groups the result into chunks of
    ``group_size`` characters separated by ``sep`` for readability.

    Args:
        length: The number of random (non-check) characters to
            generate. Must be positive.
        group_size: The number of characters per group in the
            formatted output. Must be positive. Pass a number greater
            than or equal to ``length + 1`` (or ``sep=""``) to disable
            grouping.
        sep: The separator inserted between groups. Must not contain a
            character from the extended Crockford Base32 alphabet, or
            ``verify_checksummed_id`` would not be able to tell a
            separator character apart from a payload/check one.

    Returns:
        A string of ``length`` random characters plus one check
        symbol, grouped by ``group_size`` and joined by ``sep``.

    Raises:
        ValueError: If ``length`` or ``group_size`` is not positive,
            or if ``sep`` contains a character from the extended
            Crockford Base32 alphabet.

    Example:
        ```pycon
        >>> from coola.identifier import generate_checksummed_id, verify_checksummed_id
        >>> checksummed_id = generate_checksummed_id()
        >>> verify_checksummed_id(checksummed_id)
        True
        >>> verify_checksummed_id(checksummed_id[:-1] + "0")  # doctest: +SKIP
        False

        ```
    """
    validate_positive(length, name="length")
    validate_positive(group_size, name="group_size")
    _validate_sep(sep)
    payload = "".join(_ENCODING[b % 32] for b in os.urandom(length))
    full = payload + _checksum_symbol(payload)
    return sep.join(full[i : i + group_size] for i in range(0, len(full), group_size))


def verify_checksummed_id(identifier: str, sep: str = "-") -> bool:
    r"""Verify the check symbol of an identifier from
    ``generate_checksummed_id``.

    Args:
        identifier: The identifier to verify, in the grouped form
            returned by ``generate_checksummed_id``.
        sep: The group separator used when ``identifier`` was
            generated.

    Per the Crockford Base32 spec, decoding is case-insensitive and
    normalizes the characters most often confused when an identifier is
    hand-transcribed: ``'O'`` with ``'0'``, and ``'I'``/``'L'`` with
    ``'1'``. ``identifier`` is normalized this way before its check
    symbol is verified, so e.g. a lowercase retype or an ``'O'`` typed
    for a ``'0'`` still verifies correctly.

    Returns:
        ``True`` if the trailing character is a valid check symbol for
        the characters preceding it, ``False`` otherwise (including
        when ``identifier`` contains a character outside the extended
        Crockford Base32 alphabet, or is too short to contain a
        payload and a check symbol).

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
    full = _normalize(identifier.replace(sep, "") if sep else identifier)
    if len(full) < 2:
        return False
    payload, check = full[:-1], full[-1]
    if any(char not in _ENCODING for char in payload):
        return False
    # payload was just validated above, so this cannot raise.
    return check == _checksum_symbol(payload)
