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

# Crockford's Base32 alphabet, matching coola.identifier.ulid.
_ENCODING = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
# The 5 extra check symbols from the Crockford spec, extending the
# alphabet from 32 to the 37 values needed for a mod-37 checksum.
_CHECK_SYMBOLS = "*~$=U"

_DEFAULT_LENGTH = 12
_DEFAULT_GROUP_SIZE = 4


def _checksum_symbol(payload: str) -> str:
    r"""Compute the mod-37 Crockford check symbol for ``payload``."""
    value = 0
    for char in payload:
        value = value * 32 + _ENCODING.index(char)
    remainder = value % 37
    if remainder < 32:
        return _ENCODING[remainder]
    return _CHECK_SYMBOLS[remainder - 32]


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
        sep: The separator inserted between groups.

    Returns:
        A string of ``length`` random characters plus one check
        symbol, grouped by ``group_size`` and joined by ``sep``.

    Raises:
        ValueError: If ``length`` or ``group_size`` is not positive.

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
    if length <= 0:
        msg = f"length must be positive, got {length}"
        raise ValueError(msg)
    if group_size <= 0:
        msg = f"group_size must be positive, got {group_size}"
        raise ValueError(msg)
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

        ```
    """
    full = identifier.replace(sep, "") if sep else identifier
    if len(full) < 2:
        return False
    payload, check = full[:-1], full[-1]
    if any(char not in _ENCODING for char in payload):
        return False
    # payload was just validated above, so this cannot raise.
    return check == _checksum_symbol(payload)
