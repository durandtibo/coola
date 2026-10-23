r"""Provide a ULID identifier for time-ordered records.

Unlike ``generate_stable_uuid5`` and ``generate_stable_content_id``, a
ULID is not derived from data content: two calls with the same input
produce different values. Use it when what you need is a unique,
lexicographically sortable-by-creation-time identifier (e.g. a record ID
that should sort roughly by insertion order), not a reproducible
identifier for deduplication or caching.
"""

from __future__ import annotations

__all__ = ["extract_ulid_timestamp_ms", "generate_ulid"]

import os

from coola.identifier.monotonic import MonotonicRandom
from coola.identifier.validation import (
    CROCKFORD_BASE32_ALPHABET as _ENCODING,
    decode_crockford_base32,
    validate_timestamp_ms,
)

# State of the ULIDs generated with the current time: they increase within a millisecond.
_MONOTONIC = MonotonicRandom(bits=80)


def generate_ulid(timestamp_ms: int | None = None) -> str:
    r"""Generate a ULID (Universally Unique Lexicographically Sortable
    Identifier).

    A ULID packs a 48-bit millisecond timestamp followed by 80 bits of
    randomness into a 26-character Crockford Base32 string. Because the
    timestamp is the most significant part, ULIDs generated later sort
    (lexicographically, as plain strings) after ULIDs generated
    earlier, unlike ``uuid.uuid4`` which sorts randomly.

    Note:
        When the current time is used (``timestamp_ms=None``), ULIDs
        are monotonic within a process, as recommended by the ULID
        specification: ULIDs generated in the same millisecond (or
        after the clock moved backward) keep the timestamp and
        increment the random part by one, so they still sort in the
        order of generation. The state is per process, protected by a
        lock, and reset in the child process of a fork. Across
        processes, ULIDs generated in the same millisecond are not
        ordered. When ``timestamp_ms`` is given, the random part is
        always random: two ULIDs with the same timestamp are not
        ordered.

    Args:
        timestamp_ms: The Unix timestamp in milliseconds to encode. If
            ``None`` (default), the current time is used and the ULIDs
            are monotonic (see below). Exposed mainly for
            deterministic testing and for backfilling.

    Returns:
        A 26-character uppercase Crockford Base32 ULID string.

    Raises:
        ValueError: If ``timestamp_ms`` does not fit in 48 bits (i.e.
            is negative or exceeds ``2**48 - 1``).

    Example:
        ```pycon
        >>> from coola.identifier import generate_ulid
        >>> ulid = generate_ulid()
        >>> len(ulid)
        26

        ```
    """
    if timestamp_ms is None:
        timestamp_ms, randomness = _MONOTONIC.next()
        validate_timestamp_ms(timestamp_ms)
        payload = timestamp_ms.to_bytes(6, byteorder="big") + randomness.to_bytes(
            10, byteorder="big"
        )
    else:
        validate_timestamp_ms(timestamp_ms)
        payload = timestamp_ms.to_bytes(6, byteorder="big") + os.urandom(10)
    return _encode_base32(payload)


def extract_ulid_timestamp_ms(ulid: str) -> int:
    r"""Extract the millisecond timestamp encoded in a ULID.

    Inverse of the encoding done by ``generate_ulid``: decodes the
    26-character Crockford Base32 string back to its 128-bit value and
    returns the top 48 bits, which is the timestamp ``generate_ulid``
    packed in.

    Args:
        ulid: The ULID string previously returned by ``generate_ulid``.

    Returns:
        The Unix timestamp in milliseconds that was encoded in
        ``ulid``.

    Raises:
        ValueError: If ``ulid`` is not 26 characters long, or contains
            a character outside the Crockford Base32 alphabet used by
            ``generate_ulid`` (the case is ignored), or encodes a value
            larger than 128 bits.

    Example:
        ```pycon
        >>> from coola.identifier import extract_ulid_timestamp_ms, generate_ulid
        >>> extract_ulid_timestamp_ms(generate_ulid(timestamp_ms=1704067200000))
        1704067200000

        ```
    """
    if len(ulid) != 26:
        msg = f"ulid must be 26 characters long, got {len(ulid)}"
        raise ValueError(msg)
    value = decode_crockford_base32(ulid, name="ulid")
    if value >> 128:
        msg = f"ulid is larger than 2**128 - 1 (the first character must be at most '7'), got {ulid!r}"
        raise ValueError(msg)
    return value >> 80


def _encode_base32(payload: bytes) -> str:
    r"""Encode a 16-byte payload as a 26-character Crockford Base32
    string.

    Args:
        payload: The 128 bits (16 bytes) to encode.

    Returns:
        The Crockford Base32 encoding of ``payload``.
    """
    value = int.from_bytes(payload, byteorder="big")
    chars = [""] * 26
    for i in range(25, -1, -1):
        chars[i] = _ENCODING[value & 0x1F]
        value >>= 5
    return "".join(chars)
