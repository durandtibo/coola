r"""Provide a ULID identifier for time-ordered records.

Unlike ``generate_stable_uuid`` and ``generate_stable_content_id``, a
ULID is not derived from data content: two calls with the same input
produce different values. Use it when what you need is a unique,
lexicographically sortable-by-creation-time identifier (e.g. a record
ID that should sort roughly by insertion order), not a reproducible
identifier for deduplication or caching.
"""

from __future__ import annotations

__all__ = ["generate_ulid"]

import os
import time

# Crockford's Base32 alphabet (excludes I, L, O, U to avoid
# transcription ambiguity), as specified by the ULID spec.
_ENCODING = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def generate_ulid(timestamp_ms: int | None = None) -> str:
    r"""Generate a ULID (Universally Unique Lexicographically Sortable
    Identifier).

    A ULID packs a 48-bit millisecond timestamp followed by 80 bits of
    randomness into a 26-character Crockford Base32 string. Because the
    timestamp is the most significant part, ULIDs generated later sort
    (lexicographically, as plain strings) after ULIDs generated
    earlier, unlike ``uuid.uuid4`` which sorts randomly.

    Args:
        timestamp_ms: The Unix timestamp in milliseconds to encode. If
            ``None`` (default), the current time is used. Exposed
            mainly for deterministic testing.

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
        timestamp_ms = time.time_ns() // 1_000_000
    if not 0 <= timestamp_ms <= 0xFFFFFFFFFFFF:
        msg = f"timestamp_ms must fit in 48 bits (0 to 2**48 - 1), got {timestamp_ms}"
        raise ValueError(msg)
    payload = timestamp_ms.to_bytes(6, byteorder="big") + os.urandom(10)
    return _encode_base32(payload)


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
