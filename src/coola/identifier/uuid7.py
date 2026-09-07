r"""Provide a UUIDv7 identifier for time-ordered records.

Like ``generate_ulid``, a UUIDv7 is not derived from data content: two
calls produce different values even for identical input. Unlike
``generate_ulid``, the result is a valid UUID string (RFC 9562), so it
fits a UUID-typed database column or any API expecting ``uuid.UUID``
formatting, something neither ``generate_ulid`` nor
``generate_stable_content_id`` provide. Prefer ``generate_ulid`` when
UUID format compliance does not matter, since it packs more randomness
(80 bits) than UUIDv7's 74 bits of usable random space.
"""

from __future__ import annotations

__all__ = ["extract_uuid7_timestamp_ms", "generate_uuid7"]

import os
import uuid

from coola.identifier.validation import resolve_timestamp_ms


def generate_uuid7(timestamp_ms: int | None = None) -> str:
    r"""Generate a UUIDv7 (RFC 9562) identifier.

    A UUIDv7 packs a 48-bit millisecond timestamp, followed by the
    4-bit version, 12 bits of randomness, the 2-bit variant, and 62
    more bits of randomness, into a standard 128-bit UUID layout.
    Because the timestamp is the most significant part, UUIDv7 values
    generated later sort (lexicographically, as canonical UUID
    strings) after UUIDv7 values generated earlier, unlike
    ``uuid.uuid4`` which sorts randomly.

    Args:
        timestamp_ms: The Unix timestamp in milliseconds to encode. If
            ``None`` (default), the current time is used. Exposed
            mainly for deterministic testing.

    Returns:
        A lowercase UUID string of the form
        ``'xxxxxxxx-xxxx-7xxx-yxxx-xxxxxxxxxxxx'``.

    Raises:
        ValueError: If ``timestamp_ms`` does not fit in 48 bits (i.e.
            is negative or exceeds ``2**48 - 1``).

    Example:
        ```pycon
        >>> from coola.identifier import generate_uuid7
        >>> uuid7 = generate_uuid7()
        >>> len(uuid7)
        36

        ```
    """
    timestamp_ms = resolve_timestamp_ms(timestamp_ms)

    rand = os.urandom(10)
    # 48-bit timestamp, followed by 80 bits of randomness that the
    # version/variant bits below are then stamped onto.
    value = (timestamp_ms << 80) | int.from_bytes(rand, byteorder="big")

    # Stamp the 4-bit version (7) into bits 76-79 (the top nibble of
    # the 16 bits following the timestamp).
    value &= ~(0xF << 76)
    value |= 0x7 << 76

    # Stamp the 2-bit variant (10) into bits 62-63 (the top two bits of
    # the 64 bits following the version/rand_a field).
    value &= ~(0b11 << 62)
    value |= 0b10 << 62

    return str(uuid.UUID(int=value))


def extract_uuid7_timestamp_ms(uuid7: str) -> int:
    r"""Extract the millisecond timestamp encoded in a UUIDv7.

    Inverse of the encoding done by ``generate_uuid7``: the timestamp
    is the top 48 bits of the UUID, unaffected by the version/variant
    bits stamped into the lower 80 bits.

    Args:
        uuid7: The UUIDv7 string previously returned by
            ``generate_uuid7``.

    Returns:
        The Unix timestamp in milliseconds that was encoded in
        ``uuid7``.

    Raises:
        ValueError: If ``uuid7`` is not a valid UUID string, or is not
            a version 7 UUID (RFC 9562 variant, version nibble ``7``).

    Example:
        ```pycon
        >>> from coola.identifier import extract_uuid7_timestamp_ms, generate_uuid7
        >>> extract_uuid7_timestamp_ms(generate_uuid7(timestamp_ms=1704067200000))
        1704067200000

        ```
    """
    parsed = uuid.UUID(uuid7)
    if parsed.version != 7:
        msg = f"uuid7 must be a version 7 UUID, got version {parsed.version} ({uuid7!r})"
        raise ValueError(msg)
    return parsed.int >> 80
