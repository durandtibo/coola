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
import re
import uuid

from coola.identifier.sequence import TickSequence
from coola.validation import validate_closed_range, validate_int, validate_str

_TIMESTAMP_BITS = 48
_MAX_TIMESTAMP_MS = (1 << _TIMESTAMP_BITS) - 1
_UUID7_PATTERN = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

# State of the UUIDv7 generated with the current time: they increase within a millisecond.
# A UUIDv7 has 74 random bits (12 + 62), the others are the version and the variant.
_SEQUENCE = TickSequence(bits=74)


def generate_uuid7(*, timestamp_ms: int | None = None) -> str:
    r"""Generate a UUIDv7 (RFC 9562) identifier.

    A UUIDv7 packs a 48-bit millisecond timestamp, followed by the
    4-bit version, 12 bits of randomness, the 2-bit variant, and 62
    more bits of randomness, into a standard 128-bit UUID layout.
    Because the timestamp is the most significant part, UUIDv7 values
    generated later sort (lexicographically, as canonical UUID
    strings) after UUIDv7 values generated earlier, unlike
    ``uuid.uuid4`` which sorts randomly.

    Note:
        When the current time is used (``timestamp_ms=None``), UUIDv7
        values are monotonic within a process (the "method 1"
        counter of RFC 9562): values generated in the same
        millisecond (or after the clock moved backward) keep the
        timestamp and increment the 74 random bits by one, so they
        still sort in the order of generation, which keeps database
        indexes append-only. The state is per process, protected by a
        lock, and reset in the child process of a fork. Across
        processes, values generated in the same millisecond are not
        ordered. When ``timestamp_ms`` is given, the random bits are
        always random: two values with the same timestamp are not
        ordered.

    Args:
        timestamp_ms: The Unix timestamp in milliseconds to encode. If
            ``None`` (default), the current time is used and the UUIDs
            are monotonic (see below). Exposed mainly for
            deterministic testing and for backfilling.

    Returns:
        A lowercase UUID string of the form
        ``'xxxxxxxx-xxxx-7xxx-yxxx-xxxxxxxxxxxx'``.

    Raises:
        TypeError: If ``timestamp_ms`` is not an ``int`` (a ``bool`` is
            rejected).
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
    if timestamp_ms is None:
        timestamp_ms, rand_bits = _SEQUENCE.next()
        if timestamp_ms > _MAX_TIMESTAMP_MS:  # the tick of the sequence is never negative
            validate_closed_range(timestamp_ms, 0, _MAX_TIMESTAMP_MS, name="timestamp_ms")
        # Spread the 74 monotonic random bits around the version and variant bits:
        # 12 bits (rand_a) before the variant, and 62 bits (rand_b) after.
        rand_a, rand_b = rand_bits >> 62, rand_bits & ((1 << 62) - 1)
        value = (timestamp_ms << 80) | (rand_a << 64) | rand_b
    else:
        validate_int(timestamp_ms, name="timestamp_ms")
        validate_closed_range(timestamp_ms, 0, _MAX_TIMESTAMP_MS, name="timestamp_ms")
        # 48-bit timestamp, followed by 80 bits of randomness that the
        # version/variant bits below are then stamped onto.
        value = (timestamp_ms << 80) | int.from_bytes(os.urandom(10), byteorder="big")

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
        TypeError: If ``uuid7`` is not a ``str`` (pass ``str(uuid)`` for
            a ``uuid.UUID``).
        ValueError: If ``uuid7`` is not a UUID in its canonical
            36-character form (no braces, no ``urn:uuid:`` prefix, no
            missing hyphens), or is not a version 7 UUID of the RFC 9562
            variant. The case is ignored.

    Example:
        ```pycon
        >>> from coola.identifier import extract_uuid7_timestamp_ms, generate_uuid7
        >>> extract_uuid7_timestamp_ms(generate_uuid7(timestamp_ms=1704067200000))
        1704067200000

        ```
    """
    validate_str(uuid7, name="uuid7")
    if _UUID7_PATTERN.fullmatch(uuid7) is None:
        msg = f"uuid7 must be a UUID in its canonical 36-character form, got {uuid7!r}"
        raise ValueError(msg)
    if uuid7[14] != "7":
        msg = f"uuid7 must be a version 7 UUID, got version {uuid7[14]!r} ({uuid7!r})"
        raise ValueError(msg)
    if uuid7[19] not in "89abAB":
        msg = f"uuid7 must have the RFC 9562 variant, got variant {uuid7[19]!r} ({uuid7!r})"
        raise ValueError(msg)
    return int(uuid7[:8] + uuid7[9:13], 16)
